"""记忆装配核心。

三段式记忆（开窗注入，窗口生命周期内冻结，保证 DeepSeek 前缀缓存稳定）：
  一、近期每日 —— 最近 RECENT_DAYS 天的详细概括
  二、月度概览 —— 最近 RECENT_MONTHS 个月的概览
  三、季度概览 —— 最近 RECENT_QUARTERS 个季的概览
  四、年度概览 —— 最近 RECENT_YEARS 年的概览
  五、长期记忆 —— 旧数据兜底（OB 搬运 + 早期合并的概括）
  六、上个窗口原始聊天记录 —— 最后 LAST_WINDOW_MESSAGES 条原文（零加工）

防幻觉原则：
  - OB 搬运和原始记录是"原样复制"，不经过任何 LLM；
  - 各级概括用 deepseek-flash + 防幻觉铁律 prompt，且每层有字数/ token 上限。
"""
import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import config
import ob_client
import store
import summarizer

log = logging.getLogger(__name__)
TZ = ZoneInfo(config.TIMEZONE)

_windows: dict = {}  # window_key -> {"block": str|None, "ts": float}


def _now() -> datetime:
    return datetime.now(TZ)


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text", ""))
            elif isinstance(p, str):
                parts.append(p)
        return "\n".join(parts)
    return ""


def analyze_messages(messages: list) -> dict:
    """提取窗口特征：system 文本、第一条用户消息、assistant 消息条数。"""
    system_text = ""
    for m in messages or []:
        if m.get("role") == "system":
            system_text = _text_of(m.get("content"))
            break
    first_user = ""
    assistant_count = 0
    for m in messages or []:
        role = m.get("role")
        if role == "assistant":
            assistant_count += 1
        if role == "user" and not first_user:
            first_user = _text_of(m.get("content")).strip()
    return {"system": system_text, "first_user": first_user, "assistant_count": assistant_count}


def _window_key(system_text: str, first_user: str) -> str:
    return hashlib.sha256(((system_text or "") + "\x00" + (first_user or "")).encode("utf-8")).hexdigest()[:16]


def _parse_walltime(value) -> "datetime | None":
    """解析 App 写入的"墙上时钟"字符串（yyyy-MM-dd HH:mm:ss，被 Postgres 错标为 UTC）。
    返回 naive datetime，用于与 _now() 的墙上时钟直接比较（不做时区换算）。"""
    s = str(value or "").strip()
    if not s:
        return None
    try:
        return datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")
    except Exception:
        try:
            return datetime.strptime(s[:10], "%Y-%m-%d")
        except Exception:
            return None


async def previous_window_rows(first_user_text: str) -> tuple[list, bool]:
    """从 Supabase 最近的聊天记录里找到"上一个窗口"，取它的最后 N 条。

    返回 (rows, confident_no_history)：
    - confident_no_history=True 表示库里除当前窗口外没有任何其他会话，
      "没有上个窗口"是事实而非同步延迟（用于缓存完整判断，避免每10分钟空重建）。
    """
    rows = await store.fetch_recent_chats(limit=600)
    rows = [r for r in rows
            if r.get("role") in ("user", "assistant") and (r.get("content") or "").strip()]
    if not rows:
        return [], True
    # 同一秒同内容的重复行去重
    seen, dedup = set(), []
    for r in rows:
        k = (r.get("role"), r.get("content"), str(r.get("created_at")))
        if k in seen:
            continue
        seen.add(k)
        dedup.append(r)
    rows = dedup

    # 用当前窗口的第一条用户消息，在最近同步的记录里定位"当前窗口"的会话。
    now_wall = _now().replace(tzinfo=None)
    max_age = timedelta(hours=config.CUR_WINDOW_MAX_AGE_HOURS)
    cur_conv = None
    probe = (first_user_text or "").strip()[:50]
    if probe:
        for r in reversed(rows):
            if r.get("role") != "user":
                continue
            if not str(r.get("content") or "").strip().startswith(probe):
                continue
            wt = _parse_walltime(r.get("created_at"))
            if wt is None or (now_wall - wt) > max_age:
                continue  # 太老的匹配视为误匹配
            cur_conv = r.get("conversation_id") or "__none__"
            break

    # 按会话分组（保持时间顺序）
    order, convs = [], {}
    for r in rows:
        c = r.get("conversation_id") or "__none__"
        if c not in convs:
            convs[c] = []
            order.append(c)
        convs[c].append(r)

    prev = None
    if cur_conv is not None:
        for c in reversed(order):  # 从最新往回找，第一个不是当前窗口的会话
            if c != cur_conv:
                prev = c
                break
        if prev is None:
            return [], True  # 库里只有当前窗口自己：确认无历史
    else:
        if not order:
            return [], True  # 库里没有任何会话：确认无历史
        prev = order[-1] if order else None  # 当前窗口还没同步上去：最新会话就是上个窗口
    if prev is None:
        return [], True
    return convs[prev][-config.LAST_WINDOW_MESSAGES:], False


def build_block_text(longterm, dailies, monthlies, quarterlies, yearlies, last_rows) -> str:
    parts = []
    if dailies:
        seg = []
        for d in dailies:
            c = (d.get("content") or "").strip()
            if c:
                seg.append(f"〔{d.get('scope', '')}〕\n{c}")
        if seg:
            parts.append("=== 一、近期每日记忆（最近几天，最详细） ===\n" + "\n\n".join(reversed(seg)))
    if monthlies:
        seg = []
        for m in monthlies:
            c = (m.get("content") or "").strip()
            if c:
                seg.append(f"〔{m.get('scope', '')}〕\n{c}")
        if seg:
            parts.append("=== 二、月度概览（最近几个月） ===\n" + "\n\n".join(reversed(seg)))
    if quarterlies:
        seg = []
        for q in quarterlies:
            c = (q.get("content") or "").strip()
            if c:
                seg.append(f"〔{q.get('scope', '')}〕\n{c}")
        if seg:
            parts.append("=== 三、季度概览 ===\n" + "\n\n".join(reversed(seg)))
    if yearlies:
        seg = []
        for y in yearlies:
            c = (y.get("content") or "").strip()
            if c:
                seg.append(f"〔{y.get('scope', '')}〕\n{c}")
        if seg:
            parts.append("=== 四、年度概览 ===\n" + "\n\n".join(reversed(seg)))
    if (longterm or "").strip():
        parts.append("=== 五、长期记忆（更早的历史兜底） ===\n" + longterm.strip())
    if last_rows:
        parts.append(
            f"=== 六、上个窗口聊天记录（最后 {len(last_rows)} 条原文摘录） ===\n"
            + summarizer.transcript(last_rows)
        )
    if not parts:
        return ""
    header = (
        "【记忆系统注入】以下是你自己的记忆，由系统在开窗时自动提供，都是真实发生过的事。"
        "直接当作你的记忆使用，不需要再调用任何外部记忆、聊天记录读取工具。"
    )
    return header + "\n\n" + "\n\n".join(parts)


async def _build_block(first_user_text: str) -> tuple[str, bool]:
    """构建记忆块。返回 (block, complete)：complete 表示"没有第三段"是确定事实
    （库里确实没有历史），而非同步延迟——用于冻结缓存判断。"""
    if not config.supabase_ready():
        return "", True
    longterm = await store.get_longterm()
    min_day = (_now() - timedelta(days=config.RECENT_DAYS - 1)).strftime("%Y-%m-%d")
    dailies = await store.get_recent_dailies(min_day)
    monthlies = await store.get_memories("monthly", "", config.RECENT_MONTHS, desc=True)
    quarterlies = await store.get_memories("quarterly", "", config.RECENT_QUARTERS, desc=True)
    yearlies = await store.get_memories("yearly", "", config.RECENT_YEARS, desc=True)
    last_rows, confident_no_history = await previous_window_rows(first_user_text)
    block = build_block_text(longterm, dailies, monthlies, quarterlies, yearlies, last_rows)
    return block, (bool(block) or confident_no_history)


async def get_window_block(system_text: str, first_user_text: str, assistant_count: int = 0):
    """返回 (block, window_key)。窗口生命周期内冻结：活跃窗口滑动续期，永不中途重建，
    保证前缀缓存稳定；空结果 10 分钟后允许重试。

    bugfix（Bug 2）：缓存命中但当前请求 0 条 assistant 回复、且缓存已超过
    NEW_WINDOW_REDETECT_SECONDS —— 说明是"用相同开场白开的新窗"（真正活跃的
    窗口在几分钟内几乎必然产生过 assistant 回复），强制重建，避免把上上个
    窗口的内容当成"上个窗口"注入。
    """
    key = _window_key(system_text, first_user_text)
    now = time.time()
    hit = _windows.get(key)
    if hit:
        complete = bool(hit.get("complete"))
        ttl = config.WINDOW_TTL_HOURS * 3600 if complete else 600
        if now - hit["ts"] < ttl:
            stale_new_window = (
                assistant_count == 0
                and (now - hit["ts"]) > config.NEW_WINDOW_REDETECT_SECONDS
            )
            if not stale_new_window:
                if complete:
                    hit["ts"] = now  # 活跃窗口滑动续期
                return hit["block"], key
    block, complete = "", False
    try:
        block, complete = await _build_block(first_user_text)
    except Exception:
        log.exception("构建窗口记忆失败（本次请求将不注入）")
        block, complete = "", False
    _windows[key] = {"block": block or None, "ts": now, "complete": bool(complete)}
    if len(_windows) > 200:
        # bugfix（Bug 4）：按最近使用时间驱逐（LRU），而不是 FIFO。
        oldest_key = min(_windows, key=lambda k: _windows[k]["ts"])
        _windows.pop(oldest_key)
    return block or None, key


async def build_preview(first_user_hint: str = "") -> dict:
    """管理接口用：预览一次开窗会注入什么。"""
    if not config.supabase_ready():
        return {"error": "Supabase 未配置"}
    rows = await store.fetch_recent_chats(limit=50)
    first_user = (first_user_hint or "").strip()
    if not first_user:
        for r in reversed(rows):
            if r.get("role") == "user":
                first_user = (r.get("content") or "").strip()
                break
    longterm = await store.get_longterm()
    min_day = (_now() - timedelta(days=config.RECENT_DAYS - 1)).strftime("%Y-%m-%d")
    dailies = await store.get_recent_dailies(min_day)
    monthlies = await store.get_memories("monthly", "", config.RECENT_MONTHS, desc=True)
    quarterlies = await store.get_memories("quarterly", "", config.RECENT_QUARTERS, desc=True)
    yearlies = await store.get_memories("yearly", "", config.RECENT_YEARS, desc=True)
    last_rows, _no_history = await previous_window_rows(first_user)
    block = build_block_text(longterm, dailies, monthlies, quarterlies, yearlies, last_rows)
    return {
        "first_user_matched": first_user[:50],
        "longterm_chars": len((longterm or "").strip()),
        "recent_days": [{"date": d.get("scope"), "chars": len(d.get("content") or "")} for d in dailies],
        "monthlies": [{"scope": m.get("scope"), "chars": len(m.get("content") or "")} for m in monthlies],
        "quarterlies": [{"scope": q.get("scope"), "chars": len(q.get("content") or "")} for q in quarterlies],
        "yearlies": [{"scope": y.get("scope"), "chars": len(y.get("content") or "")} for y in yearlies],
        "last_window_messages": len(last_rows),
        "block": block,
    }


def _row_date(value):
    # App 写入的 created_at 是"手机本地墙上时钟"字符串（yyyy-MM-dd HH:mm:ss，无时区），
    # 但数据库列是 TIMESTAMPTZ，Postgres 把它错标成了 UTC。
    # 所以绝不能按 UTC 转回北京时间（那样日期会被推后 8 小时：27号晚上变成28号凌晨）——
    # 字面写的日期就是真实的北京时间，直接取字面日期即可。
    s = str(value or "").strip()
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def _parse_date(s: str):
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None


def _meta_messages(row: dict) -> int:
    try:
        return int(json.loads(row.get("meta") or "{}").get("messages", -1))
    except Exception:
        return -1


def _meta_tz_fixed(row: dict) -> bool:
    """带 tz_fix 标记 = 时区修复之后生成的概括；旧概括缺标记，需强制重生成一次。"""
    try:
        return bool(json.loads(row.get("meta") or "{}").get("tz_fix"))
    except Exception:
        return False


def _meta_merged(row: dict) -> bool:
    """rough 行的 meta.merged=1 表示已并入长期记忆（防止重复合并）。"""
    try:
        return bool(json.loads(row.get("meta") or "{}").get("merged"))
    except Exception:
        return False


_refresh_lock = asyncio.Lock()      # 同一时间只允许一个刷新任务（防撞车双倍烧 API）
_MAX_CONSEC_FAIL = 5                # 连续失败 N 次本轮收工，等下一轮
_MAX_ARCHIVE_PER_CYCLE = 40         # 每轮最多归档多少个老日子


async def refresh_once(force_days: int = 0) -> dict:
    """定时刷新（按用户要求的分层记忆结构）：

    1. 详细每日概括 —— 只做最近 RECENT_DAYS 天（给"近期记忆"用）
    2. 粗略归档 —— 老于 RECENT_DAYS 的日子，一天压成一两句话
    3. 长期记忆 —— 粗略归档合并成的"这个月大概"（旧逻辑兜底）
    4. 分层滚动 —— 日 → 月 → 季 → 年，逐层向上汇总
    """
    if _refresh_lock.locked():
        return {"skipped": "上一轮刷新还在跑，本轮跳过"}
    async with _refresh_lock:
        return await _refresh_once_inner(force_days)


async def _refresh_once_inner(force_days: int = 0) -> dict:
    stats = {"days_summarized": [], "days_archived": [], "longterm_updated": False, "day_errors": {}}
    if not config.supabase_ready():
        stats["error"] = "Supabase 未配置，跳过"
        return await _finish_refresh(stats)
    if not config.deepseek_ready():
        stats["error"] = "DEEPSEEK_API_KEY 未配置，跳过"
        return await _finish_refresh(stats)

    today = _now().date()
    earliest = today - timedelta(days=config.MAX_BACKFILL_DAYS)
    rows = await store.fetch_chats_between(earliest, today + timedelta(days=1))
    by_day = {}
    for r in rows:
        d = _row_date(r.get("created_at"))
        if d and earliest <= d <= today:
            by_day.setdefault(d, []).append(r)

    fails = 0  # 连续失败计数

    # 1) 详细每日概括：只做最近 RECENT_DAYS 天（含今天）
    recent_from = today - timedelta(days=config.RECENT_DAYS - 1)
    for day in sorted(by_day.keys()):
        if day < recent_from:
            continue  # 老日子走粗略归档，不做详细概括
        day_rows = by_day[day]
        age = (today - day).days
        existing = await store.get_daily(str(day))
        if force_days and age <= force_days:
            pass  # 强制重新总结
        elif age <= 1:
            # 今天/昨天是"开放日"：消息数没变就不重复花 token
            if existing and _meta_messages(existing) == len(day_rows):
                continue
        else:
            if existing and _meta_tz_fixed(existing):
                continue  # 已冻结
            # 没带 tz_fix 标记的旧概括是时区错位时期生成的：强制重生成一次
        try:
            summary = await summarizer.summarize_day(str(day), day_rows)
        except Exception as exc:
            log.exception("总结 %s 失败", day)
            stats["day_errors"][str(day)] = f"{type(exc).__name__}: {str(exc)[:180]}"
            fails += 1
            if fails >= _MAX_CONSEC_FAIL:
                stats["aborted"] = "连续失败过多，本轮提前结束"
                break
            continue
        fails = 0
        if not summary.strip():
            stats["day_errors"][str(day)] = "模型返回空内容（推理烧光了max_tokens）"
            continue
        await store.set_memory("daily", str(day), summary, {"messages": len(day_rows), "tz_fix": 1})
        stats["days_summarized"].append(str(day))
        log.info("已生成 %s 的详细概括（%d 条消息）", day, len(day_rows))

    # 2) 粗略归档：老于 RECENT_DAYS 且还没有 rough 行的日子，一天压成一两句话。
    if not stats.get("aborted"):
        try:
            rough_rows_all = await store.get_all_roughs(limit=400)
        except Exception:
            log.exception("读取粗略归档失败")
            rough_rows_all = []
        rough_scopes = {str(r.get("scope") or "") for r in rough_rows_all}
        archive_days = [d for d in sorted(by_day.keys())
                        if (today - d).days >= config.RECENT_DAYS
                        and str(d) not in rough_scopes]
        for day in archive_days[:_MAX_ARCHIVE_PER_CYCLE]:
            day_rows = by_day[day]
            existing = await store.get_daily(str(day))
            try:
                if existing and (existing.get("content") or "").strip():
                    note = await summarizer.rough_from_summary(str(day), existing["content"])
                else:
                    note = await summarizer.rough_from_raw(str(day), day_rows)
            except Exception as exc:
                log.exception("归档 %s 失败", day)
                stats["day_errors"]["archive:" + str(day)] = f"{type(exc).__name__}: {str(exc)[:180]}"
                fails += 1
                if fails >= _MAX_CONSEC_FAIL:
                    stats["aborted"] = "连续失败过多，本轮提前结束"
                    break
                continue
            fails = 0
            if not note.strip():
                stats["day_errors"]["archive:" + str(day)] = "模型返回空内容（推理烧光了max_tokens）"
                continue
            await store.set_memory("rough", str(day), note, {"merged": 0})
            # absorbed_until 仅作展示/兼容保留，不再参与消化判断
            await store.set_memory("state", "absorbed_until", str(day), {})
            stats["days_archived"].append(str(day))

    # 3) 长期记忆：把"还没合并过"的粗略归档合并成"这个月的大概"（旧逻辑兜底）。
    if not stats.get("aborted"):
        try:
            rough_rows = await store.get_all_roughs(limit=400)
        except Exception:
            log.exception("读取粗略归档失败")
            rough_rows = []
        pending_roughs = [r for r in rough_rows if not _meta_merged(r)]
        roughs = [f"〔{r.get('scope')}〕{(r.get('content') or '').strip()[:300]}"
                  for r in pending_roughs if (r.get("content") or "").strip()]
        if roughs:
            existing_lt = (await store.get_longterm() or "").strip()
            try:
                merged = await summarizer.merge_longterm(existing_lt, roughs)
            except Exception as exc:
                log.exception("长期记忆合并失败")
                stats["merge_error"] = f"{type(exc).__name__}: {str(exc)[:180]}"
                merged = ""
            if merged.strip():
                await store.set_memory("longterm", "", merged, {"source": "gateway_merge"})
                stats["longterm_updated"] = True
                marked = 0
                for r in pending_roughs:
                    try:
                        await store.set_memory("rough", str(r.get("scope") or ""),
                                               r.get("content") or "", {"merged": 1})
                        marked += 1
                    except Exception:
                        log.exception("标记 rough merged 失败 scope=%s", r.get("scope"))
                log.info("长期记忆已合并（%d 条未合并粗略归档，标记 %d 条）", len(roughs), marked)

    # 4) 分层滚动（日→月→季→年）：在粗略归档生成之后向上汇总
    if not stats.get("aborted"):
        try:
            stats.update(await _rollup_layers(today))
        except Exception:
            log.exception("分层滚动失败")

    # 5) 清理：滑出近期窗口的每日概括是死数据（内容已并入长期记忆），删掉防止表越积越大
    if not stats.get("aborted"):
        try:
            stats["stale_dailies_deleted"] = await store.delete_dailies_before(str(recent_from))
            if stats["stale_dailies_deleted"]:
                log.info("已清理 %d 条滑出近期窗口的旧每日概括", stats["stale_dailies_deleted"])
        except Exception:
            stats["stale_dailies_deleted"] = -1
            log.exception("清理过期每日概括失败")

    return await _finish_refresh(stats)


def _quarter_of(month_scope: str) -> str:
    """'2026-09' -> '2026-Q3'"""
    try:
        y, m = month_scope.split("-")
        q = (int(m) - 1) // 3 + 1
        return f"{y}-Q{q}"
    except Exception:
        return month_scope


async def _rollup_layers(today) -> dict:
    """月/季/年分层滚动归档。只在对应周期结束后滚动，且每个 scope 只生成一次。"""
    stats = {"months_rolled": [], "quarters_rolled": [], "years_rolled": []}
    if not config.supabase_ready() or not config.deepseek_ready():
        return stats

    # 月度：把已结束的月的 rough 压成 monthly
    try:
        rough_rows = await store.get_memories("rough", "", 500, desc=False)
        months = {}
        for r in rough_rows:
            scope = str(r.get("scope") or "")
            m = scope[:7]
            if len(m) == 7:
                months.setdefault(m, []).append(r)
        today_month = today.strftime("%Y-%m")
        for m in sorted(months):
            if m >= today_month:
                continue
            if await store.get_memory("monthly", m):
                continue
            texts = [f"〔{r.get('scope')}〕{(r.get('content') or '').strip()}" for r in months[m]]
            try:
                monthly = await summarizer.summarize_month(m, texts)
            except Exception:
                log.exception("月度归档失败 %s", m)
                continue
            if monthly.strip():
                await store.set_memory("monthly", m, monthly, {"source": "monthly_rollup"})
                stats["months_rolled"].append(m)
    except Exception:
        log.exception("月度滚动失败")

    # 季度：把已结束的季的 monthly 压成 quarterly
    try:
        monthly_rows = await store.get_memories("monthly", "", 200, desc=False)
        quarters = {}
        for r in monthly_rows:
            m = str(r.get("scope") or "")
            if len(m) == 7:
                q = _quarter_of(m)
                quarters.setdefault(q, []).append(r)
        today_quarter = _quarter_of(today.strftime("%Y-%m"))
        for q in sorted(quarters):
            if q >= today_quarter:
                continue
            if await store.get_memory("quarterly", q):
                continue
            texts = [f"〔{r.get('scope')}〕{(r.get('content') or '').strip()}" for r in quarters[q]]
            try:
                quarterly = await summarizer.summarize_quarter(q, texts)
            except Exception:
                log.exception("季度归档失败 %s", q)
                continue
            if quarterly.strip():
                await store.set_memory("quarterly", q, quarterly, {"source": "quarterly_rollup"})
                stats["quarters_rolled"].append(q)
    except Exception:
        log.exception("季度滚动失败")

    # 年度：把已结束的年的 quarterly 压成 yearly
    try:
        quarterly_rows = await store.get_memories("quarterly", "", 100, desc=False)
        years = {}
        for r in quarterly_rows:
            y = str(r.get("scope") or "")[:4]
            if len(y) == 4:
                years.setdefault(y, []).append(r)
        today_year = str(today.year)
        for y in sorted(years):
            if y >= today_year:
                continue
            if await store.get_memory("yearly", y):
                continue
            texts = [f"〔{r.get('scope')}〕{(r.get('content') or '').strip()}" for r in years[y]]
            try:
                yearly = await summarizer.summarize_year(y, texts)
            except Exception:
                log.exception("年度归档失败 %s", y)
                continue
            if yearly.strip():
                await store.set_memory("yearly", y, yearly, {"source": "yearly_rollup"})
                stats["years_rolled"].append(y)
    except Exception:
        log.exception("年度滚动失败")

    return stats


async def _finish_refresh(stats: dict) -> dict:
    """收尾：记录刷新时间和错误摘要，方便管理接口排查。"""
    try:
        await store.set_memory("state", "last_refresh", _now().isoformat(), {})
        err = {k: v for k, v in stats.items() if k in ("day_errors", "merge_error")}
        await store.set_memory("state", "last_errors", json.dumps(err, ensure_ascii=False), {})
    except Exception:
        log.exception("错误状态写入失败")
    return stats


async def seed_from_ob_if_empty() -> bool:
    """长期记忆为空时，把 OB 的浮现记忆原样搬进来（逐字复制，零幻觉）。"""
    if not (config.OB_MCP_URL and config.supabase_ready()):
        return False
    try:
        existing = await store.get_longterm()
    except Exception:
        return False
    if (existing or "").strip():
        return False
    text = await ob_client.fetch_breath()
    text = (text or "").strip()
    if not text:
        return False
    await store.set_memory("longterm", "", text, {"source": "ombre_breath_verbatim"})
    log.info("已把 Ombre Brain 浮现记忆原样搬进长期记忆（%d 字）", len(text))
    return True
