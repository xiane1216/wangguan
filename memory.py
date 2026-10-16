"""记忆装配核心。

三段式记忆（开窗注入，窗口生命周期内冻结，保证 DeepSeek 前缀缓存稳定）：
  一、近期每日 —— 最近 RECENT_DAYS 天的详细概括
  二、月度概览 —— 最近 RECENT_MONTHS 个月的概览（含进行中的当月）
  三、季度概览 —— 最近 RECENT_QUARTERS 个季的概览
  四、年度概览 —— 最近 RECENT_YEARS 年的概览
  五、长期记忆 —— 旧数据兜底（OB 搬运 + 早期合并的概括）
  六、最近原始聊天记录 —— 全局最近 LAST_WINDOW_MESSAGES 条原文（零加工）

防幻觉原则：
  - OB 搬运和原始记录是"原样复制"，不经过任何 LLM；
  - 各级概括用 deepseek-flash + 防幻觉铁律 prompt，且每层有字数/ token 上限。
"""
import asyncio
import hashlib
import json
import logging
import re
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


def _window_key(first_user: str) -> str:
    """窗口指纹只用第一条用户消息（对话锚点），不再掺 system 文本。

    原因：App 的心跳和正常聊天共用同一段对话历史（first_user 相同），
    但 system 提示词是两套（心跳版多一段规则）。之前 system 参与指纹，
    心跳被当成独立窗口：心跳间隔一超过 TTL 就重拼记忆块 → system 内容
    变化 → DeepSeek 前缀缓存整体作废，心跳请求命中率暴跌。记忆块内容
    本来就是全局的（不区分对话），同一段历史共用一个冻结块即可。
    "同开场白新窗"的误命中仍由 assistant_count==0 + REDETECT 兜底强制重建。
    """
    return hashlib.sha256((first_user or "").encode("utf-8")).hexdigest()[:16]


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


async def recent_rows(limit: int | None = None) -> tuple[list, bool]:
    """取全局最近 N 条原始聊天记录（零加工，跨窗口，不区分会话）。

    返回 (rows, confident_empty)：
    - rows 是最近 N 条，按时间升序（最新在最后）。
    - confident_empty=True 表示库里确实没有任何聊天记录。
    """
    n = limit or config.LAST_WINDOW_MESSAGES
    rows = await store.fetch_recent_chats(limit=max(n * 4, 200))
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
    return dedup[-n:], False


def build_block_text(longterm, dailies, monthlies, quarterlies, yearlies, last_rows, profile: str = "") -> str:
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
    if (profile or "").strip():
        # 宝宝画像：长期记忆之后、最近原文之前；免责声明装配时追加，不存库。
        u = config.USER_LABEL
        parts.append(
            f"=== {u}的画像（我对{u}的长期认知） ===\n" + profile.strip()
            + f"\n（这份画像只是过往聊天里的记录和观察，不一定代表当下{u}的想法、情绪和感情。"
              f"此刻的{u}，以此刻TA说的话为准。）"
        )
    if last_rows:
        parts.append(
            f"=== 六、最近聊天记录（最后 {len(last_rows)} 条原文） ===\n"
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
    """构建记忆块。返回 (block, complete)：complete 表示"没有更多可注入的历史"是确定事实
    （库里确实没有历史记录），而非同步延迟——用于冻结缓存判断。"""
    if not config.supabase_ready():
        return "", True
    longterm = await store.get_longterm()
    min_day = (_now() - timedelta(days=config.RECENT_DAYS - 1)).strftime("%Y-%m-%d")
    dailies = await store.get_recent_dailies(min_day)
    monthlies = await store.get_memories("monthly", "", config.RECENT_MONTHS, desc=True)
    quarterlies = await store.get_memories("quarterly", "", config.RECENT_QUARTERS, desc=True)
    yearlies = await store.get_memories("yearly", "", config.RECENT_YEARS, desc=True)
    last_rows, confident_no_history = await recent_rows()
    profile_row = await store.get_memory("profile", "")
    profile = (profile_row or {}).get("content") or "" if profile_row else ""
    block = build_block_text(longterm, dailies, monthlies, quarterlies, yearlies, last_rows, profile)
    return block, (bool(block) or confident_no_history)


async def get_window_block(first_user_text: str, assistant_count: int = 0):
    """返回 (block, window_key)。窗口生命周期内冻结：活跃窗口滑动续期，永不中途重建，
    保证前缀缓存稳定；空结果 10 分钟后允许重试。

    bugfix（Bug 2）：缓存命中但当前请求 0 条 assistant 回复、且缓存已超过
    NEW_WINDOW_REDETECT_SECONDS —— 说明是"用相同开场白开的新窗"（真正活跃的
    窗口在几分钟内几乎必然产生过 assistant 回复），强制重建，避免把上上个
    窗口的内容当成"上个窗口"注入。
    """
    key = _window_key(first_user_text)
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
    last_rows, _no_history = await recent_rows()
    profile_row = await store.get_memory("profile", "")
    profile = (profile_row or {}).get("content") or "" if profile_row else ""
    block = build_block_text(longterm, dailies, monthlies, quarterlies, yearlies, last_rows, profile)
    return {
        "first_user_matched": first_user[:50],
        "longterm_chars": len((longterm or "").strip()),
        "profile_chars": len(profile.strip()),
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
            stats["day_errors"][str(day)] = "当天没有可总结的聊天记录"
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
                stats["day_errors"]["archive:" + str(day)] = "当天没有可归档的聊天记录"
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


async def regenerate(kind: str, scope: str) -> dict:
    """定向重总结某一层的一个周期（管理接口用，不影响正常刷新流程）。

    kind: daily / monthly / quarterly / yearly
    scope: daily -> '2026-10-14'；monthly -> '2026-10'；quarterly -> '2026-Q4'；yearly -> '2026'
    只重做指定周期，不碰其他任何日期/周期。
    """
    kind = (kind or "").strip().lower()
    scope = (scope or "").strip()
    if kind not in ("daily", "monthly", "quarterly", "yearly"):
        return {"ok": False, "error": "kind 必须是 daily / monthly / quarterly / yearly"}

    if not config.supabase_ready():
        return {"ok": False, "error": "Supabase 未配置"}
    if not config.deepseek_ready():
        return {"ok": False, "error": "DEEPSEEK_API_KEY 未配置"}

    # ---- 日总结：拉当天全部原文，与 refresh_once 同款 summarize_day ----
    if kind == "daily":
        d = _parse_date(scope)
        if not d:
            return {"ok": False, "error": "scope 格式应为 YYYY-MM-DD"}
        rows = await store.fetch_chats_between(d, d + timedelta(days=1))
        if not rows:
            return {"ok": False, "error": f"{scope} 没有聊天记录", "messages": 0}
        summary = await summarizer.summarize_day(scope, rows)
        if not summary.strip():
            return {"ok": False, "error": "总结结果为空（当天没有可总结的聊天内容）", "messages": len(rows)}
        await store.set_memory("daily", scope, summary, {"messages": len(rows), "tz_fix": 1})
        return {"ok": True, "kind": kind, "scope": scope,
                "messages": len(rows), "chars": len(summary)}

    # ---- 月总结：从该月的 rough（粗略归档）重新汇总 ----
    if kind == "monthly":
        if len(scope) != 7 or not scope.startswith(("19", "20")):
            return {"ok": False, "error": "scope 格式应为 YYYY-MM"}
        rough_rows = await store.get_memories("rough", "", 500, desc=False)
        texts = [f"〔{r.get('scope')}〕{(r.get('content') or '').strip()}"
                 for r in rough_rows
                 if str(r.get("scope") or "")[:7] == scope and (r.get("content") or "").strip()]
        if not texts:
            return {"ok": False, "error": f"{scope} 没有可用的粗略归档（日总结要先归档成 rough）"}
        monthly = await summarizer.summarize_month(scope, texts)
        if not monthly.strip():
            return {"ok": False, "error": "月度总结结果为空"}
        await store.set_memory("monthly", scope, monthly, {"source": "monthly_regenerate"})
        return {"ok": True, "kind": kind, "scope": scope, "days": len(texts), "chars": len(monthly)}

    # ---- 季总结：从该季的 monthly 重新汇总 ----
    if kind == "quarterly":
        if not re.match(r"^\d{4}-Q[1-4]$", scope):
            return {"ok": False, "error": "scope 格式应为 YYYY-Qn（如 2026-Q4）"}
        monthly_rows = await store.get_memories("monthly", "", 200, desc=False)
        texts = [f"〔{r.get('scope')}〕{(r.get('content') or '').strip()}"
                 for r in monthly_rows
                 if _quarter_of(str(r.get("scope") or "")) == scope and (r.get("content") or "").strip()]
        if not texts:
            return {"ok": False, "error": f"{scope} 没有可用的月度总结"}
        quarterly = await summarizer.summarize_quarter(scope, texts)
        if not quarterly.strip():
            return {"ok": False, "error": "季度总结结果为空"}
        await store.set_memory("quarterly", scope, quarterly, {"source": "quarterly_regenerate"})
        return {"ok": True, "kind": kind, "scope": scope, "months": len(texts), "chars": len(quarterly)}

    # ---- 年总结：从该年的 quarterly 重新汇总 ----
    if kind == "yearly":
        if not re.match(r"^\d{4}$", scope):
            return {"ok": False, "error": "scope 格式应为 YYYY"}
        quarterly_rows = await store.get_memories("quarterly", "", 100, desc=False)
        texts = [f"〔{r.get('scope')}〕{(r.get('content') or '').strip()}"
                 for r in quarterly_rows
                 if str(r.get("scope") or "")[:4] == scope and (r.get("content") or "").strip()]
        if not texts:
            return {"ok": False, "error": f"{scope} 没有可用的季度总结"}
        yearly = await summarizer.summarize_year(scope, texts)
        if not yearly.strip():
            return {"ok": False, "error": "年度总结结果为空"}
        await store.set_memory("yearly", scope, yearly, {"source": "yearly_regenerate"})
        return {"ok": True, "kind": kind, "scope": scope, "quarters": len(texts), "chars": len(yearly)}


async def _rollup_layers(today) -> dict:
    """月/季/年分层滚动归档。
    已结束的月/季/年只在周期结束后滚动一次；当月（进行中）每次刷新都增量合并。"""
    stats = {"months_rolled": [], "quarters_rolled": [], "years_rolled": []}
    if not config.supabase_ready() or not config.deepseek_ready():
        return stats

    # 月度：已结束的月一次性压成 monthly；当月（进行中）增量合并
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
            if m > today_month:
                continue
            texts = [f"〔{r.get('scope')}〕{(r.get('content') or '').strip()}" for r in months[m]]
            if m == today_month:
                # 进行中的月：增量合并，每次刷新都更新，不跳过
                existing = await store.get_memory("monthly", m)
                existing_text = (existing.get("content") or "").strip() if existing else ""
                try:
                    monthly = await summarizer.merge_month_in_progress(m, existing_text, texts)
                except Exception:
                    log.exception("进行中月度合并失败 %s", m)
                    continue
                if monthly.strip():
                    await store.set_memory("monthly", m, monthly, {"source": "monthly_in_progress"})
                    stats["months_rolled"].append(m)
            else:
                # 已结束的月：一次性生成，只做一次
                if await store.get_memory("monthly", m):
                    continue
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


# ---- 宝宝画像（手动触发的用户画像总结，kind=profile） ----
# 两层隔离：提取层每批消息→事实清单；成稿层只见清单不见原文。
# 进度供 /admin/profile_status 轮询；提取断点实时存 state，中断重跑可续。

_profile_state: dict = {
    "running": False, "phase": "", "done": 0, "total": 0,
    "error": "", "chars": 0, "started_at": "", "finished_at": "",
}


def profile_status() -> dict:
    return dict(_profile_state)


def start_profile_task() -> dict:
    """启动画像总结后台任务（同一时间只允许一个）。立即返回，进度靠轮询。"""
    if _profile_state.get("running"):
        return {"ok": False, "error": "画像总结已在运行中，等它跑完（进度见 /admin/profile_status）"}
    if not config.supabase_ready():
        return {"ok": False, "error": "Supabase 未配置"}
    if not config.deepseek_ready():
        return {"ok": False, "error": "DEEPSEEK_API_KEY 未配置"}
    _profile_state.update({
        "running": True, "phase": "preparing", "done": 0, "total": 0,
        "error": "", "chars": 0,
        "started_at": _now().isoformat(timespec="seconds"), "finished_at": "",
    })
    asyncio.create_task(_profile_run())
    return {"ok": True, "hint": "画像总结已启动。首次全量要跑很久（几万条消息逐批提取），可用 /admin/profile_status 轮询进度"}


async def _profile_run():
    """画像主流程：定位断点 → 拉新消息 → 分批提取事实 → 成稿存库。

    断点续传：
    - profile 行 meta.checkpoint = 上次成稿完成时最后处理到的消息时间戳
    - state 行 profile_extract = {"ts": 提取中途断点, "facts": 已累积事实}（JSON）
      提取每批后实时写；成稿成功后清空。中途挂掉重跑时从断点继续，不重复花 token。
    """
    st = _profile_state
    try:
        profile_row = await store.get_memory("profile", "")
        old_profile = (profile_row or {}).get("content") or "" if profile_row else ""
        base_cp = ""
        if profile_row:
            try:
                base_cp = str(json.loads(profile_row.get("meta") or "{}").get("checkpoint") or "")
            except Exception:
                base_cp = ""
        saved_ts, facts = "", ""
        try:
            saved = json.loads(await store.get_state("profile_extract") or "{}")
            saved_ts = str(saved.get("ts") or "")
            facts = str(saved.get("facts") or "")
        except Exception:
            saved_ts, facts = "", ""
        # 上次提取中途断掉（断点比已存画像新）→ 续跑，事实清单接着用
        resume = bool(saved_ts) and (not base_cp or saved_ts >= base_cp)
        start_ts = saved_ts if resume else base_cp
        if not resume:
            facts = ""

        rows = await store.fetch_chats_after(start_ts)
        rows = [r for r in rows
                if r.get("role") in ("user", "assistant") and (r.get("content") or "").strip()]
        total = len(rows)
        st["total"] = total
        if not rows and not facts.strip():
            st.update(phase="done", done=0, chars=len(old_profile))
            log.info("画像总结：没有新消息，画像保持原样（%d 字）", len(old_profile))
            return

        # ---- 提取层：分批过全部消息 ----
        batch = max(20, config.PROFILE_BATCH_MESSAGES)
        st["phase"] = "extracting"
        for i in range(0, total, batch):
            chunk = rows[i:i + batch]
            st["done"] = min(i + len(chunk), total)
            part = await summarizer.extract_profile_facts(chunk)
            if part:
                facts = (facts + "\n" + part).strip() if facts else part
            last_ts = str(chunk[-1].get("created_at") or "")
            # 断点实时落库：任何点挂掉，重跑只补本批
            await store.set_memory("state", "profile_extract",
                                   json.dumps({"ts": last_ts, "facts": facts}, ensure_ascii=False), {})

        # ---- 成稿层 ----
        st["phase"] = "composing"
        st["done"] = total
        if not facts.strip():
            st.update(phase="done", chars=len(old_profile))
            await store.set_memory("state", "profile_extract", "", {})
            return
        final_cp = str(rows[-1].get("created_at") or "") if rows else (start_ts or base_cp)
        text = await summarizer.compose_profile(facts, old_profile)
        if not text.strip():
            raise RuntimeError("成稿层返回了空画像（旧画像未动）")
        await store.set_memory("profile", "", text,
                                {"checkpoint": final_cp, "source": "profile_summarize"})
        await store.set_memory("state", "profile_extract", "", {})  # 清断点
        st.update(phase="done", chars=len(text), error="")
        log.info("宝宝画像已生成：%d 字（本轮处理 %d 条消息）", len(text), total)
    except Exception as exc:
        log.exception("宝宝画像总结失败")
        st.update(phase="error", error=f"{type(exc).__name__}: {exc}"[:500])
    finally:
        st["running"] = False
        st["finished_at"] = _now().isoformat(timespec="seconds")
