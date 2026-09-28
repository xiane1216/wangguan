"""记忆装配核心。

三段式记忆（开窗注入，窗口生命周期内冻结，保证 DeepSeek 前缀缓存稳定）：
  一、长期记忆 —— OB 搬运来的原文 + 网关自己从聊天记录总结并逐段合并的概括
  二、近期记忆 —— 最近 RECENT_DAYS 天的每日概括
  三、上个窗口原始聊天记录 —— 最后 LAST_WINDOW_MESSAGES 条原文（零加工）

防幻觉原则：
  - OB 搬运和原始记录是"原样复制"，不经过任何 LLM；
  - 每日概括和长期记忆合并用 deepseek-flash + 防幻觉铁律 prompt。
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


async def previous_window_rows(first_user_text: str) -> list:
    """从 Supabase 最近的聊天记录里找到"上一个窗口"，取它的最后 N 条。"""
    rows = await store.fetch_recent_chats(limit=600)
    rows = [r for r in rows
            if r.get("role") in ("user", "assistant") and (r.get("content") or "").strip()]
    if not rows:
        return []
    # 同一秒同内容的重复行去重
    seen, dedup = set(), []
    for r in rows:
        k = (r.get("role"), r.get("content"), str(r.get("created_at")))
        if k in seen:
            continue
        seen.add(k)
        dedup.append(r)
    rows = dedup

    # 用当前窗口的第一条用户消息，在最近同步的记录里定位"当前窗口"的会话
    cur_conv = None
    probe = (first_user_text or "").strip()[:50]
    if probe:
        for r in reversed(rows):
            if r.get("role") == "user" and str(r.get("content") or "").strip().startswith(probe):
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
    else:
        prev = order[-1] if order else None  # 当前窗口还没同步上去：最新会话就是上个窗口
    if prev is None:
        return []
    return convs[prev][-config.LAST_WINDOW_MESSAGES:]


def build_block_text(longterm, dailies, last_rows) -> str:
    parts = []
    if (longterm or "").strip():
        parts.append("=== 一、长期记忆（我们的记忆库：真实发生过的事的概括） ===\n" + longterm.strip())
    if dailies:
        seg = []
        for d in dailies:
            c = (d.get("content") or "").strip()
            if c:
                seg.append(f"〔{d.get('scope', '')}〕\n{c}")
        if seg:
            parts.append("=== 二、近期记忆（最近几天真实聊过的事） ===\n" + "\n\n".join(reversed(seg)))
    if last_rows:
        parts.append(
            f"=== 三、上个窗口聊天记录（最后 {len(last_rows)} 条原文摘录） ===\n"
            + summarizer.transcript(last_rows)
        )
    if not parts:
        return ""
    header = (
        "【记忆系统注入】以下是你自己的记忆，由系统在开窗时自动提供，都是真实发生过的事。"
        "直接当作你的记忆使用，不需要再调用任何外部记忆、聊天记录读取工具。"
    )
    return header + "\n\n" + "\n\n".join(parts)


async def _build_block(first_user_text: str) -> str:
    if not config.supabase_ready():
        return ""
    longterm = await store.get_longterm()
    min_day = (_now() - timedelta(days=config.RECENT_DAYS - 1)).strftime("%Y-%m-%d")
    dailies = await store.get_recent_dailies(min_day)
    last_rows = await previous_window_rows(first_user_text)
    return build_block_text(longterm, dailies, last_rows)


async def get_window_block(system_text: str, first_user_text: str):
    """返回 (block, window_key)。窗口生命周期内冻结：活跃窗口滑动续期，永不中途重建，
    保证前缀缓存稳定；空结果 10 分钟后允许重试。"""
    key = _window_key(system_text, first_user_text)
    now = time.time()
    hit = _windows.get(key)
    if hit:
        ttl = config.WINDOW_TTL_HOURS * 3600 if hit["block"] else 600
        if now - hit["ts"] < ttl:
            if hit["block"]:
                hit["ts"] = now  # 活跃窗口滑动续期
            return hit["block"], key
    block = ""
    try:
        block = await _build_block(first_user_text)
    except Exception:
        log.exception("构建窗口记忆失败（本次请求将不注入）")
        block = ""
    _windows[key] = {"block": block or None, "ts": now}
    if len(_windows) > 200:
        _windows.pop(next(iter(_windows)))
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
    last_rows = await previous_window_rows(first_user)
    block = build_block_text(longterm, dailies, last_rows)
    return {
        "first_user_matched": first_user[:50],
        "longterm_chars": len((longterm or "").strip()),
        "recent_days": [{"date": d.get("scope"), "chars": len(d.get("content") or "")} for d in dailies],
        "last_window_messages": len(last_rows),
        "block": block,
    }


def _row_date(value):
    s = str(value or "").strip()
    if not s:
        return None
    try:
        if s.endswith("Z") or ("+" in s[10:]):
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            return dt.astimezone(TZ).date()
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


_refresh_lock = asyncio.Lock()      # 同一时间只允许一个刷新任务（防撞车双倍烧 API）
_MAX_CONSEC_FAIL = 5                # 连续失败 N 次本轮收工，等下一轮
_MAX_ARCHIVE_PER_CYCLE = 40         # 每轮最多归档多少个老日子


async def refresh_once(force_days: int = 0) -> dict:
    """定时刷新（按用户要求的两级记忆结构）：

    1. 详细每日概括 —— 只做最近 RECENT_DAYS 天（给"近期记忆"用）
    2. 粗略归档 —— 老于 RECENT_DAYS 的日子，一天压成一两句话
    3. 长期记忆 —— 粗略归档合并成的"这个月大概"（800 字内），合并完清掉粗归档
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
            if existing:
                continue  # 前天的概括已冻结
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
        await store.set_memory("daily", str(day), summary, {"messages": len(day_rows)})
        stats["days_summarized"].append(str(day))
        log.info("已生成 %s 的详细概括（%d 条消息）", day, len(day_rows))

    # 2) 粗略归档：老于 RECENT_DAYS 且没归档过的日子，一天压成一两句话
    if not stats.get("aborted"):
        absorbed_until = _parse_date(await store.get_state("absorbed_until") or "")
        archive_days = [d for d in sorted(by_day.keys())
                        if (today - d).days >= config.RECENT_DAYS
                        and (absorbed_until is None or d > absorbed_until)]
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
            await store.set_memory("rough", str(day), note, {})
            await store.set_memory("state", "absorbed_until", str(day), {})
            stats["days_archived"].append(str(day))

    # 3) 长期记忆：把全部粗略归档合并成"这个月的大概"，合并成功后清掉粗归档
    if not stats.get("aborted"):
        try:
            rough_rows = await store.get_all_roughs()
        except Exception:
            log.exception("读取粗略归档失败")
            rough_rows = []
        roughs = [f"〔{r.get('scope')}〕{(r.get('content') or '').strip()}"
                  for r in rough_rows if (r.get("content") or "").strip()]
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
                try:
                    await store.delete_roughs()  # 已消化，清掉
                except Exception:
                    log.exception("清理已消化的粗略归档失败")
                log.info("长期记忆已合并（%d 条粗略归档）", len(roughs))

    return await _finish_refresh(stats)


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
