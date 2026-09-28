"""记忆装配核心。

三段式记忆（开窗注入，窗口生命周期内冻结，保证 DeepSeek 前缀缓存稳定）：
  一、长期记忆 —— OB 搬运来的原文 + 网关自己从聊天记录总结并逐段合并的概括
  二、近期记忆 —— 最近 RECENT_DAYS 天的每日概括
  三、上个窗口原始聊天记录 —— 最后 LAST_WINDOW_MESSAGES 条原文（零加工）

防幻觉原则：
  - OB 搬运和原始记录是"原样复制"，不经过任何 LLM；
  - 每日概括和长期记忆合并用 deepseek-flash + 防幻觉铁律 prompt。
"""
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


async def refresh_once(force_days: int = 0) -> dict:
    """定时任务：更新每日概括 + 把超出"近期窗口"的每日概括并进长期记忆。"""
    stats = {"days_summarized": [], "merged_days": [], "longterm_updated": False, "day_errors": {}}
    if not config.supabase_ready():
        stats["error"] = "Supabase 未配置，跳过"
        return stats
    if not config.deepseek_ready():
        stats["error"] = "DEEPSEEK_API_KEY 未配置，跳过"
        return stats

    today = _now().date()
    earliest = today - timedelta(days=config.MAX_BACKFILL_DAYS)
    rows = await store.fetch_chats_between(earliest, today + timedelta(days=1))
    by_day = {}
    for r in rows:
        d = _row_date(r.get("created_at"))
        if d and earliest <= d <= today:
            by_day.setdefault(d, []).append(r)

    merged_until_raw = await store.get_state("merged_until")
    merged_until = _parse_date(merged_until_raw) if merged_until_raw else None

    # 1) 每日概括
    for day in sorted(by_day.keys()):
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
                continue  # 冻结日已有概括
            if merged_until and day <= merged_until:
                continue  # 已经并进长期记忆了
        try:
            summary = await summarizer.summarize_day(str(day), day_rows)
        except Exception as exc:
            log.exception("总结 %s 失败", day)
            stats["day_errors"][str(day)] = f"{type(exc).__name__}: {str(exc)[:180]}"
            continue
        if not summary.strip():
            continue
        await store.set_memory("daily", str(day), summary, {"messages": len(day_rows)})
        stats["days_summarized"].append(str(day))
        log.info("已生成 %s 的每日概括（%d 条消息）", day, len(day_rows))

    # 2) 长期记忆合并：把超出"近期窗口"的每日概括并进长期记忆（每轮最多 12 天）
    merge_days = [d for d in sorted(by_day.keys())
                  if (today - d).days >= config.RECENT_DAYS
                  and (merged_until is None or d > merged_until)]
    merge_days = merge_days[:12]
    if merge_days:
        dailies = []
        for d in merge_days:
            row = await store.get_daily(str(d))
            c = ((row or {}).get("content") or "").strip()
            if c:
                dailies.append(f"〔{d}〕\n{c}")
        if dailies:
            existing = (await store.get_longterm() or "").strip()
            try:
                merged = await summarizer.merge_longterm(existing, dailies)
            except Exception as exc:
                log.exception("长期记忆合并失败")
                stats["merge_error"] = f"{type(exc).__name__}: {str(exc)[:180]}"
                merged = ""
            if merged.strip():
                await store.set_memory("longterm", "", merged,
                                       {"source": "gateway_merge", "merged_until": str(merge_days[-1])})
                await store.set_memory("state", "merged_until", str(merge_days[-1]), {})
                stats["merged_days"] = [str(d) for d in merge_days]
                stats["longterm_updated"] = True
                log.info("长期记忆已合并 %d 天的概括", len(merge_days))

    await store.set_memory("state", "last_refresh", _now().isoformat(), {})
    try:
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
