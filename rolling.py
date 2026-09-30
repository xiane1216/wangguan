"""窗口内滚动压缩。

App 每次请求都会带上完整对话历史，网关若原样转发，上下文越滚越长，
DeepSeek 思考链一长就超时断流（报 stream was reset: PROTOCOL_ERROR）。

这里在转发前做一次压缩：
- user/assistant 消息超过 ROLLING_TRIGGER 条时，把最老的一批压成一段摘要；
- 只保留最近 ROLLING_KEEP 条原文；
- 摘要按窗口缓存，只对新出现的部分增量总结，不重复烧 token；
- 总结失败（上游抖动等）则退化为纯截断，绝不阻塞转发；
- 一旦请求里出现非 system/user/assistant 的消息（如 tool 调用），
  说明有工具调用链，不做压缩、原样透传，保证透传承诺不破。
"""
import hashlib
import logging
import time

import config
import summarizer

log = logging.getLogger(__name__)

# window_key -> {"summary": str, "compressed": int, "ts": float}
_rolling: dict = {}
_MAX_CACHE = 100
_TALK_ROLES = ("user", "assistant")


def _key(system_text: str, first_user: str) -> str:
    raw = (system_text or "") + "\x00" + (first_user or "")
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _split(messages):
    """拆成 (system 消息列表, 参与压缩的 user/assistant 消息列表)。"""
    systems, talk = [], []
    for m in messages or []:
        role = m.get("role")
        if role == "system":
            systems.append(m)
        elif role in _TALK_ROLES:
            talk.append(m)
    return systems, talk


async def _summarize(rows) -> str:
    """把一批消息压成一段滚动摘要；失败返回空串（由调用方退化为纯截断）。"""
    try:
        return await summarizer.summarize_rollup(rows)
    except Exception:
        log.exception("滚动摘要生成失败")
        return ""


async def compress(messages, system_text: str, first_user: str) -> list:
    """返回压缩后的 messages。未超阈值或不可压缩时原样返回（对象不变）。"""
    if not config.ROLLING_TRIGGER:
        return messages

    # 有 tool 等特殊消息时不压缩，保透传，避免打断工具调用链
    for m in messages or []:
        if m.get("role") not in ("system", "user", "assistant"):
            return messages

    systems, talk = _split(messages)
    if len(talk) <= config.ROLLING_TRIGGER:
        return messages

    keep = max(1, config.ROLLING_KEEP)
    compress_count = len(talk) - keep
    if compress_count <= 0:
        return messages

    key = _key(system_text, first_user)
    now = time.time()
    cache = _rolling.get(key)

    summary = ""
    if cache and cache.get("compressed", 0) >= compress_count:
        # 这批老消息已经压过，直接复用，不重复烧 token
        summary = cache.get("summary") or ""
        cache["ts"] = now
    else:
        prev_summary = (cache.get("summary") or "") if cache else ""
        prev_count = cache.get("compressed", 0) if cache else 0
        new_rows = talk[prev_count:compress_count]
        new_summary = await _summarize(new_rows)
        if prev_summary and new_summary:
            summary = prev_summary + "\n\n" + new_summary
        elif new_summary:
            summary = new_summary
        else:
            summary = prev_summary
        _rolling[key] = {"summary": summary, "compressed": compress_count, "ts": now}
        if len(_rolling) > _MAX_CACHE:
            oldest = min(_rolling, key=lambda k: _rolling[k]["ts"])
            _rolling.pop(oldest)

    kept = talk[-keep:]
    new_messages = list(systems)
    if summary.strip():
        block = (
            "【本窗口更早的对话摘要（网关自动压缩，用于延续上下文，非原文）】\n"
            + summary.strip()
        )
        new_messages.append({"role": "system", "content": block})
    new_messages.extend(kept)
    return new_messages
