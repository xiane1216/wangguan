"""DeepSeek 总结器。

模型：deepseek-flash（DeepSeek-V4.1-Flash 的官方 API 名）。
所有总结都带防幻觉铁律：只概括真实聊过、原文明确出现的内容，禁止编造。
"""
import logging
import re

import httpx

import config
import prompts

log = logging.getLogger(__name__)

# 思考标签的 unicode 转义写法（防止个别模型把标签混进正文）
_THINK_OPEN = "\u003c\u0074\u0068\u0069\u006e\u006b\u003e"
_THINK_CLOSE = "\u003c\u002f\u0074\u0068\u0069\u006e\u006b\u003e"


async def _chat(messages: list, max_tokens: int, temperature: float) -> str:
    if not config.deepseek_ready():
        raise RuntimeError("DEEPSEEK_API_KEY 未配置")
    payload = {
        "model": config.SUMMARY_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=20, read=180, write=20, pool=20)) as client:
        r = await client.post(
            f"{config.DEEPSEEK_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {config.DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
    if r.status_code >= 400:
        raise RuntimeError(f"DeepSeek HTTP {r.status_code}: {r.text[:300]}")
    data = r.json()
    message = ((data.get("choices") or [{}])[0]).get("message") or {}
    content = message.get("content") or ""
    if not str(content).strip():
        # 个别情况正文为空、思考里有内容，兜底取思考文本
        content = message.get("reasoning_content") or ""
    content = re.sub(re.escape(_THINK_OPEN) + r".*?" + re.escape(_THINK_CLOSE), "", str(content), flags=re.S)
    content = content.replace(_THINK_OPEN, "").replace(_THINK_CLOSE, "")
    return content.strip()


def _clip(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + "…(过长已截断)"


def transcript(rows: list, max_messages: int = 400, clip: int = 2000) -> str:
    """把聊天记录转成「[时间] 角色: 内容」纯文本（原文摘录用，零加工）。"""
    out = []
    for r in rows[:max_messages]:
        role = r.get("role")
        if role not in ("user", "assistant"):
            continue
        who = config.USER_LABEL if role == "user" else config.AI_LABEL
        ts = str(r.get("created_at") or "").replace("T", " ")
        hm = ts[11:16] if len(ts) >= 16 else ""
        prefix = f"[{hm}] " if hm else ""
        out.append(f"{prefix}{who}: {_clip(r.get('content'), clip)}")
    return "\n".join(out)


async def summarize_day(date_str: str, rows: list) -> str:
    text = transcript(rows)
    if not text.strip():
        return ""
    prompt = prompts.DAILY_SUMMARY.format(
        user_label=config.USER_LABEL,
        ai_label=config.AI_LABEL,
        date=date_str,
        transcript=text,
    )
    return await _chat(
        [{"role": "user", "content": prompt}],
        config.SUMMARY_MAX_TOKENS,
        config.SUMMARY_TEMPERATURE,
    )


async def merge_longterm(existing: str, daily_texts: list) -> str:
    prompt = prompts.LONGTERM_MERGE.format(
        longterm=(existing or "").strip() or "（暂无）",
        dailies="\n\n".join(daily_texts),
    )
    return await _chat(
        [{"role": "user", "content": prompt}],
        config.SUMMARY_MAX_TOKENS + 800,
        config.SUMMARY_TEMPERATURE,
    )
