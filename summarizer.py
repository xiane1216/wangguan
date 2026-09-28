"""DeepSeek 总结器。

模型：deepseek-flash（DeepSeek-V4.1-Flash 的官方 API 名）。
所有总结都带防幻觉铁律：只概括真实聊过、原文明确出现的内容，禁止编造。

两级总结：
- 详细每日（最近3天用）：summarize_day
- 粗略归档（老日子用，一天一两句话）：rough_from_raw / rough_from_summary
- 长期记忆合并（大概级别）：merge_longterm
"""
import asyncio
import logging
import re

import httpx

import config
import prompts

log = logging.getLogger(__name__)

# 思考标签的 unicode 转义写法（防止个别模型把标签混进正文）
_THINK_OPEN = "\u003c\u0074\u0068\u0069\u006e\u006b\u003e"
_THINK_CLOSE = "\u003c\u002f\u0074\u0068\u0069\u006e\u006b\u003e"

# 正文起始标记：prompt 要求模型输出第一行必须是它。
# 只取「最后一次出现」之后的内容，标记之前的推理/复述/废话一律丢弃。
_MARKER = "「记忆归档开始」"


def _clean(text: str) -> str:
    text = str(text or "")
    # 1) 去掉成对思考标签及其中内容
    text = re.sub(re.escape(_THINK_OPEN) + r".*?" + re.escape(_THINK_CLOSE), "", text, flags=re.S)
    # 2) 去掉落单的思考标签
    text = text.replace(_THINK_OPEN, "").replace(_THINK_CLOSE, "")
    # 3) 截取正文起始标记之后的内容（防推理泄漏）
    idx = text.rfind(_MARKER)
    if idx >= 0:
        text = text[idx + len(_MARKER):]
    return text.strip()


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
        for attempt in range(2):
            r = await client.post(
                f"{config.DEEPSEEK_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {config.DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            if r.status_code == 429 and attempt == 0:
                await asyncio.sleep(30)  # 限流：等半分钟重试一次
                continue
            break
    if r.status_code >= 400:
        raise RuntimeError(f"DeepSeek HTTP {r.status_code}: {r.text[:300]}")
    data = r.json()
    message = ((data.get("choices") or [{}])[0]).get("message") or {}
    content = str(message.get("content") or "").strip()
    if not content:
        # 思考型模型可能把 max_tokens 全部烧在推理上导致正文为空。
        # 铁律：reasoning_content 是思考过程，不是答案，绝不能存成记忆。
        choice = (data.get("choices") or [{}])[0]
        finish = choice.get("finish_reason")
        usage = data.get("usage") or {}
        rc = str(message.get("reasoning_content") or "")
        raise RuntimeError(
            f"模型没有产出正文 finish={finish} completion_tokens={usage.get('completion_tokens')} "
            f"reasoning_chars={len(rc)}（推理烧光了max_tokens，需减小输入或加大上限）"
        )
    return _clean(content)


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
    """详细每日概括（最近3天用）。"""
    # 大日子输入瘦身：条数和单条长度都收着，防推理把max_tokens烧光
    text = transcript(rows, max_messages=300, clip=1500)
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


async def rough_from_raw(date_str: str, rows: list) -> str:
    """从一天的原始聊天记录直接生成粗略归档（1~3 句话）。"""
    text = transcript(rows)
    if not text.strip():
        return ""
    prompt = prompts.ROUGH_ARCHIVE.format(
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


async def rough_from_summary(date_str: str, daily_text: str) -> str:
    """把已有的详细每日概括压缩成粗略归档（更省输入）。"""
    prompt = prompts.ROUGH_FROM_SUMMARY.format(date=date_str, daily=daily_text)
    return await _chat(
        [{"role": "user", "content": prompt}],
        1200,
        config.SUMMARY_TEMPERATURE,
    )


async def merge_longterm(existing: str, rough_texts: list) -> str:
    """把粗略归档合并进长期记忆（大概级别，全文 800 字内）。"""
    prompt = prompts.LONGTERM_MERGE.format(
        longterm=(existing or "").strip() or "（暂无）",
        roughs="\n".join(rough_texts),
        max_chars=config.LONGTERM_MAX_CHARS,
    )
    return await _chat(
        [{"role": "user", "content": prompt}],
        config.SUMMARY_MAX_TOKENS + 1500,  # 输出2000字(约1400token)的余量
        config.SUMMARY_TEMPERATURE,
    )
