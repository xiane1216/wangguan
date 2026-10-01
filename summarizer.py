"""DeepSeek 总结器。

模型：deepseek-flash（DeepSeek-V4.1-Flash 的官方 API 名）。
所有总结都带防幻觉铁律：只概括真实聊过、原文明确出现的内容，禁止编造。

记忆分层：
- 详细每日（最近3天用）：summarize_day
- 粗略归档（老日子用，一天一两句话）：rough_from_raw / rough_from_summary
- 月度概览：summarize_month
- 季度概览：summarize_quarter
- 年度概览：summarize_year
- 长期记忆合并（旧逻辑兜底）：merge_longterm
- 窗口内滚动摘要（治越聊越卡）：summarize_rollup
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


async def _chat(messages: list, max_tokens: int, temperature: float, strict_full: bool = False) -> str:
    if not config.deepseek_ready():
        raise RuntimeError("DEEPSEEK_API_KEY 未配置")
    payload = {
        "model": config.SUMMARY_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
        # 关闭思考模式：deepseek-flash 默认 high 强度推理，会把 max_tokens 烧光导致无正文
        "thinking": {"type": "disabled"},
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
    if strict_full:
        finish = ((data.get("choices") or [{}])[0]).get("finish_reason") or ""
        if finish == "length":
            # 输出顶到 max_tokens 被截断——总结按时间顺序写，被砍的正是"晚上"，
            # 绝不能当正常结果存库，必须让调用方重试。
            raise RuntimeError(f"输出达到max_tokens上限被截断（completion_tokens={usage.get('completion_tokens')}）")
    return _clean(content)


def _clip(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + "…(过长已截断)"


def _hard_cap(text: str, max_chars: int) -> str:
    """硬截断：最终字数绝不超过 max_chars（在段落边界截，尽量不砍半句）。"""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    # 尽量回退到最近的换行，避免把一句话砍成两半
    idx = cut.rfind("\n")
    if idx > max_chars * 0.6:
        cut = cut[:idx]
    return cut


def transcript(rows: list, max_messages: int = 400, clip: int = 2000) -> str:
    """把聊天记录转成「[时间] 角色: 内容」纯文本（原文摘录用，零加工）。

    注意：rows 按时间升序。日总结要覆盖全天，必须取"最近 max_messages 条"（rows[-max_messages:]），
    不能取 rows[:max_messages]（那样会把最新的消息丢掉）。"""
    picked = rows[-max_messages:] if len(rows) > max_messages else rows
    out = []
    for r in picked:
        role = r.get("role")
        if role not in ("user", "assistant"):
            continue
        who = config.USER_LABEL if role == "user" else config.AI_LABEL
        ts = str(r.get("created_at") or "").replace("T", " ")
        hm = ts[11:16] if len(ts) >= 16 else ""
        prefix = f"[{hm}] " if hm else ""
        out.append(f"{prefix}{who}: {_clip(r.get('content'), clip)}")
    return "\n".join(out)


_RETRY_SUFFIX = ("\n\n注意：你上一次的输出超过长度上限被截断了。这次必须更狠地压缩："
                "每个话题最多一两句、次要内容半句带过，但全天每个聊过的时段都必须保留，"
                "务必写到当天最后一条消息所在的时段（含深夜）。")


async def _head_notes(date_str: str, rows: list) -> str:
    """一天消息太多时，把"更早时段"先压成要点（治"只总结尾巴、丢上午"）。失败返回空串，退化为旧行为。"""
    text = transcript(rows, max_messages=2500, clip=400)
    if not text.strip():
        return ""
    prompt = prompts.DAY_HEAD.format(user_label=config.USER_LABEL, date=date_str, transcript=text)
    try:
        head = await _chat(
            [{"role": "user", "content": prompt}],
            config.HEAD_MAX_TOKENS,
            config.SUMMARY_TEMPERATURE,
        )
    except Exception:
        log.exception("较早时段要点生成失败（本次将只总结较晚时段）")
        return ""
    return _hard_cap(head, config.HEAD_MAX_CHARS)


async def _day_material(date_str: str, rows: list, clip: int) -> str:
    """把一整天的记录组装成总结素材：消息不超过 DAILY_RAW_MESSAGES 条时直接全量；
    超过时把更早的先压成"较早时段要点"拼在开头，保证全天覆盖。"""
    n = config.DAILY_RAW_MESSAGES
    if len(rows) <= n:
        return transcript(rows, max_messages=n, clip=clip)
    head = await _head_notes(date_str, rows[:-n])
    body = transcript(rows[-n:], max_messages=n, clip=clip)
    if not head.strip():
        return body
    return ("【当天较早时段聊天要点（已压缩，非原文）】\n" + head.strip()
            + "\n\n【当天较晚时段聊天原文】\n" + body)


async def summarize_day(date_str: str, rows: list) -> str:
    """详细每日概括（最近3天用）。覆盖全天，别漏掉最晚时段。

    治"总结不完"三道保险：
    1. 超量天拼"较早时段要点"，不再只看最后N条丢上午；
    2. 输出顶到 max_tokens（strict_full）会被拦下，附言重试一次"压缩更狠但覆盖全天"；
    3. 模型写超字数上限时，单独做一轮"只减细节不减时段"的压缩，硬截断只做最后兜底。
    """
    text = await _day_material(date_str, rows, clip=800)
    if not text.strip():
        return ""
    base_prompt = prompts.DAILY_SUMMARY.format(
        user_label=config.USER_LABEL,
        ai_label=config.AI_LABEL,
        date=date_str,
        max_chars=config.DAILY_MAX_CHARS,
        transcript=text,
    )
    summary = ""
    try:
        summary = await _chat(
            [{"role": "user", "content": base_prompt}],
            config.DAILY_MAX_TOKENS,
            config.SUMMARY_TEMPERATURE,
            strict_full=True,
        )
    except RuntimeError as exc:
        if "被截断" not in str(exc):
            raise  # 其他错误（空正文/HTTP）照常上抛，让按天循环记录失败
        log.warning("每日概括输出被截断，压缩后重试一次：%s", date_str)
        summary = await _chat(
            [{"role": "user", "content": base_prompt + _RETRY_SUFFIX}],
            config.DAILY_MAX_TOKENS,
            config.SUMMARY_TEMPERATURE,
            strict_full=True,
        )
    if len(summary) > config.DAILY_MAX_CHARS:
        # 超长但完整：做一轮"减细节不减时段"的压缩，避免硬截断砍掉晚上
        try:
            compact = await _chat(
                [{"role": "user", "content": prompts.DAILY_COMPRESS.format(
                    date=date_str,
                    max_chars=config.DAILY_MAX_CHARS,
                    summary=summary,
                    user_label=config.USER_LABEL,
                )}],
                config.DAILY_MAX_TOKENS,
                config.SUMMARY_TEMPERATURE,
                strict_full=True,
            )
            if compact.strip():
                summary = compact
        except Exception:
            log.exception("超长总结压缩失败：%s（退化为硬截断）", date_str)
    if len(summary) > config.DAILY_MAX_CHARS:
        log.warning("每日概括最终超长，硬截断兜底：%s（%d 字）", date_str, len(summary))
    return _hard_cap(summary, config.DAILY_MAX_CHARS)


async def rough_from_raw(date_str: str, rows: list) -> str:
    """从一天的原始聊天记录直接生成粗略归档（1~3 句话）。

    超量天同样拼"较早时段要点"——历史回填（比如补8月的粗归档）时，
    不能让月度概览只建立在"每天最后400条"的尾巴视角上。"""
    text = await _day_material(date_str, rows, clip=400)
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
    """把已有的详细每日概括压缩成粗略归档（更省输入）。
    bugfix：max_tokens 从 1200 提到 ROUGH_MAX_TOKENS（默认 8000）——
    flash 推理烧 token 是常态，1200 会被推理烧光导致空正文抛错，正好触发归档失败。"""
    prompt = prompts.ROUGH_FROM_SUMMARY.format(date=date_str, daily=daily_text, user_label=config.USER_LABEL)
    return await _chat(
        [{"role": "user", "content": prompt}],
        config.ROUGH_MAX_TOKENS,
        config.SUMMARY_TEMPERATURE,
    )


async def summarize_month(month_scope: str, texts: list) -> str:
    """把某月的每日/粗略记录压成月度概览。"""
    joined = "\n".join(t for t in texts if (t or "").strip())
    if not joined.strip():
        return ""
    prompt = prompts.MONTHLY_SUMMARY.format(
        user_label=config.USER_LABEL,
        scope=month_scope,
        max_chars=config.MONTHLY_MAX_CHARS,
        transcript=joined,
    )
    result = await _chat(
        [{"role": "user", "content": prompt}],
        config.MONTHLY_MAX_TOKENS,
        config.SUMMARY_TEMPERATURE,
    )
    return _hard_cap(result, config.MONTHLY_MAX_CHARS)


async def summarize_quarter(quarter_scope: str, texts: list) -> str:
    """把某季的月度概览压成季度概览。"""
    joined = "\n".join(t for t in texts if (t or "").strip())
    if not joined.strip():
        return ""
    prompt = prompts.QUARTERLY_SUMMARY.format(
        user_label=config.USER_LABEL,
        scope=quarter_scope,
        max_chars=config.QUARTERLY_MAX_CHARS,
        transcript=joined,
    )
    result = await _chat(
        [{"role": "user", "content": prompt}],
        config.QUARTERLY_MAX_TOKENS,
        config.SUMMARY_TEMPERATURE,
    )
    return _hard_cap(result, config.QUARTERLY_MAX_CHARS)


async def summarize_year(year_scope: str, texts: list) -> str:
    """把某年的季度概览压成年度概览。"""
    joined = "\n".join(t for t in texts if (t or "").strip())
    if not joined.strip():
        return ""
    prompt = prompts.YEARLY_SUMMARY.format(
        user_label=config.USER_LABEL,
        scope=year_scope,
        max_chars=config.YEARLY_MAX_CHARS,
        transcript=joined,
    )
    result = await _chat(
        [{"role": "user", "content": prompt}],
        config.YEARLY_MAX_TOKENS,
        config.SUMMARY_TEMPERATURE,
    )
    return _hard_cap(result, config.YEARLY_MAX_CHARS)


async def restyle_longterm(text: str) -> str:
    """把旧格式（第三人称/"用户"称呼）的长期记忆改写为第一人称，内容不变。"""
    if not (text or "").strip():
        return ""
    prompt = prompts.RESTYLE.format(longterm=text.strip(), user_label=config.USER_LABEL)
    return await _chat(
        [{"role": "user", "content": prompt}],
        config.SUMMARY_MAX_TOKENS,
        config.SUMMARY_TEMPERATURE,
    )


async def merge_longterm(existing: str, rough_texts: list) -> str:
    """把粗略归档合并进长期记忆（大概级别，全文 800 字内）。"""
    prompt = prompts.LONGTERM_MERGE.format(
        longterm=(existing or "").strip() or "（暂无）",
        roughs="\n".join(rough_texts),
        max_chars=config.LONGTERM_MAX_CHARS,
        user_label=config.USER_LABEL,
    )
    return await _chat(
        [{"role": "user", "content": prompt}],
        config.SUMMARY_MAX_TOKENS + 1500,  # 输出2000字(约1400token)的余量
        config.SUMMARY_TEMPERATURE,
    )


async def summarize_rollup(rows: list) -> str:
    """把窗口内被裁掉的早期消息压成滚动摘要（第一人称，防幻觉）。"""
    text = transcript(rows, max_messages=200, clip=800)
    if not text.strip():
        return ""
    prompt = prompts.ROLLUP.format(
        user_label=config.USER_LABEL,
        ai_label=config.AI_LABEL,
        transcript=text,
    )
    return await _chat(
        [{"role": "user", "content": prompt}],
        config.ROLLUP_MAX_TOKENS,
        config.SUMMARY_TEMPERATURE,
    )
