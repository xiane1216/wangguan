"""Ombre Brain MCP 客户端。

只用于「搬运」：breath 返回的就是记忆桶的逐字内容（OB 明确不调 LLM 摘要/改写），
所以搬运过程零幻觉。只在长期记忆为空时用一次，之后网关完全自理。
"""
import json
import logging
import re

import httpx

import config

log = logging.getLogger(__name__)

_PIN_LINE = re.compile(r"^📌\s*\[([0-9a-f]+)\]")
_NAME_RE = re.compile(r"《([^》]+)》")
_DATE_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2}\s+\d{2}-\d{2}-\d{2}\s*")


def _parse(text: str) -> dict:
    t = (text or "").strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    # SSE 帧格式兜底
    for line in t.splitlines():
        if line.startswith("data:"):
            try:
                return json.loads(line[5:].strip())
            except Exception:
                continue
    raise RuntimeError(f"OB 返回了无法解析的内容: {t[:200]}")


async def call_tool(tool: str, args: dict) -> str:
    """调用 OB 的一个 MCP 工具，返回 text 结果。无会话状态；OB 开了 token 鉴权则自动携带。"""
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": tool, "arguments": args}}
    headers = {"Content-Type": "application/json",
               "Accept": "application/json, text/event-stream"}
    if config.OB_MCP_TOKEN:
        headers["Ombre-MCP-Token"] = config.OB_MCP_TOKEN
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=15, read=120, write=15, pool=15)) as client:
        r = await client.post(config.OB_MCP_URL, headers=headers, json=body)
    if r.status_code >= 400:
        raise RuntimeError(f"OB MCP HTTP {r.status_code}: {r.text[:200]}")
    data = _parse(r.text)
    if "error" in data:
        raise RuntimeError(f"OB MCP 错误: {data['error']}")
    result = data.get("result") or {}
    if result.get("isError"):
        raise RuntimeError(f"OB 工具执行失败: {json.dumps(result, ensure_ascii=False)[:300]}")
    for part in result.get("content") or []:
        if isinstance(part, dict) and part.get("type") == "text":
            return part.get("text", "")
    return json.dumps(result, ensure_ascii=False)


def _extract_bucket(raw: str, bucket_id: str) -> str:
    """从 breath_search 输出里抠出指定桶的逐字正文。

    目标块形如：`📌 [核心准则] [bucket_id:<id>]` 行之后、`👣 Footprint` /
    `---` / 下一桶头之前的所有行。没找到返回空串。
    """
    collecting = False
    out: list[str] = []
    for line in (raw or "").splitlines():
        s = line.strip()
        if not collecting:
            if ("[核心准则]" in s and f"[bucket_id:{bucket_id}]" in s):
                collecting = True
            continue
        if (s.startswith("👣 Footprint") or s == "---"
                or "[bucket_id:" in s or s.startswith("===")):
            break
        out.append(line)
    return "\n".join(out).strip("\n")


async def fetch_core_rules() -> tuple[list[str], list[str]]:
    """只拉核心准则订桶（pinned，逐字正文，全程 0 LLM）。

    兼容线上 OB 版本（pulse 行有 📌 标记、breath_search 命中钉选桶时渲染
    `📌 [核心准则] [bucket_id:xxx]` + 逐字正文）：
      1. pulse 列出全部钉选桶 id 与名称
      2. 逐桶用名称主题词 breath_search，按 bucket_id 精确匹配提取正文
    返回 (成功列表, 未取回的桶名列表)。
    """
    pulse = await call_tool("pulse", {})
    ids: list[str] = []
    names: dict[str, str] = {}
    for line in (pulse or "").splitlines():
        m = _PIN_LINE.match(line.strip())
        if not m:
            continue
        pid = m.group(1)
        ids.append(pid)
        nm = _NAME_RE.search(line)
        names[pid] = nm.group(1) if nm else pid
    if not ids:
        return [], []

    items: list[str] = []
    missing: list[str] = []
    for pid in ids:
        name = names[pid]
        q = _DATE_PREFIX.sub("", name).strip() or name
        body = ""
        for query in (q, name):  # 先用主题词搜，不行再退全名
            try:
                raw = await call_tool("breath_search", {"query": query})
            except Exception as exc:
                log.warning("breath_search(%s) 失败: %s", query, exc)
                continue
            body = _extract_bucket(raw, pid)
            if body:
                break
        if body:
            items.append(body)
        else:
            missing.append(name)
    return items, missing


async def fetch_breath() -> str:
    """原样取回 OB 的浮现记忆 + 核心准则（逐字内容，不做任何加工）。"""
    return await call_tool("breath", {})
