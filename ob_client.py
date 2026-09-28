"""Ombre Brain MCP 客户端。

只用于「搬运」：breath 返回的就是记忆桶的逐字内容（OB 明确不调 LLM 摘要/改写），
所以搬运过程零幻觉。只在长期记忆为空时用一次，之后网关完全自理。
"""
import json
import logging

import httpx

import config

log = logging.getLogger(__name__)


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
    """调用 OB 的一个 MCP 工具，返回 text 结果。OB 无需鉴权、无会话状态。"""
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": tool, "arguments": args}}
    headers = {"Content-Type": "application/json",
               "Accept": "application/json, text/event-stream"}
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


async def fetch_breath() -> str:
    """原样取回 OB 的浮现记忆 + 核心准则（逐字内容，不做任何加工）。"""
    return await call_tool("breath", {})
