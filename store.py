"""Supabase REST 读写。

只读 chat_messages（聊天记录原文）；
读写 gateway_memory（网关自己的记忆：longterm / daily / rough / monthly / quarterly / yearly / state）。
永远不碰 memory_summaries（那个日记总结表有幻觉，已弃用，网关从头到尾不读它）。
"""
import json
import logging
from datetime import datetime, timezone

import httpx

import config

log = logging.getLogger(__name__)
TIMEOUT = httpx.Timeout(connect=15, read=30, write=15, pool=15)


def _headers(prefer: str | None = None) -> dict:
    h = {
        "apikey": config.SUPABASE_KEY,
        "Authorization": f"Bearer {config.SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    if prefer:
        h["Prefer"] = prefer
    return h


async def _sb_get(table: str, query: str) -> list:
    if not config.supabase_ready():
        return []
    url = f"{config.SUPABASE_URL}/rest/v1/{table}?{query}"
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        r = await client.get(url, headers=_headers())
    if r.status_code >= 400:
        log.warning("Supabase GET %s HTTP %s: %s", table, r.status_code, r.text[:200])
        return []
    data = r.json()
    return data if isinstance(data, list) else []


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def set_memory(kind: str, scope: str, content: str, meta: dict | None = None):
    """upsert 一行到 gateway_memory（按 kind+scope 唯一约束合并）。"""
    if not config.supabase_ready():
        raise RuntimeError("Supabase 未配置（SUPABASE_URL / SUPABASE_KEY）")
    body = {
        "kind": kind,
        "scope": scope,
        "content": content,
        "meta": json.dumps(meta or {}, ensure_ascii=False),
        "updated_at": _utc_now_iso(),
    }
    url = f"{config.SUPABASE_URL}/rest/v1/{config.MEMORY_TABLE}?on_conflict=kind,scope"
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        r = await client.post(url, headers=_headers("resolution=merge-duplicates"), json=body)
    if r.status_code >= 400:
        raise RuntimeError(f"Supabase 写入 {config.MEMORY_TABLE} 失败: HTTP {r.status_code} {r.text[:300]}")


# ---- 聊天记录（chat_messages，只读） ----

async def fetch_recent_chats(limit: int = 600) -> list:
    """最近 N 条聊天记录，按时间升序返回。"""
    q = "select=id,assistant_id,conversation_id,role,content,created_at&order=created_at.desc,id.desc"
    if config.ASSISTANT_ID:
        q += f"&assistant_id=eq.{config.ASSISTANT_ID}"
    q += f"&limit={max(1, min(limit, 5000))}"
    rows = await _sb_get(config.CHAT_TABLE, q)
    rows.reverse()
    return rows


async def fetch_chats_between(start_date, end_date, limit: int = 20000) -> list:
    """取 [start_date, end_date) 之间的聊天记录，按时间升序。start/end 是 date 对象。

    重要：Supabase 托管接口单次请求最多只返回约 1000 行。
    必须用 offset 翻页拉全，否则只能拿到范围内最旧的一批
    （这就是之前"9/5 之后的记录凭空消失"的原因）。
    """
    base = (
        f"select=id,assistant_id,conversation_id,role,content,created_at"
        f"&created_at=gte.{start_date.isoformat()}"
        f"&created_at=lt.{end_date.isoformat()}"
        f"&order=created_at.asc,id.asc"
    )
    if config.ASSISTANT_ID:
        base += f"&assistant_id=eq.{config.ASSISTANT_ID}"
    cap = max(1, min(limit, 20000))
    out, offset = [], 0
    while offset < cap:
        rows = await _sb_get(config.CHAT_TABLE, f"{base}&limit=1000&offset={offset}")
        if not rows:
            break
        out.extend(rows)
        offset += len(rows)  # 按实际返回条数推进，兼容服务端更小的单页上限
    return out[:cap]


# ---- 网关记忆（gateway_memory） ----

async def get_longterm() -> str | None:
    rows = await _sb_get(config.MEMORY_TABLE, "select=content,meta&kind=eq.longterm&limit=1")
    if not rows:
        return None
    return rows[0].get("content") or ""


async def get_daily(date_str: str) -> dict | None:
    rows = await _sb_get(config.MEMORY_TABLE, f"select=content,meta&kind=eq.daily&scope=eq.{date_str}&limit=1")
    return rows[0] if rows else None


async def get_recent_dailies(min_date_str: str, limit: int = 30) -> list:
    """取某日期之后（含）的每日概括，按日期倒序。"""
    q = (
        f"select=scope,content,meta&kind=eq.daily"
        f"&scope=gte.{min_date_str}&order=scope.desc&limit={max(1, min(limit, 60))}"
    )
    return await _sb_get(config.MEMORY_TABLE, q)


async def get_all_roughs(limit: int = 200) -> list:
    """取全部粗略归档（kind=rough），按日期升序。meta 用于 merged 标记。"""
    q = f"select=scope,content,meta&kind=eq.rough&order=scope.asc&limit={max(1, min(limit, 400))}"
    return await _sb_get(config.MEMORY_TABLE, q)


async def get_memory(kind: str, scope: str) -> dict | None:
    """读取某个 kind+scope 的一行（用于判断某月/季/年是否已生成）。"""
    rows = await _sb_get(config.MEMORY_TABLE, f"select=content,meta&kind=eq.{kind}&scope=eq.{scope}&limit=1")
    return rows[0] if rows else None


async def get_memories(kind: str, min_scope: str = "", limit: int = 50, desc: bool = True) -> list:
    """按 kind 批量读取，可按 scope 过滤、排序、限量。

    desc=True（默认）用于注入（取最近 N 条）；desc=False 用于滚动（按时间升序全量处理）。
    """
    q = f"select=scope,content,meta&kind=eq.{kind}"
    if min_scope:
        q += f"&scope=gte.{min_scope}"
    q += f"&order=scope.{'desc' if desc else 'asc'}&limit={max(1, min(limit, 500))}"
    return await _sb_get(config.MEMORY_TABLE, q)


async def delete_roughs():
    """长期记忆合并成功后，清掉已消化的粗略归档。"""
    if not config.supabase_ready():
        return
    url = f"{config.SUPABASE_URL}/rest/v1/{config.MEMORY_TABLE}?kind=eq.rough"
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        r = await client.delete(url, headers=_headers())
    if r.status_code >= 400:
        raise RuntimeError(f"清理 rough 失败: HTTP {r.status_code} {r.text[:200]}")


async def delete_dailies_before(min_date_str: str) -> int:
    """删除某日期之前的每日概括（已滑出近期窗口、并入长期记忆的死数据）。返回删除条数。"""
    if not config.supabase_ready():
        return 0
    url = f"{config.SUPABASE_URL}/rest/v1/{config.MEMORY_TABLE}?kind=eq.daily&scope=lt.{min_date_str}"
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        r = await client.delete(url, headers=_headers("return=representation"))
    if r.status_code >= 400:
        raise RuntimeError(f"清理 daily 失败: HTTP {r.status_code} {r.text[:200]}")
    try:
        data = r.json()
        return len(data) if isinstance(data, list) else 0
    except Exception:
        return 0


async def get_state(scope: str) -> str | None:
    rows = await _sb_get(config.MEMORY_TABLE, f"select=content&kind=eq.state&scope=eq.{scope}&limit=1")
    if not rows:
        return None
    return rows[0].get("content") or ""
