"""记忆网关入口。

工作方式：
- App 里 DeepSeek 供应商只把 API 地址改成网关域名，其余一切照旧；
- 网关把请求原样透传给 DeepSeek（流式、思考链 reasoning_content、工具调用，一个字节不改）；
- 唯一的改动：检测到"新窗口"时把三段式记忆注入 system，且窗口期间冻结，
  不破坏 DeepSeek 的前缀缓存。
- 新增：窗口内滚动压缩——上下文太长时把老消息压成摘要，防止越聊越卡。
"""
import asyncio
import contextlib
import hmac
import json
import logging
import os
from datetime import timedelta

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Route

import config
import memory
import ob_client
import rolling
import store
import summarizer

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
log = logging.getLogger("gateway")


def _check_gateway(request: Request) -> bool:
    if not config.GATEWAY_KEY:
        return True  # 没配 Key 就不拦（不推荐）
    raw = request.headers.get("authorization", "")
    token = raw.split(" ", 1)[-1].strip() if " " in raw else raw.strip()
    return hmac.compare_digest(token, config.GATEWAY_KEY)


def _check_admin(request: Request) -> bool:
    secret = config.ADMIN_SECRET
    if not secret:
        return True
    # 支持浏览器直接访问：?key=管理钥匙
    if hmac.compare_digest(request.query_params.get("key", ""), secret):
        return True
    raw = request.headers.get("authorization", "")
    token = raw.split(" ", 1)[-1].strip() if " " in raw else raw.strip()
    return hmac.compare_digest(token, secret)


def _inject_system(messages: list, block: str) -> list:
    """把记忆块追加到第一条 system 消息后面（不覆盖人设，只在其后补充）。"""
    msgs = [dict(m) for m in messages]
    for i, m in enumerate(msgs):
        if m.get("role") == "system":
            base = memory._text_of(m.get("content")).rstrip()
            msgs[i]["content"] = f"{base}\n\n{block}" if base else block
            return msgs
    msgs.insert(0, {"role": "system", "content": block})
    return msgs


# ---- 请求透视（诊断用：只记数字和构成，不存任何聊天内容） ----
_xray: list = []  # 最近若干条经过网关的请求统计


def _content_stats(content) -> tuple:
    """返回 (文本字符数, 非文本分片数)。图片/音频等非文本分片也占token。"""
    if isinstance(content, str):
        return len(content), 0
    if isinstance(content, list):
        chars, nontext = 0, 0
        for p in content:
            if isinstance(p, dict):
                if p.get("type") == "text":
                    chars += len(p.get("text") or "")
                else:
                    nontext += 1
            elif isinstance(p, str):
                chars += len(p)
        return chars, nontext
    return 0, 0


def _xray_base(payload: dict, messages: list) -> dict:
    msgs, reasoning_msgs, reasoning_chars = [], 0, 0
    total_chars, total_nontext = 0, 0
    for m in messages or []:
        chars, nontext = _content_stats(m.get("content"))
        rc = str(m.get("reasoning_content") or "")
        total_chars += chars
        total_nontext += nontext
        if rc:
            reasoning_msgs += 1
            reasoning_chars += len(rc)
        msgs.append({"role": str(m.get("role") or "?"), "chars": chars,
                     "nontext_parts": nontext, "reasoning_chars": len(rc)})
    tools = payload.get("tools") or []
    return {
        "ts": memory._now().isoformat(timespec="seconds"),
        "model": payload.get("model"),
        "stream": bool(payload.get("stream")),
        "n_messages": len(messages or []),
        "total_message_chars": total_chars,
        "total_nontext_parts": total_nontext,
        "messages": msgs,
        "reasoning_messages": reasoning_msgs,
        "reasoning_chars": reasoning_chars,
        "tools_count": len(tools),
        "tools_chars": len(json.dumps(tools, ensure_ascii=False)) if tools else 0,
        "injected_block_chars": None,
    }


# ---- 普通接口 ----

async def index(_: Request):
    supabase_on = config.supabase_ready()
    longterm_chars = None
    dailies = []
    last_refresh = None
    if supabase_on:
        try:
            lt = await store.get_longterm()
            longterm_chars = len((lt or "").strip())
            min_day = (memory._now() - timedelta(days=config.RECENT_DAYS - 1)).strftime("%Y-%m-%d")
            dailies = await store.get_recent_dailies(min_day)
            last_refresh = await store.get_state("last_refresh")
        except Exception:
            log.exception("状态页读取失败")
    return JSONResponse({
        "ok": True,
        "说明": "记忆网关运行中。开窗自动注入：长期记忆 + 近期记忆 + 上个窗口原始聊天记录。",
        "deepseek_configured": config.deepseek_ready(),
        "supabase_configured": supabase_on,
        "ob_configured": bool(config.OB_MCP_URL),
        "summary_model": config.SUMMARY_MODEL,
        "longterm_chars": longterm_chars,
        "recent_daily": [{"date": d.get("scope"), "chars": len(d.get("content") or "")} for d in dailies],
        "cached_windows": len(memory._windows),
        "last_refresh": last_refresh,
    }, headers={"Cache-Control": "no-store"})


async def models_endpoint(_: Request):
    if not config.deepseek_ready():
        return JSONResponse({"object": "list", "data": []})
    client = _get_shared_client()
    r = await client.get(f"{config.DEEPSEEK_BASE_URL}/models",
                         headers={"Authorization": f"Bearer {config.DEEPSEEK_API_KEY}"})
    return Response(r.content, status_code=r.status_code, media_type="application/json",
                    headers={"Cache-Control": "no-store"})


# ---- 对话代理（核心） ----

# 全局共享 httpx 客户端：所有请求复用连接池，TLS 握手只做一次。
# 直连快而网关慢的主因就是这里每次请求都新建+关闭客户端（每条消息多一次 TLS 握手）。
_shared_client: httpx.AsyncClient | None = None


def _get_shared_client() -> httpx.AsyncClient:
    global _shared_client
    if _shared_client is None or _shared_client.is_closed:
        timeout = httpx.Timeout(connect=30, read=config.UPSTREAM_READ_TIMEOUT, write=30, pool=30)
        _shared_client = httpx.AsyncClient(timeout=timeout, http2=False)
    return _shared_client


async def chat_completions(request: Request):
    if not _check_gateway(request):
        return JSONResponse({"error": {"message": "Unauthorized：网关 Key 不对"}}, status_code=401)
    if not config.deepseek_ready():
        return JSONResponse({"error": {"message": "网关没配置 DEEPSEEK_API_KEY"}}, status_code=500)
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": {"message": "Invalid JSON"}}, status_code=400)

    messages = list(payload.get("messages") or [])
    info = memory.analyze_messages(messages)
    xrec = _xray_base(payload, messages)
    if info["first_user"]:
        try:
            block, wkey = await memory.get_window_block(info["system"], info["first_user"], info["assistant_count"])
            if block:
                total_chars = sum(len(memory._text_of(x.get("content"))) for x in messages)
                log.info("开窗注入：块 %d 字符；请求 %d 条消息 / %d 字符；窗口 %s",
                         len(block), len(messages), total_chars, wkey)
        except Exception:
            log.exception("窗口记忆装配异常")
            block = None
        if block:
            payload["messages"] = _inject_system(messages, block)
            xrec["injected_block_chars"] = len(block)
            log.info("窗口 %s 注入记忆 %d 字（历史 assistant 消息 %d 条）",
                     wkey, len(block), info["assistant_count"])
    _xray.append(xrec)
    del _xray[:-10]

    # 窗口内滚动压缩：上下文太长时压掉最老的一批，防止越聊越卡/超时断流
    try:
        payload["messages"] = await rolling.compress(
            payload["messages"], info["system"], info["first_user"]
        )
    except Exception:
        log.exception("滚动压缩失败（本次请求将原样转发）")

    want_stream = bool(payload.get("stream", False))
    target = f"{config.DEEPSEEK_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {config.DEEPSEEK_API_KEY}",
               "Content-Type": "application/json"}
    client = _get_shared_client()  # 复用全局连接池，不新建不关闭

    if not want_stream:
        try:
            r = await client.post(target, headers=headers, json=payload)
        except Exception as exc:
            log.exception("上游请求失败")
            return JSONResponse({"error": {"message": f"上游请求失败: {type(exc).__name__}"}}, status_code=502)
        return Response(r.content, status_code=r.status_code, media_type="application/json",
                        headers={"Cache-Control": "no-store"})

    async def relay():
        try:
            async with client.stream("POST", target, headers=headers, json=payload) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode("utf-8", errors="replace")[:500]
                    err = {"error": {"message": f"上游 DeepSeek HTTP {resp.status_code}: {body}"}}
                    yield f"data: {json.dumps(err, ensure_ascii=False)}\n\ndata: [DONE]\n\n".encode("utf-8")
                    return
                # 逐字节原样转发：流式、思考链、工具调用全部不动。
                # 注意：不要改成 aiter_lines() 逐行转发——SSE 事件可能是多行的
                # （event:/data:/空行），逐行+补换行会把一个事件拆成两个，
                # 破坏事件边界。aiter_bytes() 原样转发才是正确做法。
                async for chunk in resp.aiter_bytes():
                    yield chunk
        except Exception as exc:
            log.exception("流式转发中断")
            err = {"error": {"message": f"网关到上游的连接中断: {type(exc).__name__}"}}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\ndata: [DONE]\n\n".encode("utf-8")
        # 不 close 客户端：全局复用，留给下一个请求

    return StreamingResponse(relay(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---- 管理接口 ----

async def admin_memory(request: Request):
    if not _check_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    longterm = await store.get_longterm()
    min_day = (memory._now() - timedelta(days=config.RECENT_DAYS - 1)).strftime("%Y-%m-%d")
    dailies = await store.get_recent_dailies(min_day)
    latest = await store.get_recent_dailies("0000-00-00", limit=10)
    absorbed_until = await store.get_state("absorbed_until")
    last_refresh = await store.get_state("last_refresh")
    last_errors = await store.get_state("last_errors")
    return JSONResponse({
        "longterm": longterm,
        "last_errors": last_errors,
        "longterm_chars": len((longterm or "").strip()),
        "recent_daily": [{"date": d.get("scope"), "content": d.get("content")} for d in dailies],
        "latest_daily": [{"date": d.get("scope"), "chars": len(d.get("content") or "")} for d in latest],
        "absorbed_until": absorbed_until,
        "last_refresh": last_refresh,
    }, headers={"Cache-Control": "no-store"})


async def admin_refresh(request: Request):
    if not _check_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    try:
        days = int(request.query_params.get("days", "0"))
    except Exception:
        days = 0
    try:
        seeded = await memory.seed_from_ob_if_empty()
    except Exception:
        log.exception("OB 搬运失败")
        seeded = False
    try:
        stats = await memory.refresh_once(days)
    except Exception as exc:
        log.exception("手动刷新失败")
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
    return JSONResponse({"ok": True, "ob_seeded": seeded, **stats})


async def admin_restyle(request: Request):
    """把旧格式的长期记忆改写为第一人称（称呼统一为 USER_LABEL）。"""
    if not _check_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    try:
        existing = (await store.get_longterm() or "").strip()
    except Exception as exc:
        return JSONResponse({"ok": False, "error": f"读取失败: {exc}"}, status_code=500)
    if not existing:
        return JSONResponse({"ok": False, "error": "长期记忆为空，无需改写"})
    try:
        text = await summarizer.restyle_longterm(existing)
    except Exception as exc:
        log.exception("长期记忆改写失败")
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
    if not text.strip():
        return JSONResponse({"ok": False, "error": "模型返回了空内容，未做修改"})
    try:
        await store.set_memory("longterm", "", text, {"source": "restyle"})
    except Exception as exc:
        return JSONResponse({"ok": False, "error": f"保存失败: {exc}"}, status_code=500)
    return JSONResponse({"ok": True, "before_chars": len(existing), "after_chars": len(text)})


async def admin_import_ob(request: Request):
    if not _check_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not config.OB_MCP_URL:
        return JSONResponse({"ok": False, "error": "OB_MCP_URL 未配置"})
    mode = request.query_params.get("mode", "auto")
    try:
        text = (await ob_client.fetch_breath()).strip()
    except Exception as exc:
        return JSONResponse({"ok": False, "error": f"OB 读取失败: {exc}"})
    if not text:
        return JSONResponse({"ok": False, "error": "OB 返回了空内容"})
    existing = ""
    try:
        existing = (await store.get_longterm() or "").strip()
    except Exception:
        pass
    if existing and mode != "replace":
        return JSONResponse({
            "ok": False,
            "ob_fetched_chars": len(text),
            "hint": "长期记忆已有内容。确认要覆盖请用 /admin/import_ob?mode=replace&key=... （会丢掉网关自己总结的内容！）",
        })
    try:
        await store.set_memory("longterm", "", text, {"source": "ombre_breath_verbatim"})
    except Exception as exc:
        return JSONResponse({"ok": False, "ob_fetched_chars": len(text), "error": f"保存失败: {exc}"})
    return JSONResponse({"ok": True, "ob_fetched_chars": len(text), "saved": True, "mode": mode})


async def admin_preview(request: Request):
    if not _check_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    hint = request.query_params.get("first_user", "")
    try:
        result = await memory.build_preview(hint)
    except Exception as exc:
        log.exception("预览失败")
        result = {"error": str(exc)}
    return JSONResponse(result, headers={"Cache-Control": "no-store"})


async def debug_xray(request: Request):
    """请求透视：最近若干条经过网关的请求构成（只含数字，不含聊天内容）。

    鉴权：网关 Key（Authorization: Bearer）或管理钥匙（?key=）均可。
    """
    if not (_check_gateway(request) or _check_admin(request)):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return JSONResponse({"recent": _xray}, headers={"Cache-Control": "no-store"})


# ---- 后台任务 ----

async def background_worker():
    if not config.AUTO_REFRESH:
        log.info("AUTO_REFRESH=0，后台自动刷新已关闭。请在 /console 手动触发。")
        return
    await asyncio.sleep(3)
    while True:
        try:
            if await memory.seed_from_ob_if_empty():
                log.info("已从 Ombre Brain 搬运初始长期记忆")
        except Exception:
            log.exception("OB 初始搬运失败（下次再试）")
        try:
            stats = await memory.refresh_once(0)
            log.info("记忆刷新完成: %s", stats)
        except Exception:
            log.exception("记忆刷新失败")
        await asyncio.sleep(max(600.0, config.REFRESH_HOURS * 3600.0))


@contextlib.asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(background_worker())
    yield
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    # 关闭共享连接池
    global _shared_client
    if _shared_client is not None and not _shared_client.is_closed:
        await _shared_client.aclose()
    _shared_client = None


async def console(_: Request):
    return FileResponse("console.html", media_type="text/html")


app = Starlette(routes=[
    Route("/", index, methods=["GET"]),
    Route("/console", console, methods=["GET"]),
    Route("/models", models_endpoint, methods=["GET"]),
    Route("/v1/models", models_endpoint, methods=["GET"]),
    Route("/chat/completions", chat_completions, methods=["POST"]),
    Route("/v1/chat/completions", chat_completions, methods=["POST"]),
    Route("/admin/memory", admin_memory, methods=["GET"]),
    Route("/admin/refresh", admin_refresh, methods=["GET", "POST"]),
    Route("/admin/restyle", admin_restyle, methods=["GET", "POST"]),
    Route("/admin/import_ob", admin_import_ob, methods=["GET", "POST"]),
    Route("/admin/preview", admin_preview, methods=["GET"]),
    Route("/debug/xray", debug_xray, methods=["GET"]),
], lifespan=lifespan)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, access_log=False, timeout_keep_alive=120)
