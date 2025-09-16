# mcp_dynamic_tools_url.py
from __future__ import annotations
import os, json, asyncio
from typing import Any, Dict, List, Optional, Callable
from fastmcp import Client
from langchain.tools import DynamicTool

# Lee la URL y headers desde env
MCP_URL = os.getenv("MCP_URL", "https://potential-palm-tree-4r7vvrv49w6fjgvr-8080.app.github.dev/mcp").strip()
# Opcional: JSON con headers ({"Authorization":"Bearer ..."})
MCP_HEADERS_JSON = os.getenv("MCP_HEADERS_JSON", "").strip()

def _parse_headers() -> Dict[str, str]:
    if not MCP_HEADERS_JSON:
        return {}
    try:
        return json.loads(MCP_HEADERS_JSON)
    except Exception:
        return {}

def _run(coro):
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    try:
        asyncio.get_running_loop()  # ¿ya hay loop? (LangGraph dev lo tiene)
    except RuntimeError:
        return asyncio.run(coro)    # no hay loop → OK
    # hay loop → ejecuta en un hilo aparte
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(lambda: asyncio.run(coro))
        return fut.result()


async def _list_tools_async(only: Optional[List[str]], headers: Dict[str, str]):
    if not MCP_URL:
        raise RuntimeError("MCP_URL no está definido")
    # Puedes pasar dict de configuración (permite headers y más)
    config = {
        "mcpServers": {
            "remote": {
                "url": MCP_URL,
                # FastMCP infiere el transporte HTTP/SSE por la URL
                "headers": headers or {},
            }
        }
    }
    client = Client(config)
    async with client:
        tdefs = await client.list_tools()
        # devolvemos (client_config, lista de metadatos)
        metas = [{"name": t.name, "description": t.description or f"MCP tool: {t.name}"} for t in tdefs.tools]
        if only:
            metas = [m for m in metas if m["name"] in only]
        return metas

def _call_tool_once(tool_name: str, kwargs: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
    async def _run_once():
        config = {"mcpServers": {"remote": {"url": MCP_URL, "headers": headers or {}}}}
        client = Client(config)
        async with client:
            res = await client.call_tool(tool_name, kwargs)
            # FastMCP Client devuelve objetos con .data para resultado “friendly”
            return getattr(res, "data", getattr(res, "__dict__", {"ok": True, "tool": tool_name}))
    return _run(_run_once())

def build_tools_from_mcp_url(only: Optional[List[str]] = None) -> List[DynamicTool]:
    """
    Descubre tools en MCP_URL y crea DynamicTools de LangChain.
    Usa 'only' para quedarte con un subconjunto por nombre.
    """
    headers = _parse_headers()
    metas = _run(_list_tools_async(only, headers))
    tools: List[DynamicTool] = []

    for meta in metas:
        name = meta["name"]
        desc = meta["description"]

        # importate: fijamos _name en default arg para cerrar sobre valor actual
        def _make(_name: str) -> Callable[..., Any]:
            def _fn(**kwargs):
                return _call_tool_once(_name, kwargs, headers)
            return _fn

        tools.append(
            DynamicTool(
                name=name,
                description=desc + " (discovered via MCP_URL)",
                func=_make(name),
                return_direct=False,
            )
        )
    return tools
