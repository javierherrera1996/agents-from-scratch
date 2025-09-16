# policy_mcp_server.py
from __future__ import annotations
from typing import Any, Optional, Dict, List
import time, os, logging
from mcp.server.fastmcp import FastMCP

import nest_asyncio

nest_asyncio.apply()
# ⚙️ Server MCP (nombre que verá el cliente)
mcp = FastMCP("policies-mock")

# 🧠 Estado en memoria (mock)
POLICIES: Dict[str, Dict[str, Any]] = {}
_ID = 1000
def _next_id() -> str:
    global _ID
    _ID += 1
    return f"P{_ID}"

# ---------- TOOLS (MCP) ----------

@mcp.tool()
def ping() -> str:
    """Quick healthcheck."""
    return "pong"

@mcp.tool()
def get_policy(policy_id: str) -> Dict[str, Any]:
    """Get one policy by id."""
    p = POLICIES.get(policy_id)
    if not p:
        return {"ok": False, "error": "NOT_FOUND", "policy_id": policy_id}
    return {"ok": True, "policy": p}

@mcp.tool()
def list_policies(limit: int = 20) -> Dict[str, Any]:
    """List last N policies."""
    items = list(POLICIES.values())[-limit:]
    return {"ok": True, "items": items, "count": len(items)}

@mcp.tool()
def verify_policy_info(
    action: str,
    policy_id: Optional[str] = None,
    customer_id: Optional[str] = None,
    fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate minimum info before CRUD."""
    action_l = (action or "").lower()
    required = {
        "create": ["customer_id", "fields"],
        "update": ["policy_id", "fields"],
        "delete": ["policy_id"],
    }.get(action_l, [])
    given = {"policy_id": policy_id, "customer_id": customer_id, "fields": fields or {}}
    missing = [k for k in required if not given.get(k)]
    ok = len(missing) == 0 and action_l in ("create", "update", "delete")
    return {
        "ok": ok,
        "action": action_l,
        "missing": missing,
        "normalized_fields": given.get("fields", {}),
    }

@mcp.tool()
def mcp_create_policy(customer_id: str, fields: Dict[str, Any]) -> Dict[str, Any]:
    """Create a policy (mock)."""
    pid = _next_id()
    policy = {
        "policy_id": pid,
        "customer_id": customer_id,
        "fields": dict(fields),
        "approved": False,
        "locked": False,
        "deleted": False,
        "ts": time.time(),
    }
    POLICIES[pid] = policy
    return {"ok": True, "status": "CREATED", "policy": policy}

@mcp.tool()
def mcp_update_policy(policy_id: str, fields: Dict[str, Any]) -> Dict[str, Any]:
    """Update a policy (mock)."""
    p = POLICIES.get(policy_id)
    if not p or p.get("deleted"):
        return {"ok": False, "error": "NOT_FOUND_OR_DELETED", "policy_id": policy_id}
    p["fields"].update(fields or {})
    p["ts"] = time.time()
    return {"ok": True, "status": "UPDATED", "policy": p}

@mcp.tool()
def mcp_delete_policy(policy_id: str) -> Dict[str, Any]:
    """Delete (soft) a policy (mock)."""
    p = POLICIES.get(policy_id)
    if not p:
        return {"ok": False, "error": "NOT_FOUND", "policy_id": policy_id}
    p["deleted"] = True
    p["ts"] = time.time()
    return {"ok": True, "status": "DELETED", "policy": p}

@mcp.tool()
def approve_policy(policy_id: str) -> Dict[str, Any]:
    """Approve a policy (mock)."""
    p = POLICIES.get(policy_id)
    if not p or p.get("deleted"):
        return {"ok": False, "error": "NOT_FOUND_OR_DELETED", "policy_id": policy_id}
    p["approved"] = True
    p["ts"] = time.time()
    return {"ok": True, "status": "APPROVED", "policy": p}

@mcp.tool()
def unlock_policy(policy_id: str) -> Dict[str, Any]:
    """Unlock a policy (mock)."""
    p = POLICIES.get(policy_id)
    if not p or p.get("deleted"):
        return {"ok": False, "error": "NOT_FOUND_OR_DELETED", "policy_id": policy_id}
    p["locked"] = False
    p["ts"] = time.time()
    return {"ok": True, "status": "UNLOCKED", "policy": p}

@mcp.tool()
def request_missing_info(policy_id: str, fields: List[str]) -> Dict[str, Any]:
    """Ask for missing fields (mock)."""
    return {"ok": True, "status": "REQUESTED", "policy_id": policy_id, "missing": fields}

from starlette.applications import Starlette
from starlette.routing import Mount
import uvicorn

app = Starlette(routes=[Mount("/mcp", app=mcp.streamable_http_app())])
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)