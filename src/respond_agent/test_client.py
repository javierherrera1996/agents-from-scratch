import asyncio, os
from fastmcp import Client

MCP_URL = os.environ.get("MCP_URL", "https://potential-palm-tree-4r7vvrv49w6fjgvr-8080.app.github.dev/mcp")  # usa /mcp si lo montaste ahí

async def main():
    c = Client({"mcpServers": {"remote": {"url": MCP_URL}}})
    async with c:
        tools = await c.list_tools()
        names = [t.name for t in tools.tools]
        print("TOOLS:", names)

        r = await c.call_tool("ping", {})
        print("PING:", r.data)

        # flujo CRUD mínimo
        r = await c.call_tool("mcp_create_policy", {
            "customer_id": "C-123",
            "fields": {"status": "active", "premium": 1200}
        })
        print("CREATE:", r.data)
        pid = r.data["policy"]["policy_id"]

        r = await c.call_tool("get_policy", {"policy_id": pid})
        print("GET:", r.data)

        r = await c.call_tool("mcp_update_policy", {"policy_id": pid, "fields": {"premium": 1350}})
        print("UPDATE:", r.data)

        r = await c.call_tool("mcp_delete_policy", {"policy_id": pid})
        print("DELETE:", r.data)

asyncio.run(main())
