# 注意：MCPTool 主要用于 Stdio 和 Memory 传输
# 对于 HTTP/SSE 等远程传输，建议使用底层的 MCPClient


import asyncio
from hello_agents.protocols import MCPClient
from mcp.types import ClientNotificationType

async def test_http_transport():

    client = MCPClient("http://api.example.com/mcp")

    async with client:

        tools = await client.list_tools()
        print(f"远程服务器工具: {len(tools)} 个")

        result = await client.call_tool("process_data", {
            "data": "Hello, world!",
            "operation": "uppercase"
            })

        print(f"远程处理结果: ", result)

