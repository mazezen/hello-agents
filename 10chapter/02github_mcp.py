"""
GitHub MCP 服务示例
注意：需要设置环境变量
Windows: $env:GITHUB_PERSONAL_ACCESS_TOKEN="your_token_here"
Linux/macOS: export GITHUB_PERSONAL_ACCESS_TOKEN="your_token_here"
"""


from types import resolve_bases
from hello_agents.tools import MCPTool

github_tool = MCPTool(
        server_command=["npx", "-y", "@modelcontextprotocol/server-github"]
        )

print("📋 可用工具: ")
result = github_tool.run({"action": "list_tools"})
print(result)


# 内置服务器延迟 Memory 传输

print("内置可用的工具: ")
mcp_tool = MCPTool()
result2 = mcp_tool.run({"action": "list_tools"})
print(result2)

# 调用工具
result3 = mcp_tool.run({
    "action": "call_tool",
    "tool_name": "add",
    "arguments": {"a": 10, "b": 20}
    })
print(result3)


mcp_tool = MCPTool(server_command=["python", "my_mcp_server.py"])
mcp_tool = MCPTool(server_command=["npx", "-y", "@modelcontextprotocol/server-filesystem", "."])

result4 = mcp_tool.run({"action": "list_tools"})
print("文件服务器可用工具: ")
print(result4)

result5 = mcp_tool.run({
    "action": "call_tool",
    "tool_name": "read_file",
    "arguments": {"path": ".env.example"}
    })

print(result5)
