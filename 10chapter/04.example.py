
from hello_agents import HelloAgentsLLM, SimpleAgent
from hello_agents.tools import MCPTool

agent = SimpleAgent(name="助手", llm=HelloAgentsLLM())

mcp_tool = MCPTool(name="calculator")
agent.add_tool(mcp_tool)

response = agent.run("计算 15 乘以 16")
print(response)




# 🔗 连接外部服务器
# 示例1 连接到社区提供的文件系统服务器
agent = SimpleAgent(name="文件助手", llm=HelloAgentsLLM())
fs_tool = MCPTool(
        name="filesystem",
        description="访问本地文件系统",
        server_command=["npx", "-y", "@modelcontextprotocol/server-filesystem", "."]
        )
agent.add_tool(fs_tool)




