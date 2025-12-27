from hello_agents import SimpleAgent, HelloAgentsLLM, ToolRegistry
from hello_agents.tools import MemoryTool

# 仅用最简单的配置进行测试
llm = HelloAgentsLLM()
agent = SimpleAgent(name="记忆助手", llm=llm)

# 简化初始化
memory_tool = MemoryTool(user_id="f6f402a9")
tool_registry = ToolRegistry()
tool_registry.register_tool(memory_tool)
agent.tool_registry = tool_registry

print("记忆助手初始化完成")
