"""在 Agent 中使用天气 MCP 服务器"""

import os
from dotenv import load_dotenv
from hello_agents import SimpleAgent, HelloAgentsLLM
from hello_agents.tools import MCPTool

load_dotenv()

def create_weather_assistant():

    llm = HelloAgentsLLM()
    assistant = SimpleAgent(
        name="天气助手",
        llm=llm,
        system_prompt="""你是天气助手，可以查询城市天气。
使用 get_weather 工具查询天气，支持中文城市名。
"""
    )

    server_script = os.path.join(os.path.dirname(__file__), "weather_mcp.py")
    weather_tool = MCPTool(server_command=["python", server_script])
    assistant.add_tool(weather_tool)

    return assistant


def demo():
    assistant = create_weather_assistant()
    print("\n查询北京天气：")
    response = assistant.run("北京今天的天气怎么样?")
    print(f"回答: {response}\n")


def interactive():
    assistant = create_weather_assistant()

    while True:
        user_input = input("\n你: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            break
        response = assistant.run(user_input)
        print(f"助手: {response}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    else:
        interactive()
