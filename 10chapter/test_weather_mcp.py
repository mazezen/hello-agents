"""测试天气查询 MCP 服务器"""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'HelloAgents'))
from hello_agents.protocols.mcp.client import MCPClient

async def test_weather_server():
    server_script = os.path.join(os.path.dirname(__file__), "weather_mcp.py")
    client = MCPClient(["python", server_script])

    try:
        async with client:
            info = json.loads(await client.call_tool("get_server_info", {}))
            print(f'服务器: {info['name']} v{info['version']}')

            cities = json.loads(await client.call_tool("list_supported_cities", {}))
            print(f"支持城市: {cities['count']} 个")

            weather = json.loads(await client.call_tool("get_weather", {"city": "北京"}))
            if "error" not in weather:
                print(f"\n北京天气: {weather['temperature']}°C, {weather['condition']}")


            weather = json.loads(await client.call_tool("get_weather", {'city': "深圳"}))
            if "error" not in weather:
                print(f"深圳天气: {weather['temperature']}°C, {weather['condition']}")

            print("\n✅ 所有测试完成")
    except Exception as e:
        print(f"❌ 测试失败: {e}")


if __name__ == "__main__":
    asyncio.run(test_weather_server())
