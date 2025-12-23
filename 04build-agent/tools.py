from typing import Dict, Any
import os
from dotenv import load_dotenv
from serpapi import SerpApiClient

load_dotenv()

def search(query: str) -> str:
    """
    一个基于SerApi的实战网页搜索工具
    他会智能地解析搜索结果,优先返回直接答案或指示图谱信息
    """
    print(f"🔍 正在执行 [SerApi] 网页搜索: {query}")
    try:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return "错误: SERPAPI_API_KEY 未在 .env文件中配置."
        
        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "gl": "cn", # 国家代码
            "hl": "zh-cn", # 语言代码
        }

        client = SerpApiClient(params)
        results = client.get_dict()

        if "answer_box_list" in results:
            return "\n".join(results["answer_box_list"])
        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]
        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return results["knowledge_graph"]["description"]
        if "organic_results" in results and results["organic_results"]:
            snippets = [
                f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(results["organic_results"])
            ]
            return "\n\n".join(snippets)
        
        return f"对不起, 没有找到关于 '{query}'的信息."
    except Exception as e:
        return f"搜索时发生错误: {e}"


class ToolExecutor:

    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}
    
    def registerTool(self, name: str, description: str, func: callable):
        if name in self.tools:
            print(f"警告:工具 '{name}' 已存在,将被覆盖.")
        self.tools[name] = {"description":description, "func": func}
        print(f"工具 '{name}' 已注册. ")

    def getTool(self, name: str) -> callable:
        return self.tools.get(name, {}).get("func")
    
    def getAvailableTools(self) -> str:
        return "\n".join([
            f"- {name}: {info['description']}"
            for name, info in self.tools.items()
        ])
    
if __name__ == '__main__':
    toolExceutor = ToolExecutor()

    search_description = "一个网页搜索引擎.当你需要回答关于时事, 事实以及在你的知识库中找不到的信息时,应使用此工具."

    toolExceutor.registerTool("Search", search_description, search)

    print("\n --- 可用的工具 ---")
    print(toolExceutor.getAvailableTools())

    print("\n--- 执行 Action: Search['英伟达最新的GPU型号是什么'] ---")
    tool_name = "Search"
    tool_input = "英伟达最新的GPU型号是什么"

    tool_function = toolExceutor.getTool(tool_name)
    if tool_function:
        observation = tool_function(tool_input)
        print("--- 观察 (Observation) ---")
        print(observation)
    else:
        print(f"错误:未找到名为 '{tool_name}' 的工具。")