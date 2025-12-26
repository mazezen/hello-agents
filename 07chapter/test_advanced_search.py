
from my_advanced_search import create_advanced_search_registry, MyAdvancedSearchTool


def test_advanced_search():

    registry = create_advanced_search_registry()

    test_queries = [
        "Python编程语言的历史",
        "人工智能的最新发展",
        "2024年科技趋势"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"测试 {i}: {query}")
        result = registry.execute_tool("advanced_search", query)
        print(f"结果: {result}\n")
        print("-" * 60 + "\n")

def test_api_configuration():

    print("🔧 测试API配置检查:")
    search_tool = MyAdvancedSearchTool()

    result = search_tool.search("机器学习算法")

    print(f"搜索结果: {result}")


def test_with_agent():
    """测试与Agent的集成"""
    print("\n🤖 与Agent集成测试:")
    print("高级搜索工具已准备就绪，可以与Agent集成使用")
    # 显示工具描述
    registry = create_advanced_search_registry()
    tools_desc = registry.get_tools_description()
    print(f"工具描述:\n{tools_desc}")


if __name__ == "__main__":
    test_advanced_search()
    test_api_configuration()
    test_with_agent()
