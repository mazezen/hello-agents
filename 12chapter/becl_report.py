
from hello_agents import SimpleAgent, HelloAgentsLLM
from hello_agents.tools import BFCLEvaluationTool

llm = HelloAgentsLLM()
agent = SimpleAgent(name="TestAgent", llm=llm)

bfcl_tool = BFCLEvaluationTool()

results = bfcl_tool.run(
        agent=agent,
        category="simple_python",
        max_samples=5
        )

print(f"准确率: {results['overall_accuracy']:.2%}")
print(f"正确数: {results['correct_samples']}/{results['total_samples']}")

