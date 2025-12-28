
from hello_agents import SimpleAgent, HelloAgentsLLM
from hello_agents.tools import A2ATool
from dotenv import load_dotenv

load_dotenv()
llm = HelloAgentsLLM()


coodinator = SimpleAgent(name='协调者', llm=llm)

researcher_tool = A2ATool(
        name='researcher',
        description="研究员Agent，可以搜索和分析资料",
        agent_url="http://localhost:5000"
        )
coodinator.add_tool(researcher_tool)

response = coodinator.run('请让研究员帮我研究AI在教育领域的应用')
print(response)
