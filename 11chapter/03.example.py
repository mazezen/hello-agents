

from hello_agents.tools import RLTrainingTool
import json
rl_tool = RLTrainingTool()


reward_result = rl_tool.run({
    "action": "create_reward",
    "reward_type": "accuracy"
    })

reward_data = json.loads(reward_result)

print(f"奖励类型: {reward_data['reward_type']}")
print(f"描述: {reward_data['description']}")

