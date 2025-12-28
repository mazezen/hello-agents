
from hello_agents.tools import RLTrainingTool
import json

rl_tool = RLTrainingTool()

sft_result = rl_tool.run({
    "action": "load_dataset",
    "format": "sft",
    "max_samples": 5
    })

sft_data = json.loads(sft_result)

print(f"数据集大小: {sft_data['dataset_size']}")
print(f"数据格式: {sft_data['format']}")
print(f"样本字段: {sft_data['sample_keys']}")


rl_result = rl_tool.run({
    "action": "load_dataset",
    "format": "rl",
    "max_samples": 5
    })

rl_data = json.loads(rl_result)

print(f"数据集大小: {rl_data['dataset_size']}")
print(f"数据格式: {rl_data['format']}")
print(f"样本字段: {rl_data['sample_keys']}")


