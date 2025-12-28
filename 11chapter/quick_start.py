
import sys
import json

from hello_agents.tools import RLTrainingTool

rl_tools = RLTrainingTool()

sft_result_str = rl_tools.run({
    "action": "train",
    "algorthm": "sft",
    "model_name": "Qwen/Qwen3-0.6B",
    "output_dir": "./models/quick_test_sft",
    "max_samples": 10,
    "num_epochs": 1,
    "batch_size": 2,
    "user_lora": True
})


sft_result = json.loads(sft_result_str)
print(sft_result)
print(f"\n✓ SFT训练完成,模型保存在: {sft_result['output_dir']}")

grpo_result_str = rl_tools.run({
    "action": "train",
    "algorithm": "grpo",
    "model_name": "Qwen/Qwen3-0.6B", # 使用基础模型
    "output_dir": "./models/quick_test_grpo",
    "max_samples": 5, # 只用5个样本快速测试
    "num_epochs": 1,
    "batch_size": 2, # 必须能被num_generations(8)整除,使用2
    "use_lora": True
})

grpo_result = json.loads(grpo_result_str)
print(f"\n✓ GRPO训练完成,模型保存在: {grpo_result['output_dir']}")

eval_result_str = rl_tools.run({
    "action": "evaluate",
    "model_path": "./models/quick_test_grpo",
    "max_samples": 10, # 在10个测试样本上评估
    "use_lora": True
})
eval_result = json.loads(eval_result_str)
print(f"\n✓ 评估完成:")
print(f" - 准确率: {eval_result['accuracy']}")
print(f" - 平均奖励: {eval_result['average_reward']}")
print(f" - 测试样本数: {eval_result['num_samples']}")

print("\n" + "=" * 50)
print("🎉 恭喜!你已经完成了第一个Agentic RL模型的训练!")
print("=" * 50)
print(f"\n模型路径:")
print(f" SFT模型: {sft_result['output_dir']}")
print(f" GRPO模型: {grpo_result['output_dir']}")



