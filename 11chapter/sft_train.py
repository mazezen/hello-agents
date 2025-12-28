

from hello_agents.tools import RLTrainingTool

rl_tool = RLTrainingTool()


result = rl_tool.run({
    "action": "train",
    "algorithm": "sft",


    "model_name": "Qwen/Qwen3-0.6B",
    "output_dir": "./models/quick_test_sft/",
    "max_samples": 100,
    "num_epochs": 3,
    "batch_size": 4,
    "learning_rate": 5e-5,
    "use_lora": True,
    "lora_rank": 8,
    "lora_alpha": 16,
    })

print(f"\n✅ 训练完成")

