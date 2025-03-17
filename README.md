# Run
```bash
bash train_grpo.sh # 可full fine-tune/lora
```

## 奖励函数
- think_format_reward_func, think-answer的格式奖励
- answer_format_reward_func, 回答为具体要求的任务格式奖励：是否转供电等等（需要更改，目前匹配规则较严格，上一版训练效果不好）
- correctness_reward_func, (回答正确性奖励，以相似度分数衡量，emb模型为多语言的mmarco-mMiniLMv2-L12-H384，可以换成其他模型)