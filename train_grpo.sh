#!/bin/bash
set -e
ts=$(date +%Y%m%d_%H%M%S)

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes 8 --config_file configs/zero2.yaml grpo.py \
    --output_dir ./ckpt/glm4-9b-chat-lora-GRPO_test \
    --model_name_or_path ./model/glm-4-9b-chat_lora_merge \
    --max_prompt_length 2048 \
    --trust_remote_code \
    --max_completion_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-7 \
    --adam_beta1 0.9 \
    --adam_beta2 0.99 \
    --weight_decay 0.1 \
    --warmup_ratio 0.1 \
    --logging_steps 1 \
    --num_generations 8 \
    --save_steps 200 \
    --max_steps 1000 \
    --torch_dtype bfloat16 \
    --save_only_model \
    --eval_strategy steps \
    --eval_steps 100 \
    --gradient_checkpointing \
    --bf16 
    # --use_peft true \
    # --lora_r 16 \
    # --lora_alpha 32 \
echo "=== Training run started! Run 'tail -f train_logs_${ts}.out' to monitor. ==="