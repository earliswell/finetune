#!/bin/bash

MODEL_NAME="microsoft/Phi-4-multimodal-instruct"

export PYTHONPATH=src:$PYTHONPATH

# If you want to tune the `embed_token` with LoRA, You need to tune `lm_head` together
# The projector is included in the vision module, so you should freeze the img_projector with it.

deepspeed src/training/train.py \
    --lora_enable True \
    --vision_lora True \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_rank 64 \
    --lora_alpha 64 \
    --use_dora False \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --deepspeed scripts/zero3.json \
    --model_id $MODEL_NAME \
    --data_path /home/dabs/InHeon/car_analysis/llm_data/train_llm_data.json \
    --image_folder /home/dabs/InHeon/car_analysis/llm_data/llm_train_images \
    --tune_img_projector False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 True \
    --output_dir output/car \
    --num_crops 16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --dataloader_num_workers 4