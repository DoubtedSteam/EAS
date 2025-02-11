#!/bin/bash

# IMPORTANT: this is the training script for the original LLaVA, NOT FOR LLaVA V1.5!
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TRANSFORMERS_OFFLINE=1

torchrun --nproc_per_node 4 --master_port 11113 \
    llava/train/train_mem.py \
    --mm_projector_lr 4e-4 \
    --adapt_dim 128 \
    --replaced_dim 256 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-7b-v1.3 \
    --version v1 \
    --data_path ../data/llava_train_QCM-ALE.json \
    --image_folder ../data/images/train \
    --vision_tower openai/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-pretrain-vicuna-7b-v1.3/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-7b-v1.3-ScienceQA-PIA-Skip \
    --num_train_epochs 20 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 4e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --skipped_num 0 \
    --finetuning True 
