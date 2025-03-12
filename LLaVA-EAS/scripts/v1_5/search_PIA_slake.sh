#!/bin/bash
export CUDA_VISIBLE_DEVICES=4,5,6,7

torchrun --nproc_per_node 4 --master_port 11122 \
    llava/train/train_mem.py \
    --mm_projector_lr 4e-4 \
    --adapt_dim 128 \
    --replaced_dim 128 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --version v1 \
    --data_path json_data/slake/train_llava_en.json \
    --image_folder path/to/slake/Slake1.0/imgs \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-slake-skip12 \
    --num_train_epochs 40 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 4e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "linear" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1536 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --skipped_num 12 \
    --finetuning False \
    --warmup_steps 2280
