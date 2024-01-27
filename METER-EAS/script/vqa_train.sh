export MASTER_ADDR='0.0.0.0'
export MASTER_PORT='8001'
export NODE_RANK='0'
export CUDA_VISIBLE_DEVICES='0,1'
python train.py with data_root=../datasets num_gpus=2 num_nodes=1 task_finetune_vqa_clip_bert per_gpu_batchsize=64 \
load_path='meter_clip16_288_roberta_pretrain.ckpt' \
clip16 text_roberta image_size=384 clip_randaug log_dir='train'
