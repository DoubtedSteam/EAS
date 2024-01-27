export MASTER_ADDR='0.0.0.0'
export MASTER_PORT='7999'
export NODE_RANK='0'
export CUDA_VISIBLE_DEVICES='2'
python train.py with data_root=./arrows num_gpus=1 num_nodes=1 task_finetune_vqa_clip_bert per_gpu_batchsize=1 clip16 text_roberta image_size=384 test_only=True \
# load_path='result/Fusion_4/checkpoints/epoch=9-step=12599.ckpt' \
