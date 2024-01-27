export MASTER_ADDR='0.0.0.0'
export MASTER_PORT='9000'
export NODE_RANK='0'
export CUDA_VISIBLE_DEVICES='2' 
python train.py with data_root=./arrows num_gpus=1 num_nodes=1  task_finetune_nlvr2_clip_bert per_gpu_batchsize=64 clip16 text_roberta image_size=288 test_only=True\
# load_path='result/finetune_nlvr2_seed0_from_meter_clip16_288_roberta_pretrain/version_37/checkpoints/epoch=9-step=3379.ckpt' \

