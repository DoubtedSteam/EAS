export MASTER_ADDR='0.0.0.0'
export MASTER_PORT='9003'
export NODE_RANK='0'
export CUDA_VISIBLE_DEVICES='3'
python train.py with data_root=./arrows num_gpus=1 num_nodes=1 task_finetune_irtr_f30k_clip_bert get_recall_metric=True \
per_gpu_batchsize=32 load_path='result/finetune_irtr_f30k_seed0_from_meter_clip16_288_roberta_pretrain/version_1/checkpoints/epoch=9-step=2939.ckpt' \
clip16 text_roberta image_size=384 test_only=True
