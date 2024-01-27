export MASTER_ADDR='0.0.0.0'
export MASTER_PORT='9005'
export NODE_RANK='0'
# export CUDA_VISIBLE_DEVICES='0,1,2,3' 
export CUDA_VISIBLE_DEVICES='4,5,6,7' 
python search.py with data_root=../datasets num_gpus=4 num_nodes=1 task_finetune_irtr_f30k_clip_bert get_recall_metric=False \
per_gpu_batchsize=4 load_path='meter_clip16_288_roberta_pretrain.ckpt' clip16 text_roberta image_size=384 clip_randaug clip_randaug