export CUDA_VISIBLE_DEVICES=0

python -m llava.eval.model_vqa_kzx \
    --model-path ./checkpoints/llava-v1.5-slake-skip12 \
    --question-file json_data/slake/testval_llava_en_new.json \
    --image-folder path/to/slake/Slake1.0/imgs \
    --answers-file ./checkpoints/llava-v1.5-slake.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_kzx.py \
    --gt json_data/slake/testval_llava_en.json \
    --pred ./checkpoints/llava-v1.5-slake.jsonl