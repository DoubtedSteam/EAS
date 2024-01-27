## Setup
### Install Package 
```bash
conda create -n lavin python=3.8 -y
conda activate lavin

# install pytorch
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch

# install dependency and lavin
pip install -r requirements.txt
pip install -e .
```
### Data Preparation
- For ScienceQA, please prepare the dataset from the [official repo](https://github.com/lupantech/ScienceQA).
- Obtain the weights of LLaMA from [this form](https://forms.gle/jk851eBVbX1m5TAv5)  (official) or Download [LLaMA-7B](https://huggingface.co/nyanko7/LLaMA-7B/tree/main) and [LLaMA-13B](https://huggingface.co/TheBloke/llama-13b) from HuggingFace (unofficial).

```bash
EAS/
  |-- eas
  |-- scripts
  |-- train.py
  |-- eval.py
  ......
data/
  |-- problem.json
  |-- pid_splits.json
  |-- captions.json
  |-- all_data.json
  |-- images
      |-- train          # ScienceQA train image
      |-- val            # ScienceQA val image
      |-- test           # ScienceQA test image
  |-- weights
      |-- tokenizer.model
          |--7B
              |-- params.json
              |-- consolidated.00.pth
```

## Fine-tuning
### ScienceQA
Reproduce the performance of EAS-7B on ScienceQA.
For 7B model, we fine-tune it on 2x A100.

#### Search
```bash
sh scripts/search_sqa_7b.sh
```

#### Train

After paste the search results into 'scripts/finetuning_sqa_7b.sh'.

```bash
sh scripts/finetuning_sqa_7b.sh
```

#### Evaluate

To enable the reparameterization, please set the max_batch_size to 1.

```bash
sh scripts/finetuning_sqa_7b.sh
```