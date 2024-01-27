## Setup
### Install Package 
```bash
pip install -r requirements.txt
pip install -e .
```
### Data Preparation
#### The pre-trained model:
- METER-CLIP16-RoBERTa (resolution: 288^2) pre-trained on GCC+SBU+COCO+VG [link](https://github.com/zdou0830/METER/releases/download/checkpoint/meter_clip16_288_roberta_pretrain.ckpt)


#### Datasets:
- We follow [ViLT](https://github.com/dandelin/ViLT) and use `pyarrow` to serialize the datasets. See [this link](https://github.com/dandelin/ViLT/blob/master/DATA.md) for details.


## Search

To find the best network skipping k layers, please edit line 109 in ```METER-EAS/meter/modules/meter_module_adapter_nas.py```:

```python
self.register_buffer('nas_gate', torch.zeros(k))
```

Then run the search for f30k, nlvr or vqa by:

```bash
sh script {task_name}_search.py
```

And the search result will be print at the shell.

## Train

fill the search result into ```METER-EAS/meter/modules/meter_module_adapter_nonas.py``` line 276:

```python
self.apply_flag(torch.LongTensor([]))
```

Then run the train and evaluate for f30k, nlvr or vqa by:

```bash
sh script {task_name}_train.py
```