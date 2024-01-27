import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import math

import random

from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings, BertModel, BertEncoder, BertLayer
# from .bert_model_compact import BertCrossLayer, BertAttention
# from .bert_model_baseline import BertCrossLayer as BertCrossLayerBaseline
from .bert_model_PIA import BertCrossLayer, BertAttention
from . import swin_transformer as swin
from . import heads, objectives, meter_utils
from .clip_model import build_model, adapt_position_encoding
from .swin_helpers import swin_adapt_position_encoding
from transformers import RobertaConfig, RobertaModel

def tensor_in_list(tensor_list, new_tensor):
    for tensor in tensor_list:
        if torch.equal(tensor, new_tensor):
            return True
    return False

class METERTransformerSS(pl.LightningModule):
    def __init__(self, 
                 config,
                 get_val_loader,
                 trainable=["classifier", "pooler", "token_type_embeddings", "rank_output", "adapter"],
                 random=None
                ):
        super().__init__()
        self.save_hyperparameters(ignore='get_val_loader')

        self.get_val_loader = get_val_loader
        self.nas_val_size = config["per_gpu_batchsize"]

        self.is_clip= (not 'swin' in config['vit'])

        if 'roberta' in config['tokenizer']:
            bert_config = RobertaConfig(
                vocab_size=config["vocab_size"],
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_layers"],
                num_attention_heads=config["num_heads"],
                intermediate_size=config["hidden_size"] * config["mlp_ratio"],
                max_position_embeddings=config["max_text_len"],
                hidden_dropout_prob=config["drop_rate"],
                attention_probs_dropout_prob=config["drop_rate"],
            )
        else:
            bert_config = BertConfig(
                vocab_size=config["vocab_size"],
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_layers"],
                num_attention_heads=config["num_heads"],
                intermediate_size=config["hidden_size"] * config["mlp_ratio"],
                max_position_embeddings=config["max_text_len"],
                hidden_dropout_prob=config["drop_rate"],
                attention_probs_dropout_prob=config["drop_rate"],
            )

        resolution_after=config['image_size']

        self.cross_modal_text_transform = nn.Linear(config['input_text_embed_size'], config['hidden_size'])
        self.cross_modal_text_transform.apply(objectives.init_weights)
        self.cross_modal_image_transform = nn.Linear(config['input_image_embed_size'], config['hidden_size'])
        self.cross_modal_image_transform.apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)

        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                if self.is_clip:
                    build_model(config['vit'], resolution_after=resolution_after)
                else:
                    getattr(swin, self.hparams.config["vit"])(
                        pretrained=True, config=self.hparams.config,
                    )

                if 'roberta' in config['tokenizer']:
                    RobertaModel.from_pretrained(config['tokenizer'])
                else:
                    BertModel.from_pretrained(config['tokenizer'])

            torch.distributed.barrier()

        if self.is_clip:
            self.vit_model = build_model(config['vit'], resolution_after=resolution_after)
        else:
            self.vit_model = getattr(swin, self.hparams.config["vit"])(
                pretrained=True, config=self.hparams.config,
            )
            self.avgpool = nn.AdaptiveAvgPool1d(1)

        if 'roberta' in config['tokenizer']:
            self.text_transformer = RobertaModel.from_pretrained(config['tokenizer'])
        else:
            self.text_transformer = BertModel.from_pretrained(config['tokenizer'])

        self.cross_modal_image_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(config['num_top_layer'])])
        self.cross_modal_image_layers.apply(objectives.init_weights)
        self.cross_modal_text_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(config['num_top_layer'])])
        self.cross_modal_text_layers.apply(objectives.init_weights)

        self.register_buffer('skip_num', torch.ones(1) * 10)
        self.register_buffer('nas_gate', torch.zeros(24))

        self.nas_epoch = 10
        self.warmup_epoch = 1
        self.nas_step = 3
        self.nas_turn = 10
        self.nas_count = 0
        self.output_count = 0

        self.cross_modal_image_pooler = heads.Pooler(config["hidden_size"])
        self.cross_modal_image_pooler.apply(objectives.init_weights)
        self.cross_modal_text_pooler = heads.Pooler(config["hidden_size"])
        self.cross_modal_text_pooler.apply(objectives.init_weights)

        if config["loss_names"]["mlm"] > 0:
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)

        if config["loss_names"]["itm"] > 0:
            self.itm_score = heads.ITMHead(config["hidden_size"]*2)
            self.itm_score.apply(objectives.init_weights)

        hs = self.hparams.config["hidden_size"]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            vs = self.hparams.config["vqav2_label_size"]
            self.vqa_classifier = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, vs),
            )
            self.vqa_classifier.apply(objectives.init_weights)

        # ===================== Downstream ===================== #
        if (
            self.hparams.config["load_path"] != ""
            and not self.hparams.config["test_only"]
        ):
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            if self.is_clip:
                state_dict = adapt_position_encoding(state_dict, after=resolution_after, patch_size=self.hparams.config['patch_size'])
            else:
                state_dict = swin_adapt_position_encoding(state_dict, after=resolution_after, before=config['resolution_before'])
            self.load_state_dict(state_dict, strict=False)

        self.token4classifier = self.text_transformer.embeddings(torch.LongTensor([[0]]))
        self.token4classifier = nn.Parameter(self.token4classifier)

        if self.hparams.config["loss_names"]["nlvr2"] > 0:
            self.nlvr2_classifier = nn.Sequential(
                nn.Linear(hs * 4, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 2),
            )
            self.nlvr2_classifier.apply(objectives.init_weights)
            emb_data = self.token_type_embeddings.weight.data
            self.token_type_embeddings = nn.Embedding(3, hs)
            self.token_type_embeddings.apply(objectives.init_weights)
            self.token_type_embeddings.weight.data[0, :] = emb_data[0, :]
            self.token_type_embeddings.weight.data[1, :] = emb_data[1, :]
            self.token_type_embeddings.weight.data[2, :] = emb_data[1, :]

        if self.hparams.config["loss_names"]["snli"] > 0:
            self.snli_classifier = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 3),
            )
            self.snli_classifier.apply(objectives.init_weights)

        if self.hparams.config["loss_names"]["irtr"] > 0:
            self.rank_output = nn.Linear(hs, 1)
            self.rank_output.weight.data = self.itm_score.fc.weight.data[1:, :]
            self.rank_output.bias.data = self.itm_score.fc.bias.data[1:]
            self.margin = 0.2
            for p in self.itm_score.parameters():
                p.requires_grad = False

        meter_utils.set_metrics(self)
        self.current_tasks = list()

        self.trainable_param = trainable
        for n, p in self.named_parameters():
            if not any(t in n for t in self.trainable_param) or any(t in n for t in ['text_transformer', 'vit_model']):
                p.requires_grad = False
            else:
                print(n)

        orig_param_size = sum(p.numel() for p in self.parameters())
        trainable_size =  sum(p.numel() for p in self.parameters() if p.requires_grad)
        extra_param = sum(p.numel() for n, p in self.named_parameters() if "adapter" in n)
        print('extra parameter:{}'.format(extra_param))
        print('trainable_size:{:.4f}%({}/{})'.format(trainable_size / orig_param_size * 100, trainable_size, orig_param_size))

        # ===================== load downstream (test_only) ======================

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            if self.is_clip:
                state_dict = adapt_position_encoding(state_dict, after=resolution_after, patch_size=self.hparams.config['patch_size'])
            else:
                state_dict = swin_adapt_position_encoding(state_dict, after=resolution_after, before=config['resolution_before'])
            self.load_state_dict(state_dict, strict=True)

            select = torch.sort(self.nas_gate)[1]
            select = select[-round(self.skip_num.item()):]
            self.clean_flag()
            self.apply_flag(select)

            print()
            print(select)
            print('Fusion Image', self.nas_gate[:6])
            print('Fusion Text ', self.nas_gate[6: 12])
            print('Encoder Text', self.nas_gate[12:])
        
        # model = Backbone([
        #     self.cross_modal_text_transform,
        #     self.cross_modal_image_transform,
        #     self.token_type_embeddings,
        #     self.cross_modal_text_pooler,
        #     self.cross_modal_image_pooler,
        #     self.vit_model,
        #     self.text_transformer,
        #     self.cross_modal_image_layers,
        #     self.cross_modal_text_layers,
        #     self.adapter_list_l,
        #     self.adapter_list_v,
        #     self.adapter_list_encoder,
        #     self.skip_flag_l,
        #     self.skip_flag_v,
        #     self.skip_flag_encoder
        # ])

        # from thop import clever_format
        # from thop import profile

        # text_embeds = torch.rand(1, 40, 768)
        # image_embeds = torch.rand(1, 3, 384, 384)
        # flops, params = profile(model, inputs=(text_embeds, image_embeds, ))
        # print(flops, params)
        # flops, params = clever_format([flops, params], "%.3f")
        # print(flops, params)

        # from fvcore.nn import FlopCountAnalysis, parameter_count_table,flop_count_table
        # flops = FlopCountAnalysis(model, (text_embeds, image_embeds, ))
        # print("FLOPs: ", flops.total()/1e9)

        # exit()

    def infer(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        img=None,
    ):
        if img is None:
            if f"image_{image_token_type_idx - 1}" in batch:
                imgkey = f"image_{image_token_type_idx - 1}"
            else:
                imgkey = "image"
            img = batch[imgkey][0].cuda()

        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"].cuda()
        text_labels = batch[f"text_labels{do_mlm}"].cuda()
        text_masks = batch[f"text_masks"].cuda()

        with torch.no_grad():
            text_embeds = self.text_transformer.embeddings(input_ids=text_ids)
            
        device = text_embeds.device
        input_shape = text_masks.size()
        extend_text_masks = self.text_transformer.get_extended_attention_mask(text_masks, input_shape, device)
        for layer in self.text_transformer.encoder.layer:
            text_embeds = layer(text_embeds, extend_text_masks)[0]
        text_embeds = self.cross_modal_text_transform(text_embeds)

        with torch.no_grad():
            image_embeds = self.vit_model(img)

        image_embeds = self.cross_modal_image_transform(image_embeds)
        image_masks = torch.ones((image_embeds.size(0), image_embeds.size(1)), dtype=torch.long, device=device)
        extend_image_masks = self.text_transformer.get_extended_attention_mask(image_masks, image_masks.size(), device)

        if self.token4classifier is not None:
            token4classifiers = self.token4classifier.repeat(text_embeds.shape[0], 1, 1)
            text_embeds = torch.cat([token4classifiers, text_embeds[:, 1:, :]], dim=1)

        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds+ self.token_type_embeddings(torch.full_like(image_masks, image_token_type_idx)),
        )

        x, y = text_embeds, image_embeds
        for i, (text_layer, image_layer) in  enumerate(zip(self.cross_modal_text_layers, self.cross_modal_image_layers)):
            x1 = text_layer(x, y, extend_text_masks, extend_image_masks)
            y1 = image_layer(y, x, extend_image_masks, extend_text_masks)
            x, y = x1, y1

        text_feats, image_feats = x, y
        cls_feats_text = self.cross_modal_text_pooler(x)
        if self.is_clip:
            cls_feats_image = self.cross_modal_image_pooler(y)
        else:
            avg_image_feats = self.avgpool(image_feats.transpose(1, 2)).view(image_feats.size(0), 1, -1)
            cls_feats_image = self.cross_modal_image_pooler(avg_image_feats)
        cls_feats = torch.cat([cls_feats_text, cls_feats_image], dim=-1)

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
        }

        return ret
    

    def calculate_loss(self, ret, batch):
        ret = dict()

        # Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        # Image Text Matching
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm(self, batch))

        # Visual Question Answering
        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_vqa(self, batch))

        # Natural Language for Visual Reasoning 2
        if "nlvr2" in self.current_tasks:
            ret.update(objectives.compute_nlvr2(self, batch))

        # SNLI Visual Entailment
        if "snli" in self.current_tasks:
            ret.update(objectives.compute_snli(self, batch))

        # Image Retrieval and Text Retrieval
        if "irtr" in self.current_tasks:
            ret.update(objectives.compute_irtr(self, batch))

        return ret

    def clean_flag(self):
        for i in range(6):
            self.cross_modal_text_layers[i].skip_self_attn = False
            self.cross_modal_text_layers[i].skip_cross_attn = False
            self.cross_modal_image_layers[i].skip_self_attn = False
            self.cross_modal_image_layers[i].skip_cross_attn = False

    def apply_flag(self, select):
        for i in select:
            i = i.item()
            if i < 6:
                self.cross_modal_text_layers[i].skip_self_attn = True
            elif i < 12:
                self.cross_modal_text_layers[i - 6].skip_cross_attn = True
            elif i < 18:
                self.cross_modal_image_layers[i - 12].skip_self_attn = True
            elif i < 24:
                self.cross_modal_image_layers[i - 18].skip_self_attn = True

    def get_prob(self):
        # prob = torch.softmax(self.nas_gate * 4 ** self.current_epoch, dim=-1)
        prob = torch.sigmoid(self.nas_gate)

        return prob


    def forward(self, batch):
        ret = dict()

        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret
        
        if self.training:
            if self.current_epoch < self.nas_epoch:
                self.clean_flag()

                prob = self.get_prob()
                select = torch.multinomial(prob, round(self.skip_num.item()))
                self.apply_flag(select)
                self.nas_count += 1

                ret = self.calculate_loss(ret, batch)

                if self.nas_count >= self.nas_turn and self.current_epoch >= self.warmup_epoch:

                    self.output_count += 1
                    if self.output_count == 50:
                        if self.current_epoch < self.nas_epoch and torch.cuda.current_device() == 0:
                            print(self.nas_gate)
                        self.output_count = 0

                    # print('here')
                    val_loader = self.get_val_loader(self.nas_val_size)
                    val_batch = next(iter(val_loader))

                    rets = []
                    selects = []
                    self.nas_count = 0

                    prob = self.get_prob()
                    for k in range(self.nas_step):
                        self.clean_flag()

                        # select = torch.sort(torch.multinomial(prob, round(self.skip_num.item() + random.random() - 0.5)))[0]
                        select = torch.sort(torch.multinomial(prob, round(self.skip_num.item())))[0]
                        while tensor_in_list(selects, select):
                            # select = torch.sort(torch.multinomial(prob, round(self.skip_num.item() + random.random() - 0.5)))[0]
                            select = torch.sort(torch.multinomial(prob, round(self.skip_num.item())))[0]
                        selects.append(select)

                        self.apply_flag(select)

                        val_ret = dict()
                        with torch.no_grad():
                            val_ret = self.calculate_loss(val_ret, val_batch)

                        rets.append(val_ret)

                    rewards = []
                    for i in range(self.nas_step):
                        rewards.append(math.exp(-sum([v.item() for k, v in rets[i].items() if "loss" in k])))
                    rewardb = sum(rewards) / self.nas_step

                    lr = 1.0 # 1.0 for VQA
                    for k in range(self.nas_step):
                        for i in selects[k]:
                            i = i.item()
                            self.nas_gate[i] += (rewards[k] - rewardb) * prob[i] * (1 - prob[i])
                
                    # total_devices = torch.cuda.device_count() # 获取总GPU个数
                    # y = [torch.zeros_like(self.nas_gate) for _ in range(total_devices)]
                    # torch.distributed.all_gather(y, self.nas_gate)
                    # y = torch.stack(y)
                    # self.nas_gate = y.mean(0)
                    
                    # momentum = self.nas_gate[self.nas_gate > 0].min().item() + 0.001
                    # self.skip_num = self.skip_num * (1 - momentum) + (self.nas_gate > 0).float().sum().item() * momentum
                    # if self.skip_num.item() < 1:
                    #     self.skip_num.data = torch.ones_like(self.skip_num.data)
            else:
                ret = self.calculate_loss(ret, batch)
        else:
            ret = self.calculate_loss(ret, batch) 

        return ret

    def training_step(self, batch, batch_idx):
        meter_utils.set_task(self)
        output = self(batch)

        # if self.current_epoch < self.nas_epoch and torch.cuda.current_device() == 0 and random.randint(1, 100) == 100:
        #     print()
        #     print(self.skip_num)
        #     prob = self.get_prob()
        #     print('Fusion Image', prob[:6])
        #     print('Fusion Text ', prob[6: 12])
        #     print('Encoder Text', prob[12:])

        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outs):
        meter_utils.epoch_wrapup(self)

        self.clean_flag()

        select = torch.sort(self.nas_gate)[1]
        select = select[-round(self.skip_num.item()):]
        if self.current_epoch < self.nas_epoch and torch.cuda.current_device() == 0:
            print(select)
        self.apply_flag(select)

    # def validation_step(self, batch, batch_idx):
    #     meter_utils.set_task(self)
    #     output = self(batch)

    # def validation_epoch_end(self, outs):
    #     meter_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        meter_utils.set_task(self)
        output = self(batch)
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))

        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name)
        meter_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return meter_utils.set_schedule(self)
