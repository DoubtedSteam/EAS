import torch
import transformers

from torch import nn
from typing import Optional, Tuple
from  torch.cuda.amp import autocast

import llava

import torch.nn.functional as F


class PIAdapter_Attn(nn.Module):
    def __init__(
        self,
        in_features=768,
        hidden_dim=8,
        groups=2,
        scale=1,
        t=10.
    ):
        super().__init__()

        self.conv_A = nn.Conv1d(in_features, hidden_dim, 1, groups=1, bias=True)
        self.conv_C = nn.Conv1d(in_features, hidden_dim, 1, groups=1, bias=True)
        
        self.conv_B = nn.Conv1d(hidden_dim, in_features, 1, groups=groups, bias=True)
        self.conv_D = nn.Conv1d(hidden_dim, in_features, 1, groups=groups, bias=True)

        self.expert_weights = nn.Linear(in_features, 2)
        self.dropout = nn.Dropout(0.1)

        self.groups = groups
        self.scale = scale
        self.t = t

        nn.init.xavier_uniform_(self.conv_A.weight)
        nn.init.zeros_(self.conv_A.bias)
        nn.init.xavier_uniform_(self.conv_C.weight)
        nn.init.zeros_(self.conv_C.bias)

        nn.init.xavier_uniform_(self.conv_B.weight)
        nn.init.zeros_(self.conv_B.bias)
        nn.init.xavier_uniform_(self.conv_D.weight)
        nn.init.zeros_(self.conv_D.bias)

        self.cache_x = None

    def forward(self, x, question_mask, start_pos=True):
        with autocast():
            if self.training:
                weight = (x * question_mask[:, :, None]).sum(dim=1) # B x C
                self.weights = torch.softmax(self.expert_weights(weight) / self.t, -1) # B x 2
            elif start_pos:
                self.weights = torch.softmax(self.expert_weights(x.mean(dim=1)) / self.t, -1)
            weights = self.weights
                
            x = x.transpose(1, 2) # B x C x N

            x_= self.conv_A(x)
            if self.training:
                x_ = (x_ * question_mask[:, None, :]).sum(dim=-1, keepdim=True) # B x C x 1
                self.cache_x = x_
            elif start_pos:
                x_ = x_.mean(dim=-1, keepdim=True) 
                self.cache_x = x_
            x_= self.cache_x

            x_= self.dropout(x_ + self.conv_C(x)) * 0.5
            x = self.conv_B(x_) * self.scale * weights[:, 0, None, None] + self.conv_D(x_) * self.scale * weights[:, 1, None, None] + x
            x = x.transpose(1, 2).contiguous()
        
        return x


class PIAdapter_FFN(nn.Module):
    """ Pytorch Implemention of RepAdapter for 1d tensor"""

    def __init__(
        self,
        in_features=768,
        hidden_dim=8,
        groups=2,
        scale=1,
        t=10.
    ):
        super().__init__()

        self.conv_A = nn.Conv1d(in_features, hidden_dim, 1, groups=1, bias=True)

        self.conv_B = nn.Conv1d(hidden_dim, in_features, 1, groups=groups, bias=True)
        self.conv_D = nn.Conv1d(hidden_dim, in_features, 1, groups=groups, bias=True)

        self.expert_weights = nn.Linear(in_features, 2, bias=False)
        self.dropout = nn.Dropout(0.1)

        self.groups = groups
        self.scale = scale
        self.t = t

        nn.init.xavier_uniform_(self.conv_A.weight)
        nn.init.zeros_(self.conv_A.bias)
        nn.init.zeros_(self.conv_B.weight)
        nn.init.zeros_(self.conv_B.bias)
        nn.init.zeros_(self.conv_D.weight)
        nn.init.zeros_(self.conv_D.bias)


    def forward(self, x, question_mask, start_pos=True):
        with autocast():
            if self.training:
                weight = (x * question_mask[:, :, None]).sum(dim=1)
                self.weights = torch.softmax(self.expert_weights(weight) / self.t, -1)
            elif start_pos:
                self.weights = torch.softmax(self.expert_weights(x.mean(dim=1)) / self.t, -1)
            weights = self.weights
            
            x = x.transpose(1, 2)
            x_= self.dropout(self.conv_A(x))
            x = (self.conv_B(x_) * weights[:, 0:1, None] + self.conv_D(x_) * weights[:, 1:, None]) * self.scale + x
            x = x.transpose(1, 2).contiguous()
        return x
    

# class PIAdapter_FFN(nn.Module):
#     """ Pytorch Implemention of RepAdapter for 1d tensor"""
#     def __init__(
#         self,
#         in_features=768,
#         hidden_dim=8,
#         groups=2,
#         scale=1,
#         t=10.
#     ):
#         super().__init__()
        
#         self.conv_A = nn.Linear(in_features, hidden_dim, bias=False)
#         self.act = nn.ReLU(inplace=True)
#         self.conv_B = nn.Linear(hidden_dim, in_features, bias=False)

#     def forward(self, x, question_mask, start_pos=True):
#         with autocast():
#             out = self.conv_A(x)
#             out = self.act(out)
#             out = self.conv_B(out)
#             out = x + out
#         return out


# class PIAdapter_Attn(nn.Module):
#     """ Pytorch Implemention of RepAdapter for 1d tensor"""
#     def __init__(
#         self,
#         in_features=768,
#         hidden_dim=8,
#         groups=2,
#         scale=1,
#         t=10.
#     ):
#         super().__init__()
        
#         self.conv_A = nn.Linear(in_features, hidden_dim, bias=False)
#         self.act = nn.ReLU(inplace=True)
#         self.conv_B = nn.Linear(hidden_dim, in_features, bias=False)

#     def forward(self, x, question_mask, start_pos=True):
#         with autocast():
#             out = self.conv_A(x)
#             out = self.act(out)
#             out = self.conv_B(out)
#             out = x + out
#         return out


def forward_llama(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        question_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        
        if self.skipped_flag < 0:
            residual = hidden_states
            
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states = self.adapter_attn(hidden_states, question_mask=question_mask, start_pos=past_key_value is None).type_as(residual)
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = residual + hidden_states

            # Fully Connected
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            # hidden_states = self.adapter_ffn(hidden_states, question_mask=question_mask, start_pos=past_key_value is None).type_as(residual)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
        else:
            residual = hidden_states
            
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states = self.replaced_adapter(hidden_states, question_mask=question_mask, start_pos=past_key_value is None).type_as(residual) * self.post_attention_layernorm.weight
            
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
            
            self_attn_weights = torch.empty(0)
            present_key_value = torch.empty(0)
            
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)
            
        # if self.training:
        #     outputs += (0, )

        return outputs


def set_PIAdapter(model, adapt_dim=128, replaced_dim=256, s=1, set_forward=True, t=10, gradient_checkpointing=False):
    for _ in model.children():
        if type(_) == llava.model.language_model.modeling_llama.LlamaDecoderLayer:
            _.dim = 4096
            _.adapter_attn = PIAdapter_FFN(_.dim, hidden_dim=adapt_dim, scale=s, t=t)
            # _.adapter_ffn = PIAdapter_FFN(_.dim, hidden_dim=adapt_dim // 2, scale=s, t=t)
            _.replaced_adapter = PIAdapter_Attn(_.dim, hidden_dim=replaced_dim, scale=s, t=t)
            _.s = 1
            _.t = 1
            _.skipped_flag = -1.
            
            bound_method = forward_llama.__get__(_, _.__class__)            
            setattr(_, 'forward', bound_method)
            
        elif len(list(_.children())) != 0:
            set_PIAdapter(_, adapt_dim=adapt_dim, replaced_dim=replaced_dim, s=s, set_forward=set_forward, t=t, gradient_checkpointing=gradient_checkpointing)
            
