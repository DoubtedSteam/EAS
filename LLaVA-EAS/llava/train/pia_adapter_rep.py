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

        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.group_out = in_features // groups
        self.group_in = hidden_dim // groups

        self.conv_A = nn.Conv1d(in_features, hidden_dim, 1, groups=1, bias=True)
        self.conv_C = nn.Conv1d(in_features, hidden_dim, 1, groups=1, bias=True)
        
        self.conv_B = nn.Conv1d(hidden_dim, in_features, 1, groups=groups, bias=True)
        self.conv_D = nn.Conv1d(hidden_dim, in_features, 1, groups=groups, bias=True)

        self.expert_weights = nn.Linear(in_features, 2)
        self.dropout = nn.Dropout(0.1)

        # self.dropout = nn.Dropout(0.1)
        self.groups = groups
        self.scale = scale
        self.t = t

        nn.init.xavier_uniform_(self.conv_A.weight)
        nn.init.zeros_(self.conv_A.bias)
        nn.init.xavier_uniform_(self.conv_C.weight)
        nn.init.zeros_(self.conv_C.bias)

        nn.init.zeros_(self.conv_B.weight)
        nn.init.zeros_(self.conv_B.bias)
        nn.init.zeros_(self.conv_D.weight)
        nn.init.zeros_(self.conv_D.bias)

        self.cache_x = None
        self.rep_matrix = None
        self.rep_bias = None

    def forward(self, x, question_mask, start_pos=True):
        with autocast():
            weights = torch.softmax(self.expert_weights(x.mean(dim=1)) / self.t, -1)
            x = x.transpose(1, 2) # B x C x N
            x_= self.conv_A(x)
            x_ = x_.mean(dim=-1, keepdim=True)

            matrix_1 = self.conv_C.weight[None, :, :, 0] * 0.5 # D x C
            bias_1 = (self.conv_C.bias[None, :, None] + x_) * 0.5

            matrix_2 = torch.zeros_like(matrix_1).transpose(1, 2)
            for i in range(self.groups):
                out_start = i * self.group_out
                out_end = (i+1) * self.group_out
                in_start = i * self.group_in
                in_end = (i+1) * self.group_in
                
                B_weight = self.conv_B.weight[out_start:out_end, :, 0]
                D_weight = self.conv_D.weight[out_start:out_end, :, 0]
                
                matrix_2[0, out_start:out_end, in_start:in_end] = \
                    B_weight * self.scale * weights[0, 0] + \
                    D_weight * self.scale * weights[0, 1]
                
            bias_2 = self.conv_B.bias * self.scale * weights[0, 0] \
                    + self.conv_D.bias * self.scale * weights[0, 1]
            bias_2 = bias_2[None, :, None]
            
            self.rep_matrix = torch.bmm(matrix_2, matrix_1)
            self.rep_matrix = self.rep_matrix + torch.eye(self.rep_matrix.shape[-1], out=torch.empty_like(self.rep_matrix))
            self.rep_bias =  torch.bmm(matrix_2, bias_1) + bias_2

            self.rep_matrix = self.rep_matrix[0].contiguous()
            self.rep_bias = self.rep_bias[0, :, 0].contiguous()
        
        return None


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

        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.group_out = in_features // groups
        self.group_in = hidden_dim // groups

        self.conv_A = nn.Conv1d(in_features, hidden_dim, 1, groups=1, bias=True)

        self.conv_B = nn.Conv1d(hidden_dim, in_features, 1, groups=groups, bias=True)
        self.conv_D = nn.Conv1d(hidden_dim, in_features, 1, groups=groups, bias=True)

        self.expert_weights = nn.Linear(in_features,2)
        self.dropout = nn.Dropout(0.1)

        self.groups = groups
        self.scale = scale
        self.t = t

        nn.init.xavier_uniform_(self.conv_A.weight)
        nn.init.zeros_(self.conv_A.bias)

        nn.init.xavier_uniform_(self.conv_B.weight)
        nn.init.zeros_(self.conv_B.bias)
        nn.init.xavier_uniform_(self.conv_D.weight)
        nn.init.zeros_(self.conv_D.bias)


    def forward(self, x, question_mask, start_pos=True):
        with autocast():
            weights = torch.softmax(self.expert_weights(x.mean(dim=1)) / self.t, -1)
            x = x.transpose(1, 2) # B x C x N

            matrix_1 = self.conv_A.weight[None, :, :, 0] # D x C
            bias_1 = self.conv_A.bias[None, :, None]
            matrix_2 = torch.zeros_like(matrix_1).transpose(1, 2)

            for i in range(self.groups):
                out_start = i * self.group_out
                out_end = (i+1) * self.group_out
                in_start = i * self.group_in
                in_end = (i+1) * self.group_in
                
                B_weight = self.conv_B.weight[out_start:out_end, :, 0]
                D_weight = self.conv_D.weight[out_start:out_end, :, 0]
                
                matrix_2[0, out_start:out_end, in_start:in_end] = \
                    B_weight * self.scale * weights[0, 0] + \
                    D_weight * self.scale * weights[0, 1]
                
            bias_2 = self.conv_B.bias * self.scale * weights[0, 0] \
                + self.conv_D.bias * self.scale * weights[0, 1]
            bias_2 = bias_2[None, :, None]
        
            self.rep_matrix = torch.bmm(matrix_2, matrix_1)
            self.rep_matrix = self.rep_matrix + torch.eye(self.rep_matrix.shape[-1], out=torch.empty_like(self.rep_matrix))
            self.rep_bias =  torch.bmm(matrix_2, bias_1) + bias_2

            self.rep_matrix = self.rep_matrix[0].contiguous() # C_out x C_in
            self.rep_bias = self.rep_bias[0, :, 0].contiguous() # C_out
        return None


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
            if past_key_value is None:
                _ = self.adapter_attn(hidden_states, question_mask=question_mask, start_pos=past_key_value is None)
                self.adapter_attn.rep_matrix = self.adapter_attn.rep_matrix.type_as(self.self_attn.q_proj.weight.data)
                self.adapter_attn.rep_bias = self.adapter_attn.rep_bias.type_as(self.self_attn.q_proj.weight.data)
                self.self_attn.q_proj.weight.data = self.self_attn.q_proj.weight.data @ self.adapter_attn.rep_matrix
                self.self_attn.q_proj.bias.data = self.self_attn.q_proj.weight.data @ self.adapter_attn.rep_bias
                self.self_attn.k_proj.weight.data = self.self_attn.k_proj.weight.data @ self.adapter_attn.rep_matrix
                self.self_attn.k_proj.bias.data = self.self_attn.k_proj.weight.data @ self.adapter_attn.rep_bias
                self.self_attn.v_proj.weight.data = self.self_attn.v_proj.weight.data @ self.adapter_attn.rep_matrix
                self.self_attn.v_proj.bias.data = self.self_attn.v_proj.weight.data @ self.adapter_attn.rep_bias
            
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

            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
        else:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

            if past_key_value is None:
                _ = self.replaced_adapter(hidden_states, question_mask=question_mask, start_pos=past_key_value is None)
                self.replaced_adapter.rep_matrix = self.replaced_adapter.rep_matrix.type_as(self.self_attn.q_proj.weight.data)
                self.replaced_adapter.rep_bias = self.replaced_adapter.rep_bias.type_as(self.self_attn.q_proj.weight.data)
                
                self.replaced_adapter.rep_matrix = self.replaced_adapter.rep_matrix * self.post_attention_layernorm.weight.reshape(-1, 1)
                self.replaced_adapter.rep_bias = self.replaced_adapter.rep_bias * self.post_attention_layernorm.weight

                self.mlp.up_proj.weight.data = self.mlp.up_proj.weight.data @ self.replaced_adapter.rep_matrix
                self.mlp.up_proj.bias.data = self.mlp.up_proj.weight.data @ self.replaced_adapter.rep_bias

                self.mlp.gate_proj.weight.data = self.mlp.gate_proj.weight.data @ self.replaced_adapter.rep_matrix
                self.mlp.gate_proj.bias.data = self.mlp.gate_proj.weight.data @ self.replaced_adapter.rep_bias

            hidden_states = residual + self.mlp.forward(hidden_states)
            
            self_attn_weights = torch.zeros(1, 1, 1, 1)
            present_key_value = torch.zeros(1, 1, 1, 1)
            
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)
            
        if self.training:
            outputs += (0, )

        return outputs


def set_PIAdapter(model, dim=8, s=1, set_forward=True, t=10, gradient_checkpointing=False):
    adapt_dim = 128
    replace_dim = 256
    for _ in model.children():
        if type(_) == llava.model.language_model.modeling_llama.LlamaDecoderLayer:
            _.dim = 4096
            _.adapter_attn = PIAdapter_FFN(_.dim, hidden_dim=adapt_dim, scale=s, t=t)
            _.replaced_adapter = PIAdapter_Attn(_.dim, hidden_dim=replace_dim, scale=s, t=t)
            _.s = 1
            _.t = 10
            _.skipped_flag = -1.
            
            bound_method = forward_llama.__get__(_, _.__class__)            
            setattr(_, 'forward', bound_method)
            
        elif len(list(_.children())) != 0:
            set_PIAdapter(_, dim, s, set_forward=set_forward, t=t, gradient_checkpointing=gradient_checkpointing)
            
