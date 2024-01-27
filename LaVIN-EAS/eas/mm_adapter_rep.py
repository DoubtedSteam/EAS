
import torch
from torch import nn
import eas
from typing import Optional, Tuple
from  torch.cuda.amp import autocast
import eas.eval_model


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
        self.hiddden_dim = hidden_dim // groups
        self.in_features = in_features // groups
        self.groups = groups

        self.expert_weights = nn.Linear(in_features, 2)

        self.dropout = nn.Dropout(0.1)
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

    def forward(self, x, res=True, weights=None, start_pos=0):
        x = x.float()
        x = x.transpose(1, 2)
        
        # if start_pos == 0:
        x_= self.conv_A(x)
        x_ = x_.mean(dim=-1, keepdim=True)

        matrix_1 = self.conv_C.weight[None, :, :, 0] * 0.5 # D x C
        bias_1 = (self.conv_C.bias[None, :, None] + x_) * 0.5

        matrix_2 = torch.zeros_like(matrix_1).transpose(1, 2)

        for i in range(self.groups):
            matrix_2[0, i * self.in_features:(i + 1) * self.in_features, i * self.hiddden_dim: (i+1) * self.hiddden_dim] \
                = self.conv_B.weight[i * self.in_features:(i + 1) * self.in_features, :, 0] * self.scale * weights[0, 0] \
                + self.conv_D.weight[i * self.in_features:(i + 1) * self.in_features, :, 0] * self.scale * weights[0, 1]
                
            
        bias_2 = self.conv_B.bias * self.scale * weights[0, 0] \
                + self.conv_D.bias * self.scale * weights[0, 1]
        bias_2 = bias_2[None, :, None]
        
        self.rep_matrix = torch.bmm(matrix_2, matrix_1)
        self.rep_matrix = self.rep_matrix + torch.eye(self.rep_matrix.shape[-1], out=torch.empty_like(self.rep_matrix))
        self.rep_bias =  torch.bmm(matrix_2, bias_1) + bias_2

        out = torch.bmm(self.rep_matrix, x) + self.rep_bias
    
        self.rep_matrix = self.rep_matrix[0].contiguous()
        self.rep_bias = self.rep_bias[0, :, 0].contiguous()
        # x = torch.bmm(self.rep_matrix, x) + self.rep_bias
        # x = x.transpose(1, 2)
        
        return out.transpose(1, 2)

class PIAdapter_FFN_Visual(nn.Module):
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

        self.expert_weights = nn.Linear(in_features,2)

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


    def forward(self, x, res=True, weights=None, trans=False):
        with autocast():
            x = x.transpose(0,1)
            weights = torch.softmax(self.expert_weights(x[:, 1:].mean(dim=1)) / self.t, -1).half()
            x = x.transpose(1, 2)
            x_= self.dropout(self.conv_A(x))
            x = self.conv_B(x_) * self.scale * weights[:, 0, None, None] + self.conv_D(x_) * self.scale * weights[:, 1, None, None] + x
            x = x.transpose(1, 2).contiguous()
            x = x.transpose(0,1)
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
        self.hiddden_dim = hidden_dim // groups
        self.in_features = in_features // groups
        self.groups = groups

        self.expert_weights = nn.Linear(in_features,2)

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


    def forward(self, x, res=True, weights=None, trans=False, start_pos=0):
        # with autocast():
        x = x.transpose(1, 2)

        matrix_1 = self.conv_A.weight[None, :, :, 0] # D x C
        bias_1 = self.conv_A.bias[None, :, None]

        matrix_2 = torch.zeros_like(matrix_1).transpose(1, 2)

        for i in range(self.groups):
            matrix_2[0, i * self.in_features:(i + 1) * self.in_features, i * self.hiddden_dim: (i+1) * self.hiddden_dim] \
            = self.conv_B.weight[i * self.in_features:(i + 1) * self.in_features, :, 0] * self.scale * weights[0, 0] \
            + self.conv_D.weight[i * self.in_features:(i + 1) * self.in_features, :, 0] * self.scale * weights[0, 1]
            
        bias_2 = self.conv_B.bias * self.scale * weights[0, 0] \
            + self.conv_D.bias * self.scale * weights[0, 1]
        bias_2 = bias_2[None, :, None]
        
        self.rep_matrix = torch.bmm(matrix_2, matrix_1)
        self.rep_matrix = self.rep_matrix + torch.eye(self.rep_matrix.shape[-1], out=torch.empty_like(self.rep_matrix))
        self.rep_bias =  torch.bmm(matrix_2, bias_1) + bias_2

        # print(torch.bmm(self.rep_matrix, x) + self.rep_bias)
    
        out = torch.bmm(self.rep_matrix, x) + self.rep_bias

        self.rep_matrix = self.rep_matrix[0].contiguous() # C_out x C_in
        self.rep_bias = self.rep_bias[0, :, 0].contiguous() # C_out
        
        return out.transpose(1, 2)


def forward_llama_attn_cache(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None):
    bs_ = x.shape[0]
    if self.skipped_flag < 0:
        norm_x = self.attention_norm(x)
        if start_pos == 0:
            self.cache_weights[:bs_] = torch.softmax(self.adapter_attn.expert_weights(norm_x[:, 1:].mean(dim=1).float()) / self.t, -1).half()
            out = self.adapter_attn(norm_x, weights=self.cache_weights[:bs_], start_pos=start_pos)
            
            self.attention.wq.weight.data = self.attention.wq.weight.data @ self.adapter_attn.rep_matrix
            self.attention.wq.bias.data = self.attention.wq.weight.data @ self.adapter_attn.rep_bias
            self.attention.wk.weight.data = self.attention.wk.weight.data @ self.adapter_attn.rep_matrix
            self.attention.wk.bias.data = self.attention.wk.weight.data @ self.adapter_attn.rep_bias
            self.attention.wv.weight.data = self.attention.wv.weight.data @ self.adapter_attn.rep_matrix
            self.attention.wv.bias.data = self.attention.wv.weight.data @ self.adapter_attn.rep_bias
            
        h = x + self.attention.forward(norm_x, start_pos, freqs_cis, mask, adapter)
        out = h + self.drop_path(self.feed_forward.forward(self.ffn_norm(h)))
    else:
        norm_x = self.attention_norm(x)
        if start_pos == 0:
            self.cache_weights[:bs_] = torch.softmax(self.replaced_adapter.expert_weights(norm_x[:, 1:].mean(dim=1).float()) / self.t, -1).half()
            out = self.replaced_adapter(norm_x, weights=self.cache_weights[:bs_])

            self.replaced_adapter.rep_matrix = self.replaced_adapter.rep_matrix * self.ffn_norm.weight.reshape(-1, 1)
            self.replaced_adapter.rep_bias = self.replaced_adapter.rep_bias * self.ffn_norm.weight

            self.feed_forward.w1.weight.data = self.feed_forward.w1.weight.data @ self.replaced_adapter.rep_matrix
            self.feed_forward.w1.bias.data = self.feed_forward.w1.weight.data @ self.replaced_adapter.rep_bias

            self.feed_forward.w3.weight.data = self.feed_forward.w3.weight.data @ self.replaced_adapter.rep_matrix
            self.feed_forward.w3.bias.data = self.feed_forward.w3.weight.data @ self.replaced_adapter.rep_bias
            
        out = x + self.drop_path(self.feed_forward.forward(norm_x))

    return out


def forward_clip(self, x: torch.Tensor):
    x = x + self.attention(self.adapter_attn(self.ln_1(x), trans=True))
    x = x + self.mlp(self.ln_2(x))
    return x


def set_MMAdapter(model, method, dim=8, s=1, set_forward=True,t=10,gradient_checkpointing=False):
    if method == 'mlp':
        # not support right now
        assert NotImplementedError
        for _ in model.children():
            if type(_) ==  eas.model.TransformerBlock or type(_) == eas.eval_model.TransformerBlock:
                _.adapter_attn = PIAdapter_FFN(_.dim, hidden_dim=dim, scale=s, t=t)
                _.s = s
                _.t = t
                _.skipped_flag = -1.
                _.replaced_adapter = PIAdapter_Attn(_.dim, hidden_dim=dim * 4, scale=s, t=t)
                _.gradient_checkpointing = gradient_checkpointing
                if type(_) == eas.eval_model.TransformerBlock:
                    bound_method = forward_llama_mlp_cache.__get__(_, _.__class__)
                else:
                    bound_method = forward_llama_mlp.__get__(_, _.__class__)
                if set_forward:
                    setattr(_, 'forward', bound_method)
            elif len(list(_.children())) != 0:
                set_MMAdapter(_, method, dim, s,set_forward=set_forward,t=t,gradient_checkpointing=gradient_checkpointing)

    else:
        for _ in model.children():
            if type(_) == eas.model.TransformerBlock or type(_) == eas.eval_model.TransformerBlock:
                
                _.adapter_attn = PIAdapter_FFN(_.dim,hidden_dim=dim,scale=s,t=t)
                _.s = s
                _.t = t
                _.skipped_flag = -1.
                _.replaced_adapter = PIAdapter_Attn(_.dim, hidden_dim=dim * 4, scale=s, t=t)
                _.gradient_checkpointing = gradient_checkpointing
                if type(_) == eas.eval_model.TransformerBlock:
                    bound_method = forward_llama_attn_cache.__get__(_, _.__class__)
                else:
                    bound_method = forward_llama_attn.__get__(_, _.__class__)
                if set_forward:
                    setattr(_, 'forward', bound_method)
            elif len(list(_.children())) != 0:
                set_MMAdapter(_, method, dim, s, set_forward=set_forward,t=t,gradient_checkpointing=gradient_checkpointing)


from clip.model import ResidualAttentionBlock
def set_Clip_Adapter(model, method, dim=8, s=1, set_forward=True, t=10.):
    for _ in model.children():
        if type(_) == ResidualAttentionBlock:
            if method=='router':
                _.adapter_attn = PIAdapter_FFN_Visual(1024, hidden_dim=dim, scale=s,  t=t)
            else:
                _.adapter_attn = PIAdapter_FFN_Visual(1024, hidden_dim=dim, scale=s,  t=t)
                _.adapter_mlp = PIAdapter_FFN_Visual(1024, hidden_dim=dim, scale=s,  t=t)
            _.s = s
            if method=='router_block':
                bound_method = forward_clip_full.__get__(_, _.__class__)
            else:
                bound_method = forward_clip.__get__(_, _.__class__)
            if set_forward:
                setattr(_, 'forward', bound_method)
        elif len(list(_.children())) != 0:
            set_Clip_Adapter(_, method, dim, s, set_forward=set_forward, t=t)
