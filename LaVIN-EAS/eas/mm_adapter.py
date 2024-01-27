
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

    def forward(self, x, res=True, weights=None, start_pos=0):
        x = x.float()
        if weights is None:
            weights = torch.softmax(self.expert_weights(x[:, 1:].mean(dim=1).float()) / self.t, -1).half()
        x = x.transpose(1, 2) # B x C x N

        x_= self.conv_A(x)
        bsz, dim, seqlen = x_.shape
        if start_pos == 0:
            x_ = x_.mean(dim=-1, keepdim=True) 
            self.cache_x = x_
        else:
            x_ = self.cache_x * start_pos + x_.mean(dim=-1, keepdim=True) * seqlen
            x_ = x_ / (start_pos + seqlen)
            self.cache_x = x_

        x_= self.dropout(x_ + self.conv_C(x)) * 0.5
        x = self.conv_B(x_) * self.scale * weights[:, 0, None, None] + self.conv_D(x_) * self.scale * weights[:, 1, None, None] + x
        x = x.transpose(1, 2).contiguous()
        
        return x.half()


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
            if trans:
                x = x.transpose(0,1)
            if weights is None:
                weights = torch.softmax(self.expert_weights(x[:, 1:].mean(dim=1)) / self.t, -1).half()
            x = x.transpose(1, 2)
            x_= self.dropout(self.conv_A(x))
            x = self.conv_B(x_) * self.scale * weights[:, 0, None, None] + self.conv_D(x_) * self.scale * weights[:, 1, None, None] + x
            x = x.transpose(1, 2).contiguous()
            if trans:
                x = x.transpose(0,1)
        return x


def forward_llama_attn(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None):
    if self.training and self.gradient_checkpointing:
        if self.skipped_flag < 0:
            h = x + self.drop_path(torch.utils.checkpoint.checkpoint(self.attention, self.adapter_attn(self.attention_norm(x)), start_pos, freqs_cis, mask))
            out = h + self.drop_path(torch.utils.checkpoint.checkpoint(self.feed_forward, self.ffn_norm(h)))
        else:
            h = x + self.replaced_adapter(self.attention_norm(x))
            out = h + self.drop_path(torch.utils.checkpoint.checkpoint(self.feed_forward, self.ffn_norm(h)))
    else:
        if self.skipped_flag < 0:
            h = x + self.drop_path(self.attention.forward(self.adapter_attn(self.attention_norm(x)), start_pos, freqs_cis, mask, adapter))
            out = h + self.drop_path(self.feed_forward.forward(self.ffn_norm(h)))
        else:
            h = self.replaced_adapter(self.attention_norm(x)) * self.ffn_norm.weight
            out = x + self.drop_path(self.feed_forward.forward(h))
    return out
    

def forward_llama_attn_cache(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None):
    bs_ = x.shape[0]
    if self.skipped_flag < 0:
        if start_pos == 0:
            self.cache_weights[:bs_] = torch.softmax(self.adapter_attn.expert_weights(self.attention_norm(x)[:, 1:].mean(dim=1).float()) / self.t, -1).half()
        h = x + self.drop_path(self.attention.forward(self.adapter_attn(self.attention_norm(x), weights=self.cache_weights[:bs_]), start_pos, freqs_cis, mask, adapter))
        out = h + self.drop_path(self.feed_forward.forward(self.ffn_norm(h)))
    else:
        if start_pos == 0:
            self.cache_weights[:bs_] = torch.softmax(self.replaced_adapter.expert_weights(self.attention_norm(x)[:, 1:].mean(dim=1).float()) / self.t, -1).half()
        h = self.replaced_adapter(self.attention_norm(x), weights=self.cache_weights[:bs_]) * self.ffn_norm.weight
        out = x + self.drop_path(self.feed_forward.forward(h))
    return out


def forward_clip(self, x: torch.Tensor):
    x = x + self.attention(self.adapter_attn(self.ln_1(x), trans=True))
    x = x + self.mlp(self.ln_2(x))
    return x


def forward_clip_full(self, x: torch.Tensor):
    x = x + self.attention(self.adapter_attn(self.ln_1(x)))
    x = x + self.mlp(self.adapter_mlp(self.ln_2(x)))
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
                _.adapter_attn = PIAdapter_FFN(1024, hidden_dim=dim, scale=s,  t=t)
            else:
                _.adapter_attn = PIAdapter_FFN(1024, hidden_dim=dim, scale=s,  t=t)
                _.adapter_mlp = PIAdapter_FFN(1024, hidden_dim=dim, scale=s,  t=t)
            _.s = s
            if method=='router_block':
                bound_method = forward_clip_full.__get__(_, _.__class__)
            else:
                bound_method = forward_clip.__get__(_, _.__class__)
            if set_forward:
                setattr(_, 'forward', bound_method)
        elif len(list(_.children())) != 0:
            set_Clip_Adapter(_, method, dim, s, set_forward=set_forward, t=t)
