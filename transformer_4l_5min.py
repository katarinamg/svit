import torch
import torch.nn as nn
import numpy as np

from collections import OrderedDict


class GELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=0.2)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", GELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(
            dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class EEGTransformer(nn.Module):

    def __init__(self,
                 num_channels: int = 12,
                 num_frequencies: int = 25,
                 num_timesteps: int = 1500,
                 width: int = 768,
                 layers: int = 4,
                 heads: int = 12):
        super().__init__()
        scale = width ** -0.5
        self.ln_input = nn.LayerNorm(num_channels * num_frequencies)
        self.in_proj = nn.Parameter(
            scale * torch.randn(num_channels * num_frequencies, width))
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(num_timesteps + 2, width))
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = Transformer(width, layers=layers, heads=heads)
        self.ln_post = nn.LayerNorm(width)
        self.out_proj = nn.Parameter(scale * torch.randn(width, 1)) #2 here if not binary
        self.dropout1 = torch.nn.Dropout(0.1)

    def forward(self,
                x: torch.Tensor):  # [batch x channels x freq x time]
        x = x.permute(0, 3, 1, 2)  # [batch x time x channels x freq]
        x = x.reshape(*x.shape[:-2], -1)  # [batch x time x channels * freq]
        x = self.ln_input(x)
        x = x @ self.in_proj
        cls_embedding = self.class_embedding + \
            torch.zeros(x.shape[0], 1, x.shape[-1],
                        dtype=x.dtype, device=x.device)
        x = torch.cat([cls_embedding, x, cls_embedding], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # [time x batch x channels * freq]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # [batch x time x channels * freq]
        x = self.dropout1(self.ln_post(x[:, 0, :]))
        x = x @ self.out_proj
        return x
