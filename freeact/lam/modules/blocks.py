"""Transformer and vector-quantizer building blocks of the latent action model.

Adapted from UniVLA (https://github.com/OpenDriveLab/UniVLA, Apache-2.0), itself derived from Genie.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        exponent = torch.arange(0, model_dim, 2).float() * -(math.log(10000.0) / model_dim)
        div_term = torch.exp(exponent)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # A plain tensor rather than a buffer, so it is not part of the state_dict.
        self.pos_enc = pe

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pos_enc[: x.shape[2]].to(x.device, x.dtype)


class SelfAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0, rot_emb: bool = False) -> None:
        super().__init__()
        inner_dim = model_dim // num_heads
        self.scale = inner_dim**-0.5
        self.heads = num_heads

        self.to_q = nn.Linear(model_dim, model_dim, bias=False)
        self.to_k = nn.Linear(model_dim, model_dim, bias=False)
        self.to_v = nn.Linear(model_dim, model_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(model_dim, model_dim), nn.Dropout(dropout))

        self.rot_emb = rot_emb
        if rot_emb:
            self.rotary_embedding = RotaryEmbedding(dim=inner_dim // 2)

    def scaled_dot_product_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        is_causal: bool = False,
        attn_mask: Tensor = None,
    ) -> Tensor:
        L, S = query.shape[-2], key.shape[-2]
        attn_bias = torch.zeros(L, S, dtype=query.dtype).to(query)
        if is_causal:
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0).to(attn_bias)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

        if attn_mask is not None:
            attn_bias = attn_bias.unsqueeze(0).repeat(query.shape[0], 1, 1)
            attn_bias.masked_fill_((attn_mask > 0).logical_not().unsqueeze(1), float("-inf"))
            attn_bias = attn_bias.unsqueeze(1)

        attn_weight = query @ key.transpose(-2, -1) * self.scale
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        return attn_weight @ value

    def forward(self, x: Tensor, is_causal: bool = False, attn_mask: Tensor = None) -> Tensor:
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v))

        if self.rot_emb:
            q = self.rotary_embedding.rotate_queries_or_keys(q)
            k = self.rotary_embedding.rotate_queries_or_keys(k)

        out = self.scaled_dot_product_attention(q, k, v, is_causal=is_causal, attn_mask=attn_mask)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class SpatioTemporalBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.spatial_attn = SelfAttention(model_dim, num_heads, dropout=dropout)
        self.temporal_attn = SelfAttention(model_dim, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim),
        )

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)

    def forward(self, x: Tensor, causal_temporal: bool = False, attn_mask: Tensor = None) -> Tensor:
        t_len, s_len = x.shape[1:3]

        # Spatial attention
        x = rearrange(x, "b t s e -> (b t) s e")
        x = x + self.spatial_attn(self.norm1(x), is_causal=False, attn_mask=attn_mask)
        x = rearrange(x, "(b t) s e -> b t s e", t=t_len)

        # Temporal attention
        x = rearrange(x, "b t s e -> (b s) t e")
        x = x + self.temporal_attn(self.norm2(x), is_causal=causal_temporal)
        x = rearrange(x, "(b s) t e -> b t s e", s=s_len)

        # Feedforward
        return x + self.ffn(self.norm3(x))


class SpatioTemporalTransformer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        model_dim: int,
        out_dim: int,
        num_blocks: int,
        num_heads: int,
        dropout: float = 0.0,
        causal_temporal: bool = False,
        to_out: bool = True,
    ) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, model_dim), nn.LayerNorm(model_dim)
        )
        self.pos_enc = PositionalEncoding(model_dim)
        self.transformer_blocks = nn.ModuleList(
            [SpatioTemporalBlock(model_dim, num_heads, dropout) for _ in range(num_blocks)]
        )
        self.out = nn.Linear(model_dim, out_dim) if to_out else nn.Identity()
        self.causal_temporal = causal_temporal

    def forward(self, x: Tensor, attn_mask: Tensor = None) -> Tensor:
        x = self.pos_enc(self.ffn(x))
        for block in self.transformer_blocks:
            x = block(x, self.causal_temporal, attn_mask)
        return self.out(x)


class SpatioBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.spatial_attn = SelfAttention(model_dim, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim),
        )
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, x: Tensor, attn_mask: Tensor = None) -> Tensor:
        t_len = x.shape[1]
        x = rearrange(x, "b t s e -> (b t) s e")
        x = x + self.spatial_attn(self.norm1(x), attn_mask=attn_mask)
        x = rearrange(x, "(b t) s e -> b t s e", t=t_len)
        return x + self.ffn(self.norm2(x))


class SpatioTransformer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        model_dim: int,
        out_dim: int,
        num_blocks: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, model_dim), nn.LayerNorm(model_dim)
        )
        self.pos_enc = PositionalEncoding(model_dim)
        self.transformer_blocks = nn.ModuleList(
            [SpatioBlock(model_dim, num_heads, dropout) for _ in range(num_blocks)]
        )
        self.out = nn.Linear(model_dim, out_dim)

    def forward(self, x: Tensor, attn_mask: Tensor = None) -> Tensor:
        x = self.pos_enc(self.ffn(x))
        for block in self.transformer_blocks:
            x = block(x, attn_mask=attn_mask)
        return self.out(x)


class VectorQuantizer(nn.Module):
    def __init__(self, num_latents: int, latent_dim: int, code_restart: bool = True) -> None:
        super().__init__()
        self.codebook = nn.Embedding(num_latents, latent_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_latents, 1.0 / num_latents)

        # Code usage counter (used to restart dead codes at the end of each epoch)
        self.register_buffer("usage", torch.zeros(num_latents), persistent=False)
        self.num_latents = num_latents
        self.code_restart = code_restart

    def update_usage(self, indices: Tensor) -> None:
        self.usage += torch.bincount(indices.flatten(), minlength=self.num_latents).to(self.usage)

    def random_restart(self, epoch: int = 0) -> None:
        if not self.code_restart:
            return

        # Sync usage across ranks for consistent dead code detection
        dist = torch.distributed
        if dist.is_initialized():
            dist.all_reduce(self.usage, op=dist.ReduceOp.SUM)

        dead_codes = torch.nonzero(self.usage < 1).flatten()
        if len(dead_codes) == 0:
            return

        # Epoch-varying seed ensures different replacements each epoch while syncing ranks
        g = torch.Generator(device=self.usage.device).manual_seed(42 + epoch)
        rand_codes = torch.randperm(self.num_latents, generator=g, device=self.usage.device)

        with torch.no_grad():
            self.codebook.weight[dead_codes] = self.codebook.weight[rand_codes[: len(dead_codes)]]

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"[VQ] Restarted {len(dead_codes)} dead codes")

    def reset_usage(self) -> None:
        if self.code_restart:
            self.usage.zero_()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        distance = torch.cdist(x, self.codebook.weight)
        indices = torch.argmin(distance, dim=-1)
        z_code = self.codebook(indices)

        if not self.training or self.code_restart:
            self.update_usage(indices)

        # Straight-through estimator; returns (z_q, codebook vectors, encoder output z_e, indices)
        z_q = x + (z_code - x).detach()
        return z_q, z_code, x, indices
