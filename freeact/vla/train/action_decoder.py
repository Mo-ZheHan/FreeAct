"""MAP action head for stage 3 (multihead attention pooling, as in UniVLA / Set Transformer).

Follows the action decoder of UniVLA (https://github.com/OpenDriveLab/UniVLA, Apache-2.0).
"""

from typing import Optional

import torch
from einops import repeat
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.scale, self.eps = dim**-0.5, eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class SwishGLU(nn.Module):
    """Gated Linear Unit with Swish activation."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.act = nn.SiLU()
        self.project = nn.Linear(in_dim, 2 * out_dim)

    def forward(self, x: Tensor) -> Tensor:
        projected, gate = self.project(x).tensor_split(2, dim=-1)
        return projected * self.act(gate)


class MAPAttention(nn.Module):
    """Multi-Input Multi-Headed Attention for pooling."""

    def __init__(self, embed_dim: int, n_heads: int):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.n_heads = n_heads
        self.scale = (embed_dim // n_heads) ** -0.5
        self.q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.kv = nn.Linear(embed_dim, 2 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, seed: Tensor, x: Tensor) -> Tensor:
        B, K, C = seed.shape
        N = x.shape[1]
        q = (
            self.q(seed)
            .reshape(B, K, self.n_heads, C // self.n_heads)
            .permute(0, 2, 1, 3)
        )
        kv = (
            self.kv(x)
            .reshape(B, N, 2, self.n_heads, C // self.n_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)
        attn = (q @ (k.transpose(-2, -1) * self.scale)).softmax(dim=-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(B, K, C))


class MAPBlock(nn.Module):
    """Multiheaded Attention Pooling Block (from UniVLA/Set Transformer)."""

    def __init__(
        self,
        n_latents: int,
        input_dim: int,
        embed_dim: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.projection = nn.Linear(input_dim, embed_dim)
        self.latents = nn.Parameter(torch.zeros(n_latents, embed_dim))
        nn.init.normal_(self.latents, std=0.02)

        self.attn_norm = RMSNorm(embed_dim)
        self.attn = MAPAttention(embed_dim, n_heads)
        self.mlp_norm = RMSNorm(embed_dim)
        self.mlp = nn.Sequential(
            SwishGLU(embed_dim, int(mlp_ratio * embed_dim)),
            nn.Linear(int(mlp_ratio * embed_dim), embed_dim),
        )

    def forward(self, x: Tensor, init_embed: Optional[Tensor] = None) -> Tensor:
        latents = repeat(self.latents, "n d -> b n d", b=x.shape[0])
        if init_embed is not None:
            latents = latents + init_embed.unsqueeze(1)
        latents = self.attn_norm(latents + self.attn(latents, self.projection(x)))
        latents = self.mlp_norm(latents + self.mlp(latents))
        return latents.squeeze(1)


class VLAActionDecoder(nn.Module):
    """Continuous action head of stage 3 (saved as ``action_decoder.pt``).

    MAP-pools the hidden states at the latent action token positions, initialized from the pooled
    image-token hidden states, and projects them to a normalized action chunk.
    """

    def __init__(
        self,
        embed_dim: int,  # VLM hidden size (text_config.hidden_size)
        hidden_dim: int,
        action_dim: int,
        window_size: int,
        num_codes: int,  # latent action tokens per frame pair
        n_heads: int = None,  # Default: hidden_dim // 64
    ):
        if n_heads is None:
            n_heads = hidden_dim // 64
        super().__init__()
        self.action_dim = action_dim
        self.window_size = window_size
        self.num_codes = num_codes
        self.latent_action_pool = MAPBlock(1, embed_dim, hidden_dim, n_heads)
        self.visual_pool = MAPBlock(1, embed_dim, hidden_dim, n_heads)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, action_dim * window_size),
            nn.Tanh(),  # Output in [-1, 1]
        )

    def forward(
        self,
        latent_action_hidden: Tensor,
        visual_embed: Tensor,
    ) -> Tensor:
        """
        Args:
            latent_action_hidden: [B, num_codes, embed_dim] hidden states at the latent action tokens
            visual_embed: [B, num_visual, embed_dim]
        Returns:
            actions: [B, window_size, action_dim] in [-1, 1]
        """
        visual_pooled = self.visual_pool(visual_embed)
        out = self.proj(
            self.latent_action_pool(latent_action_hidden, init_embed=visual_pooled)
        )
        return out.reshape(-1, self.window_size, self.action_dim)
