import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .embeddings import RotaryEmbedding
from torch import Tensor


def _match_module_dtype(x: Tensor, module: nn.Module) -> Tensor:
    parameter = next(module.parameters(), None)
    if parameter is None or not torch.is_floating_point(x):
        return x
    return x.to(dtype=parameter.dtype)


def patchify(videos: Tensor, size: int) -> Tensor:
    """把图像帧切成 patch token，供原始 pixel-space LAM 使用。"""
    B, T, H, W, C = videos.shape
    videos = videos[:, :, :H - (H % size), :W - (W % size), :]
    x = rearrange(videos, "b t (hn hp) (wn wp) c -> b t (hn wn) (hp wp c)", hp=size, wp=size)
    return x


def unpatchify(patches: Tensor, size: int, h_out: int, w_out: int) -> Tensor:
    """把 patch token 还原成图像帧。当前 feature-space LAM 不走这条重建图像路径。"""
    h_pad = -h_out % size
    hn = (h_out + h_pad) // size
    x = rearrange(patches, "b t (hn wn) (hp wp c) -> b t (hn hp) (wn wp) c", hp=size, wp=size, hn=hn)
    return x[:, :, :h_out, :w_out]


class PositionalEncoding(nn.Module):
    """标准 sinusoidal position encoding，用在空间/时空 transformer 输入上。"""

    def __init__(self, model_dim: int, max_len: int = 5000) -> None:
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        exponent = torch.arange(0, model_dim, 2).float() * -(math.log(10000.0) / model_dim)
        div_term = torch.exp(exponent)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pos_enc = pe

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pos_enc[:x.shape[2]].cuda()


class SelfAttention(nn.Module):
    """LAM 内部轻量 self-attention，支持可选 RoPE 和 causal temporal mask。"""

    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0, rot_emb: bool = False) -> None:
        super(SelfAttention, self).__init__()
        inner_dim = model_dim // num_heads
        self.scale = inner_dim ** -0.5
        self.heads = num_heads

        self.to_q = nn.Linear(model_dim, model_dim, bias=False)
        self.to_k = nn.Linear(model_dim, model_dim, bias=False)
        self.to_v = nn.Linear(model_dim, model_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.Dropout(dropout)
        )

        self.rot_emb = rot_emb
        if rot_emb:
            self.rotary_embedding = RotaryEmbedding(dim=inner_dim)

    def scaled_dot_product_attention(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            is_causal: bool = False
    ) -> Tensor:
        L, S = query.shape[-2], key.shape[-2]
        attn_bias = torch.zeros(L, S, dtype=query.dtype).to(query)
        if is_causal:
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0).to(attn_bias)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

        attn_weight = query @ key.transpose(-2, -1) * self.scale
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        return attn_weight @ value

    def forward(self, x: Tensor, is_causal: bool = False) -> Tensor:
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v))
        if self.rot_emb:
            q = self.rotary_embedding.rotate_queries_or_keys(q, self.rotary_embedding.freqs)
            k = self.rotary_embedding.rotate_queries_or_keys(k, self.rotary_embedding.freqs)
            q, k = map(lambda t: t.contiguous(), (q, k))
        out = self.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        del q, k, v
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class SpatioBlock(nn.Module):
    """只在同一帧的空间 patch 之间做 attention。"""

    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super(SpatioBlock, self).__init__()
        self.spatial_attn = SelfAttention(model_dim, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim)
        )

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, x: Tensor) -> Tensor:
        t_len = x.shape[1]

        # Spatial attention
        x = rearrange(x, "b t s e -> (b t) s e")
        x = _match_module_dtype(x, self.norm1)
        x_ = self.norm1(x)
        x_ = self.spatial_attn(x_)
        x = x + x_
        x = rearrange(x, "(b t) s e -> b t s e", t=t_len)

        # Feedforward
        x = _match_module_dtype(x, self.norm2)
        x_ = self.norm2(x)
        x_ = self.ffn(x_)
        x = x + x_
        return x


class SpatioTemporalBlock(nn.Module):
    """先做空间 attention，再做时间 attention，用于建模两帧之间的变化。"""

    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super(SpatioTemporalBlock, self).__init__()
        self.spatial_attn = SelfAttention(model_dim, num_heads, dropout=dropout)
        self.temporal_attn = SelfAttention(model_dim, num_heads, dropout=dropout, rot_emb=True)#适合编码序列信息，相对位置
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim)
        )

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)

    def forward(self, x: Tensor, causal_temporal: bool = False) -> Tensor:
        t_len, s_len = x.shape[1:3]

        # Spatial attention
        x = rearrange(x, "b t s e -> (b t) s e")
        x = _match_module_dtype(x, self.norm1)
        x_ = self.norm1(x)
        x_ = self.spatial_attn(x_)
        x = x + x_
        x = rearrange(x, "(b t) s e -> b t s e", t=t_len)

        # Temporal attention
        x = rearrange(x, "b t s e -> (b s) t e")
        x = _match_module_dtype(x, self.norm2)
        x_ = self.norm2(x)
        if causal_temporal:
            x_ = self.temporal_attn(x_, is_causal=True)#is_causal决定当前步能不能看到未来步的数据
        else:
            x_ = self.temporal_attn(x_)
        x = x + x_
        x = rearrange(x, "(b s) t e -> b t s e", s=s_len)

        # Feedforward
        x = _match_module_dtype(x, self.norm3)
        x_ = self.norm3(x)
        x_ = self.ffn(x_)
        x = x + x_
        return x


class SpatioTransformer(nn.Module):
    """空间 transformer decoder，把 action latent 和第一帧 feature 解码到目标 feature。"""

    def __init__(
            self,
            in_dim: int,
            model_dim: int,
            out_dim: int,
            num_blocks: int,
            num_heads: int,
            dropout: float = 0.0
    ) -> None:
        super(SpatioTransformer, self).__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, model_dim),
            nn.LayerNorm(model_dim)
        )
        self.pos_enc = PositionalEncoding(model_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                SpatioBlock(
                    model_dim,
                    num_heads,
                    dropout
                ) for _ in range(num_blocks)
            ]
        )
        self.out = nn.Linear(model_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.ffn(x)
        x = self.pos_enc(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.out(x)
        return x  # (B, T, E)


class SpatioTemporalTransformer(nn.Module):
    """LAM encoder，用时空 attention 从两帧输入中提取 latent action。"""

    def __init__(
            self,
            in_dim: int,
            model_dim: int,
            out_dim: int,
            num_blocks: int,
            num_heads: int,
            dropout: float = 0.0,
            causal_temporal: bool = False
    ) -> None:
        super(SpatioTemporalTransformer, self).__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, model_dim),
            nn.LayerNorm(model_dim)
        )
        self.pos_enc = PositionalEncoding(model_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                SpatioTemporalBlock(
                    model_dim,
                    num_heads,
                    dropout
                ) for _ in range(num_blocks)
            ]
        )
        self.out = nn.Linear(model_dim, out_dim)
        self.causal_temporal = causal_temporal

    def forward(self, x: Tensor) -> Tensor:
        x = self.ffn(x)
        x = self.pos_enc(x)
        for block in self.transformer_blocks:
            x = block(x, self.causal_temporal)
        x = self.out(x)
        return x  # (B, T, E)


class VectorQuantizer(nn.Module):
    """VQ 代码本保留自原 LAM 实现；当前 alignment 路径主要使用 VAE latent。"""

    def __init__(self, num_latents: int, latent_dim: int, code_restart: bool = False) -> None:
        super(VectorQuantizer, self).__init__()
        self.codebook = nn.Embedding(num_latents, latent_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_latents, 1.0 / num_latents)

        # Initialize a usage buffer
        self.register_buffer("usage", torch.zeros(num_latents), persistent=False)
        self.num_latents = num_latents

        self.code_restart = code_restart

    def update_usage(self, min_enc) -> None:
        for idx in min_enc:
            self.usage[idx] = self.usage[idx] + 1  # Add used code

    def random_restart(self) -> None:
        if self.code_restart:
            # Randomly restart all dead codes
            dead_codes = torch.nonzero(self.usage < 1).squeeze(1)
            rand_codes = torch.randperm(self.num_latents)[0:len(dead_codes)]
            print(f"Restarting {len(dead_codes)} codes")
            with torch.no_grad():
                self.codebook.weight[dead_codes] = self.codebook.weight[rand_codes]

            if hasattr(self, "inner_vq"):
                self.inner_vq.random_restart()

    def reset_usage(self) -> None:
        if self.code_restart:
            # Reset usage between epochs
            self.usage.zero_()

            if hasattr(self, "inner_vq"):
                self.inner_vq.reset_usage()

    def forward(self, x: Tensor, delta_psnr: bool = False) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # Compute distances
        distance = torch.cdist(x, self.codebook.weight)

        # Get indices and embeddings
        indices = torch.argmin(distance, dim=1)
        if delta_psnr:
            shape = indices.shape
            rand_indices = torch.randint(0, self.num_latents, shape).to(distance.device)
            while torch.any(rand_indices == indices):
                new_indices = torch.randint(0, self.num_latents, shape).to(distance.device)
                rand_indices = torch.where(rand_indices == indices, new_indices, rand_indices)
            z = self.codebook(rand_indices)
        else:
            z = self.codebook(indices)

        # Update code usage
        if not self.training or self.code_restart:
            self.update_usage(indices)

        # Straight through estimator
        z_q = x + (z - x).detach()
        return z_q, z, x, indices


class ResidualVectorQuantizer(VectorQuantizer):
    """残差 VQ 版本，当前实验未直接使用。"""

    def __init__(self, num_latents: int, latent_dim: int) -> None:
        super(ResidualVectorQuantizer, self).__init__(num_latents, latent_dim)
        self.inner_vq = VectorQuantizer(num_latents, latent_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        # Compute distances
        distance = torch.cdist(x, self.codebook.weight)

        # Get indices and embeddings
        indices = torch.argmin(distance, dim=1)
        z = self.codebook(indices)

        # Residual quantization
        residual = x - z.detach()
        inner_z_q, inner_z, inner_x, inner_indices = self.inner_vq(residual)

        # Update code usage
        if not self.training or self.code_restart:
            self.update_usage(indices)
            self.inner_vq.update_usage(inner_indices)

        # Straight through estimator
        z_q = x + (z - x).detach()
        return z_q + inner_z_q, z, x, indices, inner_z, inner_x, inner_indices
