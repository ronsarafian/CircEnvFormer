"""CircEnvFormer model components.

The implementation follows the model used in the manuscript analysis:
patch embedding -> month embedding -> transformer encoder -> latent embedding.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """Convert gridded fields into non-overlapping patch tokens."""

    def __init__(
        self,
        img_size: tuple[int, int] = (40, 94),
        patch_size: tuple[int, int] = (10, 10),
        in_channels: int = 3,
        embed_dim: int = 256,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return tokens with shape ``[batch, patches, embedding]``."""
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class MonthEmbedding(nn.Module):
    """Encode calendar month using a circular sine/cosine representation."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(2, embed_dim)

    def forward(self, month: torch.Tensor) -> torch.Tensor:
        month = month.float()
        x = (month - 1.0) / 12.0
        sin_m = torch.sin(2.0 * math.pi * x)
        cos_m = torch.cos(2.0 * math.pi * x)
        return self.proj(torch.stack([sin_m, cos_m], dim=1))


class MultiheadAttentionWithReturn(nn.Module):
    """Multi-head self-attention that also returns per-head attention weights."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out, attn = self.mha(
            x, x, x,
            need_weights=True,
            average_attn_weights=False,
        )
        return out, attn


class TransformerEncoderLayerWithAttn(nn.Module):
    """Transformer encoder block matching the analysis implementation."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.self_attn = MultiheadAttentionWithReturn(embed_dim, num_heads, dropout)
        self.linear1 = nn.Linear(embed_dim, mlp_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(mlp_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_map = self.self_attn(src)
        src = self.norm1(src + self.dropout1(attn_out))
        mlp_out = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(mlp_out))
        return src, attn_map


class CircEnvEncoder(nn.Module):
    """CircEnvFormer transformer encoder producing a normalized latent embedding."""

    def __init__(
        self,
        img_size: tuple[int, int] = (40, 94),
        patch_size: tuple[int, int] = (10, 10),
        in_channels: int = 3,
        embed_dim: int = 256,
        depth: int = 8,
        num_heads: int = 8,
        mlp_dim: int = 512,
        dropout: float = 0.0,
        global_embed_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.month_embed = MonthEmbedding(embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderLayerWithAttn(embed_dim, num_heads, mlp_dim, dropout)
                for _ in range(depth)
            ]
        )
        self.global_embed_weight = float(global_embed_weight)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    @property
    def embedding_dim(self) -> int:
        return self.pos_embed.shape[-1]

    def forward(
        self,
        x: torch.Tensor,
        month: torch.Tensor,
        output_attentions: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if month is None:
            raise ValueError("month must be provided.")

        batch = x.size(0)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch, -1, -1)
        month_emb = self.month_embed(month).unsqueeze(1)
        x = x + month_emb
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(x + self.pos_embed)

        attentions: list[torch.Tensor] = []
        for block in self.blocks:
            x, attn = block(x)
            if output_attentions:
                attentions.append(attn)

        local_embed = F.normalize(x[:, 0], dim=1)
        global_embed = F.normalize(x[:, 1:].mean(dim=1), dim=1)
        embedding = self.global_embed_weight * global_embed + (1.0 - self.global_embed_weight) * local_embed
        embedding = F.normalize(embedding, dim=1)

        if output_attentions:
            return embedding, attentions
        return embedding


class RainfallClassifier(nn.Module):
    """Linear station-wise binary rainfall classifier used after the encoder."""

    def __init__(self, embed_size: int = 256, n_stations: int = 27, n_classes: int = 2) -> None:
        super().__init__()
        self.n_stations = int(n_stations)
        self.n_classes = int(n_classes)
        self.fc = nn.Linear(embed_size, n_classes * n_stations)
        nn.init.trunc_normal_(self.fc.weight, std=0.02)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).reshape(-1, self.n_classes, self.n_stations)
