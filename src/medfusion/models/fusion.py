"""Image-only baseline and the four image-metadata fusion models compared in the paper.

Every model takes ``(image, cat, cont, cont_mask)`` and returns logits, and shares the same
``ImageEncoder`` and ``MetadataTokenizer``, so differences come from the fusion mechanism alone.

    B0 ImageOnly        image features -> pooled -> head
    B3 ConcatFusion     [pooled image ; metadata vector] -> head
    B4 FiLMFusion       metadata -> per-channel (gamma, beta) on the feature map
    B5 MetaBlockFusion  Pacheco & Krohling 2021: sigmoid(tanh(F * f(m)) + g(m))
    M1 CrossAttnFusion  image patch tokens attend to metadata field tokens
"""

from __future__ import annotations

import torch
from torch import nn

from medfusion.models.encoder import ImageEncoder


class MetadataTokenizer(nn.Module):
    """One token per clinical field. Categorical index ``cardinality`` is the MISSING category."""

    def __init__(self, cat_cardinalities: list[int], n_continuous: int, dim: int):
        super().__init__()
        self.cat_embeds = nn.ModuleList(nn.Embedding(c + 1, dim) for c in cat_cardinalities)
        self.cont_proj = nn.ModuleList(nn.Linear(1, dim) for _ in range(n_continuous))
        self.cont_missing = nn.Parameter(torch.zeros(n_continuous, dim))
        self.n_fields = len(cat_cardinalities) + n_continuous
        self.field_pos = nn.Parameter(torch.zeros(1, self.n_fields, dim))
        nn.init.trunc_normal_(self.field_pos, std=0.02)
        nn.init.trunc_normal_(self.cont_missing, std=0.02)
        self.norm = nn.LayerNorm(dim)

    def forward(self, cat: torch.Tensor, cont: torch.Tensor, cont_mask: torch.Tensor):
        tokens = [emb(cat[:, j]) for j, emb in enumerate(self.cat_embeds)]
        for j, proj in enumerate(self.cont_proj):
            value = proj(cont[:, j : j + 1])
            tokens.append(torch.where(cont_mask[:, j : j + 1], value, self.cont_missing[j]))
        return self.norm(torch.stack(tokens, dim=1) + self.field_pos)


class FusionModel(nn.Module):
    """Common interface. Subclasses implement ``fuse(feature_map, meta_tokens) -> logits``."""

    uses_metadata = True

    def __init__(self, encoder: ImageEncoder, tokenizer: MetadataTokenizer | None):
        super().__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer

    def forward(self, image, cat=None, cont=None, cont_mask=None) -> torch.Tensor:
        feats = self.encoder(image)
        meta = self.tokenizer(cat, cont, cont_mask) if self.uses_metadata else None
        return self.fuse(feats, meta)

    def fuse(self, feats: torch.Tensor, meta: torch.Tensor | None) -> torch.Tensor:
        raise NotImplementedError


def _head(in_dim: int, n_classes: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.LayerNorm(in_dim), nn.Dropout(dropout), nn.Linear(in_dim, n_classes))


class _MetaVector(nn.Module):
    """Flatten field tokens and embed them into one vector (used by B3/B4/B5)."""

    def __init__(self, n_fields: int, meta_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(n_fields * meta_dim, out_dim), nn.GELU(), nn.Dropout(dropout)
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(tokens)


class ImageOnly(FusionModel):
    uses_metadata = False

    def __init__(self, encoder, n_classes, head_dropout=0.3, **_):
        super().__init__(encoder, None)
        self.head = _head(encoder.num_features, n_classes, head_dropout)

    def fuse(self, feats, meta):
        return self.head(feats.mean(dim=(2, 3)))


class ConcatFusion(FusionModel):
    def __init__(self, encoder, tokenizer, n_classes, meta_dim, head_dropout=0.3, hidden=256, **_):
        super().__init__(encoder, tokenizer)
        self.meta = _MetaVector(tokenizer.n_fields, meta_dim, hidden, head_dropout)
        self.head = _head(encoder.num_features + hidden, n_classes, head_dropout)

    def fuse(self, feats, meta):
        return self.head(torch.cat([feats.mean(dim=(2, 3)), self.meta(meta)], dim=1))


class FiLMFusion(FusionModel):
    def __init__(self, encoder, tokenizer, n_classes, meta_dim, head_dropout=0.3, hidden=256, **_):
        super().__init__(encoder, tokenizer)
        c = encoder.num_features
        self.meta = _MetaVector(tokenizer.n_fields, meta_dim, hidden, head_dropout)
        self.film = nn.Linear(hidden, 2 * c)
        nn.init.zeros_(self.film.weight)  # start as identity: gamma = 0 -> scale 1, beta = 0
        nn.init.zeros_(self.film.bias)
        self.head = _head(c, n_classes, head_dropout)

    def fuse(self, feats, meta):
        gamma, beta = self.film(self.meta(meta)).chunk(2, dim=1)
        feats = feats * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        return self.head(feats.mean(dim=(2, 3)))


class MetaBlockFusion(FusionModel):
    """MetaBlock (Pacheco & Krohling, IEEE JBHI 2021), applied channel-wise to the feature map."""

    def __init__(self, encoder, tokenizer, n_classes, meta_dim, head_dropout=0.3, hidden=256, **_):
        super().__init__(encoder, tokenizer)
        c = encoder.num_features
        self.meta = _MetaVector(tokenizer.n_fields, meta_dim, hidden, head_dropout)
        self.f = nn.Sequential(nn.Linear(hidden, c), nn.BatchNorm1d(c))
        self.g = nn.Sequential(nn.Linear(hidden, c), nn.BatchNorm1d(c))
        self.head = _head(c, n_classes, head_dropout)

    def fuse(self, feats, meta):
        m = self.meta(meta)
        f, g = self.f(m)[:, :, None, None], self.g(m)[:, :, None, None]
        feats = torch.sigmoid(torch.tanh(feats * f) + g)
        return self.head(feats.mean(dim=(2, 3)))


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention + FFN with residuals. Keeps head-averaged attention weights."""

    def __init__(self, dim: int, n_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm_q, self.norm_kv = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim)
        )
        self.last_attn: torch.Tensor | None = None

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        kv = self.norm_kv(context)
        out, weights = self.attn(self.norm_q(query), kv, kv, need_weights=True,
                                 average_attn_weights=True)
        self.last_attn = weights.detach()
        query = query + out
        return query + self.ffn(self.norm_ffn(query))


class CrossAttnFusion(FusionModel):
    DIRECTIONS = ("img2meta", "meta2img", "bidirectional")

    def __init__(self, encoder, tokenizer, n_classes, n_blocks=2, n_heads=8,
                 direction="img2meta", head_dropout=0.3, **_):
        super().__init__(encoder, tokenizer)
        if direction not in self.DIRECTIONS:
            raise ValueError(f"direction must be one of {self.DIRECTIONS}, got {direction!r}")
        dim = encoder.num_features
        self.direction = direction
        self.img2meta = (
            nn.ModuleList(CrossAttentionBlock(dim, n_heads) for _ in range(n_blocks))
            if direction in ("img2meta", "bidirectional") else None
        )
        self.meta2img = (
            nn.ModuleList(CrossAttentionBlock(dim, n_heads) for _ in range(n_blocks))
            if direction in ("meta2img", "bidirectional") else None
        )
        out_dim = dim * (2 if direction == "bidirectional" else 1)
        self.head = _head(out_dim, n_classes, head_dropout)
        self._grid: tuple[int, int] | None = None

    def fuse(self, feats, meta):
        b, c, h, w = feats.shape
        self._grid = (h, w)
        patches = feats.flatten(2).transpose(1, 2)  # (B, N, C)
        pooled = []
        if self.img2meta is not None:
            p = patches
            for block in self.img2meta:
                p = block(p, meta)
            pooled.append(p.mean(dim=1))
        if self.meta2img is not None:
            m = meta
            for block in self.meta2img:
                m = block(m, patches)
            pooled.append(m.mean(dim=1))
        return self.head(torch.cat(pooled, dim=1))

    def attention_maps(self) -> torch.Tensor | None:
        """(B, n_fields, H, W): where each metadata field was consulted, from the last img2meta
        block of the most recent forward pass. ``None`` if unavailable."""
        if self.img2meta is None or self._grid is None or self.img2meta[-1].last_attn is None:
            return None
        attn = self.img2meta[-1].last_attn  # (B, N, M)
        h, w = self._grid
        return attn.transpose(1, 2).reshape(attn.shape[0], attn.shape[2], h, w)
