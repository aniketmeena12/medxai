"""Build a model from its paper ID (B0, B3, B4, B5, M1). B1 (metadata-only) lives in tabular.py."""

from __future__ import annotations

from medfusion.data.metadata import MetadataSchema
from medfusion.models.encoder import ImageEncoder
from medfusion.models.fusion import (
    ConcatFusion,
    CrossAttnFusion,
    FiLMFusion,
    FusionModel,
    ImageOnly,
    MetaBlockFusion,
    MetadataTokenizer,
)

MODELS = {
    "B0": ImageOnly,
    "B3": ConcatFusion,
    "B4": FiLMFusion,
    "B5": MetaBlockFusion,
    "M1": CrossAttnFusion,
}


def build_model(model_id: str, cfg: dict, schema: MetadataSchema, n_classes: int) -> FusionModel:
    if model_id not in MODELS:
        raise ValueError(f"unknown model {model_id!r}; choose from {sorted(MODELS)} (B1: tabular)")
    mcfg = cfg["model"]
    encoder = ImageEncoder(mcfg["backbone"], mcfg["pretrained"], mcfg["freeze_stages"])
    cls = MODELS[model_id]
    if cls is ImageOnly:
        return ImageOnly(encoder, n_classes, head_dropout=mcfg["head_dropout"])
    # Cross-attention needs tokens in the image feature space; vector fusions use meta_dim.
    dim = encoder.num_features if cls is CrossAttnFusion else mcfg["meta_dim"]
    tokenizer = MetadataTokenizer(schema.cardinalities, len(schema.continuous), dim)
    return cls(
        encoder,
        tokenizer,
        n_classes,
        meta_dim=mcfg["meta_dim"],
        head_dropout=mcfg["head_dropout"],
        **(mcfg.get("crossattn", {}) if cls is CrossAttnFusion else {}),
    )
