"""PyTorch dataset over a processed ``samples.csv`` (written by scripts/prepare_data.py).

Required columns: ``image_path`` (cached square image), ``y`` (int label), plus metadata columns.
Optional: ``mask_path`` (lesion mask cached at the same size, used only for evaluation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

from medfusion.data.metadata import MetadataSchema


def build_transform(cfg: dict, train: bool) -> T.Compose:
    img, aug = cfg["image"], cfg["augment"]
    size = img["input_size"]
    ops: list = [T.ToImage()]
    if train:
        ops += [
            T.RandomResizedCrop(size, scale=tuple(aug["rrc_scale"]), antialias=True),
            T.RandomHorizontalFlip(aug["hflip"]),
            T.RandomVerticalFlip(aug["vflip"]),
            T.RandomRotation(aug["rotate"]),
            T.ColorJitter(*(3 * [aug["color_jitter"]])),
        ]
    else:
        ops += [T.Resize((size, size), antialias=True)]
    ops += [T.ToDtype(torch.float32, scale=True), T.Normalize(img["mean"], img["std"])]
    return T.Compose(ops)


class FusionDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        schema: MetadataSchema,
        transform,
        with_mask: bool = False,
        mask_size: int | None = None,
    ):
        self.df = df.reset_index(drop=True)
        self.cat, self.cont, self.cont_mask = schema.encode(self.df)
        self.y = self.df["y"].to_numpy(dtype=np.int64)
        self.transform = transform
        self.with_mask = with_mask and "mask_path" in self.df.columns
        self.mask_size = mask_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict:
        image = Image.open(self.df.at[i, "image_path"]).convert("RGB")
        item = {
            "image": self.transform(image),
            "cat": torch.from_numpy(self.cat[i]),
            "cont": torch.from_numpy(self.cont[i]),
            "cont_mask": torch.from_numpy(self.cont_mask[i]),
            "y": torch.tensor(self.y[i]),
            "index": torch.tensor(i),
        }
        if self.with_mask:
            path = self.df.at[i, "mask_path"]
            mask = Image.open(path).convert("L") if isinstance(path, str) else None
            if mask is not None and self.mask_size:
                mask = mask.resize((self.mask_size, self.mask_size), Image.NEAREST)
            item["lesion_mask"] = (
                torch.from_numpy(np.asarray(mask) > 127)
                if mask is not None
                else torch.zeros(self.mask_size, self.mask_size, dtype=torch.bool)
            )
            item["has_mask"] = torch.tensor(mask is not None)
        return item
