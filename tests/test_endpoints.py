"""Endpoint computation (docs/04-preregistration.md §5): P1, P4, P5 on tiny synthetic data."""

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from medfusion.data.metadata import MetadataSchema
from medfusion.eval import endpoints as E
from medfusion.models.factory import build_model

CFG = {
    "model": {
        "backbone": "convnext_tiny", "pretrained": False, "freeze_stages": 0, "meta_dim": 16,
        "head_dropout": 0.0, "meta_field_dropout": 0.0,
        "crossattn": {"n_blocks": 1, "n_heads": 4, "direction": "img2meta"},
    },
    "image": {"input_size": 32, "mean": [0.5] * 3, "std": [0.5] * 3},
    "augment": {"hflip": 0, "vflip": 0, "rotate": 0, "color_jitter": 0, "rrc_scale": [0.9, 1.0]},
}


@pytest.fixture(scope="module")
def schema():
    s = MetadataSchema(["sex"], ["age"])
    s.vocabs = {"sex": ["female", "male"]}
    s.means, s.stds = {"age": 50.0}, {"age": 10.0}
    return s


@pytest.fixture(scope="module")
def frame(tmp_path_factory):
    """Four samples with an image and a centred square lesion mask."""
    d = tmp_path_factory.mktemp("endpoints")
    rows = []
    for i in range(4):
        img, mask = d / f"{i}.jpg", d / f"{i}.png"
        Image.fromarray(np.random.default_rng(i).integers(0, 255, (48, 48, 3), dtype=np.uint8)
                        ).save(img)
        m = np.zeros((48, 48), np.uint8)
        m[16:32, 16:32] = 255
        Image.fromarray(m).save(mask)
        rows.append({"sample_id": f"s{i}", "patient_id": f"p{i // 2}", "y": i % 2,
                     "sex": ["male", "female"][i % 2], "age": 40 + i,
                     "image_path": str(img), "mask_path": str(mask)})
    return pd.DataFrame(rows)


def _model(schema, model_id="M1"):
    return build_model(model_id, CFG, schema, n_classes=2).eval()


def test_grounding_rows_and_ranges(schema, frame):
    out = E.grounding(_model(schema), frame, schema, CFG, torch.device("cpu"), "patient_id",
                      batch_size=2)
    assert len(out) == len(frame)
    assert set(out.columns) >= {"sample_id", "patient_id", "energy_in_mask", "mask_area_fraction",
                                "pointing_game", "iou_otsu"}
    # NaN is legitimate: an all-zero Grad-CAM map has no positive mass to attribute (an
    # untrained model produces those). Everything else must be a share in [0, 1].
    e = out["energy_in_mask"].dropna()
    assert ((e >= 0) & (e <= 1)).all()
    # the mask covers (16:32)^2 of 48^2 scaled to the 32px input, i.e. about a ninth to a quarter
    assert 0.05 < out["mask_area_fraction"].mean() < 0.5


def test_grounding_skips_rows_without_a_mask(schema, frame):
    partial = frame.copy()
    partial.loc[0, "mask_path"] = None
    out = E.grounding(_model(schema), partial, schema, CFG, torch.device("cpu"), "patient_id",
                      batch_size=2)
    assert len(out) == len(frame) - 1


def test_counterfactual_one_row_per_image_and_field(schema, frame):
    fields = {"relevant": ["sex"], "irrelevant": ["age"]}
    out = E.counterfactual(_model(schema), frame, schema, CFG, torch.device("cpu"), "patient_id",
                           fields, batch_size=2)
    assert len(out) == len(frame) * 2
    assert set(out["field"]) == {"sex", "age"}
    assert set(out["field_kind"]) == {"relevant", "irrelevant"}
    assert ((out["map_shift"] >= 0) & (out["map_shift"] <= 2)).all()


def test_image_only_model_has_zero_map_shift(schema, frame):
    """B0 ignores metadata, so perturbing a field cannot move its map (§7 assumes this)."""
    out = E.counterfactual(_model(schema, "B0"), frame, schema, CFG, torch.device("cpu"),
                           "patient_id", {"irrelevant": ["sex"]}, batch_size=2)
    assert out["map_shift"].abs().max() == pytest.approx(0.0, abs=1e-9)


def test_reliance_shapes_and_zero_metadata_drop_for_b0(schema, frame):
    out = E.reliance(_model(schema, "B0"), frame, schema, CFG, torch.device("cpu"), batch_size=2)
    assert len(out) == len(frame)
    assert (out["pred_full"] == out["pred_meta_shuffled"]).all()
    drops = E.reliance_drops(out)
    assert drops["drop_metadata"] == pytest.approx(0.0)
    assert set(drops) == {"bacc_full", "bacc_meta_shuffled", "bacc_mean_image",
                          "drop_metadata", "drop_image"}


def test_reliance_drops_subset_rows(schema, frame):
    out = E.reliance(_model(schema), frame, schema, CFG, torch.device("cpu"), batch_size=2)
    assert E.reliance_drops(out, rows=np.array([0, 1])) != {}
