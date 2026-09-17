import pytest
import torch

from medfusion.data.metadata import MetadataSchema
from medfusion.models.factory import build_model
from medfusion.models.fusion import CrossAttnFusion

CFG = {
    "model": {
        "backbone": "convnext_tiny", "pretrained": False, "freeze_stages": 2, "meta_dim": 16,
        "head_dropout": 0.0, "meta_field_dropout": 0.0,
        "crossattn": {"n_blocks": 1, "n_heads": 4, "direction": "img2meta"},
    }
}


@pytest.fixture(scope="module")
def schema():
    s = MetadataSchema(["sex", "site"], ["age"])
    s.vocabs = {"sex": ["female", "male"], "site": ["arm", "back", "face"]}
    s.means, s.stds = {"age": 50.0}, {"age": 10.0}
    return s


def _batch(b=2, size=64):
    return (torch.randn(b, 3, size, size), torch.tensor([[0, 1], [2, 3]]), torch.randn(b, 1),
            torch.tensor([[True], [False]]))


@pytest.mark.parametrize("model_id", ["B0", "B3", "B4", "B5", "M1"])
def test_output_shape(schema, model_id):
    model = build_model(model_id, CFG, schema, n_classes=3).eval()
    assert model(*_batch()).shape == (2, 3)


def test_frozen_stages(schema):
    model = build_model("B0", CFG, schema, n_classes=3)
    bb = model.encoder.backbone
    assert not any(p.requires_grad for p in bb.stages[1].parameters())
    assert all(p.requires_grad for p in bb.stages[3].parameters())


def test_image_only_ignores_metadata(schema):
    model = build_model("B0", CFG, schema, n_classes=3).eval()
    img, cat, cont, mask = _batch()
    with torch.no_grad():
        assert torch.allclose(model(img, cat, cont, mask), model(img, cat * 0, cont + 5, ~mask))


@pytest.mark.parametrize("model_id", ["B3", "B4", "B5", "M1"])
def test_fusion_uses_metadata(schema, model_id):
    torch.manual_seed(0)
    model = build_model(model_id, CFG, schema, n_classes=3).eval()
    for p in model.parameters():  # FiLM starts as identity; perturb so metadata has an effect
        p.data += 0.05 * torch.randn_like(p)
    img, cat, cont, mask = _batch()
    with torch.no_grad():
        assert not torch.allclose(model(img, cat, cont, mask), model(img, 1 - cat.clamp(max=1),
                                                                     cont, mask))


def test_masked_continuous_value_is_ignored(schema):
    model = build_model("M1", CFG, schema, n_classes=3).eval()
    img, cat, cont, mask = _batch()
    mask = torch.zeros_like(mask)
    with torch.no_grad():
        assert torch.allclose(model(img, cat, cont, mask), model(img, cat, cont + 100, mask),
                              atol=1e-5)


def test_attention_maps(schema):
    model = build_model("M1", CFG, schema, n_classes=3).eval()
    img, cat, cont, mask = _batch(size=64)
    with torch.no_grad():
        model(img, cat, cont, mask)
    maps = model.attention_maps()
    assert maps.shape == (2, 3, 2, 2)  # 3 fields, 64/32 = 2x2 grid
    # Each image location's attention over the metadata fields is a distribution.
    assert torch.allclose(maps.sum(dim=1), torch.ones(2, 2, 2), atol=1e-5)


def test_bad_direction(schema):
    with pytest.raises(ValueError):
        cfg = {"model": {**CFG["model"], "crossattn": {"direction": "sideways"}}}
        build_model("M1", cfg, schema, n_classes=3)


def test_meta2img_has_no_spatial_maps(schema):
    cfg = {"model": {**CFG["model"], "crossattn": {"n_blocks": 1, "n_heads": 4,
                                                   "direction": "meta2img"}}}
    model = build_model("M1", cfg, schema, n_classes=3).eval()
    assert isinstance(model, CrossAttnFusion)
    with torch.no_grad():
        model(*_batch())
    assert model.attention_maps() is None
