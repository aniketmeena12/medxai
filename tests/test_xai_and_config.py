import pytest
import torch

from medfusion.data.metadata import MetadataSchema
from medfusion.models.factory import build_model
from medfusion.utils.config import load_config, parse_overrides
from medfusion.xai.attribution import gradcam, passes_sanity_gate, randomized_copy, rise
from tests.test_models import CFG


@pytest.fixture(scope="module")
def model_and_batch():
    s = MetadataSchema(["sex"], ["age"])
    s.vocabs, s.means, s.stds = {"sex": ["f", "m"]}, {"age": 0.0}, {"age": 1.0}
    torch.manual_seed(0)
    model = build_model("M1", CFG, s, n_classes=2)
    batch = {"image": torch.randn(2, 3, 64, 64), "cat": torch.tensor([[0], [1]]),
             "cont": torch.randn(2, 1), "cont_mask": torch.ones(2, 1, dtype=torch.bool)}
    return model, batch


def test_gradcam_shape_and_range(model_and_batch):
    model, batch = model_and_batch
    maps = gradcam(model, batch)
    assert maps.shape == (2, 64, 64)
    assert maps.min() >= 0 and maps.max() <= 1


def test_rise_shape(model_and_batch):
    model, batch = model_and_batch
    maps = rise(model, batch, n_masks=16, chunk=8)
    assert maps.shape == (2, 64, 64)


def test_sanity_gate():
    a = torch.rand(4, 16, 16)
    assert not passes_sanity_gate(a, a.clone())[0]
    ok, score = passes_sanity_gate(a, torch.rand(4, 16, 16))
    assert ok and score < 0.5


def test_randomized_copy_differs(model_and_batch):
    model, _ = model_and_batch
    rnd = randomized_copy(model)
    p0, p1 = next(model.parameters()), next(rnd.parameters())
    assert not torch.equal(p0, p1)


def test_config_inherit_and_env(tmp_path, monkeypatch):
    (tmp_path / "base.yaml").write_text("a: {b: 1, c: [1, 2]}\npath: ${ROOT_X:/default}/x\n")
    (tmp_path / "child.yaml").write_text("inherit: [base.yaml]\na: {b: 2}\n")
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg["a"] == {"b": 2, "c": [1, 2]}
    assert cfg["path"] == "/default/x"
    monkeypatch.setenv("ROOT_X", "D:/data")
    assert load_config(tmp_path / "child.yaml")["path"] == "D:/data/x"
    assert parse_overrides(["train.epochs=2", "model.name=M1"]) == {
        "train": {"epochs": 2}, "model": {"name": "M1"}}
