import numpy as np
import pandas as pd
import pytest
import torch

from medfusion.data.metadata import MetadataSchema, field_dropout
from medfusion.data.splits import assert_disjoint_groups, grouped_folds, inner_val_split
from medfusion.data.trap import agreement_fraction, resample_metadata_trap, synthetic_attribute


def _df():
    return pd.DataFrame({
        "sex": ["male", "female", "UNK", None, "male", "other"],
        "age": [30, np.nan, 50, 70, 40, 60],
    })


def test_schema_missing_and_unseen():
    df = _df()
    train = np.array([True, True, True, True, False, False])
    s = MetadataSchema(["sex"], ["age"]).fit(df.iloc[:5], train_mask=train[:5])
    assert s.vocabs["sex"] == ["female", "male"]  # UNK / None are missing, "other" unseen
    cat, cont, mask = s.encode(df)
    assert cat[:, 0].tolist() == [1, 0, 2, 2, 1, 2]
    assert mask[:, 0].tolist() == [True, False, True, True, True, True]
    assert s.means["age"] == pytest.approx(50.0)  # mean of 30, 50, 70 (training rows only)
    assert cont[1, 0] == 0.0


def test_schema_roundtrip():
    s = MetadataSchema(["sex"], ["age"]).fit(_df())
    assert MetadataSchema.from_dict(s.to_dict()).to_dict() == s.to_dict()


def test_field_dropout():
    g = torch.Generator().manual_seed(0)
    cat = torch.zeros(1000, 2, dtype=torch.long)
    mask = torch.ones(1000, 1, dtype=torch.bool)
    new_cat, new_mask = field_dropout(cat, mask, [3, 5], p=0.3, generator=g)
    assert set(new_cat[:, 0].unique().tolist()) == {0, 3}
    assert set(new_cat[:, 1].unique().tolist()) == {0, 5}
    assert 0.25 < (new_cat[:, 0] == 3).float().mean() < 0.35
    assert 0.25 < (~new_mask).float().mean() < 0.35
    assert field_dropout(cat, mask, [3, 5], p=0.0)[0] is cat


def test_grouped_folds_no_leak():
    rng = np.random.default_rng(0)
    groups = rng.integers(0, 200, 1000)
    labels = rng.integers(0, 3, 1000)
    folds = grouped_folds(labels, groups, n_folds=5, seed=0)
    assert set(folds) == {0, 1, 2, 3, 4}
    for k in range(5):
        assert_disjoint_groups(groups[folds == k], groups[folds != k])
    val = inner_val_split(labels, groups, 0.15, seed=1)
    assert_disjoint_groups(groups[val], groups[~val])
    with pytest.raises(ValueError):
        assert_disjoint_groups([1, 2], [2, 3])


@pytest.mark.parametrize("rho,reverse,expected", [(0.0, False, 0.5), (0.9, False, 0.95),
                                                  (0.9, True, 0.05), (0.5, True, 0.25)])
def test_metadata_trap_resampling(rho, reverse, expected):
    rng = np.random.default_rng(0)
    n = 4000
    df = pd.DataFrame({"y": rng.integers(0, 2, n)})
    attr = rng.integers(0, 2, n).astype(float)
    attr[:50] = np.nan
    out = resample_metadata_trap(df.assign(a=attr), attr, "y", rho, reverse, rng)
    assert agreement_fraction(out["a"], out["y"]) == pytest.approx(expected, abs=0.01)
    # Class balance of the source is kept.
    assert out["y"].mean() == pytest.approx(df["y"].mean(), abs=0.03)
    assert len(out) > 0


def test_synthetic_attribute():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 20000)
    a = synthetic_attribute(y, 0.8, reverse=False, rng=rng)
    assert agreement_fraction(a, y) == pytest.approx(0.9, abs=0.01)
    a = synthetic_attribute(y, 0.8, reverse=True, rng=rng)
    assert agreement_fraction(a, y) == pytest.approx(0.1, abs=0.01)
