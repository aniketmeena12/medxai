import numpy as np
import pytest
import torch

from medfusion.data.metadata import MetadataSchema
from medfusion.eval.counterfactual import map_shift, perturb_field
from medfusion.eval.localization import (
    binarize,
    energy_in_mask,
    iou,
    mask_area_fraction,
    miou,
    otsu_threshold,
    pointing_game,
)
from medfusion.eval.shortcut import DegradationCurve, offloading_index, stratified_auc
from medfusion.eval.stats import cluster_bootstrap_diff, holm, mean_ci, t_ci, tost_equivalent


def _square_mask(size=32, lo=8, hi=24):
    m = np.zeros((size, size), bool)
    m[lo:hi, lo:hi] = True
    return m


def test_energy_in_mask():
    m = _square_mask()
    assert energy_in_mask(m.astype(float), m) == pytest.approx(1.0)
    assert energy_in_mask((~m).astype(float), m) == pytest.approx(0.0)
    assert energy_in_mask(np.ones_like(m, float), m) == pytest.approx(mask_area_fraction(m))
    assert np.isnan(energy_in_mask(-np.ones_like(m, float), m))
    with pytest.raises(ValueError):
        energy_in_mask(np.ones((4, 4)), m)


def test_pointing_game_inside_mask_no_tolerance():
    m = _square_mask()
    sal = np.zeros(m.shape)
    sal[10, 10] = 1
    assert pointing_game(sal, m)
    sal = np.zeros(m.shape)
    sal[7, 7] = 1  # one pixel outside the mask counts as a miss
    assert not pointing_game(sal, m)


def test_otsu_separates_bimodal():
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.normal(0.2, 0.02, 500), rng.normal(0.8, 0.02, 500)])
    assert 0.3 < otsu_threshold(values) < 0.7


def test_iou_and_miou():
    m = _square_mask()
    assert iou(m, m) == 1.0
    assert iou(m, ~m) == 0.0
    assert np.isnan(iou(np.zeros_like(m), np.zeros_like(m)))
    sal = m.astype(float) + 0.01 * np.random.default_rng(0).random(m.shape)
    assert binarize(sal).sum() == m.sum()
    assert miou([sal, sal], [m, m]) == pytest.approx(1.0)


def test_degradation_curve():
    curve = DegradationCurve("B0", {0.0: 0.9, 0.5: 0.8, 1.0: 0.6})
    assert curve.drop() == pytest.approx(0.3)
    assert curve.normalized_area() == pytest.approx(0.85 * 0.5 + 0.7 * 0.5)
    assert curve.as_row()["bias_0.5"] == 0.8
    flat = DegradationCurve("M1", {0.0: 0.9, 1.0: 0.9})
    assert flat.normalized_area() > curve.normalized_area()


def test_offloading_index_sign():
    assert offloading_index(np.array([0.6, 0.7]), np.array([0.4, 0.5])) == pytest.approx(-0.2)


def test_stratified_auc_both_strata():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    score = np.array([0.1, 0.9, 0.2, 0.8, 0.6, 0.4, 0.3, 0.7])
    stratum = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    out = stratified_auc(y, score, stratum)
    assert out["present"] == 1.0
    assert out["absent"] == pytest.approx(0.75)
    assert out["gap"] == pytest.approx(0.25)


def test_stats_are_seeded():
    v = np.random.default_rng(0).normal(size=50)
    assert mean_ci(v, seed=3) == mean_ci(v, seed=3)
    m, lo, hi = t_ci(v)
    assert lo < m < hi


def test_cluster_bootstrap_detects_difference():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(100), 3)
    a = rng.normal(0.5, 0.1, 300)
    b = a + 0.1

    est, lo, hi, p = cluster_bootstrap_diff(groups, lambda idx: (b[idx] - a[idx]).mean(), n=500)
    assert est == pytest.approx(0.1)
    assert lo > 0 and p < 0.05

    noise = rng.normal(0, 0.1, 300)
    est, lo, hi, p = cluster_bootstrap_diff(groups, lambda idx: noise[idx].mean(), n=500)
    assert lo < 0 < hi and p > 0.05


def test_holm():
    assert holm([0.001, 0.02, 0.04, 0.5]).tolist() == [True, False, False, False]
    assert holm([0.01, 0.012, 0.3]).tolist() == [True, True, False]


def test_tost():
    assert tost_equivalent(-0.01, 0.015, margin=0.02)
    assert not tost_equivalent(-0.01, 0.03, margin=0.02)


def test_map_shift():
    a = np.random.default_rng(0).random((8, 8))
    assert map_shift(a, a) == pytest.approx(0.0)
    assert map_shift(a, -a) == pytest.approx(2.0)


def test_perturb_field_changes_only_that_field():
    s = MetadataSchema(["sex", "site"], ["age"])
    s.vocabs = {"sex": ["f", "m"], "site": ["a", "b", "c"]}
    cat = torch.tensor([[0, 0], [1, 3], [2, 1]])  # row 2 sex is MISSING (index 2)
    cont, mask = torch.zeros(3, 1), torch.tensor([[True], [False], [True]])
    g = torch.Generator().manual_seed(0)
    new_cat, new_cont, new_mask = perturb_field(s, "sex", cat, cont, mask, generator=g)
    assert (new_cat[:, 1] == cat[:, 1]).all()
    assert new_cat[0, 0] == 1 and new_cat[1, 0] == 0 and new_cat[2, 0] in (0, 1)
    _, new_cont, new_mask = perturb_field(s, "age", cat, cont, mask, generator=g)
    assert new_cont.abs().eq(1).all() and new_mask.all()
    with pytest.raises(KeyError):
        perturb_field(s, "nope", cat, cont, mask)
