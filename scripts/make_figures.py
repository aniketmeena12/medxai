"""Generate the figures used in the slide deck and technical document, from the result tables.

    python scripts/make_figures.py     ->  results/figures/*.png
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from medfusion.eval.stats import mean_ci

REPO = Path(__file__).resolve().parents[1]
TABLES = REPO / "results" / "tables"
FIGS = REPO / "results" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# A calm, colour-blind-safe palette; the image-only reference is grey, fusion models coloured.
COL = {"B0": "#6b7280", "B1": "#9ca3af", "B3": "#2563eb", "B4": "#0891b2",
       "B5": "#dc2626", "M1": "#7c3aed"}
NAME = {"B0": "Image only", "B1": "Metadata only", "B3": "Concat", "B4": "FiLM",
        "B5": "MetaBlock", "M1": "Cross-attn"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})


def _protocol(dataset, metric="balanced_accuracy"):
    out = {}
    for m in ("B0", "B1", "B3", "B4", "B5", "M1"):
        fs = sorted(glob.glob(f"data/experiments/{dataset}_{m}_*/metrics.json"))
        d = next((json.load(open(f)) for f in reversed(fs)
                  if json.load(open(f))["complete_protocol"]), None)
        if d:
            v = np.array([x[metric] for x in d["per_fold_seed"]])
            out[m] = mean_ci(v)
    return out


def fig_accuracy():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, ds, title in ((axes[0], "padufes20", "PAD-UFES-20 (rich metadata)"),
                          (axes[1], "ham10000", "HAM10000 (thin metadata)")):
        d = _protocol(ds)
        ms = list(d)
        means = [d[m][0] for m in ms]
        err = [[d[m][0] - d[m][1] for m in ms], [d[m][2] - d[m][0] for m in ms]]
        ax.bar(range(len(ms)), means, yerr=err, capsize=4,
               color=[COL[m] for m in ms], alpha=0.9)
        ax.set_xticks(range(len(ms)))
        ax.set_xticklabels([NAME[m] for m in ms], rotation=30, ha="right")
        ax.axhline(d["B0"][0], ls="--", color="#6b7280", lw=1, alpha=0.7)
        ax.set_title(title)
        ax.set_ylabel("Balanced accuracy")
        ax.set_ylim(0.25, 0.85)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("Accuracy: fusion helps a lot on rich metadata, barely on thin metadata",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_grounding():
    d = pd.read_csv(TABLES / "endpoints_ham10000_gradcam_descriptive.csv")
    piv = d[d["endpoint"] == "P1"].pivot_table(index="model", columns="metric", values="value")
    ms = ["B0", "B3", "B4", "B5", "M1"]
    energy = [piv.loc[m, "energy_in_mask"] for m in ms]
    chance = piv.loc["B0", "mask_area_fraction"]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    bars = ax.bar(range(len(ms)), energy, color=[COL[m] for m in ms], alpha=0.9)
    ax.axhline(chance, ls="--", color="#dc2626", lw=1.5)
    ax.text(len(ms) - 0.4, chance + 0.006, f"random-guess level ({chance:.2f})",
            color="#dc2626", ha="right", fontsize=10)
    ax.axhline(energy[0], ls=":", color="#6b7280", lw=1.2)
    ax.text(0.0, energy[0] + 0.006, "image-only", color="#6b7280", fontsize=10)
    ax.set_xticks(range(len(ms)))
    ax.set_xticklabels([NAME[m] for m in ms])
    ax.set_ylabel("Share of attention inside the lesion")
    ax.set_ylim(0.25, 0.55)
    ax.set_title("Where does the model look? (higher = better grounded)", fontweight="bold")
    for b, v in zip(bars, energy, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.2f}", ha="center", fontsize=10)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_grounding.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_reliance():
    d = pd.read_csv(TABLES / "endpoints_padufes20_gradcam_descriptive.csv")
    piv = d[d["endpoint"] == "P5"].pivot_table(index="model", columns="metric", values="value")
    ms = ["B0", "B3", "B4", "B5", "M1"]
    meta = [piv.loc[m, "drop_meta_shuffled"] for m in ms]
    img = [piv.loc[m, "drop_mean_image"] for m in ms]
    x = np.arange(len(ms))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.bar(x - w / 2, meta, w, label="Lost without patient details", color="#2563eb", alpha=0.9)
    ax.bar(x + w / 2, img, w, label="Lost without the image", color="#dc2626", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([NAME[m] for m in ms])
    ax.set_ylabel("Balanced accuracy lost")
    ax.set_title("What does each model rely on? (PAD-UFES-20)", fontweight="bold")
    ax.legend(frameon=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_reliance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_degradation():
    c = pd.read_csv(TABLES / "trap_P2_curves.csv")
    levels = [0, 0.3, 0.5, 0.7, 0.9, 1.0]
    cols = [f"bias_{lv:g}" for lv in levels]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for m in ("B0", "B3", "M1"):
        sub = c[c["model"] == m]
        if sub.empty:
            continue
        mean = [sub[col].mean() for col in cols]
        ax.plot(levels, mean, "o-", color=COL[m], label=NAME[m], lw=2)
    ax.axhline(0.5, ls="--", color="#9ca3af", lw=1)
    ax.text(0.02, 0.51, "chance", color="#9ca3af", fontsize=9)
    ax.set_xlabel("Strength of the artifact shortcut (0 = none, 1 = reversed at test)")
    ax.set_ylabel("Test AUROC")
    ax.set_title("Every model collapses the same way under image shortcuts",
                 fontweight="bold")
    ax.legend(frameon=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_degradation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_story():
    """The one-slide summary: grounding (x) vs metadata reliance (y), per model."""
    g = pd.read_csv(TABLES / "endpoints_ham10000_gradcam_descriptive.csv")
    gp = g[g["endpoint"] == "P1"].pivot_table(index="model", columns="metric", values="value")
    r = pd.read_csv(TABLES / "endpoints_padufes20_gradcam_descriptive.csv")
    rp = r[r["endpoint"] == "P5"].pivot_table(index="model", columns="metric", values="value")
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for m in ("B0", "B3", "B4", "B5", "M1"):
        if m not in gp.index or m not in rp.index:
            continue
        x, y = gp.loc[m, "energy_in_mask"], rp.loc[m, "drop_meta_shuffled"]
        ax.scatter(x, y, s=260, color=COL[m], alpha=0.9, edgecolor="white", zorder=3)
        ax.annotate(NAME[m], (x, y), fontsize=11, ha="center", va="center",
                    color="white", fontweight="bold", zorder=4)
    chance = gp.loc["B0", "mask_area_fraction"]
    ax.axvline(gp.loc["B0", "energy_in_mask"], ls=":", color="#6b7280", lw=1)
    ax.axvline(chance, ls="--", color="#dc2626", lw=1)
    ax.text(chance, ax.get_ylim()[1], " chance", color="#dc2626", fontsize=9, va="top")
    ax.text(gp.loc["B0", "energy_in_mask"], ax.get_ylim()[0], "image-only ",
            color="#6b7280", fontsize=9, ha="right", va="bottom")
    ax.set_xlabel("Grounding: attention inside the lesion  ->  better")
    ax.set_ylabel("Reliance on metadata  ->  more")
    ax.set_title("The whole story on one chart: better-grounded models rely on metadata less",
                 fontweight="bold", fontsize=12)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_story.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_accuracy()
    fig_grounding()
    fig_reliance()
    fig_degradation()
    fig_story()
    print("figures written to", FIGS)
    for p in sorted(FIGS.glob("*.png")):
        print(" ", p.name, f"{p.stat().st_size // 1024} KB")
