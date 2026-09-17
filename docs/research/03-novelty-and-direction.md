# 03: Novelty analysis and research direction

Date: 2026-09-17. Builds on `02-venues-and-prior-art.md`; dataset facts are pending `01-dataset-verification.md`.
Items marked **[VERIFY]** are assumptions to confirm before they go into the paper.

---

## 1. Where the original plan stands after the prior-art check

| Original element | Status now | Why |
|---|---|---|
| C1: cross-attention fusion block | **Not novel** | MetaNet 2020, MetaBlock 2021, MAT 2021, Cheslerean-Boghiu 2023, and Mridha & Islam 2026 (same design, same dataset) |
| C2: attention scored against human annotation | **Novel, but has a data gap** | No fusion paper scores its maps, but PAD-UFES-20 has no masks or concepts, and SkinCon images have almost no metadata |
| C3: shortcut resistance of fusion | **Novel, and the strongest element** | Trap sets, NEATX and "Mask of truth" only test image-only models |
| C4: "accuracy gain is small" | **Expectation is wrong** | PAD-UFES-20 fusion gains are usually 7–10 BACC points; cross-attention in particular has shown little or no gain (DualRefNet) |

**Conclusion:** a paper built on "we propose cross-attention fusion and show better heatmaps" will be rejected as incremental. The novelty has to move from the *architecture* to a *question nobody has answered*:

> **When a model is given patient metadata, does its image evidence get better grounded, or does the model stop looking?**

---

## 2. The core idea: grounding vs. offloading

All existing fusion papers assume metadata helps the image branch. There is a second, untested possibility:

- **Grounding hypothesis:** metadata tells the model *what to look for* (e.g. "elderly, sun-exposed site, bleeding" → look for BCC features), so image attention becomes *more* aligned with the lesion.
- **Offloading hypothesis:** metadata is an easier signal (itch, bleed, grew, age are highly predictive), so the network reduces its reliance on the image. Image attention becomes *less* aligned, and the model now carries a **metadata shortcut** on top of any image shortcut.

Accuracy cannot tell these apart, since both raise it. Only the evaluation this project is built for can. **Either answer is publishable**, which removes the biggest risk for an evaluation paper (the "negative result" problem).

Working title options:
- *Same Accuracy, Different Evidence: Does Patient Metadata Ground or Distract Skin Lesion Classifiers?*
- *Look or Lean? Auditing Cross-Modal Shortcuts in Image–Metadata Fusion for Dermatology*

---

## 3. Proposed contributions (revised)

| # | Contribution | Type | Why it is new |
|---|---|---|---|
| **N1** | **A cross-modal shortcut benchmark:** controlled *metadata* trap sets (a metadata field correlated with the label in training, decorrelated/reversed at test), crossed with Bissoto's *image-artifact* trap sets, which gives a **2-D bias grid** (image bias × metadata bias) | Benchmark | Trap sets exist only for image artifacts. No benchmark tests how the two shortcut sources interact. |
| **N2** | **Counterfactual metadata attention test:** change one metadata field while holding the image fixed (e.g. flip sex, shift age, toggle "bleed") and measure how much the *image* attention map and prediction move. Irrelevant fields should not move the map; clinically relevant ones may. | Evaluation method | Needs **no human annotation**, so it runs on PAD-UFES-20 and closes the data gap from §1. No prior work does it. |
| **N3** | **Grounding audit across fusion families:** image-only, concat, FiLM, MetaBlock, cross-attention, all scored on (a) mask alignment, (b) artifact attention mass, (c) modality reliance, (d) counterfactual consistency (N2) | Empirical finding | First *quantitative* comparison of *where* different fusion mechanisms look. The expected headline: models with the same accuracy differ in grounding. |
| **N4** (optional, only if time allows) | **Counterfactual metadata consistency loss:** penalize change in image attention under irrelevant metadata perturbations | Small method | Turns the diagnosis into a fix, which reviewers like, but the paper must not depend on it working |

**Headline sentence for the abstract (to be earned by the results):**
"Fusion models with statistically indistinguishable accuracy differ by X in lesion-mask alignment and by Y in shortcut reliance, and N% of the image-attention shift is caused by clinically irrelevant metadata fields."

---

## 4. Research questions and the evidence for each

| RQ | Question | Experiment | Metric | Data |
|---|---|---|---|---|
| RQ1 | Does fusion change *where* the image branch looks? | Saliency / attention vs lesion masks | Pointing game, attention mass in mask, IoU at matched thresholds | HAM10000 + lesion masks (**verified 2026-09-17**: Harvard Dataverse doi:10.7910/DVN/DBW86T, CC BY-NC 4.0, dermatologist-corrected masks; 9,083 HAM images also appear in the ISIC 2019 trap sets) |
| RQ2 | Does fusion reduce *image* shortcut reliance? | Bissoto trap sets at 6 bias levels | AUC vs bias level (degradation curve); attention mass in artifact regions | ISIC 2018 + Bissoto annotations **[VERIFY: which metadata these images have]** |
| RQ3 | Does fusion create *metadata* shortcut reliance, and does that erode image grounding? | Metadata trap sets (N1) and the 2-D grid | Degradation surface; change in mask alignment as metadata bias grows | HAM10000 / ISIC (site or age as the controlled field) |
| RQ4 | Is attention counterfactually consistent? | N2 perturbations | Map shift (1 − SSIM or rank correlation), prediction flip rate, split into irrelevant vs relevant fields | **PAD-UFES-20** (21 fields) + HAM10000 |
| RQ5 | How much does each model rely on each modality? | Modality ablation / modality Shapley | Performance drop when image or metadata is removed or shuffled | All derm sets |

**Accuracy (T1) stays in the paper** as a sanity check against published baselines, with 95% CIs, including **MetaBlock** so no reviewer can say the fusion baselines are weak.

---

## 5. Scope for the conference paper

**Decision D-1 recommendation: dermatology only.** Reasons:
1. The novelty (N1–N3) works fully in derm, and PAD-UFES-20 is where the metadata is rich.
2. 6 GB laptop GPU; the CXR track adds ~58 GB and weeks of compute.
3. The research suggests MIDL, whose page limit fits one well-analysed track better than two thin ones.

**Dataset set for the paper**

| Dataset | Role |
|---|---|
| PAD-UFES-20 | Rich metadata; accuracy vs literature; counterfactual test (RQ4); modality reliance (RQ5) |
| HAM10000 + masks | Localization alignment with metadata (RQ1); metadata trap sets (RQ3) |
| ISIC 2018 + Bissoto annotations/trap sets | Image-artifact shortcuts (RQ2); image axis of the 2-D grid |
| DDI | External test stratified by skin tone (fairness table, if space allows) |
| SkinCon | **Move to future work.** Its images have almost no metadata, so it cannot test a fusion claim cleanly. |

**Moved to future work:** CheXlocalize, NIH-CXR14, NEATX. The CXR view-position confound (AP films ≈ sicker patients) is a good *real* cross-modal shortcut and is the natural second study for the journal version.

**Compute budget (6 GB GPU):** ConvNeXt-T at 224², mixed precision, frozen early stages, gradient accumulation to effective batch 32. Budget the grid before running:
5 models × 6 image-bias levels × 3 seeds = 90 runs for RQ2, and the 2-D grid should use a reduced 3 × 3 level set and 3 models. Cut seeds before levels. Cache frozen features where the protocol allows.

---

## 6. How reviewers will attack it, and the answers

| Likely objection | Answer built into the design |
|---|---|
| "Architecture isn't new." | Correct, and we don't claim it. The contribution is the benchmark (N1), the test (N2) and the findings (N3). |
| "Grad-CAM is unreliable, so your alignment numbers are too." | Use ≥3 attribution methods including one gradient-free method; run a sanity check (randomize weights, Adebayo et al. 2018) before any alignment number is trusted; report agreement between methods. |
| "Synthetic metadata trap sets are artificial." | Use *real* fields (anatomical site, age) resampled to control correlation, and back them with the natural shortcuts from Bissoto's real artifacts. |
| "Differences are noise." | 5 folds × 3 seeds, 95% CIs, paired tests, multiple-comparison correction; patient-level splits. |
| "Why should clinicians care?" | Offloading means a model can pass accuracy checks while ignoring the lesion. That is a deployment safety issue, especially when metadata is missing or entered incorrectly. |
| "Mridha & Islam already did cross-attention on PAD-UFES-20." | Cite it directly; their maps are qualitative. We measure what they only illustrate. |

---

## 7. Novelty risks to close in the next 2 weeks

1. Read **Ketabi et al. 2023** and **Atiq & Fattah 2025** in full: do they score maps quantitatively?
2. Search specifically for "metadata shortcut" / "tabular shortcut" / "modality laziness" / "modality competition" / "unimodal bias" in medical fusion (general ML has work on modality imbalance, e.g. OGM-GE, CVPR 2022 **[VERIFY]**; cite it and show the medical, attention-grounded angle is new).
3. Put an **arXiv preprint** up at submission time if the venue allows it, to fix the priority date (the Cheplygina group is active in this area).

---

## 8. Future directions (journal version and follow-ups)

1. **CXR with a real cross-modal confound:** NIH-CXR14 / CheXpert view position + NEATX drains, CheXlocalize masks for alignment. Then MIMIC-CXR + MIMIC-IV for rich EHR metadata (needs PhysioNet credentialing, so file it now).
2. **Grounding-aware training:** extend N4 into a full method (counterfactual consistency, mask-guided attention supervision on a small annotated subset) and show it removes offloading without losing accuracy.
3. **Foundation models:** repeat the audit on dermatology/medical foundation models with metadata or text prompts. Does large pretraining reduce offloading?
4. **Free-text as metadata:** replace structured fields with clinical notes or LLM-generated descriptions; test whether language conditioning creates stronger shortcuts.
5. **Missing and wrong metadata:** robustness when fields are missing or deliberately wrong, which is common in real clinics (links to MetaBlock-SE).
6. **Fairness:** is offloading worse for darker skin tones or for under-represented age/sex groups? (DDI, Fitzpatrick17k.)
7. **Clinician study:** do dermatologists trust or catch offloading models differently when shown their maps?
8. **Causal analysis:** mediation analysis of how much of the prediction flows through image vs metadata pathways.

---

## 9. Immediate next steps

- [ ] Decide: accept the grounding-vs-offloading framing and dermatology-only scope (D-1), and MIDL 2027 as the target (D-2).
- [ ] Wait for `01-dataset-verification.md`; add HAM10000 masks and ISIC 2018 metadata to the verification list.
- [ ] Close the novelty risks in §7.
- [ ] Update `PROJECT_HANDBOOK.md` §1, §5, §8 and §13 once decided.
- [ ] Write a one-page pre-registration (RQs, metrics, thresholds, stats) *before* training, so the analysis can't be accused of being tuned after the fact.
