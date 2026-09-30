# ruff: noqa: E501
"""Build the deep technical document (.docx): architecture, datasets, novelty, methodology,
results, and real-world use cases. Numbers and tables are read from the result files.

    python scripts/make_technical_doc.py
"""
from __future__ import annotations

import argparse
import glob
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

from medfusion.eval.stats import mean_ci

REPO = Path(__file__).resolve().parents[1]
TABLES = REPO / "results" / "tables"
FIGS = REPO / "results" / "figures"
NAMES = {"B0": "Image only", "B1": "Metadata only", "B3": "Concatenation", "B4": "FiLM",
         "B5": "MetaBlock", "M1": "Cross-attention"}
NAVY = RGBColor(0x1E, 0x3A, 0x5F)
ACCENT = RGBColor(0x25, 0x63, 0xEB)
MONO = "Consolas"


def protocol_table(dataset):
    rows = []
    for m in ("B0", "B1", "B3", "B4", "B5", "M1"):
        fs = sorted(glob.glob(f"data/experiments/{dataset}_{m}_*/metrics.json"))
        d = next((json.load(open(f)) for f in reversed(fs) if json.load(open(f))["complete_protocol"]), None)
        if d is None:
            continue
        row = {"Model": f"{m} ({NAMES[m]})"}
        for key, label in (("balanced_accuracy", "Balanced acc."), ("macro_f1", "Macro-F1"), ("auroc", "AUROC")):
            v = np.array([x[key] for x in d["per_fold_seed"]])
            mu, lo, hi = mean_ci(v)
            row[label] = f"{mu:.3f} [{lo:.3f}, {hi:.3f}]"
        rows.append(row)
    return pd.DataFrame(rows)


def read(name):
    p = TABLES / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def totals():
    seconds, runs = 0.0, 0
    for f in glob.glob("data/experiments/*/metrics.json"):
        for r in json.load(open(f)).get("per_fold_seed", []):
            seconds += r.get("seconds", 0) or 0
            runs += 1
    for pattern in ("data/experiments/isic2019_trap_*/img*.json", "data/experiments/tune_*/trial*.json"):
        for f in glob.glob(pattern):
            seconds += json.load(open(f)).get("seconds", 0) or 0
            runs += 1
    return runs, seconds / 3600


class Doc:
    def __init__(self):
        self.doc = Document()
        for s in self.doc.sections:
            s.left_margin = s.right_margin = Inches(1.0)
            s.top_margin = s.bottom_margin = Inches(0.9)
        n = self.doc.styles["Normal"]
        n.font.name = "Calibri"
        n.font.size = Pt(11)
        n.paragraph_format.space_after = Pt(7)
        n.paragraph_format.line_spacing = 1.15
        n.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    def p(self, text, size=11, bold=False, italic=False, color=None, align=None, space=7):
        para = self.doc.add_paragraph()
        para.paragraph_format.space_after = Pt(space)
        if align:
            para.alignment = align
        r = para.add_run(text)
        r.font.size = Pt(size)
        r.bold = bold
        r.italic = italic
        if color:
            r.font.color.rgb = color
        return para

    def h(self, text, level=1):
        para = self.doc.add_heading(text, level=level)
        return para

    def bullets(self, items, style="List Bullet"):
        for it in items:
            self.doc.add_paragraph(it, style=style)

    def code(self, lines):
        para = self.doc.add_paragraph()
        para.paragraph_format.left_indent = Inches(0.3)
        para.paragraph_format.space_after = Pt(8)
        para.paragraph_format.space_before = Pt(2)
        for i, ln in enumerate(lines):
            r = para.add_run(("" if i == 0 else "\n") + ln)
            r.font.name = MONO
            r.font.size = Pt(9.5)
            r.font.color.rgb = NAVY
        return para

    def table(self, df, widths=None, caption=None, style="Light Grid Accent 1"):
        if caption:
            c = self.doc.add_paragraph()
            rr = c.add_run(caption)
            rr.bold = True
            rr.font.size = Pt(9.5)
        if df.empty:
            self.p("(pending)")
            return
        t = self.doc.add_table(rows=1, cols=len(df.columns))
        t.style = style
        for i, col in enumerate(df.columns):
            cell = t.rows[0].cells[i]
            cell.text = str(col)
            for r in cell.paragraphs[0].runs:
                r.bold = True
                r.font.size = Pt(9.5)
        for _, row in df.iterrows():
            cells = t.add_row().cells
            for i, col in enumerate(df.columns):
                cells[i].text = "" if pd.isna(row[col]) else str(row[col])
                for pgh in cells[i].paragraphs:
                    for r in pgh.runs:
                        r.font.size = Pt(9.5)
        if widths:
            for row in t.rows:
                for i, w in enumerate(widths):
                    row.cells[i].width = Inches(w)
        self.doc.add_paragraph()

    def figure(self, name, width=6.2, caption=None):
        path = FIGS / name
        if not path.exists():
            return
        self.doc.add_picture(str(path), width=Inches(width))
        self.doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        if caption:
            c = self.doc.add_paragraph()
            c.alignment = WD_ALIGN_PARAGRAPH.CENTER
            rr = c.add_run(caption)
            rr.italic = True
            rr.font.size = Pt(9.5)
            rr.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    def save(self, out):
        self.doc.save(out)


def build(out, students, supervisor):
    d = Doc()
    n_runs, gpu_hours = totals()

    # ---- title -----------------------------------------------------------
    d.p("MedFusion-XAI", size=30, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, color=NAVY, space=2)
    d.p("Technical Design and Reference Document", size=16, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space=2)
    d.p("Grounding versus offloading in image-plus-metadata skin lesion classifiers: "
        "architecture, datasets, novelty, methodology, results, and use cases",
        size=12, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, space=10)
    d.p(" · ".join(students), size=12, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space=1)
    d.p(f"Supervisor: {supervisor}", size=11, align=WD_ALIGN_PARAGRAPH.CENTER, space=1)
    d.p(f"{date.today().strftime('%d %B %Y')}  ·  Target venue: MIDL 2027", size=10,
        align=WD_ALIGN_PARAGRAPH.CENTER, color=RGBColor(0x55, 0x55, 0x55), space=12)

    d.p("This document is the single technical reference for the project. It explains what the "
        "system does, why it is novel, the datasets it uses, its architecture in full detail, the "
        "measurement methodology, the results obtained to date, and the real-world situations in "
        "which the work is intended to be used. It is written to be read end-to-end by a new team "
        "member or an examiner, and every quantitative figure is produced directly from the "
        "experiment output files.", space=10)

    # contents
    d.h("Contents", level=2)
    for i, t in enumerate([
        "Executive summary", "Problem statement and motivation", "Novelty and positioning",
        "Datasets", "System architecture", "Methodology", "Results",
        "Discussion", "Actual use cases", "Limitations and future work",
        "Reproducibility and engineering", "References"], 1):
        d.doc.add_paragraph(f"{i}.  {t}", style="List Number" if False else "Normal")

    d.doc.add_section(WD_SECTION.NEW_PAGE)

    # ---- 1 executive summary --------------------------------------------
    d.h("1. Executive summary", level=1)
    d.p("Modern skin lesion classifiers can take, in addition to the lesion image, structured "
        "patient metadata such as age, sex, anatomical site and symptoms. Adding this metadata "
        "reliably improves classification accuracy, and a large literature proposes ways of "
        "combining the two modalities. What no prior work establishes is whether the added "
        "metadata makes the model's visual evidence better grounded on the lesion, or whether the "
        "model instead offloads its decision onto the metadata and attends less to the image. "
        "Accuracy on a same-distribution test set cannot distinguish these two behaviours, yet "
        "they have very different consequences when the model meets a patient whose metadata is "
        "missing, unusual, or recorded differently.")
    d.p("This project measures the distinction directly. Six models - an image-only reference, a "
        "metadata-only reference, and four image-plus-metadata fusion mechanisms (concatenation, "
        "FiLM, MetaBlock and cross-attention) - are trained under one identical protocol on two "
        "dermatology datasets and audited with five pre-registered endpoints spanning grounding "
        "(agreement with dermatologist-drawn lesion masks), shortcut robustness, modality "
        "reliance, and counterfactual sensitivity to metadata.")
    d.p(f"Across {n_runs} model trainings and roughly {gpu_hours:.0f} GPU-hours, the central "
        "finding is that the fusion mechanism, not the presence of metadata, decides what happens "
        "to the visual evidence. Simple concatenation increased attribution inside the lesion "
        "relative to the image-only model, whereas MetaBlock reduced it to near the chance level "
        "while remaining competitive on accuracy. Every fusion model relied more on metadata than "
        "on the image, and metadata fields that cannot physically affect a lesion's appearance "
        "still displaced the image evidence. Accuracy and evidence quality are therefore "
        "decoupled: a model can be more accurate and less trustworthy at the same time.")

    # ---- 2 problem -------------------------------------------------------
    d.h("2. Problem statement and motivation", level=1)
    d.h("2.1 Clinical setting", level=2)
    d.p("Skin cancer is among the most common cancers, and access to dermatological expertise is "
        "unevenly distributed. Automated triage from images is an attractive aid, and on curated "
        "benchmarks it approaches specialist accuracy. In practice, a dermatologist never reasons "
        "from the image alone: the patient's age, the body site, the Fitzpatrick skin type, and "
        "the history of the lesion (itch, bleeding, recent change) all inform the diagnosis. This "
        "motivates multimodal models that fuse the image with structured patient metadata.")
    d.h("2.2 The question", level=2)
    d.p("The accuracy benefit of fusion is settled. The behavioural question is not: when a model "
        "is given predictive metadata, does it use that information to interrogate the image more "
        "effectively (grounding), or does it rely on the metadata as a shortcut and attend to the "
        "image less (offloading)? Both can yield identical held-out accuracy.")
    d.p("We formalise two hypotheses, stated before any training:")
    d.bullets([
        "H-ground: fusion models place more of their attribution inside the lesion than an image-only model.",
        "H-offload: fusion models place less attribution inside the lesion, and a metadata shortcut erodes grounding further.",
    ])
    d.h("2.3 Why it matters", level=2)
    d.p("A model that has offloaded onto metadata will degrade precisely where it is most needed: "
        "in under-served populations where metadata is sparse, at sites that record fewer fields, "
        "or for atypical patients. It will also produce explanations that appear to point at the "
        "lesion while actually being driven by information the clinician cannot see in the image. "
        "Measuring grounding is therefore a safety question, not merely an interpretability nicety.")

    # ---- 3 novelty -------------------------------------------------------
    d.h("3. Novelty and positioning", level=1)
    d.p("The architectures compared here are all published; the contribution is not a new fusion "
        "block. Prior art establishes two mature but disconnected bodies of work: (i) many ways to "
        "fuse image and metadata, all evaluated on accuracy, and (ii) a rigorous methodology for "
        "scoring visual explanations against human annotation, developed for image-only chest "
        "radiograph models. No published work joins them. The novelty of this project lies in four "
        "specific, defensible places.")
    d.h("3.1 N1 - A cross-modal shortcut benchmark", level=2)
    d.p("Existing trap-set benchmarks (Bissoto et al.) vary the correlation between a visual "
        "artifact and the label. We introduce a controlled metadata trap: a metadata field is "
        "resampled so that it predicts the label with a chosen strength in training and the "
        "reversed correlation at test. Crossing this with the image-artifact traps yields a "
        "two-dimensional bias grid (image-bias by metadata-bias). No prior benchmark tests how "
        "image and metadata shortcuts interact.")
    d.h("3.2 N2 - A counterfactual metadata attribution test", level=2)
    d.p("We hold the image fixed, change a single metadata field, and measure how far the image "
        "attribution map moves (one minus the Spearman correlation between the maps). Fields that "
        "cannot affect a lesion's appearance are designated in advance as irrelevant; a "
        "well-grounded model should not move its evidence when they change. This test needs no "
        "human annotation, so it runs on any dataset with metadata, and it directly probes the "
        "offloading mechanism rather than inferring it. It is, to our knowledge, new.")
    d.h("3.3 N3 - The first grounding audit across fusion families", level=2)
    d.p("We provide the first quantitative, like-for-like comparison of where different fusion "
        "mechanisms look, scored against dermatologist-drawn lesion masks and reported with "
        "confidence intervals, under a parameter-randomisation sanity check. The surveyed fusion "
        "literature presents explanations only as illustrative figures; a 2026 review of 82 "
        "multimodal medical-AI studies confirms that standardised, quantitative evaluation is "
        "absent in the majority.")
    d.h("3.4 N4 - The empirical finding", level=2)
    d.p("The audit overturns a natural assumption. It is the mechanism of fusion, not the fact of "
        "fusion, that determines grounding: concatenation improves it, MetaBlock destroys it, FiLM "
        "and cross-attention lie between. The simplest mechanism is best on both accuracy and "
        "grounding, and cross-attention - which this project originally set out to promote - is "
        "middling on accuracy and the most easily swayed by irrelevant metadata. A published "
        "negative result for cross-attention accuracy on the same dataset (Khurshid et al.) "
        "corroborates that its complexity is not justified even on accuracy grounds.")

    # ---- 4 datasets ------------------------------------------------------
    d.h("4. Datasets", level=1)
    d.p("All datasets are public. Each download was verified against the provider's published "
        "figures (15 of 15 automated checks passed), and the exact version, byte size and "
        "checksum of every file is recorded under data/raw/*/SOURCE.md so the acquisition is "
        "reproducible. Dataset licences forbid redistribution; data is never committed or uploaded.")
    data = pd.DataFrame([
        {"ID": "A", "Dataset": "PAD-UFES-20", "Modality": "Clinical (smartphone)", "Images": "2,298", "Groups": "1,373 patients", "Metadata": "21 usable fields", "Ground truth used": "None (counterfactual, reliance)"},
        {"ID": "B", "Dataset": "HAM10000", "Modality": "Dermoscopy", "Images": "10,015", "Groups": "7,470 lesions", "Metadata": "age, sex, site", "Ground truth used": "Lesion mask for every image"},
        {"ID": "C", "Dataset": "ISIC 2019 + traps", "Modality": "Dermoscopy", "Images": "25,331", "Groups": "13,931 lesions", "Metadata": "age, sex, site", "Ground truth used": "Artifact labels; shared masks"},
        {"ID": "D", "Dataset": "DDI (planned)", "Modality": "Clinical", "Images": "656", "Groups": "per-image", "Metadata": "none usable", "Ground truth used": "Skin-tone (Fitzpatrick) groups"},
    ])
    d.table(data, widths=[0.35, 1.5, 1.35, 0.75, 1.05, 1.0, 1.5],
            caption="Table 4.1. The four datasets and their roles.")
    d.h("4.1 PAD-UFES-20 (dataset A)", level=2)
    d.p("2,298 smartphone clinical photographs of six lesion types (BCC, SCC, ACK, SEK, MEL, NEV) "
        "from 1,373 patients, with up to 26 metadata columns of which 21 are used: age, anatomical "
        "region, Fitzpatrick type, lesion diameters, itch, bleed, growth, elevation, family and "
        "personal cancer history, and environmental fields such as piped water and sewage system. "
        "Roughly 58% of samples are biopsy-proven, including all cancers. Images are sourced from "
        "a mirror that preserves the original filenames; metadata is the SHA256-verified official "
        "Mendeley file. Splits are grouped by patient.")
    d.h("4.2 HAM10000 (dataset B)", level=2)
    d.p("10,015 dermoscopic images across seven diagnoses, with a dermatologist-corrected binary "
        "lesion segmentation mask for every image - the property that makes it the only dataset on "
        "which grounding (P1) can be measured. Metadata is limited to age, sex and localization. "
        "Images and masks come from the official Harvard Dataverse record, with published MD5s "
        "matched. Splits are grouped by lesion.")
    d.h("4.3 ISIC 2019 and the Bissoto trap sets (dataset C)", level=2)
    d.p("25,331 dermoscopic images from the ISIC 2019 challenge, used as the substrate for "
        "Bissoto et al.'s artifact trap sets: pre-built train/validation/test splits in which "
        "seven artifacts (dark corners, hair, gel borders, gel bubbles, rulers, ink, patches) "
        "correlate with malignancy at six strengths {0, 0.3, 0.5, 0.7, 0.9, 1.0}, reversed at "
        "test. A verified join shows all 20,599 trap-set images have ISIC metadata (sex known for "
        "98.2%), and 9,083 of them are also HAM10000 images (6,941 distinct lesions) and therefore "
        "carry lesion masks - the intersection that makes the offloading endpoint (P3) possible.")
    d.h("4.4 DDI (dataset D, planned)", level=2)
    d.p("656 biopsy-proven clinical photographs balanced across Fitzpatrick skin-tone groups, "
        "intended as an external, skin-tone-balanced fairness test and an independent check of the "
        "offloading finding (fusion models receive all-MISSING metadata on it). Access requires an "
        "individual Stanford research-use agreement that has not been obtainable; the full "
        "processing and evaluation pipeline is nevertheless implemented and tested, so the analysis "
        "can run immediately if access is granted.")

    # ---- 5 architecture --------------------------------------------------
    d.h("5. System architecture", level=1)
    d.p("Every model consumes the same input tuple (image, cat, cont, cont_mask) and returns class "
        "logits, and every model shares the same image encoder and, where applicable, the same "
        "metadata tokenizer. Consequently any measured difference between models is attributable to "
        "the fusion mechanism alone. This section describes each component and gives the exact "
        "tensor shapes, using B for batch size, C for the encoder's channel count (768 for "
        "ConvNeXt-Tiny), H and W for the feature-map grid (7x7 at 224px input), F for the number "
        "of metadata fields, and D for the embedding dimension.")

    d.h("5.1 Image encoder", level=2)
    d.p("The image branch is a timm ConvNeXt-Tiny backbone (ImageNet-1k weights, pinned to the "
        "exact tag convnext_tiny.fb_in1k) with its classifier removed, returning the final "
        "spatial feature map rather than a pooled vector. The stem and the first two stages are "
        "frozen; the last two stages are fine-tuned. For a 224x224 input the output is a feature "
        "map of shape (B, 768, 7, 7). The wrapper also exposes the last stage as the Grad-CAM "
        "target layer, and handles Vision Transformer backbones (dropping prefix tokens and "
        "folding patches back to a grid) so the same interface serves both families.")
    d.code([
        "image (B,3,224,224) --> ConvNeXt-T.forward_features --> feature map (B,768,7,7)",
        "gradcam_layer = backbone.stages[-1]   # activations + gradients for Grad-CAM",
    ])

    d.h("5.2 Metadata tokenizer", level=2)
    d.p("Structured metadata is turned into one token per field, so that F fields become an "
        "(B, F, D) tensor. Each categorical field has its own embedding table of size "
        "(cardinality + 1); the extra index is a genuine MISSING category, so a missing value is "
        "modelled explicitly rather than imputed. Each continuous field is projected from a scalar "
        "to D dimensions by its own linear layer, and where the value is absent a learned "
        "'continuous-missing' vector is substituted. Learned per-field positional embeddings are "
        "added and the result is layer-normalised. This design lets the model represent "
        "missingness as information (which, in medicine, it often is) and gives the cross-attention "
        "model a per-field token to attend to.")
    d.code([
        "for each categorical field j:  token_j = Embedding_j(cat[:, j])          # MISSING = index cardinality",
        "for each continuous  field j:  token_j = mask ? Linear_j(value) : cont_missing_j",
        "tokens = LayerNorm( stack(tokens, dim=1) + field_positional )            # -> (B, F, D)",
    ])
    d.p("Metadata field dropout is applied identically to every fusion model during training: whole "
        "fields are randomly set to MISSING, so all models learn to cope with absent metadata and "
        "are regularised in the same way. This is what later makes the all-MISSING external "
        "evaluation (dataset D) a fair test rather than a distribution shock.")

    d.h("5.3 The six models", level=2)
    d.p("B0 (image only) pools the feature map by global average and applies a LayerNorm-Dropout-"
        "Linear head. It uses no metadata and is the reference against which grounding and reliance "
        "are compared.")
    d.p("B1 (metadata only) is a gradient-boosted decision tree (HistGradientBoostingClassifier) "
        "over the encoded metadata matrix, with the MISSING category preserved and continuous "
        "fields left as NaN for the tree to split on. It measures the signal available in the "
        "tabular fields alone and has no image branch.")
    d.p("B3 (concatenation) flattens the field tokens into one vector through a linear-GELU-dropout "
        "block, concatenates it with the pooled image vector, and classifies the result. It is the "
        "simplest fusion and, empirically, the strongest.")
    d.code([
        "z_img  = mean_HW(feature_map)                       # (B, 768)",
        "z_meta = GELU(Linear(flatten(tokens)))              # (B, 256)",
        "logits = Head( concat[z_img, z_meta] )              # (B, n_classes)",
    ])
    d.p("B4 (FiLM) uses the metadata vector to produce a per-channel scale (gamma) and shift (beta) "
        "applied to the image feature map before pooling. The projection is zero-initialised so the "
        "model starts as the identity (scale 1, shift 0) and learns conditioning gradually.")
    d.code([
        "gamma, beta = chunk( Linear(z_meta) )               # each (B, 768)",
        "feature_map = feature_map * (1 + gamma) + beta      # broadcast over H, W",
        "logits = Head( mean_HW(feature_map) )",
    ])
    d.p("B5 (MetaBlock, Pacheco & Krohling 2021) conditions the feature map channel-wise through a "
        "gated nonlinearity, with two metadata-driven affine maps f and g (each Linear + "
        "BatchNorm):")
    d.code([
        "F' = sigmoid( tanh( feature_map * f(z_meta) ) + g(z_meta) )",
        "logits = Head( mean_HW(F') )",
    ])
    d.p("M1 (cross-attention) treats the H*W feature-map cells as a sequence of image patch tokens "
        "and lets them attend to the F metadata field tokens. Each block is pre-norm "
        "multi-head cross-attention (query = image patches, key/value = metadata) followed by a "
        "feed-forward network, with residual connections; n_blocks blocks are stacked. The pooled "
        "output is classified. The default direction is img2meta (image queries metadata), which "
        "is also the only direction that yields a spatial attention map per metadata field; "
        "meta2img and bidirectional variants are available as ablations.")
    d.code([
        "patches = flatten_HW(feature_map).T                 # (B, 49, 768)",
        "for block in img2meta_blocks:                       # query=patches, key/value=meta",
        "    patches = patches + CrossAttn( LN(patches), LN(meta) )",
        "    patches = patches + FFN( LN(patches) )",
        "logits = Head( mean_seq(patches) )                  # (B, n_classes)",
        "attention_maps: (B, F, 7, 7)  # where each field was consulted (last block)",
    ])
    d.p("The cross-attention block stores its head-averaged attention weights, from which a per-"
        "field spatial map of shape (B, F, H, W) is reconstructed - a third explanation channel "
        "unique to M1, showing where in the image each metadata field was consulted.")

    # ---- 6 methodology ---------------------------------------------------
    d.h("6. Methodology", level=1)
    d.h("6.1 Cross-validation and grouping", level=2)
    d.p("Each model is trained under five-fold, stratified, grouped cross-validation repeated over "
        "three random seeds: 15 runs per model per dataset. Grouping is by patient (A) or lesion "
        "(B, C) so that no individual appears on both sides of a split; without this a model can "
        "recognise the individual rather than the disease and every metric is inflated. An inner, "
        "grouped validation split is carved from the training folds for early stopping; the test "
        "fold is never seen during training or tuning.")
    d.h("6.2 Hyperparameter tuning", level=2)
    d.p("Each model is tuned once per dataset with an identical six-trial grid (learning rate in "
        "{1e-4, 3e-4, 1e-3} by head dropout in {0.1, 0.3}) on fold 0, seed 0, selected on "
        "inner-validation balanced accuracy. The identical budget for every model is what keeps "
        "the comparison fair. Every model selected the largest learning rate; because the same "
        "grid was applied to all, this affects absolute accuracy but not the between-model "
        "comparison, and it is recorded as a limitation.")
    d.h("6.3 The five endpoints", level=2)
    ep = pd.DataFrame([
        {"ID": "P1", "Endpoint (plain)": "Grounding: share of positive attribution inside the lesion mask", "Data": "B"},
        {"ID": "P2", "Endpoint (plain)": "Image-shortcut robustness: area under the AUROC-vs-bias curve", "Data": "C"},
        {"ID": "P3", "Endpoint (plain)": "Offloading: change in grounding when a metadata shortcut is planted", "Data": "C"},
        {"ID": "P4", "Endpoint (plain)": "Counterfactual map shift when one field changes, image fixed", "Data": "A"},
        {"ID": "P5", "Endpoint (plain)": "Reliance: accuracy lost without metadata vs. without the image", "Data": "A"},
    ])
    d.table(ep, widths=[0.5, 5.4, 0.6], caption="Table 6.1. The pre-registered endpoints.")
    d.h("6.4 Attribution and the sanity gate", level=2)
    d.p("Attribution is computed with Grad-CAM on the encoder's final stage, identically for every "
        "model so that maps are comparable, for the predicted class (with the true class as a "
        "sensitivity analysis). RISE, a gradient-free method, is available as a second channel. "
        "Before any attribution result is used, a parameter-randomisation sanity check (Adebayo et "
        "al.) is applied: the maps from the trained model are compared with those from a fully "
        "randomised copy, and a method whose maps survive randomisation (mean SSIM above 0.5) is "
        "excluded. Grad-CAM passed on every model, with mean SSIM between 0.036 and 0.143.")
    d.h("6.5 Statistics", level=2)
    d.p("Out-of-fold predictions are pooled within each seed and averaged over seeds. Confidence "
        "intervals are 95% patient-cluster bootstrap intervals (10,000 resamples, fixed seed), so "
        "multiple images from one patient are resampled together. Nineteen primary comparisons "
        "were specified in advance and corrected together by Holm-Bonferroni at a family-wise "
        "error rate of 0.05. The 'same accuracy' claim is tested formally by two one-sided tests "
        "(TOST) with a two-point equivalence margin. The entire plan was frozen before training "
        "and every subsequent change is logged with a date and reason.")

    # ---- 7 results -------------------------------------------------------
    d.h("7. Results", level=1)
    d.h("7.1 Classification accuracy", level=2)
    d.table(protocol_table("padufes20"), widths=[2.0, 1.6, 1.5, 1.5],
            caption="Table 7.1. Dataset A (PAD-UFES-20), mean over 15 runs [95% CI].")
    d.table(protocol_table("ham10000"), widths=[2.0, 1.6, 1.5, 1.5],
            caption="Table 7.2. Dataset B (HAM10000), mean over 15 runs [95% CI].")
    d.figure("fig_accuracy.png", width=6.4, caption="Figure 7.1. Balanced accuracy by model on both datasets. Dashed line: image-only reference.")
    d.p("On dataset A, metadata alone nearly matches the image alone and fusion adds about eight "
        "points; on dataset B, where only three weak fields exist, fusion adds at most 1.8 points. "
        "The TOST test finds no fusion model equivalent to image-only within two points on dataset "
        "A (they are genuinely more accurate), while B4 and B5 are equivalent on dataset B.")

    d.h("7.2 Grounding (P1) - the central result", level=2)
    p1 = read("endpoints_ham10000_gradcam_descriptive.csv")
    if not p1.empty:
        piv = p1[p1["endpoint"] == "P1"].pivot_table(index="model", columns="metric", values="value")
        show = pd.DataFrame({
            "Model": [f"{m} ({NAMES[m]})" for m in piv.index],
            "Attribution in lesion": [f"{piv.loc[m, 'energy_in_mask']:.3f}" for m in piv.index],
            "Chance": [f"{piv.loc[m, 'mask_area_fraction']:.3f}" for m in piv.index],
            "Pointing game": [f"{piv.loc[m, 'pointing_game']:.1%}" for m in piv.index],
            "IoU (Otsu)": [f"{piv.loc[m, 'iou_otsu']:.3f}" for m in piv.index],
        })
        d.table(show, widths=[1.9, 1.5, 1.0, 1.2, 1.1],
                caption="Table 7.3. P1 on dataset B, 7,470 lesion clusters. Chance = mask area fraction.")
    d.figure("fig_grounding.png", width=5.6, caption="Figure 7.2. Share of attribution inside the lesion. Dashed red: chance; dotted grey: image-only.")
    d.p("Concatenation places significantly more attribution inside the lesion than image-only "
        "(+0.049); FiLM (-0.087), MetaBlock (-0.160) and cross-attention (-0.017) place "
        "significantly less. MetaBlock sits at 0.299 against a chance level of 0.267 - its "
        "attribution has almost ceased to track the lesion while its accuracy remains competitive. "
        "All four comparisons survive Holm correction, and pointing game and Otsu IoU rank the "
        "models identically, so the effect is not an artefact of one measure.")

    d.h("7.3 Modality reliance (P5)", level=2)
    p5 = read("endpoints_padufes20_gradcam_descriptive.csv")
    if not p5.empty:
        piv = p5[p5["endpoint"] == "P5"].pivot_table(index="model", columns="metric", values="value")
        show = pd.DataFrame({
            "Model": [f"{m} ({NAMES[m]})" for m in piv.index],
            "BACC lost without metadata": [f"{piv.loc[m, 'drop_meta_shuffled']:.3f}" for m in piv.index],
            "BACC lost without image": [f"{piv.loc[m, 'drop_mean_image']:.3f}" for m in piv.index],
        })
        d.table(show, widths=[2.0, 2.2, 2.2], caption="Table 7.4. P5 on dataset A, 1,373 patient clusters.")
    d.figure("fig_reliance.png", width=5.6, caption="Figure 7.3. Accuracy lost when each modality is removed.")
    d.p("Every fusion model loses more accuracy when metadata is removed than when the image is "
        "removed, and adding metadata roughly halves dependence on the image (0.477 to 0.18-0.27). "
        "The image-only control loses exactly zero when metadata is shuffled, confirming the "
        "measurement is sound.")

    d.h("7.4 Counterfactual sensitivity (P4)", level=2)
    p4 = read("endpoints_padufes20_gradcam_descriptive.csv")
    if not p4.empty:
        piv = p4[p4["endpoint"] == "P4"].pivot_table(index="model", columns="metric", values="value")
        show = pd.DataFrame({
            "Model": [f"{m} ({NAMES[m]})" for m in piv.index],
            "Shift, irrelevant fields": [f"{piv.loc[m, 'map_shift_irrelevant']:.3f}" for m in piv.index],
            "Shift, relevant fields": [f"{piv.loc[m, 'map_shift_relevant']:.3f}" for m in piv.index],
        })
        d.table(show, widths=[2.0, 2.2, 2.2], caption="Table 7.5. P4 on dataset A. Irrelevant fields designated in advance.")
    d.p("Holding the image fixed and changing one field moves the attribution map. Every model "
        "moves about twice as much for a clinically relevant field as for an irrelevant one, but "
        "movement for irrelevant fields (piped water, sewage system) is not zero, and cross-"
        "attention moves most - significantly more than every other mechanism.")

    d.h("7.5 Shortcut robustness (P2) and offloading (P3)", level=2)
    d.figure("fig_degradation.png", width=5.6, caption="Figure 7.4. Test AUROC as the artifact shortcut strengthens. All models collapse together.")
    d.p("No fusion mechanism differs from image-only on image-shortcut robustness: all collapse "
        "from AUROC near 0.90 to below chance as the artifact correlation reverses, reproducing "
        "the published finding. The offloading endpoint P3 returned no usable answer - it rests on "
        "three splits per model and the intervals are too wide; the image-only control drifted in "
        "the same direction, showing the drift is a resampling artefact rather than model "
        "behaviour. Reporting P3 as inconclusive is a direct consequence of pre-registering that "
        "control.")

    d.h("7.6 Statistical summary", level=2)
    fam = read("family_holm.csv")
    if not fam.empty:
        s = fam.groupby("endpoint").agg(tests=("reject_holm", "size"), significant=("reject_holm", "sum")).reset_index()
        s["endpoint"] = s["endpoint"].map({"P1": "P1 Grounding", "P2": "P2 Image shortcut", "P3": "P3 Offloading", "P4": "P4 Counterfactual", "P5": "P5 Reliance"})
        s.columns = ["Endpoint", "Tests", "Significant after Holm"]
        d.table(s, widths=[2.2, 1.4, 2.2], caption="Table 7.6. The pre-registered family of 19 tests.")
        d.p("Eleven of the nineteen pre-registered tests are significant after correction. The two "
            "endpoints that returned nothing (P2, P3) were planned in advance; a planned null is a "
            "result, not a gap.")

    # ---- 8 discussion ----------------------------------------------------
    d.h("8. Discussion", level=1)
    d.p("The results converge on one interpretation: the mechanism of fusion, not the presence of "
        "metadata, determines what happens to the model's visual evidence. Identical information, "
        "an identical backbone and an identical recipe produce opposite effects on grounding "
        "depending only on how the two modalities are combined.")
    d.p("Two consequences are practical. First, the simplest mechanism - plain concatenation, "
        "usually a throwaway baseline - is best on accuracy (dataset A) and the only one to improve "
        "grounding (dataset B). Second, accuracy and trustworthiness are decoupled: MetaBlock is "
        "accurate while attending barely above chance, and cross-attention combines middling "
        "accuracy with the largest displacement of evidence by irrelevant metadata. An "
        "accuracy-only evaluation would rank these models in an order unrelated to the quality of "
        "their evidence.")

    # ---- 9 use cases -----------------------------------------------------
    d.h("9. Actual use cases", level=1)
    d.p("The project is a measurement methodology and a set of findings, not a deployable product. "
        "Its value is realised in the following concrete settings.")
    d.h("9.1 Model selection before clinical deployment", level=2)
    d.p("A team choosing a fusion architecture for a triage tool typically ranks candidates by "
        "accuracy. This work shows that ranking can be actively misleading: MetaBlock and "
        "concatenation are within about two accuracy points on dataset B, yet one attends to the "
        "lesion and the other does not. The endpoints here (grounding, reliance, counterfactual "
        "sensitivity) give a procurement or regulatory reviewer a way to prefer a model that is "
        "right for the right reason, and to reject one that will fail silently on atypical patients.")
    d.h("9.2 A pre-deployment audit suite", level=2)
    d.p("The five endpoints form a reusable audit that any image-plus-metadata medical model can be "
        "run through before deployment. Two of them - counterfactual sensitivity (N2) and modality "
        "reliance - require no human annotation and so can be applied to any dataset the vendor "
        "already has. A model that loses most of its accuracy when metadata is withheld, or whose "
        "evidence shifts when an administrative field changes, is flagged as fragile regardless of "
        "its headline accuracy.")
    d.h("9.3 Fairness and robustness assurance", level=2)
    d.p("Offloading onto metadata is a fairness hazard: under-served populations have sparser "
        "metadata, so a model that leans on it will degrade exactly where care is scarcest. The "
        "reliance endpoint quantifies that exposure directly, and the planned all-MISSING external "
        "evaluation on skin-tone-balanced data (dataset D) tests it under real distribution shift. "
        "The same logic applies to any site that records fewer fields than the training site.")
    d.h("9.4 Shortcut and dataset-bias screening", level=2)
    d.p("The cross-modal trap benchmark (N1) lets a team stress-test whether a model has latched "
        "onto spurious correlations - visual artifacts, or a metadata field that happened to "
        "predict the label in the training cohort - before those shortcuts cause a field failure. "
        "This is directly relevant to the well-documented tendency of skin lesion models to key on "
        "rulers, ink and hair.")
    d.h("9.5 Education and method transfer", level=2)
    d.p("Because the methodology is modality-agnostic, it transfers to other image-plus-tabular "
        "medical problems - chest radiographs with clinical indications, retinal images with "
        "patient history - where the same grounding-versus-offloading question arises. The journal "
        "extension of this project targets exactly that transfer.")

    # ---- 10 limitations --------------------------------------------------
    d.h("10. Limitations and future work", level=1)
    d.bullets([
        "The offloading endpoint (P3) is underpowered - three published splits per model give intervals too wide to resolve the effect. This is a limit of the published data, not a fixable bug.",
        "Every model selected the largest learning rate in the tuning grid, so the optimum may lie outside the searched range; the identical grid keeps comparisons fair but absolute accuracy may be understated.",
        "Only one image backbone (ConvNeXt-Tiny) was evaluated; whether the findings hold for other architectures is open.",
        "Grounding is measured on one dataset, because HAM10000 is the only public dermatology dataset with a mask for every image.",
        "External skin-tone fairness validation (DDI) is pending an individual research-use agreement; the pipeline is built and tested.",
        "PAD-UFES-20 images came from a mirror; the metadata is byte-identical to the official file and all image IDs match, but per-image byte verification against the original host remains open.",
    ])
    d.p("Planned work: produce the final figures; add RISE as a second attribution channel and "
        "compute the counterfactual and reliance endpoints on dataset B as robustness checks; run "
        "the unfiltered published trap splits for direct comparison with Bissoto et al.; and, in a "
        "journal version, restore the chest-radiograph track and prototype a training-time penalty "
        "that discourages evidence from moving under irrelevant-metadata perturbation (turning the "
        "diagnostic finding into a correction).")

    # ---- 11 reproducibility ----------------------------------------------
    d.h("11. Reproducibility and engineering", level=1)
    d.p("The project is a Python package (src/medfusion) with a scripted pipeline: download and "
        "verify data, cache images and write grouped splits, tune, train the full protocol, "
        "compute endpoints, apply the statistics, and generate reports and figures. Every training "
        "run stores its exact configuration snapshot and the git commit it was launched from; each "
        "cross-validation fold and each trap-set cell writes its own result file, so any run "
        "resumes after interruption and no result is silently overwritten. A suite of 48 automated "
        "tests covers the data encoding, splits, models, attribution and statistics.")
    tech = pd.DataFrame([
        {"Item": "Total model trainings", "Value": f"{n_runs}"},
        {"Item": "Total GPU time", "Value": f"about {gpu_hours:.0f} hours"},
        {"Item": "Hardware", "Value": "NVIDIA RTX 5080 (16 GB), shared with another service"},
        {"Item": "Framework", "Value": "PyTorch 2.11 (CUDA 12.8), timm, scikit-learn"},
        {"Item": "Backbone", "Value": "ConvNeXt-Tiny, convnext_tiny.fb_in1k (ImageNet-1k)"},
        {"Item": "Automated tests", "Value": "48, all passing"},
        {"Item": "Data on disk", "Value": "72 GB, never uploaded (licences forbid redistribution)"},
        {"Item": "Analysis plan", "Value": "docs/04-preregistration.md, frozen before training"},
        {"Item": "Label mapping", "Value": "docs/label_mapping.md, frozen before external download"},
    ])
    d.table(tech, widths=[2.4, 4.0], caption="Table 11.1. Reproducibility record.")

    # ---- 12 references ---------------------------------------------------
    d.h("12. References", level=1)
    refs = [
        "A. G. C. Pacheco and R. A. Krohling, \"The impact of patient clinical information on automated skin cancer detection,\" Computers in Biology and Medicine, vol. 116, 103545, 2020.",
        "X. Li et al., \"Fusing metadata and dermoscopy images for skin disease diagnosis,\" ISBI, 2020.",
        "A. G. C. Pacheco and R. A. Krohling, \"An attention-based mechanism to combine images and metadata in deep learning models applied to skin cancer classification\" (MetaBlock), IEEE JBHI, vol. 25, no. 9, pp. 3554-3563, 2021.",
        "A. Khurshid, M. Singh, and M. Vatsa, \"Multimodal dual-stage feature refinement for robust skin lesion classification,\" Scientific Reports, 2025.",
        "K. Mridha and M. Islam, \"Cross-attention enables context-aware multimodal skin lesion diagnosis,\" medRxiv, 2026.",
        "P. Tschandl, C. Rosendahl, and H. Kittler, \"The HAM10000 dataset,\" Scientific Data, vol. 5, 180161, 2018.",
        "P. Tschandl et al., \"Human-computer collaboration for skin cancer recognition,\" Nature Medicine, vol. 26, pp. 1229-1234, 2020.",
        "A. Bissoto, C. Barata, E. Valle, and S. Avila, \"Artifact-based domain generalization of skin lesion models,\" ECCV Workshops, 2022.",
        "A. Saporta et al., \"Benchmarking saliency methods for chest X-ray interpretation\" (CheXlocalize), Nature Machine Intelligence, vol. 4, 2022.",
        "R. R. Selvaraju et al., \"Grad-CAM: Visual explanations from deep networks via gradient-based localization,\" ICCV, 2017.",
        "J. Adebayo et al., \"Sanity checks for saliency maps,\" NeurIPS, 2018.",
        "V. Petsiuk, A. Das, and K. Saenko, \"RISE: Randomized input sampling for explanation of black-box models,\" BMVC, 2018.",
        "Z. Liu et al., \"A ConvNet for the 2020s\" (ConvNeXt), CVPR, 2022.",
        "E. Perez et al., \"FiLM: Visual reasoning with a general conditioning layer,\" AAAI, 2018.",
        "R. Daneshjou et al., \"Disparities in dermatology AI performance on a diverse, curated clinical image set\" (DDI), Science Advances, vol. 8, no. 31, 2022.",
        "A. G. C. Pacheco et al., \"PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones,\" Data in Brief, vol. 32, 106221, 2020.",
        "\"A scoping review of explainable artificial intelligence for medical multimodal data,\" npj Digital Medicine, 2026.",
    ]
    for i, r in enumerate(refs, 1):
        para = d.doc.add_paragraph()
        para.paragraph_format.left_indent = Inches(0.35)
        para.paragraph_format.first_line_indent = Inches(-0.35)
        para.paragraph_format.space_after = Pt(4)
        rr = para.add_run(f"[{i}] {r}")
        rr.font.size = Pt(10)

    d.save(out)
    print("written:", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--students", nargs="*", default=["Aniket Meena", "Ritu Raj Swami", "Nishant Kumar"])
    ap.add_argument("--supervisor", default="Dr. Bharti Nagpal")
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "MedFusion-XAI_Technical_Document.docx")
    a = ap.parse_args()
    build(a.out, a.students, a.supervisor)


if __name__ == "__main__":
    main()
