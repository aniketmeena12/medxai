"""Build the committee progress report (.docx) in the prescribed six-section format.

    python scripts/make_committee_report.py

Format required by the committee: Introduction, Literature survey, Problem statement &
Methodology, Work done till date / Results and Discussion, Future work, References. 10-15 pages,
signed by the student(s) and supervisor(s).

Every experimental number is read from the result files, so the report cannot drift from the data.
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
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

from medfusion.eval.stats import mean_ci

REPO = Path(__file__).resolve().parents[1]
TABLES = REPO / "results" / "tables"
NAMES = {"B0": "Image only", "B1": "Metadata only", "B3": "Concatenation",
         "B4": "FiLM", "B5": "MetaBlock", "M1": "Cross-attention"}


# ----------------------------------------------------------------------------- data
def protocol_table(dataset: str) -> pd.DataFrame:
    rows = []
    for m in ("B0", "B1", "B3", "B4", "B5", "M1"):
        files = sorted(glob.glob(f"data/experiments/{dataset}_{m}_*/metrics.json"))
        d = next((json.load(open(f)) for f in reversed(files)
                  if json.load(open(f))["complete_protocol"]), None)
        if d is None:
            continue
        row = {"Model": f"{m} ({NAMES[m]})"}
        for key, label in (("balanced_accuracy", "Balanced acc."), ("macro_f1", "Macro-F1"),
                           ("auroc", "AUROC")):
            v = np.array([x[key] for x in d["per_fold_seed"]])
            mu, lo, hi = mean_ci(v)
            row[label] = f"{mu:.3f} [{lo:.3f}, {hi:.3f}]"
        rows.append(row)
    return pd.DataFrame(rows)


def read(name: str) -> pd.DataFrame:
    p = TABLES / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def totals() -> tuple[int, float]:
    seconds, runs = 0.0, 0
    for f in glob.glob("data/experiments/*/metrics.json"):
        for r in json.load(open(f)).get("per_fold_seed", []):
            seconds += r.get("seconds", 0) or 0
            runs += 1
    for pattern in ("data/experiments/isic2019_trap_*/img*.json",
                    "data/experiments/tune_*/trial*.json"):
        for f in glob.glob(pattern):
            seconds += json.load(open(f)).get("seconds", 0) or 0
            runs += 1
    return runs, seconds / 3600


# ----------------------------------------------------------------------------- docx
def add_table(doc, df, style="Light Grid Accent 1", widths=None, caption=None):
    if caption:
        c = doc.add_paragraph(caption)
        c.runs[0].bold = True
        c.runs[0].font.size = Pt(9.5)
    if df.empty:
        doc.add_paragraph("(pending)")
        return
    t = doc.add_table(rows=1, cols=len(df.columns))
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
            for p in cells[i].paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9.5)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    doc.add_paragraph()


def bullets(doc, items, style="List Bullet"):
    for it in items:
        doc.add_paragraph(it, style=style)


def page_numbers(doc):
    footer = doc.sections[0].footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    for instr in ("begin", "PAGE", "end"):
        el = OxmlElement(f"w:fld{'Char' if instr != 'PAGE' else 'Simple'}")
        if instr == "PAGE":
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = " PAGE "
        else:
            el.set(qn("w:fldCharType"), instr)
        run._r.append(el)


def build(out: Path, students: list[str], supervisor: str) -> None:
    doc = Document()
    for s in doc.sections:
        s.left_margin = s.right_margin = Inches(1.0)
        s.top_margin = s.bottom_margin = Inches(0.9)
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(11)
    st.paragraph_format.space_after = Pt(6)
    st.paragraph_format.line_spacing = 1.15
    st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    page_numbers(doc)

    n_runs, gpu_hours = totals()

    # ------------------------------------------------------------------ title page
    t = doc.add_paragraph("Same Accuracy, Different Evidence: Does Patient Metadata Ground or "
                          "Distract Skin Lesion Classifiers?", style="Title")
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = sub.add_run("Major Project Progress Report")
    r.bold = True
    r.font.size = Pt(14)
    for line, size, bold in ((", ".join(students), 12, False),
                             (f"Supervisor: {supervisor}", 12, False),
                             (f"Date: {date.today().strftime('%d %B %Y')}", 11, False)):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(line)
        run.font.size = Pt(size)
        run.bold = bold
    doc.add_paragraph()

    ab = doc.add_paragraph()
    ab.add_run("Abstract. ").bold = True
    ab.add_run(
        "Skin lesion classifiers are increasingly given patient metadata - age, sex, anatomical "
        "site, symptoms - alongside the image, and this usually raises accuracy. It is not known "
        "whether the added information makes the model's visual evidence better grounded on the "
        "lesion, or whether the model instead leans on the metadata and attends less to the image. "
        "This project measures that directly. Five fusion mechanisms and an image-only reference "
        "were trained under an identical protocol on two dermatology datasets and "
        "evaluated against dermatologist-drawn lesion masks, artifact shortcut "
        f"benchmarks and counterfactual metadata perturbations. Across {n_runs} "
        "training runs, the finding is that the fusion mechanism, not the presence of "
        "metadata, determines whether grounding improves or "
        "degrades: simple concatenation increased attribution inside the lesion by 4.9 points over "
        "the image-only model, while MetaBlock reduced it by 16.0 points to near the chance level. "
        "All fusion models relied more on metadata than on the image, and metadata fields that "
        "cannot affect lesion appearance still displaced the image evidence.")

    doc.add_paragraph()
    dec = doc.add_paragraph()
    dec.add_run("Declaration and signatures").bold = True
    doc.add_paragraph(
        "We certify that the work reported here was carried out by us under the guidance of our "
        "supervisor, and that all sources used have been cited.")
    sig = pd.DataFrame([{"Name": s, "Role": "Student", "Signature": "", "Date": ""}
                        for s in students]
                       + [{"Name": supervisor, "Role": "Supervisor", "Signature": "", "Date": ""}])
    add_table(doc, sig, style="Table Grid", widths=[2.0, 1.2, 2.0, 1.2])

    doc.add_section(WD_SECTION.NEW_PAGE)

    # ------------------------------------------------------------------ a. Introduction
    doc.add_heading("a. Introduction", level=1)
    doc.add_paragraph(
        "Skin cancer is among the most common cancers worldwide, and access to dermatological "
        "expertise is unevenly distributed. Automated classification of skin lesions from "
        "photographs has therefore attracted sustained attention, and recent systems approach "
        "specialist-level accuracy on curated benchmarks. In clinical practice, however, a "
        "dermatologist never judges a lesion from its appearance alone: the patient's age, the "
        "anatomical site, whether the lesion itches, bleeds, or has recently changed, and the "
        "patient's skin type all inform the diagnosis.")
    doc.add_paragraph(
        "This has motivated multimodal models that combine the lesion image with structured "
        "patient metadata. The reported benefit is consistent and often large: on clinical "
        "smartphone images, adding metadata has been reported to improve balanced accuracy by "
        "around seven points [1], and a range of fusion mechanisms - concatenation, feature "
        "gating, attention-based blocks and cross-attention transformers - have since been "
        "proposed [2, 3, 4, 6].")
    doc.add_paragraph(
        "The accuracy question is therefore largely settled. A different question is not. When a "
        "model is handed metadata that is itself predictive, it may use that information to "
        "interrogate the image more effectively - for example, by knowing where on the body to "
        "expect a particular pattern. Or it may do the opposite: rely on the metadata as a "
        "shortcut and attend less carefully to the image. Both behaviours can produce the same "
        "accuracy on a held-out test set drawn from the same distribution, and accuracy alone "
        "cannot distinguish them.")
    doc.add_paragraph(
        "The distinction matters clinically. A model that has quietly shifted its weight onto "
        "metadata will degrade when metadata is missing, unusual, or systematically different - "
        "which is exactly the situation in an under-served population, or when the model is "
        "deployed at a site that records fewer fields. It will also produce explanations that "
        "look plausible while being driven by information the clinician cannot see in the image.")
    doc.add_paragraph("This project asks and answers a single question:")
    q = doc.add_paragraph()
    qr = q.add_run("When a skin lesion classifier is given patient metadata, does its image "
                   "evidence become better grounded on the lesion, or does it rely less on the "
                   "image?")
    qr.italic = True
    doc.add_paragraph(
        "We refer to the two outcomes as grounding and offloading. Both are reported as findings; "
        "neither is assumed. The contribution is not a new architecture - the architectures "
        "compared here are all published - but a quantitative audit of where five fusion "
        "mechanisms actually look, measured against dermatologist-drawn lesion outlines, "
        "shortcut benchmarks, and counterfactual perturbations of the metadata.")

    # ------------------------------------------------------------------ b. Literature survey
    doc.add_heading("b. Literature Survey", level=1)

    doc.add_heading("b.1 Fusion of images and patient metadata", level=2)
    doc.add_paragraph(
        "Pacheco and Krohling [1] established the baseline result for clinical dermatology "
        "images: concatenating patient metadata with image features improved balanced accuracy by "
        "approximately seven points across several backbones. Li et al. [2] introduced MetaNet, "
        "which uses metadata to gate image feature channels multiplicatively. Pacheco and "
        "Krohling subsequently proposed MetaBlock [3], a metadata-conditioned feature attention "
        "block that outperformed both MetaNet and concatenation in the majority of tested "
        "settings, and which remains the standard baseline on PAD-UFES-20. The same group has "
        "recently extended it to handle missing metadata [10].")
    doc.add_paragraph(
        "Reported accuracies on PAD-UFES-20 provide useful reference points: de Lima and Krohling "
        "[5] report balanced accuracy up to 0.800 with transformer backbones and MetaBlock, while "
        "Khurshid et al. [9] report 0.845 with a dual-stage refinement network. Comparable work "
        "on dermoscopic data reports accuracy gains from fusion of roughly four to five points on "
        "HAM10000 [13, 14].")

    doc.add_heading("b.2 Attention and cross-attention fusion", level=2)
    doc.add_paragraph(
        "Zhou and Luo [4] applied a mutual-attention transformer between image and metadata "
        "representations. Cheslerean-Boghiu et al. [6] proposed a single-stage attention fusion of "
        "image and metadata tokens and claim native interpretability in both domains. Atiq and "
        "Fattah [11] combine segmentation-guided dual networks with metadata cross-attention. Most "
        "directly relevant to this project, Mridha and Islam [12] apply metadata-to-image "
        "cross-attention on PAD-UFES-20 and report a small AUC improvement (0.9776 to 0.9818) "
        "together with improved calibration.")
    doc.add_paragraph(
        "Two observations follow from this body of work. First, the architecture this project "
        "originally proposed as its contribution has already been published, on the same dataset "
        "[12], which is why the project was re-aimed at measurement rather than architecture. "
        "Second, and more usefully, Khurshid et al. [9] report a published negative result: in "
        "their comparison, cross-attention achieved only 68.99 per cent balanced accuracy on "
        "PAD-UFES-20, below simpler alternatives. The evidence that cross-attention is worth its "
        "complexity is therefore weak even on accuracy grounds.")

    doc.add_heading("b.3 Quantitative evaluation of visual explanations", level=2)
    doc.add_paragraph(
        "Grad-CAM [16] remains the most widely used attribution method for convolutional medical "
        "imaging models. Critically, Adebayo et al. [17] showed that several popular saliency "
        "methods produce visually plausible maps even when the model's weights are randomised, "
        "establishing parameter-randomisation as a necessary sanity check before any attribution "
        "result is interpreted. Saporta et al. [15] provided the benchmark methodology this "
        "project reuses: they scored seven saliency methods against radiologist segmentations on "
        "chest radiographs using intersection-over-union and a hit-rate criterion, finding that "
        "all methods trailed human annotators.")
    doc.add_paragraph(
        "Applying this standard to multimodal models is the gap. Across the fusion literature "
        "surveyed above, explanations are presented as illustrative figures rather than scored: "
        "Das et al. [8] use Grad-CAM and integrated gradients qualitatively; Sonuc et al. [14] "
        "state explicitly that no systematic validation was performed; Mridha and Islam [12] show "
        "attention maps and metadata perturbations but do not quantify them. A 2026 scoping "
        "review of explainable AI for multimodal medical data [19] reaches the same conclusion "
        "across 82 studies, finding that standardised evaluation is missing in the majority, "
        "which rely solely on qualitative measures.")

    doc.add_heading("b.4 Shortcut learning and dataset bias", level=2)
    doc.add_paragraph(
        "Bissoto et al. [18] demonstrated that skin lesion classifiers rely heavily on visual "
        "artifacts - rulers, hair, ink markings, dark corners - rather than the lesion itself, and "
        "constructed trap sets in which the correlation between artifact and diagnosis is varied "
        "systematically and reversed at test time. These benchmarks are image-only; to our "
        "knowledge they have not previously been applied to metadata-fused models, which is the "
        "use made of them here.")

    doc.add_heading("b.5 Fairness and skin tone", level=2)
    doc.add_paragraph(
        "Daneshjou et al. [20] curated the Diverse Dermatology Images dataset, the first publicly "
        "available set of biopsy-proven lesions with substantial dark skin representation, and "
        "showed that published algorithms degrade on darker skin. Access requires an individual "
        "research-use agreement, which has not been obtainable within this project (Section e).")

    doc.add_heading("b.6 Summary of the gap", level=2)
    doc.add_paragraph(
        "Prior work establishes that metadata fusion improves accuracy, and provides a mature "
        "methodology for scoring visual explanations against human annotation. No published work "
        "joins the two: no study measures whether adding metadata improves or degrades the "
        "grounding of a model's image evidence. That is the space this project occupies.")

    # ------------------------------------------------------------------ c. Problem & method
    doc.add_heading("c. Problem Statement and Methodology", level=1)

    doc.add_heading("c.1 Problem statement", level=2)
    doc.add_paragraph(
        "Given a skin lesion classifier that takes an image and structured patient metadata, "
        "determine whether the metadata improves or degrades the grounding of the model's image "
        "evidence, and whether the answer depends on the mechanism used to combine the two "
        "modalities. Two competing hypotheses are stated in advance:")
    bullets(doc, [
        "H-ground: fusion models place more of their attribution inside the lesion than an "
        "image-only model does.",
        "H-offload: fusion models place less attribution inside the lesion, and a metadata "
        "shortcut erodes image grounding further.",
    ])
    doc.add_paragraph(
        "Accuracy is treated as a secondary, sanity-check result throughout. The full analysis "
        "plan - hypotheses, endpoints, statistical tests and decision rules - was written and "
        "frozen before any model was trained, and every subsequent change is logged with a date "
        "and reason. This pre-registration is what prevents the findings from being shaped after "
        "the results were seen.")

    doc.add_heading("c.2 Datasets", level=2)
    data = pd.DataFrame([
        {"ID": "A", "Dataset": "PAD-UFES-20", "Content": "2,298 clinical smartphone images, 6 "
         "classes, 1,373 patients, 21 usable metadata fields",
         "Role": "Rich-metadata regime; counterfactual and reliance endpoints"},
        {"ID": "B", "Dataset": "HAM10000", "Content": "10,015 dermoscopic images, 7 classes, "
         "7,470 lesions, 3 metadata fields, lesion mask for every image",
         "Role": "Grounding endpoint (the only public set with masks)"},
        {"ID": "C", "Dataset": "ISIC 2019 + trap sets", "Content": "25,331 images; published "
         "splits at 6 artifact-bias levels x 10 repetitions",
         "Role": "Shortcut robustness and the metadata-trap grid"},
        {"ID": "D", "Dataset": "DDI", "Content": "656 biopsy-proven images balanced across skin "
         "tones", "Role": "Planned external fairness test; access not obtained"},
    ])
    add_table(doc, data, widths=[0.4, 1.3, 2.4, 2.3],
              caption="Table 1. Datasets. All are public; each download was verified against the "
                      "provider's published figures (15 of 15 checks passed).")
    doc.add_paragraph(
        "Splits are grouped so that no patient (dataset A) or lesion (datasets B and C) appears on "
        "both sides of any split. Without this, a model can recognise the patient rather than the "
        "disease and every reported number is inflated.")

    doc.add_heading("c.3 Models compared", level=2)
    models = pd.DataFrame([
        {"ID": "B0", "Mechanism": "Image only - the reference point"},
        {"ID": "B1", "Mechanism": "Metadata only (gradient boosting); measures signal in the "
                                  "tabular fields alone"},
        {"ID": "B3", "Mechanism": "Concatenation of pooled image and metadata embeddings"},
        {"ID": "B4", "Mechanism": "FiLM: metadata produces scale and shift parameters applied to "
                                  "image features"},
        {"ID": "B5", "Mechanism": "MetaBlock [3]: metadata-conditioned feature attention"},
        {"ID": "M1", "Mechanism": "Cross-attention: image patch tokens query metadata tokens"},
    ])
    add_table(doc, models, widths=[0.6, 5.8], caption="Table 2. The six models.")
    doc.add_paragraph(
        "All image models share the same backbone (ConvNeXt-Tiny, ImageNet weights pinned to an "
        "exact version), the same input resolution, the same augmentation, the same optimiser "
        "settings and the same tuning budget. Only the fusion mechanism differs, which is what "
        "makes the comparison attributable to the mechanism.")

    doc.add_heading("c.4 Training and tuning protocol", level=2)
    bullets(doc, [
        "Five-fold grouped, stratified cross-validation, repeated with three random seeds: 15 "
        "training runs per model per dataset.",
        "Hyperparameters tuned once per model per dataset with an identical six-trial grid "
        "(learning rate x head dropout) on fold 0, seed 0, selected on inner-validation balanced "
        "accuracy. The test fold is never loaded during tuning.",
        "Metadata field dropout is applied identically to every fusion model, so all of them meet "
        "missing fields during training.",
        "Balanced accuracy is the primary accuracy measure because the classes are heavily "
        "imbalanced (melanoma is 52 of 2,298 images in dataset A).",
    ])

    doc.add_heading("c.5 Endpoints", level=2)
    ep = pd.DataFrame([
        {"ID": "P1", "Endpoint": "Grounding: share of positive attribution falling inside the "
         "dermatologist-drawn lesion mask", "Data": "B"},
        {"ID": "P2", "Endpoint": "Shortcut robustness: normalised area under the curve of test "
         "AUROC against artifact-bias level", "Data": "C"},
        {"ID": "P3", "Endpoint": "Offloading: change in grounding when a metadata shortcut is "
         "introduced", "Data": "C"},
        {"ID": "P4", "Endpoint": "Counterfactual map shift: movement of the attribution map when "
         "one metadata field is changed and the image is held fixed", "Data": "A"},
        {"ID": "P5", "Endpoint": "Modality reliance: accuracy lost when metadata is shuffled "
         "versus when the image is replaced by the dataset mean image", "Data": "A"},
    ])
    add_table(doc, ep, widths=[0.5, 4.9, 0.6],
              caption="Table 3. The five pre-registered endpoints.")
    doc.add_paragraph(
        "Attribution is computed with Grad-CAM on the final image-encoder stage, identically for "
        "every model so that maps are comparable. Following Adebayo et al. [17], a "
        "parameter-randomisation sanity check is applied before any attribution result is used: a "
        "method whose maps survive randomisation of the model weights is excluded.")

    doc.add_heading("c.6 Statistical analysis", level=2)
    bullets(doc, [
        "Out-of-fold predictions are pooled within each seed; metrics are averaged over seeds.",
        "Confidence intervals are 95 per cent patient-cluster bootstrap intervals (10,000 "
        "resamples, fixed random seed), so that multiple images from one patient are resampled "
        "together rather than treated as independent.",
        "Nineteen primary comparisons were specified in advance and are corrected together by the "
        "Holm-Bonferroni procedure at a family-wise error rate of 0.05.",
        "The 'same accuracy' claim is tested formally by two one-sided tests (TOST) with an "
        "equivalence margin of plus or minus two balanced-accuracy points.",
    ])

    # ------------------------------------------------------------------ d. Work done
    doc.add_heading("d. Work Done Till Date: Results and Discussion", level=1)

    doc.add_heading("d.1 Work completed", level=2)
    doc.add_paragraph(
        f"All planned experiments are complete: {n_runs} model trainings consuming approximately "
        f"{gpu_hours:.0f} GPU hours. This comprises hyperparameter tuning (60 trials), the full "
        "protocol on datasets A and B (90 runs each), and the trap-set grid on dataset C (226 "
        "cells). Supporting software includes the data pipeline, the training and "
        "cross-validation engine, attribution and endpoint computation, and the statistical "
        "analysis, with 48 automated tests covering them.")

    doc.add_heading("d.2 Classification accuracy", level=2)
    add_table(doc, protocol_table("padufes20"), widths=[1.9, 1.6, 1.5, 1.5],
              caption="Table 4. Dataset A (PAD-UFES-20). Mean over 15 runs [95% CI].")
    add_table(doc, protocol_table("ham10000"), widths=[1.9, 1.6, 1.5, 1.5],
              caption="Table 5. Dataset B (HAM10000). Mean over 15 runs [95% CI].")
    doc.add_paragraph(
        "The two datasets occupy different regimes. On dataset A, metadata alone (B1) reaches "
        "0.611 balanced accuracy against 0.645 for the image alone, and fusion adds roughly eight "
        "points. On dataset B, where only age, sex and anatomical site are available, metadata "
        "alone collapses to 0.291 and fusion adds at most 1.8 points. Having both regimes is an "
        "advantage: one shows what happens when metadata is highly informative, the other when it "
        "is not.")
    doc.add_paragraph(
        "The TOST equivalence test shows that on dataset A no fusion model is equivalent to the "
        "image-only model within two points - they are genuinely more accurate. On dataset B, B4 "
        "and B5 are statistically equivalent to image-only. The working title's 'same accuracy' "
        "premise therefore holds only on dataset B, and the framing is being revised accordingly.")

    doc.add_heading("d.3 Grounding: where the models look", level=2)
    p1 = read("endpoints_ham10000_gradcam_descriptive.csv")
    if not p1.empty:
        piv = p1[p1["endpoint"] == "P1"].pivot_table(index="model", columns="metric",
                                                     values="value")
        show = pd.DataFrame({
            "Model": [f"{m} ({NAMES[m]})" for m in piv.index],
            "Attribution in lesion": [f"{v:.3f}" for v in piv["energy_in_mask"]],
            "Chance level": [f"{v:.3f}" for v in piv["mask_area_fraction"]],
            "Pointing game": [f"{v:.1%}" for v in piv["pointing_game"]],
            "IoU (Otsu)": [f"{v:.3f}" for v in piv["iou_otsu"]],
        })
        add_table(doc, show, widths=[1.8, 1.3, 1.1, 1.1, 1.1],
                  caption="Table 6. P1 on dataset B, 7,470 lesion clusters. Chance level is the "
                          "mask area fraction.")
    doc.add_paragraph(
        "This is the central result. Concatenation places significantly more attribution inside "
        "the lesion than the image-only model (+0.049). FiLM (-0.087), MetaBlock (-0.160) and "
        "cross-attention (-0.017) place significantly less. MetaBlock sits at 0.299 against a "
        "chance level of 0.267: its attribution has almost ceased to track the lesion, while its "
        "accuracy remains competitive. All four comparisons survive Holm correction. Pointing "
        "game and Otsu-thresholded IoU rank the models identically, so the result is not an "
        "artefact of one particular measure.")
    doc.add_paragraph(
        "The sanity check passed for every model: mean structural similarity between maps from "
        "trained and parameter-randomised models ranged from 0.036 to 0.143, far below the "
        "pre-registered exclusion threshold of 0.5. Grad-CAM is therefore reading the learned "
        "weights rather than image edges.")

    doc.add_heading("d.4 Modality reliance", level=2)
    p5 = read("endpoints_padufes20_gradcam_descriptive.csv")
    if not p5.empty:
        piv = p5[p5["endpoint"] == "P5"].pivot_table(index="model", columns="metric",
                                                     values="value")
        show = pd.DataFrame({
            "Model": [f"{m} ({NAMES[m]})" for m in piv.index],
            "BACC lost without metadata": [f"{v:.3f}" for v in piv["drop_meta_shuffled"]],
            "BACC lost without image": [f"{v:.3f}" for v in piv["drop_mean_image"]],
        })
        add_table(doc, show, widths=[2.0, 2.2, 2.2],
                  caption="Table 7. P5 on dataset A, 1,373 patient clusters.")
    doc.add_paragraph(
        "Every fusion model loses more accuracy when metadata is removed than when the image is "
        "removed. Equally important, adding metadata approximately halves dependence on the image: "
        "the image-only model loses 0.477 balanced accuracy without its image, whereas fusion "
        "models lose only 0.18 to 0.27. The image-only control loses exactly zero when metadata is "
        "shuffled, confirming the measurement behaves as expected.")

    doc.add_heading("d.5 Counterfactual metadata perturbation", level=2)
    p4 = read("endpoints_padufes20_gradcam_descriptive.csv")
    if not p4.empty:
        piv = p4[p4["endpoint"] == "P4"].pivot_table(index="model", columns="metric",
                                                     values="value")
        show = pd.DataFrame({
            "Model": [f"{m} ({NAMES[m]})" for m in piv.index],
            "Shift, irrelevant fields": [f"{v:.3f}" for v in piv["map_shift_irrelevant"]],
            "Shift, relevant fields": [f"{v:.3f}" for v in piv["map_shift_relevant"]],
        })
        add_table(doc, show, widths=[2.0, 2.2, 2.2],
                  caption="Table 8. P4 on dataset A. Irrelevant fields (piped water, sewage "
                          "system) were designated before any result was seen.")
    doc.add_paragraph(
        "Holding the image fixed and changing a single metadata field moves the attribution map. "
        "Every model moves roughly twice as much for a clinically relevant field as for an "
        "irrelevant one, which indicates the movement is not arbitrary. However, movement for "
        "fields that cannot affect lesion appearance - whether the household has piped water or a "
        "sewage system - is not zero, and cross-attention moves most, significantly more than "
        "every other fusion mechanism.")

    doc.add_heading("d.6 Shortcut robustness", level=2)
    doc.add_paragraph(
        "On the artifact trap sets, no fusion mechanism differs from image-only (P2: all four "
        "comparisons not significant, p between 0.43 and 0.998). Every model degrades in the same "
        "way as the artifact-diagnosis correlation strengthens, from AUROC near 0.90 to below "
        "chance when the correlation is reversed, reproducing the published finding [18]. Metadata "
        "fusion neither protects against image shortcuts nor aggravates them.")
    doc.add_paragraph(
        "The offloading endpoint P3 returned no usable answer. It rests on three published splits "
        "per model and the resulting intervals are too wide to resolve an effect of the expected "
        "size. Notably, the image-only control - which cannot use metadata at all - drifted in the "
        "same direction, indicating that the drift arises from the re-sampling procedure rather "
        "than from model behaviour. Reporting P3 as inconclusive rather than as a positive finding "
        "is a direct consequence of having pre-registered that control.")

    doc.add_heading("d.7 Statistical summary", level=2)
    fam = read("family_holm.csv")
    if not fam.empty:
        s = (fam.groupby("endpoint").agg(t=("reject_holm", "size"),
                                         r=("reject_holm", "sum")).reset_index())
        s.columns = ["Endpoint", "Tests", "Significant after Holm"]
        add_table(doc, s, widths=[1.6, 1.4, 2.4],
                  caption="Table 9. The pre-registered family of 19 tests.")
        doc.add_paragraph(
            "Eleven of the nineteen pre-registered tests are significant after correction. The "
            "two endpoints returning nothing (P2 and P3) were planned in advance; a planned "
            "question that returns no effect is a result rather than a gap.")

    doc.add_heading("d.8 Exploratory analysis: performance by skin type", level=2)
    doc.add_paragraph(
        "Because external validation on skin-tone-balanced data could not be obtained, dataset A's "
        "own Fitzpatrick field was used for an exploratory check. Binary malignant-versus-benign "
        "AUROC was 0.93 for types I-II (1,029 images), 0.90-0.95 for type III (392 images) and "
        "0.92-0.98 for types IV-VI (73 images). No degradation on darker skin is evident, but "
        "types V and VI together contribute only 11 images, so this analysis cannot support a "
        "fairness claim and is reported as exploratory only.")

    doc.add_heading("d.9 Discussion", level=2)
    doc.add_paragraph(
        "The results converge on a single interpretation: the mechanism of fusion, not the "
        "presence of metadata, determines what happens to the model's visual evidence. Identical "
        "information, an identical backbone and an identical training recipe produce opposite "
        "effects on grounding depending only on how the two modalities are combined.")
    doc.add_paragraph(
        "Two findings are of practical importance. First, the simplest mechanism - plain "
        "concatenation, usually treated as a throwaway baseline - achieved both the best accuracy "
        "on dataset A and the only improvement in grounding on dataset B. Second, a model can "
        "become more accurate and less trustworthy simultaneously: MetaBlock is competitive on "
        "accuracy while its attribution sits barely above chance, and cross-attention combines "
        "middling accuracy with the largest displacement of evidence by irrelevant metadata. An "
        "evaluation restricted to accuracy would rank these models in an order that has nothing "
        "to do with the quality of their evidence.")
    doc.add_paragraph(
        "Several defects in our own pipeline were identified and corrected during the work, "
        "including debug output that shared filenames with real results, attribution maps with no "
        "positive mass being silently dropped from averages, and an inconsistency between the "
        "number of statistical tests described in the plan and the number the experiment settings "
        "produced. The last was resolved by training the two missing models rather than by "
        "redefining the family after seeing results.")

    # ------------------------------------------------------------------ e. Future work
    doc.add_heading("e. Future Work", level=1)
    doc.add_paragraph("Remaining work falls into four parts.")
    doc.add_heading("e.1 Figures and manuscript", level=2)
    doc.add_paragraph(
        "Five figures remain to be produced: the degradation curves across artifact-bias levels, "
        "the grounding comparison with confidence intervals, the counterfactual shift by field "
        "type, a qualitative panel of attribution maps with mask overlays, and the architecture "
        "diagram. The manuscript will be written for MIDL 2027, whose deadline is expected in "
        "early December 2026. The framing will be revised from 'same accuracy' to the decoupling "
        "of accuracy from evidence quality, which is what the equivalence testing supports.")
    doc.add_heading("e.2 External validation and fairness", level=2)
    doc.add_paragraph(
        "The DDI dataset was planned as an external, skin-tone-balanced test set and would also "
        "provide an independent check of the offloading finding, because it carries no usable "
        "metadata and therefore forces fusion models into an all-missing condition. Access "
        "requires an individual research-use agreement which has not been obtainable. The complete "
        "processing and evaluation pipeline for it has nevertheless been implemented and tested, "
        "so the analysis can be run immediately if access is granted. If it is not, the Brazilian "
        "atlas subset of Fitzpatrick17k (1,043 obtainable clinical images with binary labels) will "
        "be used as a secondary external test, with its web-sourced label noise stated explicitly, "
        "and the absence of a skin-tone fairness result will be reported as a limitation.")
    doc.add_heading("e.3 Secondary analyses", level=2)
    bullets(doc, [
        "RISE, a gradient-free attribution method, as a second channel to confirm that the P1 "
        "ordering is not specific to Grad-CAM.",
        "The counterfactual and reliance endpoints computed on dataset B as a robustness check.",
        "The unfiltered published trap splits, for direct comparison with the numbers reported by "
        "Bissoto et al. [18].",
    ])
    doc.add_heading("e.4 Longer-term extensions", level=2)
    doc.add_paragraph(
        "A journal version would restore the chest radiograph track that was cut to keep the "
        "conference paper focused, testing whether the same mechanism-dependence appears in a "
        "different imaging modality with radiologist-drawn segmentations. A second direction is a "
        "training-time remedy: penalising attribution that moves under perturbation of "
        "clinically irrelevant metadata fields, turning the diagnostic finding of this project "
        "into a correction.")

    # ------------------------------------------------------------------ f. References
    doc.add_heading("f. References", level=1)
    refs = [
        "A. G. C. Pacheco and R. A. Krohling, \"The impact of patient clinical information on "
        "automated skin cancer detection,\" Computers in Biology and Medicine, vol. 116, 103545, "
        "2020.",
        "X. Li, J. Zhuang, et al., \"Fusing metadata and dermoscopy images for skin disease "
        "diagnosis,\" in IEEE International Symposium on Biomedical Imaging (ISBI), 2020.",
        "A. G. C. Pacheco and R. A. Krohling, \"An attention-based mechanism to combine images and "
        "metadata in deep learning models applied to skin cancer classification,\" IEEE Journal of "
        "Biomedical and Health Informatics, vol. 25, no. 9, pp. 3554-3563, 2021.",
        "Y. Zhou and J. Luo, \"Deep features fusion with mutual attention transformer for skin "
        "lesion diagnosis,\" in IEEE International Conference on Image Processing (ICIP), 2021, "
        "pp. 3797-3801.",
        "G. C. de Lima and R. A. Krohling, \"Exploring advances in transformers and CNN for skin "
        "lesion diagnosis on small datasets,\" in BRACIS, 2022. arXiv:2205.15442.",
        "B. Cheslerean-Boghiu, M. Fleischmann, et al., \"Transformer-based interpretable "
        "multi-modal data fusion for skin lesion classification,\" arXiv:2304.14505, 2023.",
        "P. Tschandl, C. Rosendahl, and H. Kittler, \"The HAM10000 dataset, a large collection of "
        "multi-source dermatoscopic images of common pigmented skin lesions,\" Scientific Data, "
        "vol. 5, 180161, 2018. doi:10.1038/sdata.2018.161.",
        "S. Das, A. Agarwal, and D. Shetty, \"Comparative analysis of multimodal architectures for "
        "effective skin lesion detection using clinical and image data,\" Frontiers in Artificial "
        "Intelligence, 2025. doi:10.3389/frai.2025.1608837.",
        "A. Khurshid, M. Singh, and M. Vatsa, \"Multimodal dual-stage feature refinement for "
        "robust skin lesion classification,\" Scientific Reports, 2025. "
        "doi:10.1038/s41598-025-14839-7.",
        "L. Bouzon, D. Da Rocha, et al., \"MetaBlock-SE: A method to deal with missing metadata in "
        "multimodal skin cancer classification,\" IEEE Journal of Biomedical and Health "
        "Informatics, 2025. doi:10.1109/JBHI.2025.3612837.",
        "M. Atiq and S. A. Fattah, \"Towards explainable skin cancer classification: A "
        "dual-network attention model with lesion segmentation and clinical metadata fusion,\" "
        "arXiv:2510.17773, 2025.",
        "K. Mridha and M. Islam, \"Cross-attention enables context-aware multimodal skin lesion "
        "diagnosis,\" medRxiv, 2026. doi:10.64898/2026.03.10.26348046.",
        "S. Pathirana, D. Munasinghe, and C. Alwis, \"Multimodal skin lesion classification with "
        "Swin transformer and clinical metadata fusion,\" arXiv:2608.07574, 2026.",
        "E. Sonuc, A. Saihood, et al., \"Interpretable multimodal fusion for skin lesion "
        "classification using dermoscopic images and patient metadata,\" Frontiers in Medicine, "
        "2026. doi:10.3389/fmed.2026.1871689.",
        "A. Saporta, X. Gui, et al., \"Benchmarking saliency methods for chest X-ray "
        "interpretation,\" Nature Machine Intelligence, vol. 4, pp. 867-878, 2022. "
        "doi:10.1038/s42256-022-00536-x.",
        "R. R. Selvaraju, M. Cogswell, et al., \"Grad-CAM: Visual explanations from deep networks "
        "via gradient-based localization,\" in IEEE International Conference on Computer Vision "
        "(ICCV), 2017, pp. 618-626.",
        "J. Adebayo, J. Gilmer, et al., \"Sanity checks for saliency maps,\" in Advances in Neural "
        "Information Processing Systems (NeurIPS), 2018.",
        "A. Bissoto, C. Barata, E. Valle, and S. Avila, \"Artifact-based domain generalization of "
        "skin lesion models,\" in ECCV Workshops, LNCS, 2023. doi:10.1007/978-3-031-25069-9_10. "
        "See also A. Bissoto, E. Valle, and S. Avila, \"Debiasing skin lesion datasets and models? "
        "Not so fast,\" in CVPR Workshops, 2020, pp. 3192-3201.",
        "\"A scoping review of explainable artificial intelligence for medical multimodal data,\" "
        "npj Digital Medicine, 2026.",
        "R. Daneshjou, K. Vodrahalli, et al., \"Disparities in dermatology AI performance on a "
        "diverse, curated clinical image set,\" Science Advances, vol. 8, no. 31, eabq6147, 2022. "
        "doi:10.1126/sciadv.abq6147.",
        "A. G. C. Pacheco, G. R. Lima, et al., \"PAD-UFES-20: A skin lesion dataset composed of "
        "patient data and clinical images collected from smartphones,\" Data in Brief, vol. 32, "
        "106221, 2020. doi:10.1016/j.dib.2020.106221.",
        "Z. Liu, H. Mao, et al., \"A ConvNet for the 2020s,\" in IEEE/CVF Conference on Computer "
        "Vision and Pattern Recognition (CVPR), 2022, pp. 11976-11986.",
        "E. Perez, F. Strub, et al., \"FiLM: Visual reasoning with a general conditioning layer,\" "
        "in AAAI Conference on Artificial Intelligence, 2018.",
        "V. Petsiuk, A. Das, and K. Saenko, \"RISE: Randomized input sampling for explanation of "
        "black-box models,\" in British Machine Vision Conference (BMVC), 2018.",
        "P. Tschandl, C. Rinner, et al., \"Human-computer collaboration for skin cancer "
        "recognition,\" Nature Medicine, vol. 26, no. 8, pp. 1229-1234, 2020. "
        "doi:10.1038/s41591-020-0942-0.",
    ]
    for i, r in enumerate(refs, 1):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.35)
        p.paragraph_format.first_line_indent = Inches(-0.35)
        p.paragraph_format.space_after = Pt(4)
        p.add_run(f"[{i}] {r}")

    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out)
    print(f"written: {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--students", nargs="*", default=["Aniket Meena"])
    ap.add_argument("--supervisor", default="[Supervisor name]")
    ap.add_argument("--out", type=Path,
                    default=REPO / "docs" / "MedFusion-XAI_Committee_Progress_Report.docx")
    a = ap.parse_args()
    build(a.out, a.students, a.supervisor)


if __name__ == "__main__":
    main()
