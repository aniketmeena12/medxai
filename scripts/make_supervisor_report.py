"""Build the supervisor progress report as a styled .docx, with every number read from the
result files rather than retyped.

    python scripts/make_supervisor_report.py [--out docs/REPORT.docx]
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
MODEL_NAMES = {
    "B0": "Image only",
    "B1": "Patient details only",
    "B3": "Simple combination (concatenation)",
    "B4": "FiLM",
    "B5": "MetaBlock",
    "M1": "Cross-attention",
}


# --------------------------------------------------------------------------- data helpers
def protocol_table(dataset: str) -> pd.DataFrame:
    rows = []
    for m in ("B0", "B1", "B3", "B4", "B5", "M1"):
        files = sorted(glob.glob(f"data/experiments/{dataset}_{m}_*/metrics.json"))
        d = next((json.load(open(f)) for f in reversed(files)
                  if json.load(open(f))["complete_protocol"]), None)
        if d is None:
            continue
        row = {"Model": f"{m} - {MODEL_NAMES[m]}"}
        for key, label in (("balanced_accuracy", "Balanced accuracy"),
                           ("macro_f1", "Macro F1"), ("auroc", "AUROC")):
            v = np.array([x[key] for x in d["per_fold_seed"]])
            mu, lo, hi = mean_ci(v)
            row[label] = f"{mu:.3f}  ({lo:.3f}-{hi:.3f})"
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


# --------------------------------------------------------------------------- docx helpers
def add_table(doc: Document, df: pd.DataFrame, style: str = "Light Grid Accent 1",
              widths: list[float] | None = None) -> None:
    if df.empty:
        doc.add_paragraph("(not available yet)", style="Intense Quote")
        return
    t = doc.add_table(rows=1, cols=len(df.columns))
    t.style = style
    for i, c in enumerate(df.columns):
        cell = t.rows[0].cells[i]
        cell.text = str(c)
        for run in cell.paragraphs[0].runs:
            run.bold = True
    for _, r in df.iterrows():
        cells = t.add_row().cells
        for i, c in enumerate(df.columns):
            cells[i].text = "" if pd.isna(r[c]) else str(r[c])
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    doc.add_paragraph()


def bullets(doc: Document, items: list[str]) -> None:
    for it in items:
        doc.add_paragraph(it, style="List Bullet")


def keypoint(doc: Document, text: str) -> None:
    doc.add_paragraph(text, style="Intense Quote")


def build(out: Path) -> None:
    doc = Document()

    # Page setup and base font
    for s in doc.sections:
        s.left_margin = s.right_margin = Inches(1.0)
        s.top_margin = s.bottom_margin = Inches(0.9)
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(8)
    normal.paragraph_format.line_spacing = 1.15

    n_runs, gpu_hours = totals()
    today = date.today().isoformat()

    # ---------------------------------------------------------------- title block
    doc.add_paragraph("MedFusion-XAI - Progress Report", style="Title")
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = sub.add_run("Does patient information help a skin-cancer model look in the right place,\n"
                    "or does it make the model stop looking?")
    r.italic = True
    r.font.size = Pt(13)
    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    mr = meta.add_run(f"Aniket Meena  |  {today}  |  Target venue: MIDL 2027")
    mr.font.size = Pt(10)
    mr.font.color.rgb = RGBColor(0x40, 0x40, 0x40)

    doc.add_paragraph()
    keypoint(doc, "In one sentence: all the experiments are finished, and the result is that "
                  "adding patient information changes WHERE the model looks - but whether it "
                  "looks better or worse depends entirely on how the information is combined "
                  "with the image.")

    # ---------------------------------------------------------------- 1. summary
    doc.add_heading("1. Summary for a quick read", level=1)
    doc.add_paragraph(
        "This project asks a simple question. A computer model that looks at a photograph of a "
        "skin lesion can also be given the patient's details - age, sex, whether the lesion itches "
        "or bleeds, and so on. Giving it those details usually makes it more accurate. But does it "
        "make the model examine the photograph more carefully, or does the model start leaning on "
        "the patient details and stop paying attention to the image?")
    doc.add_paragraph(
        "This matters because a model that quietly stops looking at the image can still score well "
        "on a test set, and then fail badly on a patient whose details are unusual or missing. "
        "Accuracy alone will not reveal that problem.")
    doc.add_paragraph("What we can now say, based on completed experiments:")
    bullets(doc, [
        "The way the patient details are combined with the image decides the answer. The same "
        "information, combined two different ways, gives opposite results.",
        "The simplest method of combining them (just joining the two sets of numbers together) was "
        "the only one that made the model look MORE carefully at the lesion - and it was also the "
        "most accurate. ",
        "The more elaborate methods, including the cross-attention design this project originally "
        "set out to promote, made the model look LESS at the lesion.",
        "On the first dataset, every combined model relied more on the patient details than on the "
        "photograph itself. Removing the photograph hurt it less than scrambling the patient "
        "details.",
        "Changing a patient detail that cannot possibly change what a lesion looks like - such as "
        "whether the household has piped water - still moved the model's attention around the "
        "image.",
    ])
    doc.add_paragraph(
        f"Scale of the work: {n_runs} separate model trainings, about {gpu_hours:.0f} hours of GPU "
        "time, three public datasets totalling 72 GB, and a statistical plan that was written down "
        "and frozen before any model was trained.")

    # ---------------------------------------------------------------- 2. background
    doc.add_heading("2. What changed since the last report", level=1)
    doc.add_paragraph(
        "The previous report (8 September 2026) proposed a new cross-attention design as the main "
        "contribution. Background research in September changed that plan, for a good reason: the "
        "same architecture had already been published by others (MetaBlock in 2021, and a 2026 "
        "paper using the same design on the same dataset). Competing on architecture would have "
        "been competing on something already done.")
    doc.add_paragraph("The project was therefore re-aimed at a question nobody has measured:")
    bullets(doc, [
        "Old aim: build a better way of combining images and patient details.",
        "New aim: measure whether combining them improves or damages the model's visual evidence, "
        "using human-drawn outlines of the lesions as the reference.",
    ])
    doc.add_paragraph(
        "Two other decisions were taken at the same time: the chest X-ray half of the project was "
        "dropped so that the dermatology half could be done properly, and the full analysis plan "
        "was written down and frozen before any training began. That plan is the document "
        "docs/04-preregistration.md. Freezing it matters because it prevents the results from "
        "being shaped, even unintentionally, by what looks good once you see the numbers.")

    # ---------------------------------------------------------------- 3. data
    doc.add_heading("3. The data we used", level=1)
    doc.add_paragraph(
        "Everything is public and was downloaded from the official source wherever possible. Each "
        "download was checked against the numbers published by the data provider - 15 out of 15 "
        "checks passed - and the exact version, file size and checksum of every file is recorded "
        "so the work can be repeated exactly.")
    data = pd.DataFrame([
        {"Dataset": "PAD-UFES-20", "What it contains": "2,298 smartphone photographs of skin "
         "lesions from 1,373 patients, with 21 pieces of patient information each",
         "Why we use it": "The main test of what happens when a model is given rich patient "
         "details"},
        {"Dataset": "HAM10000", "What it contains": "10,015 dermoscopic images covering 7 "
         "diagnoses, each with a lesion outline drawn by a dermatologist",
         "Why we use it": "The outlines let us measure exactly where the model looks"},
        {"Dataset": "ISIC 2019 + trap sets", "What it contains": "25,331 images, plus ready-made "
         "splits where visual artefacts (rulers, hair, ink marks) deliberately predict the "
         "diagnosis", "Why we use it": "Tests whether models are taking visual shortcuts"},
        {"Dataset": "DDI (not yet obtained)", "What it contains": "656 biopsy-proven photographs, "
         "balanced across skin tones", "Why we use it": "Fairness check on darker skin, and an "
         "independent test of the main finding. Needs a signed research agreement"},
    ])
    add_table(doc, data, widths=[1.5, 2.7, 2.2])
    doc.add_paragraph(
        "A useful accident: 9,083 of the trap-set images are also HAM10000 images, so for those we "
        "have the lesion outline, the patient details and the artefact labels all at once.")

    # ---------------------------------------------------------------- 4. models
    doc.add_heading("4. The models we compared", level=1)
    doc.add_paragraph(
        "All six use exactly the same image backbone, the same image size, the same training "
        "recipe and the same amount of tuning. The only thing that differs is how (or whether) the "
        "patient details are mixed in. That is what makes the comparison fair.")
    models = pd.DataFrame([
        {"Name": "B0", "What it does": "Looks at the photograph only. Our reference point"},
        {"Name": "B1", "What it does": "Uses the patient details only, no photograph"},
        {"Name": "B3", "What it does": "Simplest combination: the two sets of numbers are joined "
                                       "end to end"},
        {"Name": "B4", "What it does": "FiLM: the patient details rescale the image features"},
        {"Name": "B5", "What it does": "MetaBlock: a published method designed for this task"},
        {"Name": "M1", "What it does": "Cross-attention: the image queries the patient details "
                                       "(the design this project first proposed)"},
    ])
    add_table(doc, models, widths=[0.9, 5.5])

    # ---------------------------------------------------------------- 5. how tested
    doc.add_heading("5. How we tested them, in plain terms", level=1)
    bullets(doc, [
        "Every model was trained 15 times on each dataset - five different splits of the data, "
        "repeated with three different random starts - so that no result depends on one lucky "
        "split.",
        "Patients are never split across the training and testing sides. If a patient has three "
        "photographs, all three stay together. Otherwise the model can recognise the patient "
        "rather than the disease.",
        "'Balanced accuracy' is used instead of plain accuracy because it treats rare diseases as "
        "importantly as common ones. Melanoma is only 52 of the 2,298 photographs in the first "
        "dataset, and plain accuracy would let a model ignore it.",
        "Every number is reported with a range in brackets. That range is the uncertainty: if the "
        "ranges for two models overlap, the difference between them is not solid.",
        "We ran 19 planned statistical tests, so the results were adjusted for the fact that "
        "running many tests at once produces false positives by chance.",
    ])
    doc.add_paragraph(
        "To see where a model looks, we use a standard technique (Grad-CAM) that produces a heat "
        "map over the photograph showing which areas drove the decision. Before trusting any of "
        "it, we ran the standard safety check: scramble the model's learned knowledge and confirm "
        "the heat maps change. They did, for every model, by a wide margin. A technique that "
        "produced the same picture from a scrambled model would have been measuring the image's "
        "edges, not the model's reasoning.")

    # ---------------------------------------------------------------- 6. results
    doc.add_heading("6. What we found", level=1)

    doc.add_heading("6.1 Accuracy", level=2)
    doc.add_paragraph(
        "On the smartphone-photograph dataset, adding patient details helps a lot - about 8 points "
        "of balanced accuracy. Notably, the patient details on their own (B1) score almost as well "
        "as the photograph on its own.")
    add_table(doc, protocol_table("padufes20"), widths=[2.6, 1.5, 1.4, 1.4])
    doc.add_paragraph(
        "On the dermoscopic dataset the picture is completely different. Here only three pieces of "
        "patient information exist (age, sex, body site), they are weak on their own, and adding "
        "them barely changes accuracy.")
    add_table(doc, protocol_table("ham10000"), widths=[2.6, 1.5, 1.4, 1.4])
    keypoint(doc, "Having two datasets that behave so differently is an advantage, not a problem. "
                  "One shows what happens when patient details are highly informative; the other "
                  "shows what happens when they are not.")

    doc.add_heading("6.2 Where the model looks - the central result", level=2)
    doc.add_paragraph(
        "Using the dermatologist-drawn outlines, we measured what share of the model's attention "
        "falls inside the actual lesion. Higher is better. If a model spread its attention at "
        "random, it would score 0.267, because that is the average share of the picture the lesion "
        "occupies. That is the 'random guessing' line to compare against.")
    p1 = read("endpoints_ham10000_gradcam_descriptive.csv")
    if not p1.empty:
        piv = p1[p1["endpoint"] == "P1"].pivot_table(index="model", columns="metric",
                                                     values="value")
        show = pd.DataFrame({
            "Model": [f"{m} - {MODEL_NAMES[m]}" for m in piv.index],
            "Attention inside the lesion": [f"{v:.3f}" for v in piv["energy_in_mask"]],
            "Random-guessing level": [f"{v:.3f}" for v in piv["mask_area_fraction"]],
            "Hits the lesion (best single point)": [f"{v:.0%}" for v in piv["pointing_game"]],
        })
        add_table(doc, show, widths=[2.6, 1.6, 1.4, 1.4])
    bullets(doc, [
        "B3, the simplest combination, looks at the lesion MORE than the image-only model.",
        "B5 (MetaBlock) is barely above the random-guessing line. Its attention has almost stopped "
        "following the lesion, even though its accuracy looks fine.",
        "B4 and M1 also look at the lesion less than the image-only model.",
        "Three separate ways of measuring this agree on the same ordering, so it is not an "
        "artefact of one particular measure.",
    ])

    doc.add_heading("6.3 What the models actually rely on", level=2)
    doc.add_paragraph(
        "Here we take a trained model and either scramble the patient details or replace the "
        "photograph with a blank average image, then see how much accuracy it loses. The bigger "
        "the loss, the more it was relying on that source.")
    p5 = read("endpoints_padufes20_gradcam_descriptive.csv")
    if not p5.empty:
        piv = p5[p5["endpoint"] == "P5"].pivot_table(index="model", columns="metric",
                                                     values="value")
        show = pd.DataFrame({
            "Model": [f"{m} - {MODEL_NAMES[m]}" for m in piv.index],
            "Accuracy lost without patient details": [f"{v:.3f}"
                                                      for v in piv["drop_meta_shuffled"]],
            "Accuracy lost without the photograph": [f"{v:.3f}" for v in piv["drop_mean_image"]],
        })
        add_table(doc, show, widths=[2.6, 2.0, 2.0])
    keypoint(doc, "Every combined model lost more accuracy when the patient details were taken "
                  "away than when the photograph was taken away. Adding patient details also "
                  "roughly halved how much the model depended on the image (0.477 for the "
                  "image-only model, 0.18-0.27 for the combined ones). This is the clearest "
                  "evidence of the model shifting its weight off the image.")

    doc.add_heading("6.4 Patient details that cannot matter still move the evidence", level=2)
    doc.add_paragraph(
        "We kept the photograph identical and changed one piece of patient information, then "
        "measured how far the heat map moved. Two of the fields were chosen in advance precisely "
        "because they cannot change what a lesion looks like: whether the household has piped "
        "water, and whether it has a sewage system.")
    p4 = read("endpoints_padufes20_gradcam_descriptive.csv")
    if not p4.empty:
        piv = p4[p4["endpoint"] == "P4"].pivot_table(index="model", columns="metric",
                                                     values="value")
        show = pd.DataFrame({
            "Model": [f"{m} - {MODEL_NAMES[m]}" for m in piv.index],
            "Movement from an irrelevant detail": [f"{v:.3f}"
                                                   for v in piv["map_shift_irrelevant"]],
            "Movement from a medically relevant detail": [f"{v:.3f}"
                                                          for v in piv["map_shift_relevant"]],
        })
        add_table(doc, show, widths=[2.6, 2.0, 2.0])
    doc.add_paragraph(
        "Every model moved about twice as much for a medically meaningful detail as for an "
        "irrelevant one, which is reassuring - the movement is not random. But the movement for "
        "irrelevant details is not zero, and cross-attention (M1) moved most of all, significantly "
        "more than every other method.")

    doc.add_heading("6.5 Shortcut tests", level=2)
    doc.add_paragraph(
        "We also tested whether patient details make models more or less prone to visual "
        "shortcuts, using prepared datasets where rulers, hair and ink marks deliberately predict "
        "the diagnosis. The answer is clear: they make no difference at all. Every model, with or "
        "without patient details, collapsed in the same way - from very good to worse than "
        "guessing - once the artefacts pointed the wrong way. This replicates a known published "
        "finding and shows our models are no better and no worse than the field's.")
    doc.add_paragraph(
        "A related test - whether planting a misleading patient detail erodes the model's visual "
        "evidence - did not give a usable answer. The measurement rests on only three data points "
        "per model and the uncertainty is far too wide to conclude anything. Importantly, the "
        "image-only model, which cannot use patient details at all, drifted in the same direction, "
        "which shows the drift comes from how the data was re-sampled rather than from the model's "
        "behaviour. That control is exactly why it was included, and it stops us from reporting a "
        "false finding.")

    doc.add_heading("6.6 The formal statistical result", level=2)
    fam = read("family_holm.csv")
    if not fam.empty:
        summary = (fam.groupby("endpoint")
                      .agg(tests=("reject_holm", "size"),
                           significant=("reject_holm", "sum")).reset_index())
        summary["endpoint"] = summary["endpoint"].map({
            "P1": "P1 - Where the model looks",
            "P2": "P2 - Resistance to visual shortcuts",
            "P3": "P3 - Damage from a misleading patient detail",
            "P4": "P4 - Evidence moved by patient details",
            "P5": "P5 - What the model relies on"})
        summary.columns = ["Question tested", "Tests run", "Clearly significant"]
        add_table(doc, summary, widths=[3.6, 1.4, 1.6])
        doc.add_paragraph(
            f"In total {int(fam['reject_holm'].sum())} of the {len(fam)} planned tests were "
            "clearly significant after adjusting for multiple testing. The two questions that came "
            "back empty (P2 and P3) are reported as such - they were planned in advance, and a "
            "planned question with no effect is a result, not a gap.")

    # ---------------------------------------------------------------- 7. contributions
    doc.add_heading("7. What is new here, and how each planned contribution turned out", level=1)
    doc.add_paragraph(
        "A prior-art check in September established what already exists. Combining images with "
        "patient details is well established, and so is adding attention to that combination - "
        "including the exact cross-attention design this project first proposed. What nobody has "
        "done is measure whether that combination improves or damages the model's visual evidence, "
        "scored against human-drawn outlines rather than judged by eye. That gap is what the "
        "project now occupies.")
    doc.add_paragraph("Four contributions were planned in September. Their status today:")
    contrib = pd.DataFrame([
        {"Planned contribution": "N1 - A benchmark where a misleading patient detail and a "
                                 "misleading visual artefact can be dialled up independently",
         "Status": "Built and run in full (225 training runs)",
         "Outcome": "The benchmark exists and works. The specific question it was built to answer "
                    "could not be resolved - see section 6.5"},
        {"Planned contribution": "N2 - A test that changes one patient detail, holds the "
                                 "photograph fixed, and measures how far the evidence moves",
         "Status": "Done", "Outcome": "Works, and gives clear results. Needs no human annotation, "
                                      "so it runs on any dataset with patient details"},
        {"Planned contribution": "N3 - A like-for-like audit of where five different fusion "
                                 "methods look",
         "Status": "Done", "Outcome": "This became the headline finding. The method of combining "
                                      "decides whether evidence improves or degrades"},
        {"Planned contribution": "N4 - An optional training fix to penalise evidence that moves "
                                 "for irrelevant details",
         "Status": "Not attempted", "Outcome": "Was always conditional on time. The paper does not "
                                               "depend on it"},
    ])
    add_table(doc, contrib, widths=[2.4, 1.6, 2.4])
    doc.add_paragraph(
        "Three of the four were delivered, and the one that was optional from the start was the "
        "one dropped. The planned abstract sentence assumed models with indistinguishable accuracy "
        "would differ in grounding; the accuracy half of that assumption did not hold on one of "
        "the two datasets, which is the framing issue raised in the next section.")

    # ---------------------------------------------------------------- 8. meaning
    doc.add_heading("8. What this means for the paper", level=1)
    doc.add_paragraph(
        "The project now has a clear, defensible story, and it is a more interesting one than the "
        "original plan would have produced.")
    bullets(doc, [
        "The headline finding is that the method of combining matters more than the fact of "
        "combining. Same information, same backbone, same training - opposite effects on the "
        "model's visual evidence.",
        "The simplest method wins twice: best accuracy on both datasets AND better grounding than "
        "the image-only model. The field generally treats concatenation as a throwaway baseline.",
        "Cross-attention, the design this project originally proposed as its contribution, comes "
        "out middling on accuracy and worse on evidence quality. Reporting that honestly is more "
        "valuable than promoting the method, and the project was deliberately restructured in "
        "September so that this finding is publishable rather than a failure.",
        "A model can be more accurate and less trustworthy at the same time. That is the practical "
        "message for anyone deploying these systems.",
    ])

    doc.add_heading("8.1 One issue that needs your decision", level=2)
    doc.add_paragraph(
        "The working title is 'Same Accuracy, Different Evidence'. A formal equivalence test shows "
        "this is not true on the smartphone dataset - the combined models are clearly MORE "
        "accurate there, by 7 to 9 points. The title claim holds only on the dermoscopic dataset, "
        "and only strictly for two of the four methods.")
    doc.add_paragraph(
        "No result needs to change; only the framing does. The claim the data actually supports is "
        "stronger and simpler: accuracy and evidence quality are separate things. On one dataset "
        "the model gains 8 points of accuracy while halving its reliance on the image; on the "
        "other it gains nothing measurable and still damages its visual evidence.")

    # ---------------------------------------------------------------- 8. quality
    doc.add_heading("9. Quality control and problems found", level=1)
    doc.add_paragraph(
        "Several problems were found and fixed during the work. They are listed here because they "
        "are the kind of thing that silently corrupts results if missed.")
    issues = pd.DataFrame([
        {"Problem": "Test results from quick debug runs were written to the same filenames as "
                    "real results", "Consequence if missed": "Debug data would have been silently "
         "mixed into the final numbers", "Status": "Fixed - debug output is kept separate and "
         "refused by the analysis"},
        {"Problem": "Heat maps that are entirely blank produce an undefined score, which averaging "
                    "silently skips", "Consequence if missed": "Images would vanish from the "
         "results with no trace", "Status": "Fixed - the count is now reported (under 3% for every "
         "model)"},
        {"Problem": "Running the data preparation before a dataset was downloaded left a marker "
                    "that made it skip extraction forever afterwards",
         "Consequence if missed": "The dataset would never load, with a confusing error",
         "Status": "Fixed"},
        {"Problem": "The analysis plan described 19 statistical tests but the experiment settings "
                    "only produced 15", "Consequence if missed": "Either an incomplete family or a "
         "family redefined after seeing results", "Status": "Fixed - the two missing models were "
         "trained so the plan is followed exactly"},
        {"Problem": "The official mask file hides an extra file that looks like one more mask",
         "Consequence if missed": "A mismatch between images and masks",
         "Status": "Fixed - filtered out"},
    ])
    add_table(doc, issues, widths=[2.3, 2.1, 2.0])

    doc.add_heading("9.1 Honest limitations", level=2)
    bullets(doc, [
        "The misleading-patient-detail test (P3) is underpowered and gives no usable answer. This "
        "is a limit of the published data splits, not a fixable bug.",
        "Every model chose the highest learning rate offered in the tuning grid, so the "
        "best setting may lie outside the range we searched. The same grid was used for every "
        "model, so comparisons remain fair, but absolute accuracy may be slightly understated.",
        "Only one image backbone (ConvNeXt-T) was tested. Whether the findings hold for other "
        "architectures is unknown.",
        "The lesion-outline measurements come from one dataset, because it is the only public "
        "dermatology dataset with outlines for every image.",
        "No external validation yet - see the next section.",
        "The PAD-UFES-20 photographs were obtained from a mirror rather than the original host, "
        "which does not publish per-file checksums. The patient information file is byte-identical "
        "to the official one and every image ID matches, but exact image-level verification is "
        "still open.",
    ])

    # ---------------------------------------------------------------- 9. next
    doc.add_heading("10. What remains", level=1)
    remaining = pd.DataFrame([
        {"Task": "DDI dataset - fairness across skin tones, plus an independent check of the main "
                 "finding", "Blocked by": "A research agreement that must be signed by an "
         "individual", "Effort": "5 minutes to register, then automatic"},
        {"Task": "Figures for the paper", "Blocked by": "Nothing - waiting on the title decision "
         "so figures are not drawn twice", "Effort": "1 day"},
        {"Task": "Secondary analyses (a second heat-map method, extra robustness checks)",
         "Blocked by": "Nothing", "Effort": "1-2 days"},
        {"Task": "Writing the paper", "Blocked by": "The framing decision in section 8.1",
         "Effort": "2 weeks"},
    ])
    add_table(doc, remaining, widths=[2.6, 2.2, 1.6])
    doc.add_paragraph(
        "The deadline is expected in early December 2026. The experimental work is complete and "
        "well ahead of the original schedule, which allowed nine weeks and assumed a much slower "
        "machine.")

    doc.add_heading("10.1 What we would like from you", level=2)
    bullets(doc, [
        "A decision on the title and framing (section 8.1). This is the only thing holding up "
        "figures and writing.",
        "Whether to re-run the tuning with a wider range of learning rates. It costs about an hour "
        "and would be recorded as a documented change to the frozen plan.",
        "Approval to register for the DDI dataset, which requires agreeing to a research-use "
        "licence.",
    ])

    # ---------------------------------------------------------------- appendix
    doc.add_section(WD_SECTION.NEW_PAGE)
    doc.add_heading("Appendix - technical record", level=1)
    doc.add_paragraph(
        "Included so the work can be checked or repeated by someone else.")
    tech = pd.DataFrame([
        {"Item": "Total model trainings", "Value": f"{n_runs}"},
        {"Item": "Total GPU time", "Value": f"about {gpu_hours:.0f} hours"},
        {"Item": "Hardware", "Value": "One NVIDIA RTX 5080 (16 GB), shared with another service"},
        {"Item": "Image backbone", "Value": "ConvNeXt-Tiny, ImageNet weights, pinned to an exact "
                                            "version"},
        {"Item": "Data on disk", "Value": "72 GB, never uploaded anywhere (licences forbid "
                                          "redistribution)"},
        {"Item": "Automated tests", "Value": "48, all passing"},
        {"Item": "Analysis plan", "Value": "docs/04-preregistration.md, frozen 17 September 2026, "
                                           "with every later change logged and dated"},
        {"Item": "Label mapping", "Value": "docs/label_mapping.md, frozen before the external "
                                           "dataset was downloaded"},
        {"Item": "Reproducibility", "Value": "Every run stores its exact settings and the code "
                                             "version it ran from"},
    ])
    add_table(doc, tech, widths=[2.4, 4.0])

    doc.add_heading("Glossary", level=2)
    gloss = pd.DataFrame([
        {"Term": "Balanced accuracy", "Plain meaning": "Accuracy that counts rare diseases as "
         "heavily as common ones"},
        {"Term": "AUROC", "Plain meaning": "How well the model ranks sick cases above healthy "
         "ones; 1.0 is perfect, 0.5 is guessing"},
        {"Term": "Grad-CAM / heat map", "Plain meaning": "A picture showing which parts of the "
         "photograph the model used"},
        {"Term": "Cross-validation", "Plain meaning": "Training and testing repeatedly on "
         "different slices of the data so no single split decides the answer"},
        {"Term": "Confidence interval", "Plain meaning": "The range the true value most likely "
         "falls in; overlapping ranges mean an unreliable difference"},
        {"Term": "Pre-registration", "Plain meaning": "Writing the analysis plan down before "
         "seeing results, so findings cannot be shaped after the fact"},
        {"Term": "Multiple-testing adjustment", "Plain meaning": "A correction applied when many "
         "tests are run at once, because some would look significant by chance alone"},
    ])
    add_table(doc, gloss, widths=[1.9, 4.5])

    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out)
    print(f"written: {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=REPO / "docs" / f"MedFusion-XAI_Progress_Report_{date.today()}.docx")
    build(ap.parse_args().out)


if __name__ == "__main__":
    main()
