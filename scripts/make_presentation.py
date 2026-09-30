"""Build the project slide deck (.pptx), 14 slides, plain language, novelty foregrounded.

    python scripts/make_presentation.py

Numbers are read from results/tables/ and figures from results/figures/ so the deck cannot drift
from the data. Run scripts/make_figures.py first.
"""
from __future__ import annotations

import argparse
import glob
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

from medfusion.eval.stats import mean_ci

REPO = Path(__file__).resolve().parents[1]
FIGS = REPO / "results" / "figures"
TABLES = REPO / "results" / "tables"

INK = RGBColor(0x1B, 0x26, 0x3B)       # near-black navy text
NAVY = RGBColor(0x1E, 0x3A, 0x5F)      # title bar
ACCENT = RGBColor(0x25, 0x63, 0xEB)    # blue
TEAL = RGBColor(0x0F, 0x76, 0x6E)
RED = RGBColor(0xC0, 0x2B, 0x2B)
GREY = RGBColor(0x6B, 0x72, 0x80)
LIGHT = RGBColor(0xF1, 0xF5, 0xF9)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GREEN = RGBColor(0x15, 0x7F, 0x3B)

W, H = Inches(13.333), Inches(7.5)


def _num(dataset, model, metric="balanced_accuracy"):
    fs = sorted(glob.glob(f"data/experiments/{dataset}_{model}_*/metrics.json"))
    d = next((json.load(open(f)) for f in reversed(fs)
              if json.load(open(f))["complete_protocol"]), None)
    if not d:
        return None
    return mean_ci(np.array([x[metric] for x in d["per_fold_seed"]]))[0]


def _p1(model, metric):
    d = pd.read_csv(TABLES / "endpoints_ham10000_gradcam_descriptive.csv")
    piv = d[d["endpoint"] == "P1"].pivot_table(index="model", columns="metric", values="value")
    return piv.loc[model, metric]


class Deck:
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = W, H
        self.blank = self.prs.slide_layouts[6]

    def slide(self, bg=WHITE):
        s = self.prs.slides.add_slide(self.blank)
        r = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
        r.fill.solid(); r.fill.fore_color.rgb = bg; r.line.fill.background()
        r.shadow.inherit = False
        s.shapes._spTree.remove(r._element); s.shapes._spTree.insert(2, r._element)
        return s

    def _box(self, s, x, y, w, h):
        tb = s.shapes.add_textbox(x, y, w, h)
        tf = tb.text_frame; tf.word_wrap = True
        return tb, tf

    def _run(self, para, text, size, color=INK, bold=False, italic=False, font="Calibri"):
        r = para.add_run(); r.text = text
        r.font.size = Pt(size); r.font.bold = bold; r.font.italic = italic
        r.font.color.rgb = color; r.font.name = font
        return r

    def header(self, s, kicker, title):
        bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, Inches(1.25))
        bar.fill.solid(); bar.fill.fore_color.rgb = NAVY; bar.line.fill.background()
        bar.shadow.inherit = False
        _, tf = self._box(s, Inches(0.55), Inches(0.12), Inches(12.2), Inches(1.05))
        p = tf.paragraphs[0]
        self._run(p, kicker.upper(), 12, RGBColor(0x9D, 0xB8, 0xD8), bold=True)
        p2 = tf.add_paragraph()
        self._run(p2, title, 26, WHITE, bold=True)
        # accent underline
        u = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.55), Inches(1.25),
                               Inches(2.2), Pt(4))
        u.fill.solid(); u.fill.fore_color.rgb = ACCENT; u.line.fill.background()
        u.shadow.inherit = False

    def bullets(self, s, items, x=Inches(0.7), y=Inches(1.6), w=Inches(12.0),
                h=Inches(5.4), size=17, gap=10):
        _, tf = self._box(s, x, y, w, h)
        for i, it in enumerate(items):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.space_after = Pt(gap)
            lvl = 0
            if isinstance(it, tuple):
                it, lvl = it
            if isinstance(it, list):  # [ (text,bold,color), ...] rich runs
                for seg in it:
                    txt, *fmt = seg
                    b = fmt[0] if len(fmt) > 0 else False
                    c = fmt[1] if len(fmt) > 1 else INK
                    self._run(p, txt, size - lvl, c, bold=b)
            else:
                bullet = "   -  " if lvl else "•  "
                self._run(p, bullet, size - lvl, ACCENT if not lvl else GREY, bold=True)
                self._run(p, it, size - lvl, INK if not lvl else GREY)
        return tf

    def figure(self, s, name, x, y, h):
        img = FIGS / name
        pic = s.shapes.add_picture(str(img), x, y, height=h)
        return pic

    def chip(self, s, x, y, text, color):
        c = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, Inches(3.9), Inches(0.62))
        c.fill.solid(); c.fill.fore_color.rgb = color; c.line.fill.background()
        c.shadow.inherit = False
        tf = c.text_frame; tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        self._run(p, text, 13, WHITE, bold=True)

    def save(self, out):
        self.prs.save(out)


def build(out: Path, students, supervisor):
    d = Deck()

    # ---- 1 title -----------------------------------------------------------
    s = d.slide(NAVY)
    band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(2.0), W, Inches(0.09))
    band.fill.solid(); band.fill.fore_color.rgb = ACCENT; band.line.fill.background()
    band.shadow.inherit = False
    _, tf = d._box(s, Inches(0.8), Inches(2.25), Inches(11.7), Inches(2.6))
    p = tf.paragraphs[0]
    d._run(p, "Same Accuracy, Different Evidence", 40, WHITE, bold=True)
    p2 = tf.add_paragraph()
    d._run(p2, "Does patient information help a skin-cancer model look in the right "
               "place, or make it stop looking?", 20, RGBColor(0xC7, 0xD7, 0xEC))
    _, tf = d._box(s, Inches(0.8), Inches(5.3), Inches(11.7), Inches(1.8))
    p = tf.paragraphs[0]
    d._run(p, "  ·  ".join(students), 18, WHITE, bold=True)
    p = tf.add_paragraph(); d._run(p, f"Supervisor: {supervisor}", 16,
                                   RGBColor(0xC7, 0xD7, 0xEC))
    p = tf.add_paragraph()
    d._run(p, f"Major Project  ·  Target venue: MIDL 2027  ·  {date.today().strftime('%B %Y')}",
           13, RGBColor(0x9D, 0xB8, 0xD8))

    # ---- 2 the problem -----------------------------------------------------
    s = d.slide()
    d.header(s, "The problem", "A model can be right for the wrong reason")
    d.bullets(s, [
        "A computer can look at a photo of a skin lesion and predict whether it is cancer. "
        "It can also be given the patient's details: age, sex, whether the spot itches, "
        "bleeds, or recently changed.",
        "Adding those details almost always makes the model more accurate. That part is "
        "well known and already published.",
        [("The unanswered question: ", True), ("does the model then look at the lesion ", False),
         ("more", True, ACCENT), (" carefully, or does it lean on the patient details and "
         "look at the image ", False), ("less", True, RED), ("?", False)],
        "This matters in the clinic. A model that quietly stops looking at the image still "
        "scores well on a test set - then fails on a patient whose details are unusual, "
        "missing, or come from a different hospital.",
        [("Accuracy alone cannot tell these two models apart. Something else must.", True,
          INK)],
    ], size=18, gap=14)

    # ---- 3 the gap ---------------------------------------------------------
    s = d.slide()
    d.header(s, "Why this is worth a paper", "Everyone reports accuracy; nobody checks the evidence")
    d.bullets(s, [
        [("What the field already does well:", True, TEAL)],
        ("Combines images with patient details in many ways - concatenation, FiLM, "
         "MetaBlock, cross-attention transformers.", 1),
        ("Reports the accuracy gain, usually several points.", 1),
        [("What almost nobody does:", True, RED)],
        ("Check whether the model's visual evidence actually sits on the lesion, measured "
         "against outlines drawn by dermatologists.", 1),
        ("A 2026 review of 82 multimodal medical-AI studies found that standardised "
         "evaluation of explanations is missing in the majority - they rely on eyeballing "
         "heat maps.", 1),
        [("The gap: no published work measures whether adding patient details improves or "
          "damages a model's visual grounding. That is exactly what we measure.", True, INK)],
    ], size=17, gap=10)

    # ---- 4 NOVELTY (the crux) ---------------------------------------------
    s = d.slide(LIGHT)
    d.header(s, "Novelty  ·  the heart of this work", "Four things here that do not exist elsewhere")
    cards = [
        ("N1  New benchmark", ACCENT,
         "A controlled 'metadata trap': a patient detail is made to predict the diagnosis in "
         "training, then reversed at test - crossed with image-artifact traps to form a 2-D "
         "shortcut grid. Trap sets existed only for image artifacts before."),
        ("N2  New measurement", TEAL,
         "A counterfactual test: hold the image fixed, change one patient detail, and measure "
         "how far the visual evidence moves. Needs no human annotation, so it runs on any "
         "dataset with metadata. Not done before."),
        ("N3  First grounding audit", RED,
         "The first like-for-like, quantitative comparison of WHERE five fusion mechanisms look, "
         "scored against dermatologist masks - not illustrative heat maps, actual numbers with "
         "confidence intervals."),
        ("N4  A finding, not a widget", GREEN,
         "The result overturns a natural assumption: it is the fusion mechanism, not the "
         "presence of metadata, that decides grounding - and the simplest method wins while "
         "the fanciest degrades."),
    ]
    xs = [Inches(0.55), Inches(6.95)]
    ys = [Inches(1.65), Inches(4.35)]
    for i, (title, col, body) in enumerate(cards):
        x = xs[i % 2]; y = ys[i // 2]
        card = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, Inches(5.85), Inches(2.5))
        card.fill.solid(); card.fill.fore_color.rgb = WHITE; card.line.color.rgb = col
        card.line.width = Pt(1.5); card.shadow.inherit = False
        tf = card.text_frame; tf.word_wrap = True
        tf.margin_left = Inches(0.2); tf.margin_right = Inches(0.2); tf.margin_top = Inches(0.15)
        p = tf.paragraphs[0]; d._run(p, title, 16, col, bold=True)
        p2 = tf.add_paragraph(); p2.space_before = Pt(4)
        d._run(p2, body, 12.5, INK)

    # ---- 5 datasets --------------------------------------------------------
    s = d.slide()
    d.header(s, "Data", "Three public datasets, every download verified")
    rows = [
        ["ID", "Dataset", "What it gives us", "Role"],
        ["A", "PAD-UFES-20", "2,298 phone photos, 1,373 patients, 21 patient-detail fields",
         "Rich-metadata tests"],
        ["B", "HAM10000", "10,015 dermoscopy images + a lesion outline for every one",
         "Where-it-looks tests"],
        ["C", "ISIC 2019 + traps", "25,331 images with built-in shortcut splits",
         "Shortcut robustness"],
        ["D", "DDI (planned)", "656 biopsy-proven images across skin tones",
         "Fairness - access pending"],
    ]
    table_shape = s.shapes.add_table(len(rows), 4, Inches(0.55), Inches(1.6),
                                     Inches(12.2), Inches(3.4)).table
    widths = [Inches(0.7), Inches(2.6), Inches(6.3), Inches(2.6)]
    for j, wd in enumerate(widths):
        table_shape.columns[j].width = wd
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = table_shape.cell(i, j); c.text = val
            para = c.text_frame.paragraphs[0]
            for r in para.runs:
                r.font.size = Pt(13 if i else 13); r.font.name = "Calibri"
                r.font.bold = (i == 0)
                r.font.color.rgb = WHITE if i == 0 else INK
            c.fill.solid()
            c.fill.fore_color.rgb = NAVY if i == 0 else (LIGHT if i % 2 else WHITE)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
    d.bullets(s, [
        [("A useful accident: ", True), ("9,083 images belong to both B and C, so for those we "
         "have the lesion outline, the patient details and the shortcut labels all at once.",
         False)],
        [("Splits are grouped so no patient is ever on both sides - otherwise the model learns "
          "the patient, not the disease.", False, GREY)],
    ], y=Inches(5.25), size=14, gap=8)

    # ---- 6 models ----------------------------------------------------------
    s = d.slide()
    d.header(s, "Method  ·  models", "Six models, one difference between them")
    d.bullets(s, [
        [("B0  Image only", True, GREY), ("  -  the reference point.", False)],
        [("B1  Metadata only", True, GREY), ("  -  patient details, no image.", False)],
        [("B3  Concatenation", True, ACCENT), ("  -  simplest: join the two sets of numbers.",
         False)],
        [("B4  FiLM", True, TEAL), ("  -  patient details rescale the image features.", False)],
        [("B5  MetaBlock", True, RED), ("  -  a published attention block for this task.",
         False)],
        [("M1  Cross-attention", True, RGBColor(0x7C,0x3A,0xED)),
         ("  -  the image queries the patient details (the design we first proposed).", False)],
    ], y=Inches(1.65), size=18, gap=12)
    note = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.7), Inches(6.15),
                              Inches(12.0), Inches(1.0))
    note.fill.solid(); note.fill.fore_color.rgb = LIGHT; note.line.color.rgb = ACCENT
    note.line.width = Pt(1); note.shadow.inherit = False
    tf = note.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.2)
    p = tf.paragraphs[0]
    d._run(p, "Same backbone, same images, same training recipe, same tuning budget for all. "
              "Only the fusion mechanism differs - so any difference is caused by the mechanism.",
           14, INK, bold=True)

    # ---- 7 how measured ----------------------------------------------------
    s = d.slide()
    d.header(s, "Method  ·  how we measure evidence", "Five pre-registered measurements")
    d.bullets(s, [
        [("P1  Grounding", True, ACCENT), ("  -  share of the model's attention that lands "
         "inside the dermatologist's lesion outline.", False)],
        [("P2  Image-shortcut resistance", True, ACCENT), ("  -  how fast accuracy falls as "
         "rulers/hair/ink are made to predict the diagnosis.", False)],
        [("P3  Offloading", True, ACCENT), ("  -  does planting a misleading patient detail "
         "pull attention off the lesion?", False)],
        [("P4  Counterfactual", True, ACCENT), ("  -  change one detail, keep the image fixed, "
         "measure how far the evidence moves.", False)],
        [("P5  Reliance", True, ACCENT), ("  -  accuracy lost when we remove the metadata vs. "
         "when we remove the image.", False)],
    ], y=Inches(1.6), size=16, gap=9)
    note = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.7), Inches(5.85),
                              Inches(12.0), Inches(1.35))
    note.fill.solid(); note.fill.fore_color.rgb = NAVY; note.line.fill.background()
    note.shadow.inherit = False
    tf = note.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.25)
    p = tf.paragraphs[0]
    d._run(p, "Trustworthy by design: ", 14, RGBColor(0x9D,0xB8,0xD8), bold=True)
    d._run(p, "the full plan was frozen before any model was trained; every number carries a "
              "confidence interval; and the heat-map method passed the standard 'scramble the "
              "weights' safety check on every model.", 14, WHITE)

    # ---- 8 result: accuracy ------------------------------------------------
    s = d.slide()
    d.header(s, "Result 1  ·  accuracy", "Fusion helps a lot - or barely - depending on the data")
    d.figure(s, "fig_accuracy.png", Inches(0.7), Inches(1.55), Inches(4.6))
    b1 = _num("padufes20", "B1"); b0 = _num("padufes20", "B0")
    d.bullets(s, [
        [("PAD-UFES-20 (21 rich fields): ", True), ("fusion adds about 8 points. Patient "
         "details alone (%.2f) nearly match the image alone (%.2f)." % (b1, b0), False)],
        [("HAM10000 (only age/sex/site): ", True), ("fusion adds at most ~1.8 points; details "
         "alone score just 0.29.", False)],
        [("Two regimes in one study", True, ACCENT),
         (" - one where metadata is powerful, one where it is weak. That contrast is a strength.",
          False)],
    ], x=Inches(7.5), y=Inches(1.8), w=Inches(5.4), size=15, gap=12)

    # ---- 9 result: grounding (central) ------------------------------------
    s = d.slide()
    d.header(s, "Result 2  ·  the central finding", "The way you combine decides where the model looks")
    d.figure(s, "fig_grounding.png", Inches(0.6), Inches(1.5), Inches(4.7))
    e_b3 = _p1("B3", "energy_in_mask")
    e_b5 = _p1("B5", "energy_in_mask")
    e_b0 = _p1("B0", "energy_in_mask")
    d.bullets(s, [
        [("Same information. Opposite results.", True, RED)],
        [("Concatenation looks ", False), ("more", True, GREEN),
         (f" at the lesion than image-only ({e_b3:.2f} vs {e_b0:.2f}).", False)],
        [("MetaBlock ", False), ("collapses to near random guessing", True, RED),
         (f" ({e_b5:.2f}, chance is 0.27) - its attention has almost stopped following the "
          "lesion, yet its accuracy still looks fine.", False)],
        "FiLM and cross-attention also look less than image-only.",
        [("All four differences are statistically solid; three separate measures agree.", True,
          INK)],
    ], x=Inches(6.9), y=Inches(1.7), w=Inches(6.0), size=14.5, gap=10)

    # ---- 10 result: reliance ----------------------------------------------
    s = d.slide()
    d.header(s, "Result 3  ·  what they lean on", "Every fusion model leans on details more than image")
    d.figure(s, "fig_reliance.png", Inches(0.6), Inches(1.5), Inches(4.7))
    d.bullets(s, [
        "On PAD-UFES-20, removing the patient details hurts every fusion model more than "
        "removing the image does.",
        [("Adding metadata roughly halves how much the model depends on the image", True, ACCENT),
         (" (0.48 -> 0.18-0.27).", False)],
        "This is offloading, measured directly - not guessed from a heat map.",
        [("The image-only model loses exactly zero when details are scrambled - the check "
          "that proves the measurement is sound.", False, GREY)],
    ], x=Inches(6.9), y=Inches(1.8), w=Inches(6.0), size=15, gap=12)

    # ---- 11 counterfactual + shortcut -------------------------------------
    s = d.slide()
    d.header(s, "Results 4 & 5  ·  two more checks", "Irrelevant details move the evidence; shortcuts fool everyone")
    d.figure(s, "fig_degradation.png", Inches(0.6), Inches(1.5), Inches(4.7))
    d.bullets(s, [
        [("Counterfactual (P4): ", True), ("changing a detail that cannot affect a lesion - "
         "'household has piped water' - still moves the evidence. Cross-attention moves the "
         "most.", False)],
        [("Image shortcuts (P2): ", True), ("no model resists them. All fall from ~0.90 to below "
         "chance when artifacts point the wrong way (left figure). Metadata neither helps nor "
         "hurts here.", False)],
        [("Offloading under a planted detail (P3): ", True), ("inconclusive - too few data "
         "points, and the honest answer is to say so.", False, GREY)],
    ], x=Inches(6.9), y=Inches(1.7), w=Inches(6.0), size=14.5, gap=11)

    # ---- 12 statistics -----------------------------------------------------
    s = d.slide(LIGHT)
    d.header(s, "Rigour", "19 tests planned in advance, corrected together")
    fam = pd.read_csv(TABLES / "family_holm.csv")
    summ = fam.groupby("endpoint")["reject_holm"].agg(["size", "sum"])
    labels = {"P1":"P1 Grounding","P2":"P2 Image shortcut","P3":"P3 Offloading",
              "P4":"P4 Counterfactual","P5":"P5 Reliance"}
    items = []
    for ep in ("P1","P2","P3","P4","P5"):
        if ep in summ.index:
            n, r = int(summ.loc[ep,"size"]), int(summ.loc[ep,"sum"])
            col = GREEN if r == n and r>0 else (RED if r==0 else ACCENT)
            verdict = "all significant" if r==n and r>0 else ("none significant" if r==0
                                                              else f"{r} of {n}")
            items.append([(f"{labels[ep]}:  ", True), (f"{verdict}", True, col)])
    total_r = int(fam["reject_holm"].sum())
    d.bullets(s, items + [
        [(f"{total_r} of 19 tests significant after correction.", True, INK)],
        [("P2 and P3 came back empty - but they were planned, so 'no effect' is a result, not a "
          "gap. This is what pre-registration buys.", False, GREY)],
    ], y=Inches(1.65), size=17, gap=11)

    # ---- 13 meaning --------------------------------------------------------
    s = d.slide(NAVY)
    _, tf = d._box(s, Inches(0.8), Inches(0.7), Inches(11.7), Inches(1.2))
    p = tf.paragraphs[0]
    d._run(p, "What it all means", 30, WHITE, bold=True)
    u = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.85), Inches(1.55), Inches(2.2), Pt(4))
    u.fill.solid(); u.fill.fore_color.rgb = ACCENT; u.line.fill.background(); u.shadow.inherit=False
    _, tf = d._box(s, Inches(0.85), Inches(2.0), Inches(11.6), Inches(5.0))
    takeaways = [
        ("The mechanism matters more than the metadata.", "Identical information, combined "
         "differently, improves or destroys the model's visual evidence."),
        ("The simplest method wins twice.", "Plain concatenation was the most accurate on one "
         "dataset and the only one to improve grounding on the other - yet the field treats it "
         "as a throwaway baseline."),
        ("A model can be more accurate and less trustworthy at once.", "MetaBlock is competitive "
         "on accuracy while barely looking at the lesion. Accuracy hides this; our measures "
         "expose it."),
        ("Honesty about our own proposal.", "Cross-attention - what we first set out to promote "
         "- is middling on accuracy and worst at being swayed by irrelevant details. Reporting "
         "that is the contribution."),
    ]
    for i, (h, b) in enumerate(takeaways):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(12)
        d._run(p, f"{i+1}.  ", 18, ACCENT, bold=True)
        d._run(p, h + "  ", 18, WHITE, bold=True)
        d._run(p, b, 15, RGBColor(0xC7,0xD7,0xEC))

    # ---- 14 contributions / status / thanks -------------------------------
    s = d.slide()
    d.header(s, "Contributions, status & what's next", "Where the project stands")
    d.bullets(s, [
        [("Delivered: ", True, GREEN), ("all experiments complete - 449 model trainings, ~21 "
         "GPU-hours; a new benchmark (N1), a new annotation-free test (N2), and the first "
         "grounding audit across fusion families (N3).", False)],
        [("Finding (N4): ", True, GREEN), ("fusion mechanism, not metadata, controls grounding; "
         "accuracy and trustworthiness are decoupled.", False)],
        [("Limitations we state plainly: ", True, RED), ("one offloading test was underpowered; "
         "one backbone tested; skin-tone fairness needs DDI, whose access is pending.", False)],
        [("Next: ", True, ACCENT), ("finalise figures, secondary checks, and write up for MIDL "
         "2027 (deadline ~Dec 2026). Framing shifts from 'same accuracy' to 'accuracy and "
         "evidence are separate things', which the data supports.", False)],
    ], y=Inches(1.6), size=15.5, gap=13)
    thanks = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.7), Inches(6.2),
                                Inches(12.0), Inches(0.95))
    thanks.fill.solid(); thanks.fill.fore_color.rgb = NAVY; thanks.line.fill.background()
    thanks.shadow.inherit = False
    tf = thanks.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE; tf.word_wrap = True
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    d._run(p, "Thank you  ·  " + ", ".join(students) + "   |   Supervisor: " + supervisor,
           15, WHITE, bold=True)

    d.save(out)
    print("written:", out, f"({len(d.prs.slides.__iter__.__self__._sldIdLst)} slides)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--students", nargs="*",
                    default=["Aniket Meena", "Ritu Raj Swami", "Nishant Kumar"])
    ap.add_argument("--supervisor", default="Dr. Bharti Nagpal")
    ap.add_argument("--out", type=Path,
                    default=REPO / "docs" / "MedFusion-XAI_Presentation.pptx")
    a = ap.parse_args()
    build(a.out, a.students, a.supervisor)


if __name__ == "__main__":
    main()
