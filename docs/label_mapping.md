# Label mapping: our label spaces -> DDI's binary malignant/benign

**Status: frozen 2026-09-21, before any DDI image was downloaded or any external number produced.**
Dataset D (DDI) is evaluated **zero-shot** (`docs/04-preregistration.md` §2), so the mapping from our
training label spaces to DDI's binary target must be fixed in advance or it becomes a researcher
degree of freedom. Changes after this date go in the change log below.

## DDI's own labels

DDI ships `malignant` (boolean, **biopsy-proven**), `disease` (78 distinct diagnoses) and
`skin_tone` (Fitzpatrick I-II / III-IV / V-VI). We use `malignant` as ground truth and `skin_tone`
as the stratifier; `disease` is not mapped (78 classes, mostly absent from our training sets).

## Mapping our classes to malignant (1) / benign (0)

| Dataset | Class | -> | Rationale |
|---|---|---|---|
| A: PAD-UFES-20 | BCC basal cell carcinoma | **1** | Skin cancer; biopsy-proven in PAD |
| | SCC squamous cell carcinoma (incl. Bowen's) | **1** | Skin cancer; PAD clusters Bowen's into SCC |
| | MEL melanoma | **1** | Skin cancer |
| | **ACK actinic keratosis** | **1** | See the note below |
| | NEV nevus | 0 | Benign |
| | SEK seborrheic keratosis | 0 | Benign |
| B: HAM10000 | mel, bcc | **1** | Skin cancer |
| | **akiec** (actinic keratoses / intraepithelial carcinoma) | **1** | Includes Bowen's disease = SCC in situ; see below |
| | nv, bkl, df, vasc | 0 | Benign |
| C: ISIC 2019 | MEL, BCC, SCC, **AK** | **1** | Already fixed in `scripts/prepare_data.py` before this document |
| | NV, BKL, DF, VASC | 0 | Benign |

### The actinic-keratosis decision (the only debatable cell)

ACK / akiec / AK is a pre-malignant lesion, so "malignant" is arguable either way. It is mapped to
**1** for three reasons:

1. **Consistency.** `prepare_data.py` put `AK` in the malignant set for ISIC 2019 (line 189) before
   this question arose, and Bissoto et al.'s trap sets use the same convention. A different choice
   for A and B would make the three datasets mutually incomparable.
2. **Clinical direction.** AK is managed as a lesion requiring treatment, and HAM10000's `akiec`
   explicitly includes intraepithelial carcinoma (Bowen's disease), which is SCC in situ.
3. **It is the conservative choice for our claim.** ACK is 32% of PAD-UFES-20 (730/2,298); calling
   it benign would inflate the benign class and make the external AUROC easier. Mapping it to
   malignant makes the external test harder, so a positive result cannot be an artefact of the
   mapping.

**Sensitivity analysis (pre-specified):** every DDI number is also reported with ACK/akiec/AK
**excluded** from the evaluation entirely (not remapped). If the two disagree, both are reported and
the disagreement is discussed. No third variant is tried.

## What is dropped, and reported as dropped

- DDI diagnoses with no counterpart in our label spaces are **not** dropped: the binary target is
  defined for every DDI image, and a model trained on 6 or 7 classes still produces a malignant
  probability as the sum over its malignant classes. What we cannot do is per-class evaluation.
- **Probability aggregation:** malignant score = sum of the softmax probabilities of the classes
  mapped to 1. AUROC uses that score; balanced accuracy uses a 0.5 threshold on it.

## Modality shift: which models get a fair external test

DDI is **clinical photography**, like PAD-UFES-20 and unlike HAM10000's dermoscopy.

- **Dataset A (PAD) models are the primary external test** - matched modality, so a drop in AUROC
  reflects domain/skin-tone shift rather than a change of imaging device.
- **Dataset B (HAM) models are reported separately as a modality-shift result**, not pooled with A.
  Their numbers confound dermoscopic-to-clinical transfer with everything else.

## Metadata on DDI: the all-MISSING condition

DDI carries no clinical metadata we can use (§2 of the pre-registration: "none usable -> fusion
models get all-MISSING tokens"). Every categorical field is set to its MISSING index and every
continuous field to `cont_mask=False`, which is exactly the encoding those models already saw during
training through `meta_field_dropout`.

This makes DDI an **independent test of the offloading result**, not just a fairness check: a model
that has offloaded its evidence onto metadata should lose more when metadata is entirely absent than
an image-only model, which loses nothing by construction. The pre-registered P5 ordering (B5 > B4 >
B3 ~ M1 on metadata reliance) predicts the ordering of the DDI drops. That prediction is recorded
here **before the data exists**.

## Change log

| Date | Change | Reason |
|---|---|---|
| 2026-09-21 | Mapping frozen, before DDI download | Fixes a researcher degree of freedom ahead of the external test |
