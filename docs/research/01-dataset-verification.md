# 01 — Dataset & Benchmark Claim Verification

Checked on 2026-09-17 against primary sources: dataset papers (full text read from arXiv / publisher PDFs), official repositories, and official data pages or APIs. No datasets were downloaded; file sizes come from repository APIs.

Status key: **VERIFIED** means the primary source confirms the claim. **WRONG** means the primary source contradicts it (correct value given). **PARTIAL** means part of the claim is right and part is wrong or imprecise. **UNVERIFIABLE** means no primary source could be found to confirm it.

## Summary table

| # | Claim | Status | Correction / note |
|---|-------|--------|-------------------|
| 1a | PAD-UFES-20 DOI 10.17632/zr7vgbcyr2.1 | VERIFIED | Mendeley Data v1, published 2020-07-07, CC BY 4.0 |
| 1b | 2,298 smartphone clinical images | VERIFIED | from 1,641 lesions and 1,373 patients |
| 1c | 6 classes BCC, SCC, ACK, SEK, MEL, NEV | VERIFIED | The ISIC Archive copy also lists Bowen's disease (BOD), which is grouped under SCC in the original labels |
| 1d | "up to 26 clinical features" | VERIFIED (with caveat) | The metadata CSV has 26 columns, including IDs and the label. The arXiv v2 intro says "up to 22 clinical features"; the metadata section says 26 |
| 1e | 58.4% biopsy-proven, incl. 100% of cancers | VERIFIED | Table 1 of the paper |
| 1f | ~3.5 GB | VERIFIED | 3 zips = 3,592,396,763 bytes (≈3.59 GB / 3.35 GiB), plus a 316 KB metadata.csv |
| 1g | Image format .png | VERIFIED | "All images are available in .png format" |
| 1h | metadata.csv column list | **WRONG (incomplete)** | The claimed list has 25 names and is **missing `lesion_id`**. The other 25 names, including the misspelling `fitspatrick`, match the paper |
| 1i | Metadata-fusion gain ≈ +2.5% on PAD-UFES-20 | **PARTIAL / misattributed** | +2.53 pp is *accuracy* from one 2026 handcrafted-feature paper (HCHS-Net, Biomimetics). The standard deep-learning references report much larger balanced-accuracy gains (≈7% in Pacheco & Krohling 2020 on the earlier PAD data). Do not cite ~2.5% as the typical gain |
| 2 | SkinCon: 48 concepts; 3,230 Fitz17k + 656 DDI; 22 concepts ≥50 images | VERIFIED | Annotations are CSVs on skincon-dataset.github.io; the images come from Fitzpatrick17k and DDI |
| 3a | Fitzpatrick17k size | VERIFIED: 16,577 images, 114 conditions | Only a CSV of image URLs is distributed; some links are broken; a request form gives the full image set |
| 3b | Fitzpatrick17k-C | VERIFIED: 11,394 images (7,975 / 1,139 / 2,280) | Abhishek, Jain & Hamarneh, *Sci. Data* 2025. The split CSV is on GitHub and Zenodo; the images still come from Fitzpatrick17k |
| 4 | DDI: size, access, labels | VERIFIED | 656 images, 570 patients; FST I–II 208, III–IV 241, V–VI 207; 485 benign / 171 malignant; research-use-only terms; Stanford AIMI portal, also listed on Stanford Data Farm (Redivis) |
| 5a | Bissoto: 7 artifacts; 2,594 ISIC 2018 + 872 Atlas | VERIFIED | The artifacts are dark corners, hair, gel borders, gel bubbles, rulers, ink markings/staining, patches |
| 5b | Repo github.com/alceubissoto/debiasing-skin | VERIFIED exists | Contains `artefacts-annotation/{isic_bias.csv, atlas_bias.csv}`, `trap-sets/` (5 train/val/test CSV splits), `norm-background/`. ISIC 2018 Task 1/2 images, masks and metadata plus the Atlas are needed |
| 5c | "trap sets at 6 bias levels" | **WRONG source** | The CVPRW 2020 repo has one trap-set protocol (5 splits) and no bias levels. The six levels {0, 0.3, 0.5, 0.7, 0.9, 1} are in a **different** repo and paper: `alceubissoto/artifact-generalization-skin` (Bissoto, Barata, Valle, Avila, ISIC@ECCV 2022), and those sets are built on **ISIC 2019** |
| 6a | CheXlocalize: pixel segmentations for 10 pathologies on CheXpert val/test; radiologist benchmark set | VERIFIED | val 234 CXRs / 200 patients; test 668 CXRs / 500 patients; ground-truth and human-benchmark segmentations and points for both |
| 6b | Grad-CAM best of 7 saliency methods | VERIFIED (hedged) | The paper says Grad-CAM "generally" localized best; it was not best on every pathology |
| 6c | Grad-CAM hit rate 0.376 | **WRONG source** | 0.376 is not in Saporta et al. It is the mean hit rate of Grad-CAM on an **EfficientNet-B0** in *MedicalPatchNet* (Wienholt et al., arXiv 2509.07477). In Saporta et al., DenseNet121-ensemble Grad-CAM per-pathology hit rates are 0.290–0.903, unweighted mean ≈ **0.575**, and mIoU mean ≈ **0.282** (my calculation from Table 1) |
| 6d | Hit-rate tolerance region around the most representative point | **WRONG** | There is **no tolerance**. Pointing game: the heat-map argmax pixel must lie **inside** the ground-truth mask. It is computed on the "true positive slice" |
| 6e | Thresholding | VERIFIED | Otsu's method per pathology (default). A second scheme searches a per-pathology threshold that maximizes validation mIoU; there is no significant difference between the two |
| 7 | NIH ChestX-ray14: 112,120 images, 30,805 patients, 14 labels, ~45 GB | VERIFIED | Kaggle official mirror 45,077,186,807 bytes; NIH Box instant download |
| 8a | NEATX: ~3.5k drain annotations CXR14; ~1k across 4 tube types PadChest | VERIFIED | 3,543 and 1,011; Zenodo 10.5281/zenodo.14944064, CC BY-NC-SA 2.0 |
| 8b | Drain-present vs drain-absent pneumothorax AUC 0.940 / 0.770 from NEATX | **WRONG source** | These numbers are **not** in NEATX. They come from Oakden-Rayner et al., "Hidden Stratification…", ACM CHIL 2020, which reports them to 2 decimals: 0.94 with drains, 0.77 without, 0.87 overall |
| 9 | CheXpert downsampled ~11 GB | VERIFIED (via mirror) | CheXpert-v1.0-small = 11,471,429,175 bytes (Kaggle mirror). The official page doesn't state the size. Official access is now Stanford AIMI / Redivis with a research-use agreement |

---

## 1. PAD-UFES-20

**Primary sources**
- Mendeley Data: https://data.mendeley.com/datasets/zr7vgbcyr2/1 (API: `https://data.mendeley.com/public-api/datasets/zr7vgbcyr2?version=1`)
- Paper (arXiv v2 full text read): https://arxiv.org/abs/2007.00478; published version *Data in Brief* 32:106221, https://doi.org/10.1016/j.dib.2020.106221
- Code repo (no data): https://github.com/labcin-ufes/PAD-UFES-20
- Mirror: ISIC Archive collection 406, https://api.isic-archive.com/collections/406/

**Facts confirmed**
- 2,298 images, 1,641 lesions, 1,373 patients, collected 2018–2019 with a range of smartphones. Images are raw (no preprocessing), vary in size, and are all `.png`.
- Table 1 (samples / % biopsied): ACK 730 / 24.4%; BCC 845 / 100%; MEL 52 / 100%; NEV 244 / 24.6%; SCC 192 / 100%; SEK 235 / 6.4%; **total 2,298 / 58.4%**. All BCC, SCC and MEL are biopsy-proven. (The PDF table layout is scrambled when extracted; this pairing follows the paper's statement that all cancers are 100% biopsied.)
- Files on Mendeley: `metadata.csv` (316,209 B) and `images/imgs_part_1.zip` (1,245,184,680 B), `imgs_part_2.zip` (1,126,646,990 B), `imgs_part_3.zip` (1,220,565,093 B). **Total ≈ 3.59 GB.**
- **Access:** instant, no registration (Mendeley Data). **License:** CC BY 4.0.

**metadata.csv columns (paper's "Meta-data" section, 26 features)**
`patient_id, lesion_id, img_id, smoke, drink, background_father, background_mother, age, pesticide, gender, skin_cancer_history, cancer_history, has_piped_water, has_sewage_system, fitspatrick, region, diameter_1, diameter_2, diagnostic, itch, grew, hurt, changed, bleed, elevation, biopsed`

- The claimed list is missing **`lesion_id`**. Every other name matches, including the misspelled `fitspatrick` and `biopsed`.
- The paper lists the columns in the order above. The physical column order in the CSV was **not** checked, because that would mean downloading the file; check it on first load.
- `patient_id, lesion_id, img_id, age, region, biopsed` are always present. Other fields may be blank (missing) or `UNK`.
- `region` has 15 macro-regions. `diagnostic` ∈ {ACK, BCC, MEL, NEV, SCC, SEK}.
- Caveat: arXiv v2 says "up to 22 clinical features" in the Background section but "26 features" in the Meta-data section. Mendeley says 26. 26 is the column count, which includes IDs and the label; the true clinical features number about 21–22.

**Published metadata-fusion gain (claimed ~+2.5%)**
- **+2.53 pp accuracy** (95.23% → 97.76%): A. Solak, "HCHS-Net: A Multimodal Handcrafted Feature and Metadata Framework for Interpretable Skin Lesion Classification", *Biomimetics* 11(2):154, 2026, doi:10.3390/biomimetics11020154. This is a handcrafted-feature plus gradient-boosting ablation, measured in **accuracy, not balanced accuracy**. It claims patient-level splits. Its absolute accuracy (~98% on 6 classes) is far above deep-learning SOTA (BACC ~0.80–0.85), so treat it with caution.
- Pacheco & Krohling, "The impact of patient clinical information on automated skin cancer detection", *Comput. Biol. Med.* 116:103545, 2020 (arXiv 1909.12912): about **7% balanced-accuracy improvement** for all models. This used the earlier PAD-UFES data, before the -20 release.
- Pacheco & Krohling, MetaBlock, *IEEE JBHI* 25(9):3554–3563, 2021, doi:10.1109/JBHI.2021.3062002: reports BACC 0.770±0.016 (EfficientNet-B4 + MetaBlock) and AUC 0.945±0.005 (EfficientNet-B4 + concatenation) on PAD-UFES-20. These values are quoted by de Lima & Krohling (arXiv 2205.15442). I could not open the JBHI full text, so the image-only baseline from that paper is **UNVERIFIED** here.
- de Lima & Krohling (arXiv 2205.15442): MetaBlock beats concatenation by +0.020 to +0.037 BACC for PiT, CoaT and ViT. That compares two fusion methods, not fusion against image-only.
- **Recommendation:** don't write "metadata fusion yields ~2.5%". Cite a range and name the metric, or run our own image-only vs fusion ablation.

## 2. SkinCon

- Site: https://skincon-dataset.github.io/ ; paper arXiv 2302.00785 ; NeurIPS 2022 Datasets & Benchmarks.
- **Verified:** 48 clinical concepts chosen by two dermatologists; **3,230** Fitzpatrick17k images; **656** DDI images; **22** concepts with ≥50 images.
- **Access:** instant. `annotations_fitzpatrick17k.csv` and `annotations_ddi.csv` download straight from the project site. The images must come separately from Fitzpatrick17k (URLs or request form) and DDI (Stanford AIMI agreement).
- **License of the annotation CSVs:** not stated on the site (UNVERIFIED). The arXiv record uses the default non-exclusive license. The underlying images keep their own terms: Fitz17k is CC BY-NC-SA 3.0 for the atlas sources, and DDI is research-use only.
- Title differs between venues. NeurIPS: "…for fine-grained debugging and analysis". arXiv: "…for fine-grained **model** debugging and analysis".

## 3. Fitzpatrick17k and Fitzpatrick17k-C

**Fitzpatrick17k**: https://github.com/mattgroh/fitzpatrick17k
- 16,577 clinical images, 114 conditions (53–653 images each), Fitzpatrick I–VI labels. Sources: DermaAmin (12,672) and Atlas Dermatologico (3,905), as stated in Abhishek et al.
- **Access:** `fitzpatrick17k.csv` with image URLs is instant. Some links are broken; the maintainers ask users to fill a form to get a link to all images (registration-like step). Images are CC BY-NC-SA 3.0 Unported (per repo).
- Published at CVPR 2021 **Workshops** (ISIC workshop), pp. 1820–1828. It was not the main CVPR conference, although arXiv and the repo say "CVPR".

**Fitzpatrick17k-C**: Abhishek, Jain, Hamarneh, *Scientific Data* 12:196 (2025), arXiv 2401.14497
- **11,394 images**: train 7,975 (70%), val 1,139 (10%), test 2,280 (20%), stratified by diagnosis. Duplicates were removed by embedding similarity, clustering and manual review; outlier or erroneous images were removed too.
- The paper text I read does not state whether the splits are patient-disjoint (UNVERIFIED; a secondary summary claims they are).
- Distributed as a split CSV. Repo: https://github.com/kakumarabhishek/Corrected-Skin-Image-Datasets (code Apache-2.0). Zenodo: https://doi.org/10.5281/zenodo.11101337 (the resolved record 12739457 shows metadata CC BY 4.0 and NPZ images CC BY-NC 4.0). The Fitzpatrick17k-C images still come from the original Fitzpatrick17k.

## 4. DDI (Diverse Dermatology Images)

- Site: https://ddi-dataset.github.io/ ; paper Daneshjou et al., *Science Advances* 8(32):eabq6147, 2022, arXiv 2203.08807.
- **656 images, 570 patients.** Biopsy-proven, from Stanford clinics 2010–2020. The FST I–II and FST V–VI groups are matched on diagnosis, age (within 10 years), gender and photo date (within 3 years).
- **FST groups** (from the paper): I–II **208** (159 benign / 49 malignant); III–IV **241** (167 / 74); V–VI **207** (159 / 48). **Totals:** 485 benign / 171 malignant. Labels include binary malignancy, disease name and skin-tone group (12/34/56).
- **Access:** Stanford AIMI Shared Datasets portal (https://stanfordaimi.azurewebsites.net/datasets/35866158-8196-48d8-87bf-50dca81df965). It is also listed on Stanford Data Farm / Redivis (https://stanford.redivis.com/datasets/3r16-5mby7gfer), along with a separate "DDI2" dataset. Both need a login and acceptance of terms; the Redivis page did not render, so exact approval steps are UNVERIFIED.
- **Terms (site):** research use only, non-commercial. No redistribution, no sharing of download links, no re-identification, no clinical use.

## 5. Bissoto et al. artifact annotations

**Paper:** Bissoto, Valle, Avila, "Debiasing Skin Lesion Datasets and Models? Not So Fast", CVPRW 2020 (ISIC workshop), pp. 3192–3201, doi:10.1109/CVPRW50498.2020.00378, arXiv 2004.11457. Full text read.
- **7 artifacts (verified):** dark corners (vignetting), hair, gel borders, gel bubbles, rulers, ink markings/staining, patches.
- **Manual annotation (verified):** 2,594 images of **ISIC 2018 Tasks 1 & 2** and 872 images of the Interactive Atlas of Dermoscopy (dermoscopic subset, "Dermoscopic Atlas").
- The task is binary: malignant (melanoma) vs benign (nevus, seborrheic keratosis).
- **Trap sets** in this paper: train and test with amplified, opposite artifact–label correlations. The paper gives **no graded bias levels**. Results use 5 fixed splits.

**Repo** https://github.com/alceubissoto/debiasing-skin (exists)
- `artefacts-annotation/isic_bias.csv`, `atlas_bias.csv`; `trap-sets/isic_annotated_{train,val,test}{1..5}.csv`; `norm-background/`; `images/`.
- The README says users must obtain the **ISIC 2018 Task 1/2 images, masks and metadata** and the Atlas themselves. So yes, the ISIC 2018 Task 1&2 images are required.
- No license file was found on the README page (UNVERIFIED).

**"6 bias levels" actually comes from:** Bissoto, Barata, Valle, Avila, "Artifact-based Domain Generalization of Skin Lesion Models", ISIC Workshop @ ECCV 2022 (LNCS), doi:10.1007/978-3-031-25069-9_10, arXiv 2208.09756.
- Repo https://github.com/alceubissoto/artifact-generalization-skin has `trap_sets/bias_{0,0.3,0.5,0.7,0.9,1}/` with multiple train/test replicates.
- These trap sets are built on **ISIC 2019**, using artifact labels inferred for ISIC 2019 (`isic_inferred_wocarcinoma.csv`). The OOD tests are Derm7pt / edraAtlas, PH2 and PAD-UFES-20.
- The bias level is the probability of sampling by the trap procedure instead of at random (0 = random, 1 = fully biased).

## 6. CheXlocalize

**Paper:** Saporta et al., "Benchmarking saliency methods for chest X-ray interpretation", *Nature Machine Intelligence* 4(10):867–878, 2022, doi:10.1038/s42256-022-00536-x. Full text read from the author PDF.
**Repo:** https://github.com/rajpurkarlab/cheXlocalize (MIT).
**Data:** Stanford AIMI, https://aimi.stanford.edu/datasets/chexlocalize → Redivis https://stanford.redivis.com/datasets/efx9-5nspnbb4b (dataset DOI 10.71718/hap9-kn94).

- **10 pathologies:** Airspace Opacity (= CheXpert "Lung Opacity"), Atelectasis, Cardiomegaly, Consolidation, Edema, Enlarged Cardiomediastinum, Lung Lesion, Pleural Effusion, Pneumothorax, Support Devices.
- **Validation** (234 CXRs, 200 patients) and **test** (668 CXRs, 500 patients) each come with ground-truth segmentations, human-benchmark segmentations and most-representative points (radiologists), and Grad-CAM maps and segmentations.
- The paper text says a "development dataset" of 234 images with 643 segmentations was released. The current repo and data page list the test-set files (`gt_segmentations_test.json`, `hb_segmentations_test.json`) too, so both splits are now available.
- **Access:** credentialed-light. It needs a Stanford AIMI / Redivis account and acceptance of the Stanford research-use terms. The exact wording was not rendered (UNVERIFIED).
- **Methods:** 7 saliency methods (Grad-CAM, Grad-CAM++, Integrated Gradients, Eigen-CAM, DeepLIFT, LRP, Occlusion) × DenseNet121 / ResNet152 / Inception-v4. Each is an ensemble of 30 checkpoints, input 320×320. "Grad-CAM with DenseNet121 generally demonstrated better localization performance", so Grad-CAM is best *generally*, not on every pathology.
- **Hit rate (exact definition):** based on the pointing game. For each pathology, "the pixel in the saliency method heat map with the largest value" is "the single most representative point". A **hit** means that point **lies within the ground-truth segmentation**.
  - There is **no tolerance margin or neighbourhood**. (Some other pointing-game papers add a tolerance of a few pixels; CheXlocalize does not.)
  - It is computed on the true-positive slice: CXRs that have both a predicted point and a ground-truth segmentation.
  - Human benchmark: radiologists clicked one most-representative point.
  - Reported as the mean over 1,000 bootstrap replicates with 95% CIs.
- **Thresholding:** per pathology, **Otsu's method** is the default (`heatmap_to_segmentation.py` uses cv2 Otsu). An alternative scheme searches for the threshold that maximizes per-pathology mIoU on the validation set (`tune_heatmap_threshold.py`). There was no significant difference between the two when compared against the human benchmark.
- **Grad-CAM (DenseNet121) test hit rate, per pathology (Table 1):** Airspace opacity 0.498, Atelectasis 0.501, Cardiomegaly 0.903, Consolidation 0.738, Edema 0.746, Enl. cardiom. 0.818, Lung lesion 0.290, Pleural effusion 0.507, Pneumothorax 0.392, Support devices 0.355.
  - Unweighted mean ≈ 0.575 (my calculation; the paper gives no single overall number).
  - The support-devices values (hit rate 0.355, mIoU 0.163) are confirmed by the paper's own text.
- **Grad-CAM mIoU, per pathology:** 0.248, 0.254, 0.452, 0.408, 0.362, 0.379, 0.101, 0.235, 0.213, 0.163 (mean ≈ 0.282).
- Grad-CAM is on average 24.0% worse than the human benchmark on mIoU and 29.4% worse on hit rate.
- **Where "0.376" comes from:** Wienholt, Kuhl, Kather, Nebelung, Truhn, "MedicalPatchNet: A Patch-Based Self-Explainable AI Architecture for Chest X-ray Classification", arXiv 2509.07477 (2025). The abstract reads: "mean hit-rate 0.485 vs. 0.376 with Grad-CAM". That Grad-CAM is on **their EfficientNet-B0 baseline** evaluated on CheXlocalize, not on Saporta's ensemble.

## 7. NIH ChestX-ray14

- **Paper:** Wang et al., "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks…", CVPR 2017, pp. 2097–2106, arXiv 1705.02315. The original paper reported 108,948 images, 32,717 patients and 8 labels. The dataset was later expanded to ChestX-ray14, and the CVPR paper is still the one to cite.
- **Verified (NIH release and Kaggle NIH mirror):** 112,120 frontal-view images, 30,805 patients, 14 disease labels plus "No Finding". Images are 1024×1024 PNG in 12 archives. Also includes bounding boxes for a subset and a `Data_Entry_2017` metadata CSV. Labels were NLP-mined (>90% estimated accuracy).
- **Size:** 45,077,186,807 bytes ≈ **45 GB** (Kaggle `nih-chest-xrays/data`).
- **Access:** instant, no registration, from the NIH Box (https://nihcc.app.box.com/v/ChestXray-NIHCC) or Kaggle. Kaggle labels it CC0. The NIH README's exact terms (unrestricted use, acknowledge NIH CC, cite the paper) could not be rendered here (UNVERIFIED wording).

## 8. NEATX

- **Paper:** Cheplygina, Damgaard, Eriksen, Juodelyte, Jiménez-Sánchez, "Augmenting Chest X-ray Datasets with Non-Expert Annotations", MIUA 2025, LNCS 15916, pp. 133–144, doi:10.1007/978-3-031-98688-8_10, arXiv 2309.02244.
- **Data:** Zenodo "NEATX: Non-Expert Annotations of Tubes in X-rays", doi:10.5281/zenodo.14944064, v1.0, **CC BY-NC-SA 2.0**. Files: NIH-CXR14 aggregated CSV, PadChest aggregated and raw CSVs, healthsheet. Code: https://github.com/purrlab/chestxr-label-reliability.
- **Verified:** **3,543** chest-drain annotations for NIH-CXR14, drawn from the 5,302 pneumothorax images (a 3,709-image subset was annotated). **1,011** PadChest annotations across 4 tube types: chest drain, tracheostomy, nasogastric, endotracheal.
- **Access:** annotations are instant. Images need NIH CXR14 (instant) and PadChest (BIMCV request/registration; not verified here).
- **The 0.940 / 0.770 numbers:** not in NEATX. NEATX reports drain-*detector* AUCs of about 0.90–0.93.
  - The source is **Oakden-Rayner, Dunnmon, Carneiro, Ré, "Hidden stratification causes clinically meaningful failures in machine learning for medical imaging", ACM CHIL 2020, pp. 151–159, doi:10.1145/3368555.3384468** (arXiv 1909.12475).
  - On the CXR14 test set with radiologist (LOR) drain labels: pneumothorax AUC **0.87** overall, **0.94** with chest drains, **0.77** without. 80% of test pneumothoraces had a drain. PPV was 0.90 with drains vs 0.60 without.
  - The figures are published to **2 decimals**; write "0.94 / 0.77", not "0.940 / 0.770".
  - They are also restated in Jiménez-Sánchez et al., "Detecting Shortcuts in Medical Images – A Case Study in Chest X-rays", ISBI 2023 / arXiv 2211.04279. Their own CheXpert-trained model on CXR14 gave AUC ≈0.81 baseline, 0.85 with drains, 0.77 without (24k training subset).

## 9. CheXpert (downsampled)

- **Paper:** Irvin et al., AAAI 2019, arXiv 1901.07031: 224,316 radiographs, 65,240 patients, 14 observations.
- **Official page** (https://stanfordmlgroup.github.io/competitions/chexpert/) now points to Stanford AIMI / Redivis (https://stanford.redivis.com/datasets/5yyj-1a9f6ap0x). The AIMI page gives dataset DOI 10.71718/y7pj-4v93. The official page does not state the size of the small version.
- **Size of CheXpert-v1.0-small:** 11,471,429,175 bytes ≈ **11.5 GB** (≈10.7 GiB), taken from a third-party Kaggle mirror (`ashery/chexpert`) whose "CC0" label is **not** the official license. **~11 GB is consistent.**
- **Access:** credentialed-light: Stanford AIMI / Redivis account plus the Stanford University School of Medicine research use agreement (non-commercial, non-clinical). Use the official route for the paper, not the Kaggle mirror.
- A Hugging Face `StanfordAIMI/CheXpert-v1.0-512` version also exists (gated).

---

## BibTeX

```bibtex
@article{pacheco2020padufes20,
  author  = {Pacheco, Andre G. C. and Lima, Gustavo R. and Salom{\~a}o, Amanda S. and Krohling, Breno and Biral, Igor P. and de Angelo, Gabriel G. and Alves Jr, F{\'a}bio C. R. and Esgario, Jos{\'e} G. M. and Simora, Alana C. and Castro, Pedro B. C. and Rodrigues, Felipe B. and Frasson, Patricia H. L. and Krohling, Renato A. and Knidel, Helder and Santos, Maria C. S. and do Esp{\'i}rito Santo, Rachel B. and Macedo, Telma L. S. G. and Canuto, Tania R. P. and de Barros, Lu{\'i}z F. S.},
  title   = {{PAD-UFES-20}: A skin lesion dataset composed of patient data and clinical images collected from smartphones},
  journal = {Data in Brief},
  volume  = {32},
  pages   = {106221},
  year    = {2020},
  doi     = {10.1016/j.dib.2020.106221}
}

@misc{pacheco2020padufes20data,
  author       = {Pacheco, Andre G. C. and others},
  title        = {{PAD-UFES-20}: a skin lesion dataset composed of patient data and clinical images collected from smartphones},
  howpublished = {Mendeley Data, V1},
  year         = {2020},
  doi          = {10.17632/zr7vgbcyr2.1}
}

@article{pacheco2020impact,
  author  = {Pacheco, Andre G. C. and Krohling, Renato A.},
  title   = {The impact of patient clinical information on automated skin cancer detection},
  journal = {Computers in Biology and Medicine},
  volume  = {116},
  pages   = {103545},
  year    = {2020},
  doi     = {10.1016/j.compbiomed.2019.103545}
}

@article{pacheco2021metablock,
  author  = {Pacheco, Andre G. C. and Krohling, Renato A.},
  title   = {An Attention-Based Mechanism to Combine Images and Metadata in Deep Learning Models Applied to Skin Cancer Classification},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  volume  = {25},
  number  = {9},
  pages   = {3554--3563},
  year    = {2021},
  doi     = {10.1109/JBHI.2021.3062002}
}

@article{solak2026hchsnet,
  author  = {Solak, Ahmet},
  title   = {{HCHS-Net}: A Multimodal Handcrafted Feature and Metadata Framework for Interpretable Skin Lesion Classification},
  journal = {Biomimetics},
  volume  = {11},
  number  = {2},
  pages   = {154},
  year    = {2026},
  doi     = {10.3390/biomimetics11020154}
}

@inproceedings{daneshjou2022skincon,
  author    = {Daneshjou, Roxana and Yuksekgonul, Mert and Cai, Zhuo Ran and Novoa, Roberto and Zou, James},
  title     = {{SkinCon}: A skin disease dataset densely annotated by domain experts for fine-grained debugging and analysis},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS) 35, Datasets and Benchmarks Track},
  year      = {2022},
  eprint    = {2302.00785},
  archivePrefix = {arXiv}
}

@inproceedings{groh2021fitzpatrick17k,
  author    = {Groh, Matthew and Harris, Caleb and Soenksen, Luis and Lau, Felix and Han, Rachel and Kim, Aerin and Koochek, Arash and Badri, Omar},
  title     = {Evaluating Deep Neural Networks Trained on Clinical Images in Dermatology with the {Fitzpatrick 17k} Dataset},
  booktitle = {2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  pages     = {1820--1828},
  year      = {2021},
  doi       = {10.1109/CVPRW53098.2021.00201}
}

@article{abhishek2025fitz17kc,
  author  = {Abhishek, Kumar and Jain, Aditi and Hamarneh, Ghassan},
  title   = {Investigating the Quality of {DermaMNIST} and {Fitzpatrick17k} Dermatological Image Datasets},
  journal = {Scientific Data},
  volume  = {12},
  pages   = {196},
  year    = {2025},
  doi     = {10.1038/s41597-025-04382-5}
}

@article{daneshjou2022ddi,
  author  = {Daneshjou, Roxana and Vodrahalli, Kailas and Novoa, Roberto A. and Jenkins, Melissa and Liang, Weixin and Rotemberg, Veronica and Ko, Justin and Swetter, Susan M. and Bailey, Elizabeth E. and Gevaert, Olivier and Mukherjee, Pritam and Phung, Michelle and Yekrang, Kiana and Fong, Bradley and Sahasrabudhe, Rachna and Allerup, Johan A. C. and Okata-Karigane, Utako and Zou, James and Chiou, Albert S.},
  title   = {Disparities in dermatology {AI} performance on a diverse, curated clinical image set},
  journal = {Science Advances},
  volume  = {8},
  number  = {32},
  pages   = {eabq6147},
  year    = {2022},
  doi     = {10.1126/sciadv.abq6147}
}

@inproceedings{bissoto2020debiasing,
  author    = {Bissoto, Alceu and Valle, Eduardo and Avila, Sandra},
  title     = {Debiasing Skin Lesion Datasets and Models? Not So Fast},
  booktitle = {2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  pages     = {3192--3201},
  year      = {2020},
  doi       = {10.1109/CVPRW50498.2020.00378}
}

@inproceedings{bissoto2022artifact,
  author    = {Bissoto, Alceu and Barata, Catarina and Valle, Eduardo and Avila, Sandra},
  title     = {Artifact-Based Domain Generalization of Skin Lesion Models},
  booktitle = {Computer Vision -- ECCV 2022 Workshops},
  series    = {Lecture Notes in Computer Science},
  publisher = {Springer},
  year      = {2023},
  doi       = {10.1007/978-3-031-25069-9_10},
  note      = {arXiv:2208.09756}
}

@article{saporta2022chexlocalize,
  author  = {Saporta, Adriel and Gui, Xiaotong and Agrawal, Ashwin and Pareek, Anuj and Truong, Steven Q. H. and Nguyen, Chanh D. T. and Ngo, Van-Doan and Seekins, Jayne and Blankenberg, Francis G. and Ng, Andrew Y. and Lungren, Matthew P. and Rajpurkar, Pranav},
  title   = {Benchmarking saliency methods for chest {X}-ray interpretation},
  journal = {Nature Machine Intelligence},
  volume  = {4},
  number  = {10},
  pages   = {867--878},
  year    = {2022},
  doi     = {10.1038/s42256-022-00536-x}
}

@misc{wienholt2025medicalpatchnet,
  author        = {Wienholt, Patrick and Kuhl, Christiane and Kather, Jakob Nikolas and Nebelung, Sven and Truhn, Daniel},
  title         = {{MedicalPatchNet}: A Patch-Based Self-Explainable {AI} Architecture for Chest {X}-ray Classification},
  year          = {2025},
  eprint        = {2509.07477},
  archivePrefix = {arXiv}
}

@inproceedings{wang2017chestxray8,
  author    = {Wang, Xiaosong and Peng, Yifan and Lu, Le and Lu, Zhiyong and Bagheri, Mohammadhadi and Summers, Ronald M.},
  title     = {{ChestX-ray8}: Hospital-scale Chest {X}-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {2097--2106},
  year      = {2017},
  doi       = {10.1109/CVPR.2017.369}
}

@inproceedings{cheplygina2025neatx,
  author    = {Cheplygina, Veronika and Damgaard, Cathrine and Eriksen, Trine Naja and Juodelyte, Dovile and Jim{\'e}nez-S{\'a}nchez, Amelia},
  title     = {Augmenting Chest {X}-ray Datasets with Non-Expert Annotations},
  booktitle = {Medical Image Understanding and Analysis (MIUA 2025)},
  series    = {Lecture Notes in Computer Science},
  volume    = {15916},
  pages     = {133--144},
  publisher = {Springer},
  year      = {2025},
  doi       = {10.1007/978-3-031-98688-8_10}
}

@misc{neatx2025data,
  author    = {Cheplygina, Veronika and Damgaard, Cathrine and Eriksen, Trine Naja and Juodelyte, Dovile and Jim{\'e}nez-S{\'a}nchez, Amelia},
  title     = {{NEATX}: Non-Expert Annotations of Tubes in X-rays},
  publisher = {Zenodo},
  version   = {1.0},
  year      = {2025},
  doi       = {10.5281/zenodo.14944064}
}

@inproceedings{oakdenrayner2020hidden,
  author    = {Oakden-Rayner, Luke and Dunnmon, Jared and Carneiro, Gustavo and R{\'e}, Christopher},
  title     = {Hidden stratification causes clinically meaningful failures in machine learning for medical imaging},
  booktitle = {Proceedings of the ACM Conference on Health, Inference, and Learning (CHIL)},
  pages     = {151--159},
  year      = {2020},
  doi       = {10.1145/3368555.3384468}
}

@inproceedings{jimenezsanchez2023shortcuts,
  author    = {Jim{\'e}nez-S{\'a}nchez, Amelia and Juodelyte, Dovile and Chamberlain, Bethany and Cheplygina, Veronika},
  title     = {Detecting Shortcuts in Medical Images -- A Case Study in Chest {X}-rays},
  booktitle = {IEEE International Symposium on Biomedical Imaging (ISBI)},
  year      = {2023},
  note      = {arXiv:2211.04279}
}

@inproceedings{irvin2019chexpert,
  author    = {Irvin, Jeremy and Rajpurkar, Pranav and Ko, Michael and Yu, Yifan and Ciurea-Ilcus, Silviana and Chute, Chris and Marklund, Henrik and Haghgoo, Behzad and Ball, Robyn and Shpanskaya, Katie and Seekins, Jayne and Mong, David A. and Halabi, Safwan S. and Sandberg, Jesse K. and Jones, Ricky and Larson, David B. and Langlotz, Curtis P. and Patel, Bhavik N. and Lungren, Matthew P. and Ng, Andrew Y.},
  title     = {{CheXpert}: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {33},
  pages     = {590--597},
  year      = {2019},
  doi       = {10.1609/aaai.v33i01.3301590}
}
```

**BibTeX fields not independently checked (confirm before submission):**
- The CVPR 2017 DOI (10.1109/CVPR.2017.369).
- The CheXpert AAAI DOI and pages.
- The ISBI 2023 venue for arXiv 2211.04279 (the arXiv PDF itself doesn't state a venue).
- The ECCV 2022 Workshops LNCS volume and pages.
- The SkinCon NeurIPS proceedings page numbers.

All other entries were checked against Crossref, arXiv or publisher metadata.
