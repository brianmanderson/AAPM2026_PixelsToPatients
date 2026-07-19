# Session 3 — From Model to Clinic

*From Pixels to Patients · AAPM 2026 (Session 3 of 3)*

Building a model is the easy half; knowing when to trust its output is the hard half. Session 3
adds two QC tools between the Session 2 model and the clinic:

- **`qc_workflow`** — *reference-free* QC. Scores a new contour with **no ground truth**, on two
  axes: radiomic **shape** (is it an outlier?) and **position** (is it where this organ sits?),
  wrapped in **input validation** (is it a CT, at a spacing the model has seen?). After Elguindi
  et al., *Reference-Free QC of OAR Contours via Radiomic and Positional Signatures*.
- **`contour_compare`** — *reference-based* comparison. When you *do* have a manual/corrected
  contour, grades the AI against it with the metrics that predict **editing effort**: volumetric
  Dice, surface Dice @1/2/3 mm, added/total path length, estimated time saved, and a 1–5 rating.

Use `qc_workflow` at deployment (no reference exists); use `contour_compare` during validation
and audit (a reference does exist).

## Files

| File | What it is |
|------|------------|
| `NSCLC_QC_and_Deployment_Checks.ipynb` | Companion notebook for the talk. |
| `run_qc_checks.py` | Reference-free QC, end to end (characterize → build → check). |
| `score_case.py` | Score one case against a prebuilt QC panel (the deployment call). |
| `compare_contours.py` | Reference-based demo: edit a contour, then grade the edit. |
| `qc_workflow/` | Reference-free QC package. |
| `contour_compare/` | Reference-based comparison package. |
| `test_contour_compare.py` | Self-tests for `contour_compare` (no data needed). |

## Quick start

Run Session 1's NIfTI export first (a few NSCLC-Radiomics patients is enough — no trained model
needed, the checks reason about the data). Then, from the repo root:

```
pip install numpy scipy nibabel pandas matplotlib
python Session3/run_qc_checks.py       # reference-free QC over the structure set
python Session3/compare_contours.py    # reference-based AI-vs-manual comparison
```

Outputs land in `Session3/outputs/` (git-ignored).

---

## `qc_workflow` — reference-free QC

| Axis | Question | How |
|------|----------|-----|
| **Shape** | Is the contour a radiomic outlier vs. training? | Robust (median/MAD) Mahalanobis over ~19 features → z-score. |
| **Position** | Is it where this structure normally sits? | Generalized-Procrustes atlas of organ centroids → wrong-place residual. |
| **Input** | Is this a CT at a known spacing? | HU range + spacing vs. training range. |

The axes are complementary: a well-shaped GTV in the wrong lung passes shape but fails position;
an over-segmented leak fails shape while its centroid looks fine.

**Panel over the structure set.** Assuming Session 2 trains the GTV *and* the thoracic OARs, the
check fits one shape reference per structure (`gtv`, `lung_l/r`, `heart`, `esophagus`, `cord`)
plus one shared position atlas. OARs are anatomically stereotyped, so their shape distributions
are tight and outliers are meaningful; the shared atlas adds a **laterality** check that catches
a swapped `Lung_L`/`Lung_R`.

`run_qc_checks.py` writes `training_profile.json`, `panel_check.json`, `qc_report.csv`, and a
`QC_CARD.md` — the artifacts a monitoring/governance process consumes.

> **Small-cohort note.** A reference-free check is only as good as its reference distribution. On
> a few cases it is illustrative, not reliable (the shape model falls back to a robust per-feature
> form and the atlas floors its tolerance). Scale the cohort up before trusting a verdict.

---

## `contour_compare` — reference-based comparison

Volumetric Dice weights the contour interior, which editors rarely touch. **Boundary** metrics
track the actual work: in Vaassen et al. (2020), added path length correlated **R=0.87** with
correction time. `contour_compare` reports, per structure:

| Metric | Meaning |
|--------|---------|
| Volumetric Dice | Interior overlap (shown for contrast — it barely moves with quality). |
| Surface Dice @1/2/3 mm | Fraction of both surfaces agreeing within τ mm (area-weighted). |
| APL / TPL | Contour length that must be redrawn / total reference length (cm). |
| Time saved | `(TPL − APL) / rate`; `%saved = 100·(TPL−APL)/TPL` is rate-independent. |
| Rating 1–5 | Suggested clinical-acceptability score (a prior for the reviewer). |

The **1–5 rubric** is the semi-standard MD Anderson scale (Baroudi et al., *Diagnostics* 2023):
5 use-as-is · 4 minor stylistic · 3 minor necessary (faster than scratch) · 2 major (prefer
scratch) · 1 unusable. `suggest_rating` maps geometry → a score using one anchor the rubric
itself gives — level 3 is "faster than starting from scratch," i.e. `time_saved > 0` — with a
surface-Dice floor for "grossly wrong shape." Thresholds live in `RubricThresholds`; tune them.

`compare_contours.py` takes one real case, treats its manual GTV as truth, synthesizes an AI GTV
(a localized in-plane edit, written to `masks_ai/` as a real second data point), and grades it —
then sweeps edit severity to walk the rubric 5→1. The teaching point: volumetric Dice stays high
(≈0.75 even at rating 1) while surface Dice and %-saved track quality.

---

## Scope

Workshop artifacts, not clinical devices — **prioritizers for human review**, not approvers.
Trained/validated on **real CT only** (MR / pseudo-CT is rejected at input validation). The
suggested rating is a prior, never a verdict.

## References

- Elguindi et al., *Reference-Free QC of OAR Contours via Radiomic and Positional Signatures*
  — companion method for `qc_workflow`.
- Vaassen et al., *Evaluation of measures for assessing time-saving of automatic OAR segmentation.*
  Phys Imaging Radiat Oncol. 2020;13:1-6. [doi:10.1016/j.phro.2019.12.001](https://doi.org/10.1016/j.phro.2019.12.001) — APL / surface DSC vs. editing time.
- Nikolov et al., *Clinically Applicable Segmentation of Head and Neck Anatomy for Radiotherapy.*
  J Med Internet Res. 2021;23(7):e26151. [doi:10.2196/26151](https://doi.org/10.2196/26151) — surface Dice at tolerance τ.
- Baroudi et al., *Automated Contouring and Planning: What Is 'Clinically Acceptable'?*
  Diagnostics. 2023;13(4):667. [doi:10.3390/diagnostics13040667](https://doi.org/10.3390/diagnostics13040667) — the 1–5 acceptability scale.
- Data: NSCLC-Radiomics — [TCIA](https://www.cancerimagingarchive.net/collection/nsclc-radiomics/) · CC BY-NC 3.0 (cite Aerts et al., *Nat Commun.* 2014;5:4006).
