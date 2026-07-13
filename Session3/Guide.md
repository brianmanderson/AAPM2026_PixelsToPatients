# Session 3 — From Model to Clinic: Deployment and Integration

*From Pixels to Patients · AAPM 2026 (Session 3 of 3)*

Developing a model is the easy half. The hard half is knowing, in production, whether you
can *trust a given prediction* — with no ground truth to compare against. Session 3 builds
the QC layer that sits between the Session 2 model and the clinic: it **characterizes the
training data**, freezes that into a **reference-free check**, and uses it to catch the
silent failures that erode trust — out-of-distribution inputs, malformed contours, and
contours placed where the anatomy says they shouldn't be.

The method follows Elguindi et al., *"Reference-Free Quality Control of Organ-at-Risk
Contours via Radiomic and Positional Signatures"* — two orthogonal axes, neither of which
needs a reference contour at scoring time.

---

## Learning objectives

By the end of this session you should be able to:

1. **Characterize** a training cohort as a distribution — acquisition (spacing, HU), and the
   target's shape and positional signatures — and understand why that distribution *is* the
   reference.
2. **Build** a reference-free check that scores a new contour with no ground truth, on two
   axes: radiomic **shape** (is the drawing an outlier?) and **position** (is it where this
   structure normally sits?).
3. **Validate inputs and outputs** at deployment — reject non-CT inputs and out-of-range
   acquisitions before the model runs; flag implausible predictions after.
4. **Reason about drift and silent failure**: what the check catches, what it can't, and how
   it degrades when the reference cohort is small.

---

## What's in this folder

| File | What it is |
|------|------------|
| `NSCLC_QC_and_Deployment_Checks.ipynb` | The runnable companion notebook for the talk. |
| `Guide.md` | This document. |
| `run_qc_checks.py` | Script version: characterize → build → check, end to end. |
| `qc_workflow/` | Reusable QC code — features, characterization, the reference-free check. |

---

## Quick start

1. Run **Session 1** through the NIfTI export step (a handful of NSCLC-Radiomics patients is
   plenty). The check reasons about the data, so no trained model is required to try it.
2. Install the Session 3 dependencies:
   ```
   pip install numpy scipy nibabel pandas matplotlib
   ```
3. Run the end-to-end script from the repo root:
   ```
   python Session3/run_qc_checks.py
   ```
   or open `NSCLC_QC_and_Deployment_Checks.ipynb` and run top to bottom.

Outputs are written under `Session3/outputs/` (ignored by git).

> **Small-cohort note.** A reference-free check is only as good as the distribution it was
> built from. On a few cases it is illustrative, not reliable — the shape model falls back to
> a robust per-feature form and the position atlas floors its tolerance so it doesn't cry wolf.
> Scale the reference cohort up before you trust a verdict.

---

## The two axes

| Axis | Question | How |
|------|----------|-----|
| **Shape** | Is the contour's radiomic signature an outlier vs. the training targets? | Robust (median/MAD) Mahalanobis over ~19 geometry + appearance features → a z-score. |
| **Position** | Is the contour *where this structure normally sits*? | A generalized-Procrustes atlas of organ centroids; align on the surrounding anatomy (lungs/heart/cord/esophagus), then measure the target's wrong-place residual. |

They are complementary: a perfectly-shaped GTV pasted in the wrong lung passes shape but
fails position; an over-segmented leak fails shape while its centroid still looks fine.
Wrapped around both is **input validation** — is this a CT at all, at a spacing the model
has seen?

### Beyond the GTV — a panel over the whole structure set

A segmentation model rarely outputs one structure. Assuming Session 2 trains the GTV **and**
the thoracic OARs, Session 3 fits a **panel**: one shape reference per structure
(`gtv`, `lung_l`, `lung_r`, `heart`, `esophagus`, `cord`) plus a single shared positional
atlas. OARs are anatomically stereotyped, so their shape distributions are tighter than the
GTV's and outliers are more meaningful. The shared atlas adds a **laterality** check that
catches the classic OAR error — a swapped `Lung_L`/`Lung_R` — by aligning on the non-paired
midline anatomy and asking which side each lung actually landed on.

---

## The pipeline, step by step

The notebook mirrors the talk:

1. **Receive the dataset + model bundle** — start from Session 1's NIfTI tree; optionally read
   Session 2's `config.json` for the exact input assumptions (the CT normalization window).
2. **Characterize** — distill the training cohort into a profile: spacing and HU ranges, and
   each structure's shape distribution and the anatomical constellation it lives in.
3. **Build the panel** — freeze one shape reference per structure plus the shared positional
   atlas into one portable `panel_check.json`.
4. **Validate inputs** — HU range (is it CT?) and spacing vs. the training range.
5. **Check outputs** — score every structure on a case (shape + position + laterality);
   return pass / review / fail per structure with the reasons attached.
6. **See it catch failures** — a swapped `Lung_L`/`Lung_R`, plus three GTV corruptions
   (mislocated, over-segmented, non-CT input), demonstrate each axis firing.
7. **Package** — write `training_profile.json`, `panel_check.json`, a per-structure
   `qc_report.csv`, and a `QC_CARD.md` for governance and monitoring hand-off.

---

## Output layout

```
Session3/outputs/
  training_profile.json    # the characterized training distribution (per structure)
  panel_check.json         # the frozen per-structure shape models + shared position atlas
  qc_report.csv            # per-structure verdicts (holdout + synthetic corruptions)
  QC_CARD.md               # summary card for governance / monitoring
```

---

## Scope and honesty

This is a workshop artifact, not a clinical device. The check is a **prioritizer for human
review** — it ranks the riskiest contours first; it does not approve anything. It is trained
on **real CT only**; MR or MR-derived pseudo-CT is out of scope and is rejected at input
validation, because the HU-based features don't transfer.

---

## Links

- **Methodology:** Elguindi et al., *Reference-Free QC of Organ-at-Risk Contours via Radiomic
  and Positional Signatures* (companion package: `contour-qc`).
- **Prev:** Session 2 produced the model and the deployment bundle this session guards.
- **Data:** NSCLC-Radiomics — [TCIA](https://www.cancerimagingarchive.net/collection/nsclc-radiomics/) · CC BY-NC 3.0.
