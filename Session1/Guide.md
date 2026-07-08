# Session 1 — From DICOM to a Generalized Research Dataset

*From Pixels to Patients · AAPM 2026 (Session 1 of 3)*

This guide orients you to the Session 1 materials and walks you through running them.
The goal of the session is a **reproducible path from raw DICOM-RT to an analysis-ready
dataset** — NIfTI volumes and masks, a clinical-metadata sidecar, and an anonymization
key that lets the dataset keep growing.

---

## Learning objectives

By the end of this session you should be able to:

1. **Identify** the challenges in extracting, harmonizing, and curating DICOM-RT objects
   (CT/MR, RTSTRUCT, RTDOSE) for an AI pipeline.
2. **Compare** generalized storage formats — NIfTI vs NumPy — and choose the right one.
3. **Apply** a reproducible conversion that preserves the clinical metadata you care about
   (age, voxel spacing, acquisition, outcomes) and maintains an anonymization key.

---

## What's in this folder

| File | What it is |
|------|------------|
| `NSCLC_Radiomics_DICOM_to_NIfTI.ipynb` | The runnable, end-to-end companion notebook — downloads the NSCLC-Radiomics cohort from TCIA and executes the entire pipeline. |
| `Guide.md` | This document. |

The notebook is the heart of the session; this guide is the map.

---

## Quick start

1. **Install dependencies.** The first notebook cell handles this:
   ```
   pip install DicomRTTool tcia_utils SimpleITK pandas matplotlib nibabel
   ```
2. **Open the notebook** and run the cells top to bottom.
3. **Download the data.** Section 1 pulls a small subset (default: 3 patients) of the public
   **NSCLC-Radiomics** collection from TCIA — no login required. Downloading everything is
   ~36 GB, so start small.
4. **Watch the pipeline run** through to an exported NIfTI dataset and a verification plot.

> **Disk space & time:** a few patients download in a couple of minutes and use well under
> a gigabyte. Scale up only when you're ready.

---

## The pipeline, step by step

The notebook mirrors the talk. Each stage maps to a slide:

1. **Download** — grab a subset of NSCLC-Radiomics (CT + RTSTRUCT) via `tcia_utils`, or the
   NBIA Data Retriever for the full cohort.
2. **Discover** — `walk_through_folders` scans the tree, groups files by series, and links
   each RTSTRUCT to its image. `return_rois` lists every ROI found.
3. **Survey** — `create_manifest` writes one row per series: image spacing plus one
   `<roi> cc` column per structure (its mask volume). This is your cohort overview.
4. **Spot outliers** — read the manifest as a QC instrument. Odd `spacing_z` values,
   ROI volumes far from the cohort norm, and blank cells (missing structures) all surface
   here. The notebook flags them with an IQR rule and histograms.
5. **Select & normalize ROIs** — map inconsistent names (`Lung-Left`, `lung_l`, `left lung`)
   onto canonical labels with `ROIAssociationClass`.
6. **Set the output voxel size** — choose one target grid (e.g. isotropic `1×1×1 mm`, or a
   native-like `0.98×0.98×3 mm`) so every case is comparable. Resampling uses **linear**
   interpolation for the image and dose, **nearest-neighbour** for masks (never blur a label).
7. **Preserve metadata** — request extra DICOM tags (age, sex, manufacturer, kVp, slice
   thickness). They're written to a **grouped, versioned `metadata.json`** (`schema_version: 2`)
   beside every exported series — DICOM features organized by category (`image`, `structures`,
   `doses`, `plans`), with your requested tags nested under the owning category's `tags` sub-dict
   (image tags under `image["tags"]`). Empty categories are omitted, so an image-only series still
   parses cleanly. Pass `metadata_style="flat"` for the historical `{name: value}` dict.
8. **Convert** — `write_to_folder` does it all in one call: resampled `image.nii.gz`, one
   mask per ROI, dose (when present), the metadata sidecar, a cohort `manifest.csv`, and an
   `anonymization_key.json`.
9. **Verify** — load an exported case, confirm image and masks share geometry, read the grouped
   `metadata.json` (tags under `image["tags"]`, each exported ROI's `volume_cc` and
   `exported_file`), and preview a slice with the GTV overlaid.
10. **Grow the dataset** — the deterministic anonymization key and incremental manifest let
    you re-pull the same patients and fold in new imaging or follow-up without re-identifying.

---

## Output layout

After the conversion step you'll have a tidy per-case tree:

```
nifti/
  <patient>/<study>/<series>/
    image.nii.gz
    masks/
      gtv.nii.gz  lung_l.nii.gz  lung_r.nii.gz  ...
    doses/            # only when dose is present
    metadata.json     # grouped DICOM features (schema v2); your tags under image["tags"]
  manifest.csv              # identifiers, spacing, per-ROI volume (cc)
  anonymization_key.json    # reverse lookup — keep this OUT of git
```

---

## A note on privacy

The `anonymization_key.json` maps each study hash back to its MRN. **It is
re-identification data** — store it offline, access-controlled, and never commit it. The
repo's `.gitignore` already excludes it along with all imaging data; don't override that.

---

## Links

- **Tool:** DicomRTTool — <https://github.com/brianmanderson/Dicom_RT_and_Images_to_Mask> · `pip install DicomRTTool`
- **Data:** NSCLC-Radiomics — [TCIA collection](https://www.cancerimagingarchive.net/collection/nsclc-radiomics/) · DOI [10.7937/K9/TCIA.2015.PF0M9REI](https://doi.org/10.7937/K9/TCIA.2015.PF0M9REI) · CC BY-NC 3.0
- **Next:** Session 2 builds reproducible PyTorch pipelines on this exact dataset.
