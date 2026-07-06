# Session 2 - From Dataset to Model: PyTorch Workflows

*From Pixels to Patients - AAPM 2026 (Session 2 of 3)*

Session 2 starts with the NIfTI export produced in Session 1: one `image.nii.gz`, a
`masks/` folder, optional `doses/`, and a `metadata.json` sidecar for each anonymized
series.

The goal is a reproducible path from that curated dataset to a trained, evaluated, and
packaged PyTorch model. The example task is 3D GTV segmentation from CT, with optional
RT-structure context channels such as lungs, heart, esophagus, and cord.

---

## Learning objectives

By the end of this session you should be able to:

1. **Index** a Session 1 NIfTI dataset without leaking patient identity across splits.
2. **Build** PyTorch datasets and dataloaders for 3D medical volumes and RT masks.
3. **Apply** spatial and intensity augmentations that respect image and label geometry.
4. **Train, validate, and test** a compact 3D U-Net with reproducible configuration.
5. **Package** model weights, preprocessing assumptions, metrics, and split manifests for
   handoff to deployment and monitoring workflows.

---

## What's in this folder

| File | What it is |
|------|------------|
| `NSCLC_NIfTI_to_PyTorch.ipynb` | The runnable companion notebook for the talk. |
| `Guide.md` | Orientation and run instructions. |
| `train_gtv_segmentation.py` | Script version of the notebook training workflow. |
| `pytorch_workflow/` | Reusable PyTorch data, model, training, and packaging code. |

---

## Quick start

1. Run Session 1 through the NIfTI export step.
2. Confirm the default dataset path exists:
   ```
   aapm_nsclc/nifti/
   ```
3. Install the Session 2 dependencies in your Python environment:
   ```
   pip install torch nibabel numpy pandas matplotlib
   ```
4. Open `NSCLC_NIfTI_to_PyTorch.ipynb` and run top to bottom.

Training outputs are written locally under `Session2/outputs/`, which is ignored by git.

---

## The pipeline, step by step

The notebook follows the same path used in the session:

1. **Receive the dataset** - start from Session 1's anonymized NIfTI tree.
2. **Index the cases** - find each `image.nii.gz`, target mask, optional context masks,
   and metadata sidecar.
3. **Split by patient** - keep every series from the same patient hash in one split.
4. **Load 3D tensors** - convert NIfTI volumes into channel-first PyTorch tensors.
5. **Normalize and augment** - clip CT values, crop fixed-size patches, flip volumes, and
   perturb intensity without blurring labels.
6. **Train** - fit a compact 3D U-Net on GTV segmentation.
7. **Validate and test** - report held-out Dice and loss using frozen splits.
8. **Package** - save weights, config, metrics, split indexes, and a model card.
9. **Hand off** - pass the deployment bundle to Session 3 for input validation, drift
   monitoring, and governance.

---

## Expected Session 1 input layout

```
nifti/
  <patient>/<study>/<series>/
    image.nii.gz
    masks/
      gtv.nii.gz
      lung_l.nii.gz
      lung_r.nii.gz
      heart.nii.gz
      esophagus.nii.gz
      cord.nii.gz
    doses/
    metadata.json
```

Only `image.nii.gz` and `masks/gtv.nii.gz` are required for the demo model. Other masks
are used as optional anatomical context channels when present; missing context masks are
filled with zeros so the model interface stays stable.

---

## Modeling task

GTV segmentation is a natural first modeling task because it uses the image and RT
structure masks produced directly by Session 1. Clinical outcome prediction can be added
once endpoint definitions, follow-up windows, and censoring rules are ready. Starting
with segmentation keeps the session focused on the core PyTorch workflow: dataloaders,
3D augmentation, leakage-safe splits, training, evaluation, and model packaging.

---

## Privacy and reproducibility

Do not commit NIfTI data, manifests with identifiers, model checkpoints trained on
non-shareable data, or any re-identification key. This folder keeps generated artifacts
under `Session2/outputs/`, and that path is ignored by git.
