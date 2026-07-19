# From Pixels to Patients
### Practical Pipelines for Implementing AI in Radiation Oncology
**AAPM 2026 · a three-part educational track**

A start-to-finish playbook: raw treatment data -> trained model -> safe clinical deployment.
Every session pairs a short talk with runnable open-source code, notebooks, and templates.

Presented by **Brian M. Anderson, PhD**, Radiation Medicine & Applied Sciences.

## Layout

```
Session1/   DICOM -> generalized research dataset
Session2/   dataset -> model (PyTorch)
Session3/   model -> clinic (deployment QC)
```

Each `SessionN/` is self-contained: start from its `Guide.md`, then run the notebook/code.

## Sessions

**1 · DICOM to a research dataset** — [`Session1/Guide.md`](Session1/Guide.md)
Convert RT DICOM (CT/MR/RTSTRUCT/RTPLAN/RTDOSE) into AI-ready NIfTI/NumPy with metadata and an
anonymization key, using [DicomRTTool](https://github.com/brianmanderson/Dicom_RT_and_Images_to_Mask).
Downloads the public **NSCLC-Radiomics** cohort from TCIA.

**2 · Dataset to a model** — `Session2/`
3D/4D dataloaders, RT-specific augmentation, and reproducible PyTorch pipelines with
leakage-safe, patient-level splits.

**3 · Model to the clinic** — [`Session3/Guide.md`](Session3/Guide.md)
Two contour-QC tools:
- **`qc_workflow`** — *reference-free* QC (no ground truth): input validation + radiomic **shape**
  outlier + **positional** atlas (catches a mislocated GTV or swapped Lung_L/R).
- **`contour_compare`** — *reference-based* comparison (AI vs. manual): volumetric Dice, surface
  Dice @1/2/3 mm, added/total path length, estimated time saved, and a 1-5 acceptability rating.

## Getting started

```bash
git clone https://github.com/brianmanderson/AAPM2026_PixelsToPatients.git
cd AAPM2026_PixelsToPatients
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

Then open a session and follow its `Guide.md`. Session 1 installs its own deps from the notebook;
Session 3 needs `numpy scipy nibabel pandas matplotlib`.

## Data & privacy

**Code and documentation only — no patient data.** `.gitignore` excludes downloaded imaging,
generated datasets, and any `anonymization_key.json` (which maps study IDs to MRNs). Never commit
imaging data or re-identification keys.

## References

**Session 3 methods**
- Vaassen F, et al. *Evaluation of measures for assessing time-saving of automatic organ-at-risk
  segmentation in radiotherapy.* Phys Imaging Radiat Oncol. 2020;13:1-6.
  [doi:10.1016/j.phro.2019.12.001](https://doi.org/10.1016/j.phro.2019.12.001) — added path length
  & surface DSC vs. editing time.
- Nikolov S, et al. *Clinically Applicable Segmentation of Head and Neck Anatomy for Radiotherapy.*
  J Med Internet Res. 2021;23(7):e26151. [doi:10.2196/26151](https://doi.org/10.2196/26151) —
  surface Dice at tolerance τ.
- Baroudi H, et al. *Automated Contouring and Planning in Radiation Therapy: What Is 'Clinically
  Acceptable'?* Diagnostics. 2023;13(4):667.
  [doi:10.3390/diagnostics13040667](https://doi.org/10.3390/diagnostics13040667) — the 1-5
  clinical-acceptability scale.
- Elguindi S, et al. *Reference-free QC of organ-at-risk contours via radiomic and positional
  signatures* — companion method for `qc_workflow`.

**Data & tools**
- Aerts HJWL, et al. *Decoding tumour phenotype by noninvasive imaging using a quantitative
  radiomics approach.* Nat Commun. 2014;5:4006.
  [doi:10.1038/ncomms5006](https://doi.org/10.1038/ncomms5006).
- NSCLC-Radiomics — [TCIA collection](https://www.cancerimagingarchive.net/collection/nsclc-radiomics/) ·
  [doi:10.7937/K9/TCIA.2015.PF0M9REI](https://doi.org/10.7937/K9/TCIA.2015.PF0M9REI) · CC BY-NC 3.0.
- DicomRTTool — <https://github.com/brianmanderson/Dicom_RT_and_Images_to_Mask> · `pip install DicomRTTool`.
