# From Pixels to Patients
### Practical Pipelines for Implementing AI in Radiation Oncology
**AAPM 2026 · A three-part educational track**

Artificial intelligence is transforming radiation oncology, but clinical adoption is
slowed by a persistent gap between theory and practice. This track closes that gap with
a start-to-finish playbook: from raw treatment data, to a trained model, to safe clinical
deployment. Each session pairs didactic content with a demonstration of open-source tools,
and everything here — reference code, notebooks, and templates — is meant to be run.

Presented by **Brian M. Anderson, PhD**, Radiation Medicine & Applied Sciences.

---

## Repository layout

```
AAPM2026_PixelsToPatients/
├── Session1/   From DICOM to a Generalized Research Dataset
├── Session2/   From Dataset to Model: PyTorch Workflows
├── Session3/   From Model to Clinic: Deployment and Integration
└── README.md   (you are here)
```

Each `SessionN/` folder is self-contained and holds that session's materials — a
`Guide.md` orienting you to the contents, plus the runnable notebooks, code, and templates
demonstrated in the talk.

---

## The three sessions

### Session 1 — From DICOM to a Generalized Research Dataset  ·  `Session1/`
Radiation oncology data is locked inside complex DICOM objects (CT, MR, RTSTRUCT, RTPLAN,
RTDOSE). This session shows a reproducible workflow to convert them into generalized,
AI-ready formats (NIfTI, NumPy, metadata sidecars) while preserving the clinical metadata
that makes the data useful — and an anonymization key that lets the dataset keep growing.

**Inside:** an end-to-end notebook that downloads the public **NSCLC-Radiomics** cohort
from TCIA and runs the full pipeline with [DicomRTTool](https://github.com/brianmanderson/Dicom_RT_and_Images_to_Mask).
See [`Session1/Guide.md`](Session1/Guide.md) to get started.

### Session 2 — From Dataset to Model: PyTorch Workflows  ·  `Session2/`
Once the data is curated, it must be shaped for deep learning. This session walks through
dataloaders for 3D/4D medical data, augmentation strategies unique to RT, and reproducible
PyTorch pipelines — with an emphasis on version control, dataset-leakage prevention, and
scaling to large cohorts.

*Materials to be added.*

### Session 3 — From Model to Clinic: Deployment and Integration  ·  `Session3/`
Developing a model is only half the challenge; the harder part is safe deployment. This
session covers the QC infrastructure needed to manage it — input validation, output sanity
checking, and detecting model drift, out-of-distribution cases, and silent failures — drawn
from real-world experience (including our own mistakes), plus a framework for monitoring
and governance across the full model lifecycle.

**Inside:** a **reference-free** contour-QC layer that characterizes the Session 1/2 training
data and checks new contours with no ground truth — input validation (is it CT? in-range
spacing?), a radiomic **shape** outlier detector, and a **positional** atlas that flags a GTV
placed where the surrounding anatomy says it shouldn't be. Ships an end-to-end script
(`run_qc_checks.py`), a reusable `qc_workflow/` package, and a companion notebook. See
[`Session3/Guide.md`](Session3/Guide.md).

---

## Getting started

The materials are Python-based. A typical setup:

```bash
git clone https://github.com/brianmanderson/AAPM2026_PixelsToPatients.git
cd AAPM2026_PixelsToPatients
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

Then open the session you're interested in and follow its `Guide.md`. Session 1 installs
its own dependencies from within the notebook (`DicomRTTool`, `tcia_utils`, `SimpleITK`, …).

**Prerequisites:** basic familiarity with DICOM and radiation therapy workflows. Python
experience is helpful but not required — worked examples are provided throughout.

---

## Tools & data

- **DicomRTTool** — DICOM ⇄ NIfTI/NumPy conversion for RT data · <https://github.com/brianmanderson/Dicom_RT_and_Images_to_Mask> · `pip install DicomRTTool`
- **NSCLC-Radiomics** — public sample cohort from TCIA · [collection page](https://www.cancerimagingarchive.net/collection/nsclc-radiomics/) · DOI [10.7937/K9/TCIA.2015.PF0M9REI](https://doi.org/10.7937/K9/TCIA.2015.PF0M9REI) · license **CC BY-NC 3.0** (cite the collection and Aerts et al. 2014 if you use it)

---

## Notes on data & privacy

This repository holds **code and documentation only** — no patient data. The `.gitignore`
excludes downloaded DICOM/NIfTI, generated datasets, and, most importantly, any
`anonymization_key.json` (which maps study IDs back to MRNs and must never leave a secured,
access-controlled location). Keep it that way: never commit imaging data or re-identification keys.

