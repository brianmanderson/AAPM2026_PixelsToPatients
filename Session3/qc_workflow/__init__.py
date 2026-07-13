"""Reference-free QC for AI contours — Session 3 of *From Pixels to Patients* (AAPM 2026).

Characterize the training cohort, freeze it into a reference, then check new contours with
no ground truth: input validation (is it CT? plausible spacing?) + output sanity (radiomic
shape OOD + wrong-place position). Methodology after Elguindi et al., "Reference-Free
Quality Control of Organ-at-Risk Contours via Radiomic and Positional Signatures".
"""
from __future__ import annotations

from .characterize import CohortProfile, build_reference, characterize_cohort
from .checks import check_contour, check_panel, input_validation
from .config import QCConfig
from .data import CaseRecord, discover_cases, find_nifti_root, load_case, split_by_patient
from .features import SHAPE_FEATURES, extract_features
from .panel import PanelProfile, QCPanel, build_panel, characterize_panel
from .reference import ReferenceCheck

__all__ = [
    "QCConfig",
    "CaseRecord", "discover_cases", "find_nifti_root", "load_case", "split_by_patient",
    "SHAPE_FEATURES", "extract_features",
    # single structure (the GTV walk-through)
    "characterize_cohort", "build_reference", "CohortProfile",
    "ReferenceCheck", "check_contour", "input_validation",
    # the whole structure set (GTV + OARs)
    "characterize_panel", "build_panel", "PanelProfile", "QCPanel", "check_panel",
]
