"""Score a single case against a previously-built QC panel — the deployment use case.

``run_qc_checks.py`` *builds* the panel from the training cohort. In production you build it
once, then reuse it: this loads the saved ``panel_check.json`` and checks one case's contours,
which is what you'd call on each new model output.

    python Session3/score_case.py                         # a held-out case (auto-picked)
    python Session3/score_case.py path/to/<series_dir>    # a specific case (image.nii.gz + masks/)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from qc_workflow import (
    CaseRecord, QCConfig, QCPanel, check_panel, discover_cases, find_nifti_root,
    load_case, split_by_patient,
)


def main() -> None:
    cfg = QCConfig()
    panel_path = Path(cfg.output_dir) / "panel_check.json"
    if not panel_path.exists():
        raise SystemExit(f"No panel at {panel_path}. Build it first: python Session3/run_qc_checks.py")
    panel = QCPanel.load(panel_path)

    profile_path = Path(cfg.output_dir) / "training_profile.json"
    profile_spacing = json.loads(profile_path.read_text()).get("spacing") if profile_path.exists() else None

    if len(sys.argv) > 1:
        series_dir = Path(sys.argv[1])
        masks_dir = series_dir / "masks"
        rec = CaseRecord(
            case_id=series_dir.name, patient_id=series_dir.name,
            image_path=series_dir / "image.nii.gz",
            mask_paths={p.name[:-7]: p for p in sorted(masks_dir.glob("*.nii.gz"))},
        )
    else:
        root = find_nifti_root() or cfg.nifti_root
        records = discover_cases(root, required_masks=(cfg.target_mask,))
        rec = split_by_patient(records, holdout_fraction=0.3, seed=cfg.seed)["holdout"][0]
        print(f"No case given — using a held-out case: {rec.case_id}")

    image, masks, spacing = load_case(rec, tuple(panel.structures))
    print(f"\nQC for {rec.case_id} (spacing z,y,x = {tuple(round(s, 2) for s in spacing)} mm):")
    for name, rep in check_panel(image, masks, spacing, panel, cfg, profile_spacing).items():
        reasons = "; ".join(rep.get("reasons", [])) if rep["verdict"] != "pass" else ""
        print(f"  {name:<10} {rep['verdict'].upper():<8} {reasons}")


if __name__ == "__main__":
    main()
