"""End-to-end Session 3 QC over the whole structure set.

Assuming Session 2 trains the GTV *and* the thoracic OARs, this characterizes the training
data, builds a reference-free check per structure (GTV + Lung_L/R + Heart + Esophagus + Cord)
sharing one positional atlas, runs it on held-out cases, and shows it catching silent
failures — including a swapped Lung_L/Lung_R.

    python Session3/run_qc_checks.py

Reads the Session 1 export (``Session1/aapm_nsclc/nifti``), writes under ``Session3/outputs/``.
No GPU and no trained model needed — the check reasons about the data, not the network.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from qc_workflow import (
    QCConfig,
    build_panel,
    characterize_panel,
    check_contour,
    check_panel,
    discover_cases,
    find_nifti_root,
    load_case,
    split_by_patient,
)


def _gtv_corruptions(image, gtv, spacing):
    """Three ways a GTV prediction goes silently wrong: wrong place, wrong shape, wrong input."""
    from scipy import ndimage as ndi
    zz, yy, xx = np.where(gtv > 0.5)
    moved = np.zeros_like(gtv)
    yi = np.clip(yy + gtv.shape[1] // 3, 0, gtv.shape[1] - 1)
    xi = np.clip(xx + gtv.shape[2] // 3, 0, gtv.shape[2] - 1)
    moved[zz, yi, xi] = 1
    norm = 2.0 * (np.clip(image, -1000, 400) + 1000) / 1400 - 1.0
    return {
        "wrong_place": (image, moved),
        "wrong_shape": (image, ndi.binary_dilation(gtv, iterations=6).astype(np.float32)),
        "wrong_modality": (norm.astype(np.float32), gtv),
    }


def main() -> None:
    cfg = QCConfig()
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    nifti_root = find_nifti_root() or cfg.nifti_root
    records = discover_cases(nifti_root, required_masks=(cfg.target_mask,))
    if len(records) < 4:
        print(f"Found {len(records)} cases under {nifti_root}. Run Session 1's export first.")
        return

    splits = split_by_patient(records, holdout_fraction=0.3, seed=cfg.seed)
    print(f"{len(records)} cases | reference={len(splits['reference'])} holdout={len(splits['holdout'])}")

    # 1) characterize the training data (every structure) + 2) build the panel
    profile = characterize_panel(splits["reference"], cfg)
    print("\nTraining cohort — structures characterized:")
    for s in profile.structures:
        n = len(profile.feature_matrices[s])
        print(f"  {s:<12} {n:>3} examples ({profile.coverage[s]:.0%} of cases)")
    (out / "training_profile.json").write_text(json.dumps(profile.to_json_dict(), indent=2))

    panel = build_panel(profile, cfg)
    panel.save(out / "panel_check.json")
    print(f"\nBuilt QC panel: {len(panel.structures)} structures "
          f"({', '.join(panel.structures)}), shared atlas of "
          f"{len(panel.atlas.get('template', {}))} organs.")

    # 3) run the panel on the held-out cases
    rows = []
    all_masks = (cfg.target_mask, *cfg.context_masks)
    print("\nHeld-out cases (verdict per structure):")
    for rec in splits["holdout"]:
        image, masks, spacing = load_case(rec, all_masks)
        results = check_panel(image, masks, spacing, panel, cfg, profile.spacing)
        flags = [f"{n}={r['verdict']}" for n, r in results.items() if r["verdict"] != "pass"]
        print(f"  {rec.case_id[:34]:<34} {'all pass' if not flags else '  '.join(flags)}")
        for n, r in results.items():
            rows.append({"case_id": rec.case_id, "structure": n, "kind": "holdout",
                         "verdict": r["verdict"], "shape_z": r["shape"].get("z"),
                         "position_z": r["position"].get("z"), "reasons": "; ".join(r.get("reasons", []))})

    # 4a) swapped Lung_L / Lung_R — the classic OAR labelling error
    demo = splits["holdout"][0]
    image, masks, spacing = load_case(demo, all_masks)
    if masks.get("lung_l") is not None and masks.get("lung_r") is not None:
        swapped = dict(masks)
        swapped["lung_l"], swapped["lung_r"] = masks["lung_r"], masks["lung_l"]
        print("\nSwapped Lung_L / Lung_R (should flag both as wrong-place):")
        res = check_panel(image, swapped, spacing, panel, cfg, profile.spacing)
        for n in ("lung_l", "lung_r"):
            r = res[n]
            print(f"  {n:<10} {r['verdict'].upper():<7} pos z={r['position'].get('z')} "
                  f"| {'; '.join(r['reasons'])}")
            rows.append({"case_id": f"{demo.case_id}::swap_lungs", "structure": n, "kind": "synthetic",
                         "verdict": r["verdict"], "shape_z": r["shape"].get("z"),
                         "position_z": r["position"].get("z"), "reasons": "; ".join(r["reasons"])})

    # 4b) GTV corruptions — wrong place / shape / modality, against the GTV reference
    print("\nGTV corruptions (should be flagged):")
    ctx = {}
    for m in cfg.context_masks:
        arr = masks.get(m)
        if arr is not None and arr.sum() > 0:
            zz, yy, xx = np.where(arr > 0.5)
            ctx[m] = [float(zz.mean() * spacing[0]), float(yy.mean() * spacing[1]),
                      float(xx.mean() * spacing[2])]
    for kind, (img, msk) in _gtv_corruptions(image, masks[cfg.target_mask], spacing).items():
        r = check_contour(img, msk, ctx, spacing, panel.references[cfg.target_mask], cfg, profile.spacing)
        print(f"  {kind:<16} {r['verdict'].upper():<7} | {'; '.join(r['reasons'])}")
        rows.append({"case_id": f"{demo.case_id}::{kind}", "structure": cfg.target_mask,
                     "kind": "synthetic", "verdict": r["verdict"], "shape_z": r["shape"].get("z"),
                     "position_z": r["position"].get("z"), "reasons": "; ".join(r["reasons"])})

    with (out / "qc_report.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "structure", "kind", "verdict",
                                          "shape_z", "position_z", "reasons"])
        w.writeheader()
        w.writerows(rows)
    _write_qc_card(out, cfg, panel, profile, rows)
    print(f"\nWrote {out / 'qc_report.csv'}, training_profile.json, panel_check.json, QC_CARD.md")


def _write_qc_card(out: Path, cfg: QCConfig, panel, profile, rows) -> None:
    flagged = sum(1 for r in rows if r["verdict"] not in ("pass", "skipped"))
    (out / "QC_CARD.md").write_text("\n".join([
        "# Contour QC Panel — Card",
        "",
        f"Reference cohort: {panel.n_cases} cases (Session 1 NSCLC-Radiomics export).",
        f"Structures checked ({len(panel.structures)}): {', '.join(panel.structures)}.",
        f"Shared position atlas: {', '.join(sorted(panel.atlas.get('template', {}))) or 'n/a'}.",
        f"Shape features: {len(next(iter(panel.references.values())).shape['feature_names'])} "
        "radiomic geometry/appearance descriptors per structure.",
        "",
        "## Axes",
        "- **Input validation** — HU range (is it CT?) + spacing vs. training range.",
        f"- **Shape** — per-structure Mahalanobis OOD; review z≥{cfg.shape_z_review}, fail z≥{cfg.shape_z_fail}.",
        f"- **Position** — shared Procrustes atlas; wrong-place flag at z>{cfg.position_z_flag} "
        "(catches swapped L/R and mislocated contours).",
        "",
        f"## This run: {flagged} structure-verdicts flagged for review.",
        "",
        "Reference-free (no ground truth at scoring time). A prioritizer for human review,",
        "not a clinically-validated device. Real CT only. Methodology: Elguindi et al.,",
        "*Reference-Free QC of Organ-at-Risk Contours via Radiomic and Positional Signatures*.",
        "",
    ]))


if __name__ == "__main__":
    main()
