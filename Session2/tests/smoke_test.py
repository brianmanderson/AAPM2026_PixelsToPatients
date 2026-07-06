from __future__ import annotations

import shutil
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

SESSION2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SESSION2))

from make_synthetic_session1 import create_synthetic_session1_export
from pixelstopatients import NiftiSegmentationDataset, TinyUNet3D
from pixelstopatients.config import TrainingConfig
from pixelstopatients.data import discover_cases, split_by_patient
from pixelstopatients.engine import evaluate, seed_everything, train_one_epoch
from pixelstopatients.packaging import save_deployment_bundle


def main() -> None:
    seed_everything(2026)
    root = SESSION2 / "tempworkspace" / "synthetic_nifti"
    run_dir = SESSION2 / "tempworkspace" / "smoke_run"
    if root.exists():
        shutil.rmtree(root)
    if run_dir.exists():
        shutil.rmtree(run_dir)

    create_synthetic_session1_export(root, n_patients=6)
    records = discover_cases(root, required_masks=("gtv",))
    assert len(records) == 6, len(records)

    splits = split_by_patient(records, val_fraction=0.2, test_fraction=0.2, seed=2026)
    assert splits["train"] and splits["val"] and splits["test"]
    assert set(r.patient_id for r in splits["train"]).isdisjoint(r.patient_id for r in splits["val"])
    assert set(r.patient_id for r in splits["train"]).isdisjoint(r.patient_id for r in splits["test"])

    config = TrainingConfig(
        nifti_root=root,
        output_dir=run_dir,
        context_masks=("lung_l", "lung_r", "heart"),
        patch_size=(16, 32, 32),
        epochs=1,
    )
    train_ds = NiftiSegmentationDataset(
        splits["train"],
        context_masks=config.context_masks,
        patch_size=config.patch_size,
        training=True,
    )
    val_ds = NiftiSegmentationDataset(
        splits["val"],
        context_masks=config.context_masks,
        patch_size=config.patch_size,
        training=False,
    )
    sample = train_ds[0]
    assert sample["image"].shape == (1 + len(config.context_masks), *config.patch_size)
    assert sample["target"].shape == (1, *config.patch_size)

    device = torch.device("cpu")
    model = TinyUNet3D(in_channels=1 + len(config.context_masks), features=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    train_metrics = train_one_epoch(model, train_loader, optimizer, device)
    val_metrics = evaluate(model, val_loader, device)
    assert "loss" in train_metrics and "dice" in val_metrics

    bundle = save_deployment_bundle(
        model,
        config,
        {"train": train_metrics, "val": val_metrics},
        run_dir / "deployment_bundle",
    )
    assert (bundle / "model_state.pt").exists()
    assert (bundle / "config.json").exists()
    assert (bundle / "metrics.json").exists()
    print("Smoke test passed.")


if __name__ == "__main__":
    main()

