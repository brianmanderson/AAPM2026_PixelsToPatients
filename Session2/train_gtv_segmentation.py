from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pytorch_workflow import NiftiSegmentationDataset, TinyUNet3D, TrainingConfig
from pytorch_workflow.data import discover_cases, split_by_patient, write_index_csv
from pytorch_workflow.engine import evaluate, seed_everything, train_one_epoch, write_json
from pytorch_workflow.packaging import save_deployment_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a workshop 3D U-Net on Session 1 NIfTI exports.")
    parser.add_argument("--nifti-root", type=Path, default=TrainingConfig.nifti_root)
    parser.add_argument("--output-dir", type=Path, default=TrainingConfig.output_dir)
    parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        nifti_root=args.nifti_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    seed_everything(config.seed)
    device = torch.device(args.device)

    records = discover_cases(config.nifti_root, required_masks=(config.target_mask,))
    if not records:
        raise SystemExit(f"No cases with mask '{config.target_mask}' found under {config.nifti_root}")

    splits = split_by_patient(records, config.val_fraction, config.test_fraction, config.seed)
    for split_name, split_records in splits.items():
        write_index_csv(split_records, config.output_dir / f"{split_name}_index.csv")

    train_ds = NiftiSegmentationDataset(
        splits["train"],
        target_mask=config.target_mask,
        context_masks=config.context_masks,
        patch_size=config.patch_size,
        training=True,
        seed=config.seed,
    )
    val_ds = NiftiSegmentationDataset(
        splits["val"] or splits["train"],
        target_mask=config.target_mask,
        context_masks=config.context_masks,
        patch_size=config.patch_size,
        training=False,
        seed=config.seed,
    )
    test_ds = NiftiSegmentationDataset(
        splits["test"] or splits["val"] or splits["train"],
        target_mask=config.target_mask,
        context_masks=config.context_masks,
        patch_size=config.patch_size,
        training=False,
        seed=config.seed,
    )

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=config.num_workers)

    in_channels = 1 + len(config.context_masks)
    model = TinyUNet3D(in_channels=in_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    history = []
    best_dice = -1.0
    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device, threshold=config.threshold)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(row)
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            config.output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), config.output_dir / "best_model_state.pt")

    model.load_state_dict(torch.load(config.output_dir / "best_model_state.pt", map_location=device))
    test_metrics = evaluate(model, test_loader, device, threshold=config.threshold)
    metrics = {"history": history, "test": test_metrics, "n_cases": len(records)}
    write_json(metrics, config.output_dir / "training_metrics.json")
    bundle_dir = save_deployment_bundle(model, config, metrics, config.output_dir / "deployment_bundle")
    print(f"Saved deployment bundle to {bundle_dir}")


if __name__ == "__main__":
    main()
