from __future__ import annotations

from pathlib import Path

import torch

from .config import TrainingConfig
from .engine import write_json


def save_deployment_bundle(
    model: torch.nn.Module,
    config: TrainingConfig,
    metrics: dict,
    output_dir: str | Path,
) -> Path:
    """Save the minimum artifacts needed to reproduce and deploy inference."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model_state.pt")
    write_json(config.to_json_dict(), output_dir / "config.json")
    write_json(metrics, output_dir / "metrics.json")
    (output_dir / "MODEL_CARD.md").write_text(
        "\n".join(
            [
                "# GTV Segmentation Model",
                "",
                "Training data: Session 1 NIfTI export.",
                "Task: binary 3D segmentation of the gross tumor volume.",
                "Inputs: CT channel plus configured optional RT structure context masks.",
                "Outputs: one sigmoid probability map on the Session 1 image grid.",
                "",
                "Clinical note: this bundle is a workshop artifact, not a clinically",
                "validated device. Use Session 3 quality controls before deployment.",
                "",
            ]
        )
    )
    return output_dir

