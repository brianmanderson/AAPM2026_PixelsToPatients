from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class QCConfig:
    """Configuration for the Session 3 reference-free QC check.

    The defaults mirror Session 2: the same NIfTI export, the same GTV target, and the
    same anatomical context masks. Session 3 reuses those context structures as the
    *constellation* the GTV's position is judged against (a GTV should sit near a lung,
    not out in the mediastinum or off the scan).
    """

    nifti_root: Path = Path("Session1/aapm_nsclc/nifti")
    output_dir: Path = Path("Session3/outputs")
    target_mask: str = "gtv"
    context_masks: Sequence[str] = ("lung_l", "lung_r", "heart", "esophagus", "cord")

    # CT input-validation window — the same clip Session 2 trains under (transforms.normalize_ct).
    # A volume whose intensities fall outside a real-CT HU range is not a CT the model can trust.
    ct_hu_lower: float = -1000.0
    ct_hu_upper: float = 400.0

    # Shape OOD (radiomic Mahalanobis) verdict bands, in robust standard deviations.
    shape_z_review: float = 2.0
    shape_z_fail: float = 4.0

    # Position (Procrustes atlas) — a GTV whose wrong-place residual z exceeds this is flagged.
    position_min_shared: int = 3      # organs (incl. GTV) needed to align a case to the atlas
    position_z_flag: float = 3.0

    seed: int = 2026

    def to_json_dict(self) -> dict:
        data = asdict(self)
        data["nifti_root"] = str(self.nifti_root)
        data["output_dir"] = str(self.output_dir)
        data["context_masks"] = list(self.context_masks)
        return data
