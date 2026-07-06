from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dice_score_from_logits(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    pred = (torch.sigmoid(logits) >= threshold).float()
    dims = tuple(range(1, pred.ndim))
    intersection = (pred * target).sum(dims)
    denominator = pred.sum(dims) + target.sum(dims)
    return ((2 * intersection + 1e-6) / (denominator + 1e-6)).mean()


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    intersection = (probs * target).sum(dims)
    denominator = probs.sum(dims) + target.sum(dims)
    return 1 - ((2 * intersection + 1e-6) / (denominator + 1e-6)).mean()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    bce = nn.BCEWithLogitsLoss()
    losses: list[float] = []
    dices: list[float] = []
    for batch in loader:
        image = batch["image"].to(device)
        target = batch["target"].to(device)
        logits = model(image)
        loss = 0.5 * bce(logits, target) + 0.5 * dice_loss_from_logits(logits, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        dices.append(float(dice_score_from_logits(logits.detach(), target).cpu()))
    return {"loss": float(np.mean(losses)), "dice": float(np.mean(dices))}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, threshold: float = 0.5) -> dict[str, float]:
    model.eval()
    bce = nn.BCEWithLogitsLoss()
    losses: list[float] = []
    dices: list[float] = []
    for batch in loader:
        image = batch["image"].to(device)
        target = batch["target"].to(device)
        logits = model(image)
        loss = 0.5 * bce(logits, target) + 0.5 * dice_loss_from_logits(logits, target)
        losses.append(float(loss.cpu()))
        dices.append(float(dice_score_from_logits(logits, target, threshold=threshold).cpu()))
    return {"loss": float(np.mean(losses)), "dice": float(np.mean(dices))}


def write_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")

