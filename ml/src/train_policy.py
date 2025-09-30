from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import glob

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

import numpy as np

from .dataset import VrpMoveDataset


@dataclass
class TrainingDefaults:
    log_patterns: Sequence[str] = ("logs/*.jsonl",)
    epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    val_split: float = 0.2
    seed: int = 42
    export_json: Path = Path("ml/move_regressor.json")

DEFAULTS = TrainingDefaults()


class MoveRegressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def export_move_regressor(
    model: MoveRegressor,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    export_path: Path,
) -> None:
    model_cpu = model.to(torch.device("cpu")).eval()
    layers = []
    for module in model_cpu.net:
        if isinstance(module, nn.Linear):
            layers.append(
                {
                    "weights": module.weight.detach().tolist(),
                    "bias": module.bias.detach().tolist(),
                }
            )

    payload = {
        "layers": layers,
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
    }

    export_path.parent.mkdir(parents=True, exist_ok=True)
    with export_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the layered planner move regressor")
    parser.add_argument(
        "--logs",
        nargs="+",
        help="Paths or globs to JSONL solver logs (defaults to logs/*.jsonl)",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULTS.epochs)
    parser.add_argument("--batch-size", type=int, default=DEFAULTS.batch_size)
    parser.add_argument("--lr", type=float, default=DEFAULTS.lr)
    parser.add_argument("--val-split", type=float, default=DEFAULTS.val_split)
    parser.add_argument("--seed", type=int, default=DEFAULTS.seed)
    parser.add_argument(
        "--export-json",
        type=Path,
        default=DEFAULTS.export_json,
        help="Export location for JSON weights consumed by the Rust planner",
    )
    return parser.parse_args()


def expand_globs(patterns: Sequence[str]) -> tuple[List[Path], List[str]]:
    paths: List[Path] = []
    missing: List[str] = []
    for pattern in patterns:
        # Use glob for shell-style wildcards; fall back to plain Path.
        matches = [Path(p) for p in glob.glob(pattern)]
        if not matches:
            candidate = Path(pattern)
            if candidate.exists():
                paths.append(candidate)
            else:
                missing.append(pattern)
            continue
        paths.extend(matches)
    return paths, missing


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    patterns = args.logs if args.logs else list(DEFAULTS.log_patterns)
    paths, missing_patterns = expand_globs(patterns)
    if missing_patterns:
        friendly_missing = ", ".join(missing_patterns)
        raise SystemExit(
            "No solver logs found for patterns: "
            + friendly_missing
            + "\nGenerate logs (run the solver with logging enabled) or pass --logs explicitly."
        )

    dataset = VrpMoveDataset(paths)

    if len(dataset) == 0:
        friendly_patterns = ", ".join(str(pattern) for pattern in patterns)
        raise SystemExit(
            f"Parsed zero move records from logs (patterns: {friendly_patterns})."
        )

    feature_mean, feature_std = dataset.feature_stats()
    if feature_mean is None or feature_std is None:
        raise SystemExit("Unable to compute feature statistics for the move dataset")

    feature_sample, _ = dataset[0]
    model = MoveRegressor(feature_sample.shape[-1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = max(1, len(dataset) - val_size)
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for features, targets in train_loader:
            features = features.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)

            optimiser.zero_grad()
            preds = model(features)
            loss = loss_fn(preds, targets)
            loss.backward()
            optimiser.step()

            total_loss += loss.item() * features.size(0)

        train_loss = total_loss / len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            total_val = 0.0
            for features, targets in val_loader:
                features = features.to(device, dtype=torch.float32)
                targets = targets.to(device, dtype=torch.float32)
                preds = model(features)
                loss = loss_fn(preds, targets)
                total_val += loss.item() * features.size(0)

        val_loss = total_val / len(val_loader.dataset)
        print(f"epoch={epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    torch.save(model.state_dict(), "move_regressor.pt")
    print("Saved weights to move_regressor.pt")

    export_move_regressor(model, feature_mean, feature_std, args.export_json)
    print(f"Exported JSON weights to {args.export_json}")


if __name__ == "__main__":
    main()
