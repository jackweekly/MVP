from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable, Iterator, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class MoveRecord:
    iteration: int
    delta_cost: float
    current_cost: float
    best_cost: float
    operator: str
    route_lengths: List[int]
    total_duration: float
    total_distance: float
    unassigned: int


def load_events(paths: Sequence[Path]) -> Iterator[dict]:
    """Yield JSON events from one or more log files."""

    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def parse_move_records(paths: Sequence[Path]) -> List[MoveRecord]:
    """Extract `MoveAccepted` events into a structured list."""

    records: List[MoveRecord] = []
    for event in load_events(paths):
        if event.get("type") != "MoveAccepted":
            continue

        state = event.get("state", {})
        routes = state.get("routes", [])
        route_lengths = [len(route.get("customers", [])) for route in routes]

        records.append(
            MoveRecord(
                iteration=event.get("iteration", 0),
                delta_cost=float(event.get("delta_cost", 0.0)),
                current_cost=float(event.get("current_cost", 0.0)),
                best_cost=float(event.get("best_cost", 0.0)),
                operator=event.get("operator", {}).get("type", str(event.get("operator", "unknown"))),
                route_lengths=route_lengths,
                total_duration=float(state.get("total_duration", 0.0)),
                total_distance=float(state.get("total_distance", 0.0)),
                unassigned=len(state.get("unassigned", [])),
            )
        )

    return records


class VrpMoveDataset(Dataset):
    """Minimal torch dataset for supervised move scoring experiments."""

    def __init__(self, paths: Iterable[str | Path]):
        path_objs = [Path(p) for p in paths]
        self.records = parse_move_records(path_objs)
        if self.records:
            encoded = [self._encode(record) for record in self.records]
            self._feature_matrix = np.stack(encoded).astype(np.float32)
        else:
            self._feature_matrix = np.empty((0, 0), dtype=np.float32)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.records)

    def __getitem__(self, index: int):  # type: ignore[override]
        record = self.records[index]
        features = self._feature_matrix[index]
        target = np.array([record.delta_cost], dtype=np.float32)
        return torch.from_numpy(features), torch.from_numpy(target)

    def feature_stats(self) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        if self._feature_matrix.size == 0:
            return (None, None)
        mean = self._feature_matrix.mean(axis=0)
        std = self._feature_matrix.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        return mean, std

    @staticmethod
    def _encode(record: MoveRecord) -> np.ndarray:
        route_count = len(record.route_lengths) or 1
        route_mean = mean(record.route_lengths) if record.route_lengths else 0.0
        route_max = max(record.route_lengths or [0])
        feature_vector = np.array(
            [
                route_count,
                route_mean,
                route_max,
                record.delta_cost,
                record.current_cost,
                record.best_cost,
                record.total_duration,
                record.total_distance,
                record.unassigned,
            ],
            dtype=np.float32,
        )
        return feature_vector
