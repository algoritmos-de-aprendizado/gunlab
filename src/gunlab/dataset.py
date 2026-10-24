from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .model import FEATURES, gf_to_bin

RAW_COLUMNS = (*FEATURES, "gf", "gf_width", "real", "round")
COLUMNS = (*FEATURES, "gf", "gf_width", "gf_bin", "real", "round", "opponent")


@dataclass
class Dataset:
    X: np.ndarray
    gf: np.ndarray
    width: np.ndarray
    real: np.ndarray
    opponent: np.ndarray

    def subset(self, mask: np.ndarray) -> Dataset:
        return Dataset(self.X[mask], self.gf[mask], self.width[mask], self.real[mask], self.opponent[mask])


def load_csv(path: str | Path) -> Dataset:
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in (*FEATURES, "gf") if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        rows = list(reader)
    X = np.array([[float(r[c]) for c in FEATURES] for r in rows], dtype=float).reshape(len(rows), len(FEATURES))
    gf = np.array([float(r["gf"]) for r in rows], dtype=float)
    width = np.array([float(r.get("gf_width") or "nan") for r in rows], dtype=float)
    real = np.array([int(r.get("real") or 0) for r in rows], dtype=int)
    opponent = np.array([r.get("opponent") or "" for r in rows])
    return Dataset(X, gf, width, real, opponent)


def logger_files(robots_dir: Path) -> list[Path]:
    return [p for p in robots_dir.rglob("rows.csv") if p.parent.name == "LoggerBot.data"]


def clear_logger_files(robots_dir: Path) -> None:
    for path in logger_files(robots_dir):
        path.unlink()


def collect_logger_rows(robots_dir: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    for path in logger_files(robots_dir):
        with path.open(newline="", encoding="utf-8") as f:
            rows.extend(rec for rec in csv.reader(f) if len(rec) == len(RAW_COLUMNS))
        path.unlink()
    return rows


class TierWriter:
    def __init__(self, path: Path, bins: int, virtual_stride: int = 1) -> None:
        self.path = path
        self.bins = bins
        self.virtual_stride = max(1, virtual_stride)
        self.count = 0
        self._virtual_seen = 0

    def __enter__(self) -> TierWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(COLUMNS)
        return self

    def add(self, rows: list[list[str]], opponent: str) -> int:
        """Keep every real wave and one virtual wave in virtual_stride; returns the rows kept."""
        n = len(FEATURES)
        kept = 0
        for rec in rows:
            if rec[n + 2] != "1":
                self._virtual_seen += 1
                if (self._virtual_seen - 1) % self.virtual_stride:
                    continue
            gf_bin = int(gf_to_bin(float(rec[n]), self.bins))
            self._writer.writerow([*rec[: n + 2], gf_bin, rec[n + 2], rec[n + 3], opponent])
            kept += 1
        self._file.flush()
        self.count += kept
        return kept

    def __exit__(self, *exc) -> None:
        self._file.close()
