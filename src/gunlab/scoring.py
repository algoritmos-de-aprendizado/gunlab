from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from .dataset import Dataset
from .model import gf_to_bin, load_weights


def short_name(robot: str) -> str:
    return robot.split()[0].rsplit(".", 1)[-1]


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(str(cell)) for cell in column) for column in zip(headers, *rows)]
    lines = ["  ".join(str(cell).ljust(width) for cell, width in zip(row, widths)) for row in (headers, *rows)]
    lines.insert(1, "  ".join("-" * width for width in widths))
    return "\n".join(lines)


def write_csv(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def offline_board(weight_files: list[Path], data: Dataset, bins: int, by: str = "hit") -> list[dict]:
    """hit: fraction of waves where |predicted GF - GF| <= gf_width, i.e. the bullet would have hit."""
    target = gf_to_bin(data.gf, bins)
    has_width = not np.isnan(data.width).any()
    rows = []
    for path in weight_files:
        row = {"entry": path.stem, "hit": math.nan, "accuracy": math.nan, "mse": math.nan, "error": ""}
        try:
            pred = load_weights(path).guess_factor(data.X)
            if has_width:
                row["hit"] = float(np.mean(np.abs(pred - data.gf) <= data.width))
            row["accuracy"] = float(np.mean(gf_to_bin(pred, bins) == target))
            row["mse"] = float(np.mean((pred - data.gf) ** 2))
        except (OSError, ValueError) as exc:
            row["error"] = str(exc)
        rows.append(row)
    rows.sort(key=lambda r: (1, 0.0) if r["error"] else (0, r["mse"] if by == "mse" else -r[by]))
    return rows


def offline_table(rows: list[dict]) -> tuple[list[str], list[list[str]]]:
    headers = ["#", "entry", "hit", "accuracy", "MSE", "error"]
    body = [
        [str(i), r["entry"], _fmt(r["hit"]), _fmt(r["accuracy"]), _fmt(r["mse"]), r["error"]]
        for i, r in enumerate(rows, 1)
    ]
    return headers, body


class ArenaBoard:
    """Ranked by hit rate (hits / bullets fired), which does not depend on how long the robot survives."""

    def __init__(self, entries: list[str], opponents: list[str]) -> None:
        self.opponents = opponents
        self.results: dict[str, dict[str, dict]] = {entry: {} for entry in entries}

    def record(self, entry: str, opponent: str, damage: float, stats: dict[str, int]) -> None:
        self.results[entry][opponent] = {
            "damage": damage,
            "shots": stats.get("shots", 0),
            "hits": stats.get("hits", 0),
            "waves": stats.get("waves", 0),
            "would_hit": stats.get("would_hit", 0),
            "loaded": stats.get("loaded", 0),
        }

    def table(self) -> tuple[list[str], list[list[str]]]:
        ranked = sorted(self.results.items(), key=lambda item: -_totals(item[1])[0])
        headers = ["#", "entry", "hit rate", "would hit", "shots", "bullet damage", *(short_name(o) for o in self.opponents)]
        body = []
        for i, (entry, per) in enumerate(ranked, 1):
            rate, would, shots, damage = _totals(per)
            name = entry if all(r["loaded"] for r in per.values()) else f"{entry} (no weights)"
            body.append(
                [str(i), name, f"{rate:.1%}", f"{would:.1%}", str(shots), f"{damage:.0f}", *(_rate(per.get(o)) for o in self.opponents)]
            )
        return headers, body


def _totals(per: dict[str, dict]) -> tuple[float, float, int, float]:
    """Real hit rate, would-hit rate over all waves of the battle, shots, bullet damage."""
    shots = sum(r["shots"] for r in per.values())
    hits = sum(r["hits"] for r in per.values())
    waves = sum(r["waves"] for r in per.values())
    would = sum(r["would_hit"] for r in per.values())
    return (hits / shots if shots else 0.0), (would / waves if waves else 0.0), shots, sum(r["damage"] for r in per.values())


def _rate(result: dict | None) -> str:
    if result is None:
        return "-"
    return f"{result['hits'] / result['shots']:.1%}" if result["shots"] else "0.0%"


def _fmt(value: float) -> str:
    return "-" if math.isnan(value) else f"{value:.4f}"
