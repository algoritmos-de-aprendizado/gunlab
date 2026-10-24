"""Weights file format and NumPy reference forward pass, identical to gunlab/Mlp.java."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

FEATURES = (
    "distance",
    "bullet_power",
    "lateral_speed",
    "advancing_velocity",
    "enemy_speed",
    "acceleration",
    "ticks_since_reverse",
    "wall_ahead",
    "wall_behind",
    "dist_last_10",
    "last_gf",
    "mean_gf_10",
    *(f"hist_all_{i}" for i in range(5)),
    *(f"hist_real_{i}" for i in range(5)),
)

MAGIC = ("gunlab-mlp", "1")

ACTIVATIONS = {
    "identity": lambda z: z,
    "tanh": np.tanh,
    "logistic": lambda z: 1.0 / (1.0 + np.exp(-z)),
    "relu": lambda z: np.maximum(z, 0.0),
}


@dataclass
class Mlp:
    mean: np.ndarray
    std: np.ndarray
    weights: list[np.ndarray]
    biases: list[np.ndarray]
    activations: list[str]

    @property
    def outputs(self) -> int:
        return self.weights[-1].shape[0]

    def forward(self, x: np.ndarray) -> np.ndarray:
        a = (np.atleast_2d(np.asarray(x, dtype=float)) - self.mean) / self.std
        for w, b, act in zip(self.weights, self.biases, self.activations):
            a = ACTIVATIONS[act](a @ w.T + b)
        return a

    def guess_factor(self, x: np.ndarray) -> np.ndarray:
        return output_to_gf(self.forward(x))


def output_to_gf(y: np.ndarray) -> np.ndarray:
    """K = 1: the output is the GF. K > 1: center of the highest-scoring of K equal bins."""
    if y.shape[1] == 1:
        return np.clip(y[:, 0], -1.0, 1.0)
    return -1.0 + 2.0 * np.argmax(y, axis=1) / (y.shape[1] - 1)


def gf_to_bin(gf, bins: int) -> np.ndarray:
    return np.clip(np.rint((np.asarray(gf, dtype=float) + 1.0) / 2.0 * (bins - 1)), 0, bins - 1).astype(int)


def save_weights(path, weights, biases, activations, mean, std) -> Path:
    """Write a network with W of shape (n_out, n_in) per layer; zero std values become 1."""
    std = np.asarray(std, dtype=float).ravel()
    model = Mlp(
        mean=np.asarray(mean, dtype=float).ravel(),
        std=np.where(std > 0, std, 1.0),
        weights=[np.asarray(w, dtype=float) for w in weights],
        biases=[np.asarray(b, dtype=float).ravel() for b in biases],
        activations=[str(a) for a in activations],
    )
    _validate(model)
    lines = [
        " ".join(MAGIC),
        f"inputs {model.mean.size}",
        "mean " + _row(model.mean),
        "std " + _row(model.std),
        f"layers {len(model.weights)}",
    ]
    for w, b, act in zip(model.weights, model.biases, model.activations):
        lines.append(f"dense {w.shape[1]} {w.shape[0]} {act}")
        lines.extend(_row(r) for r in w)
        lines.append(_row(b))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


def load_weights(path) -> Mlp:
    tokens = _Tokens(Path(path))
    for word in MAGIC:
        tokens.expect(word)
    tokens.expect("inputs")
    n = tokens.integer()
    tokens.expect("mean")
    mean = tokens.numbers(n)
    tokens.expect("std")
    std = tokens.numbers(n)
    tokens.expect("layers")
    weights, biases, activations = [], [], []
    for _ in range(tokens.integer()):
        tokens.expect("dense")
        n_in, n_out = tokens.integer(), tokens.integer()
        activations.append(tokens.word())
        weights.append(tokens.numbers(n_in * n_out).reshape(n_out, n_in))
        biases.append(tokens.numbers(n_out))
    if tokens.remaining():
        raise ValueError(f"{path}: unexpected data after the last layer")
    model = Mlp(mean, std, weights, biases, activations)
    _validate(model)
    return model


def _validate(model: Mlp) -> None:
    n = len(FEATURES)
    if model.mean.shape != (n,) or model.std.shape != (n,):
        raise ValueError(f"mean and std must have {n} values, one per feature")
    if not model.weights or not len(model.weights) == len(model.biases) == len(model.activations):
        raise ValueError("weights, biases and activations must have the same non-zero length")
    width = n
    for i, (w, b, act) in enumerate(zip(model.weights, model.biases, model.activations), 1):
        if act not in ACTIVATIONS:
            raise ValueError(f"layer {i}: unknown activation '{act}' (use {', '.join(ACTIVATIONS)})")
        if w.ndim != 2 or w.shape[1] != width:
            raise ValueError(f"layer {i}: weight matrix must have shape (n_out, {width})")
        if b.shape != (w.shape[0],):
            raise ValueError(f"layer {i}: bias must have {w.shape[0]} values")
        width = w.shape[0]
    if not all(np.all(np.isfinite(a)) for a in (model.mean, model.std, *model.weights, *model.biases)):
        raise ValueError("all values must be finite")
    if np.any(model.std <= 0):
        raise ValueError("std values must be positive")


def _row(values) -> str:
    return " ".join(repr(float(v)) for v in values)


class _Tokens:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.items: list[str] = []
        for line in path.read_text(encoding="ascii").splitlines():
            self.items.extend(line.split("#", 1)[0].split())
        self.position = 0

    def word(self) -> str:
        if self.position >= len(self.items):
            raise ValueError(f"{self.path}: unexpected end of file")
        self.position += 1
        return self.items[self.position - 1]

    def expect(self, word: str) -> None:
        found = self.word()
        if found != word:
            raise ValueError(f"{self.path}: expected '{word}', found '{found}'")

    def integer(self) -> int:
        return int(self.word())

    def numbers(self, count: int) -> np.ndarray:
        return np.array([float(self.word()) for _ in range(count)], dtype=float)

    def remaining(self) -> bool:
        return self.position < len(self.items)
