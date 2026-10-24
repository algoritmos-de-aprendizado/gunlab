"""Reference trainer: MLP over GF bins, full-batch gradient descent with momentum, NumPy only.

Targets are soft hit windows: every bin whose center lies within gf_width of the true GF gets an equal share.
Real-bullet waves weigh REAL_WEIGHT times more than virtual ones. The saved weights are those with the best
validation hit rate on real-bullet waves.

poetry run python examples/train_mlp.py data/train.csv data/validation.csv submissions/example.txt
"""

import sys

import numpy as np

from gunlab.dataset import load_csv
from gunlab.model import Mlp, gf_to_bin, save_weights

BINS = 31
HIDDEN = 32
EPOCHS = 500
LEARNING_RATE = 0.3
MOMENTUM = 0.9
REAL_WEIGHT = 4.0
EVAL_EVERY = 25
ACTIVATIONS = ["tanh", "identity"]


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def hit_targets(gf: np.ndarray, width: np.ndarray) -> np.ndarray:
    centers = np.linspace(-1.0, 1.0, BINS)
    inside = np.abs(centers[None, :] - gf[:, None]) <= width[:, None]
    inside[np.arange(gf.size), gf_to_bin(gf, BINS)] = True
    return inside / inside.sum(axis=1, keepdims=True)


def real_hit_rate(predict, data) -> float:
    real = data.real == 1
    return float(np.mean(np.abs(predict(data.X[real]) - data.gf[real]) <= data.width[real]))


def main(train_path: str, validation_path: str, out_path: str) -> None:
    train = load_csv(train_path)
    validation = load_csv(validation_path)

    mean = train.X.mean(axis=0)
    std = train.X.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    x = (train.X - mean) / std
    targets = hit_targets(train.gf, train.width)
    weights = np.where(train.real == 1, REAL_WEIGHT, 1.0)
    weights /= weights.sum()
    d = x.shape[1]
    print(f"head-on validation hit {real_hit_rate(lambda X: np.zeros(len(X)), validation):.3f}")

    rng = np.random.default_rng(0)
    w1 = rng.normal(0.0, 1.0 / np.sqrt(d), (HIDDEN, d))
    b1 = np.zeros(HIDDEN)
    w2 = rng.normal(0.0, 1.0 / np.sqrt(HIDDEN), (BINS, HIDDEN))
    b2 = np.zeros(BINS)
    params = [w1, b1, w2, b2]
    velocity = [np.zeros_like(p) for p in params]
    best = (-1.0, 0, [p.copy() for p in params])

    for epoch in range(1, EPOCHS + 1):
        hidden = np.tanh(x @ w1.T + b1)
        probs = softmax(hidden @ w2.T + b2)
        delta2 = (probs - targets) * weights[:, None]
        delta1 = (delta2 @ w2) * (1.0 - hidden**2)
        grads = [delta1.T @ x, delta1.sum(axis=0), delta2.T @ hidden, delta2.sum(axis=0)]
        for p, v, g in zip(params, velocity, grads):
            v *= MOMENTUM
            v -= LEARNING_RATE * g
            p += v
        if epoch % EVAL_EVERY == 0:
            loss = -np.sum(weights * np.sum(targets * np.log(probs + 1e-12), axis=1))
            hit = real_hit_rate(Mlp(mean, std, [w1, w2], [b1, b2], ACTIVATIONS).guess_factor, validation)
            if hit > best[0]:
                best = (hit, epoch, [p.copy() for p in params])
            print(f"epoch {epoch:4d}  train cross-entropy {loss:.4f}  validation hit {hit:.3f}")

    hit, epoch, (w1, b1, w2, b2) = best
    save_weights(out_path, [w1, w2], [b1, b2], ACTIVATIONS, mean, std)
    print(f"saved {out_path}: epoch {epoch}, validation hit {hit:.3f}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
