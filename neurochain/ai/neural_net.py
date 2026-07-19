"""A tiny fully-connected neural network implemented in NumPy.

Deliberately dependency-light: no PyTorch/TensorFlow. Supports arbitrary
hidden layers, ReLU activations, a linear output head, and mini-batch
gradient descent with Adam. Reused by the fraud-detection autoencoder and
the validator-reputation regressor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


def _seeded_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


@dataclass
class MLP:
    layer_sizes: List[int]          # e.g. [8, 16, 8, 1]
    seed: int = 7
    lr: float = 1e-2
    weights: List[np.ndarray] = field(default_factory=list)
    biases: List[np.ndarray] = field(default_factory=list)
    # Adam state
    _mw: List[np.ndarray] = field(default_factory=list)
    _vw: List[np.ndarray] = field(default_factory=list)
    _mb: List[np.ndarray] = field(default_factory=list)
    _vb: List[np.ndarray] = field(default_factory=list)
    _t: int = 0

    def __post_init__(self) -> None:
        rng = _seeded_rng(self.seed)
        for fan_in, fan_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            scale = np.sqrt(2.0 / fan_in)  # He initialization
            self.weights.append(rng.standard_normal((fan_in, fan_out)) * scale)
            self.biases.append(np.zeros(fan_out))
        self._mw = [np.zeros_like(w) for w in self.weights]
        self._vw = [np.zeros_like(w) for w in self.weights]
        self._mb = [np.zeros_like(b) for b in self.biases]
        self._vb = [np.zeros_like(b) for b in self.biases]

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    def _forward(self, x: np.ndarray) -> Tuple[np.ndarray, list, list]:
        activations = [x]
        pre = []
        a = x
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ w + b
            pre.append(z)
            a = z if i == len(self.weights) - 1 else self._relu(z)  # linear output
            activations.append(a)
        return a, activations, pre

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        return self._forward(x)[0]

    def _adam_step(self, grads_w, grads_b) -> None:
        self._t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i in range(len(self.weights)):
            self._mw[i] = b1 * self._mw[i] + (1 - b1) * grads_w[i]
            self._vw[i] = b2 * self._vw[i] + (1 - b2) * grads_w[i] ** 2
            mhat = self._mw[i] / (1 - b1 ** self._t)
            vhat = self._vw[i] / (1 - b2 ** self._t)
            self.weights[i] -= self.lr * mhat / (np.sqrt(vhat) + eps)

            self._mb[i] = b1 * self._mb[i] + (1 - b1) * grads_b[i]
            self._vb[i] = b2 * self._vb[i] + (1 - b2) * grads_b[i] ** 2
            mhb = self._mb[i] / (1 - b1 ** self._t)
            vhb = self._vb[i] / (1 - b2 ** self._t)
            self.biases[i] -= self.lr * mhb / (np.sqrt(vhb) + eps)

    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 200,
        batch_size: int = 32,
    ) -> List[float]:
        """Fit to targets ``y`` with MSE loss. Returns per-epoch loss history."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        y = np.atleast_2d(np.asarray(y, dtype=float))
        if y.shape[0] != x.shape[0]:
            y = y.reshape(x.shape[0], -1)
        rng = _seeded_rng(self.seed + 1)
        n = x.shape[0]
        history: List[float] = []
        for _ in range(epochs):
            idx = rng.permutation(n)
            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                bi = idx[start : start + batch_size]
                xb, yb = x[bi], y[bi]
                out, acts, pre = self._forward(xb)
                m = xb.shape[0]
                delta = (out - yb) * (2.0 / m)     # dMSE/dout
                epoch_loss += float(np.mean((out - yb) ** 2)) * m
                grads_w = [None] * len(self.weights)
                grads_b = [None] * len(self.biases)
                for i in reversed(range(len(self.weights))):
                    grads_w[i] = acts[i].T @ delta
                    grads_b[i] = delta.sum(axis=0)
                    if i > 0:
                        d_relu = (pre[i - 1] > 0).astype(float)
                        delta = (delta @ self.weights[i].T) * d_relu
                self._adam_step(grads_w, grads_b)
            history.append(epoch_loss / n)
        return history
