"""Autoencoder-based transaction anomaly detection.

An unsupervised autoencoder is trained on the distribution of "normal"
transactions. At inference it reconstructs each transaction's feature vector;
a high reconstruction error means the transaction looks unlike anything in
the training distribution and is flagged as potentially fraudulent (e.g.
anomalous amount/fee combinations, dust flooding, or nonce anomalies).

Nodes can use scores to prioritise mempool admission or to attach a risk
signal to blocks — without any labelled fraud data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..core.transaction import Transaction
from .neural_net import MLP

FEATURE_DIM = 5
_CLIP = 10.0  # bound normalized features so a near-constant column can't explode


def extract_features(tx: Transaction) -> np.ndarray:
    """Map a transaction to a fixed-length numeric feature vector.

    Features are chosen to be stable and scale-free: log amount, log fee,
    fee-to-amount ratio, log nonce, and a coinbase flag. Anomalous
    combinations (huge amounts, disproportionate fees) fall far from the
    learned manifold.
    """
    amount = max(tx.amount, 0.0)
    fee = max(tx.fee, 0.0)
    fee_ratio = fee / amount if amount > 0 else 0.0
    return np.array(
        [
            math.log1p(amount),
            math.log1p(fee),
            min(fee_ratio, 1.0),
            math.log1p(max(tx.nonce, 0)),
            1.0 if tx.is_coinbase() else 0.0,
        ],
        dtype=float,
    )


@dataclass
class FraudDetector:
    threshold_sigma: float = 3.0
    seed: int = 21
    _model: Optional[MLP] = field(default=None, init=False)
    _mean: Optional[np.ndarray] = field(default=None, init=False)
    _std: Optional[np.ndarray] = field(default=None, init=False)
    _err_mean: float = field(default=0.0, init=False)
    _err_std: float = field(default=1.0, init=False)
    fitted: bool = field(default=False, init=False)

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self._mean) / self._std, -_CLIP, _CLIP)

    def fit(self, transactions: List[Transaction], epochs: int = 300) -> "FraudDetector":
        if len(transactions) < 8:
            raise ValueError("need at least 8 transactions to fit the detector")
        x = np.vstack([extract_features(t) for t in transactions])
        self._mean = x.mean(axis=0)
        raw_std = x.std(axis=0)
        # Near-constant columns get unit scale so they contribute a bounded
        # (mean-centred) signal instead of exploding on any deviation.
        self._std = np.where(raw_std < 1e-3, 1.0, raw_std)
        xn = self._normalize(x)

        # Hold out a validation split so the error baseline reflects
        # *generalization* error, not the (near-zero) memorized training error.
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(xn))
        n_val = max(2, len(xn) // 5)
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        self._model = MLP(
            layer_sizes=[FEATURE_DIM, 8, 3, 8, FEATURE_DIM],
            seed=self.seed,
            lr=5e-3,
        )
        self._model.train(xn[train_idx], xn[train_idx], epochs=epochs, batch_size=16)

        recon = self._model.predict(xn[val_idx])
        errors = np.mean((recon - xn[val_idx]) ** 2, axis=1)
        self._err_mean = float(errors.mean())
        # Guard the spread so a lucky-tight val split can't make the detector
        # hypersensitive; scale to the error magnitude.
        self._err_std = float(errors.std() + 0.1 * self._err_mean + 1e-6)
        self.fitted = True
        return self

    def score(self, tx: Transaction) -> float:
        """Reconstruction error; higher means more anomalous."""
        if not self.fitted:
            raise RuntimeError("FraudDetector must be fit() before scoring")
        xn = self._normalize(extract_features(tx).reshape(1, -1))
        recon = self._model.predict(xn)
        return float(np.mean((recon - xn) ** 2))

    def anomaly_zscore(self, tx: Transaction) -> float:
        return (self.score(tx) - self._err_mean) / self._err_std

    def is_anomalous(self, tx: Transaction) -> bool:
        return self.anomaly_zscore(tx) > self.threshold_sigma
