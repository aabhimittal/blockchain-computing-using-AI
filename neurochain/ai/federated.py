"""Byzantine-robust federated aggregation of on-chain model updates.

NeuroChain's "useful work" can be decentralised model training: validators
submit gradient (or weight-delta) vectors and the chain aggregates them into the
next global model. A naive average is trivially poisoned by a single malicious
validator, so this module offers robust aggregators -- coordinate-wise trimmed
mean, Krum, and reputation-weighted mean -- plus input sanitisation that drops
NaN/Inf updates and clips oversized ones before they reach the aggregate.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np


class AggregationError(Exception):
    pass


def _stack(updates: Sequence[np.ndarray]) -> np.ndarray:
    if len(updates) == 0:
        raise AggregationError("no updates to aggregate")
    arrs = [np.asarray(u, dtype=float).ravel() for u in updates]
    dim = arrs[0].size
    if any(a.size != dim for a in arrs):
        raise AggregationError("updates have inconsistent dimensions")
    return np.vstack(arrs)


def sanitize(
    updates: Sequence[np.ndarray], max_norm: Optional[float] = None
) -> List[np.ndarray]:
    """Drop non-finite updates and clip the rest to ``max_norm`` (L2).

    Returns the surviving updates. A validator that submits NaN/Inf is silently
    excluded rather than poisoning the batch; honest-but-large updates are scaled
    down to ``max_norm`` instead of dropped.
    """
    kept: List[np.ndarray] = []
    for u in updates:
        a = np.asarray(u, dtype=float).ravel()
        if not np.all(np.isfinite(a)):
            continue
        if max_norm is not None:
            norm = float(np.linalg.norm(a))
            if norm > max_norm > 0:
                a = a * (max_norm / norm)
        kept.append(a)
    return kept


def trimmed_mean(updates: Sequence[np.ndarray], trim: float = 0.1) -> np.ndarray:
    """Coordinate-wise mean after discarding the ``trim`` fraction of extremes
    at each end of every dimension. Tolerates up to ``trim`` Byzantine share."""
    if not 0 <= trim < 0.5:
        raise AggregationError("trim must be in [0, 0.5)")
    m = _stack(updates)
    n = m.shape[0]
    k = int(np.floor(trim * n))
    if 2 * k >= n:  # nothing would survive; fall back to the plain mean
        return m.mean(axis=0)
    ordered = np.sort(m, axis=0)
    return ordered[k : n - k].mean(axis=0)


def krum(updates: Sequence[np.ndarray], num_byzantine: int = 0) -> np.ndarray:
    """Krum (Blanchard et al., 2017): pick the update closest to its neighbours.

    With ``n`` updates and ``f`` assumed Byzantine, each update is scored by the
    sum of squared distances to its ``n - f - 2`` nearest peers; the lowest score
    wins. Robust while ``n > 2f + 2``.
    """
    m = _stack(updates)
    n = m.shape[0]
    f = max(0, num_byzantine)
    if n <= 2 * f + 2:
        raise AggregationError("krum needs n > 2*f + 2 updates")
    # Pairwise squared distances.
    diffs = m[:, None, :] - m[None, :, :]
    sq = np.sum(diffs * diffs, axis=2)
    k = n - f - 2
    scores = np.empty(n)
    for i in range(n):
        nearest = np.sort(sq[i])[1 : k + 1]  # skip self-distance (0)
        scores[i] = float(np.sum(nearest))
    return m[int(np.argmin(scores))].copy()


def reputation_weighted_mean(
    updates: Sequence[np.ndarray], weights: Sequence[float]
) -> np.ndarray:
    """Weighted mean using per-validator reputation as the weight.

    Negative weights are rejected; if every weight is zero the mean is uniform.
    """
    m = _stack(updates)
    w = np.asarray(weights, dtype=float)
    if w.shape[0] != m.shape[0]:
        raise AggregationError("weights length must match number of updates")
    if np.any(w < 0) or not np.all(np.isfinite(w)):
        raise AggregationError("weights must be finite and non-negative")
    total = float(w.sum())
    if total == 0:
        return m.mean(axis=0)
    return (m * w[:, None]).sum(axis=0) / total


def aggregate(
    updates: Sequence[np.ndarray],
    method: str = "trimmed_mean",
    *,
    trim: float = 0.1,
    num_byzantine: int = 0,
    weights: Optional[Sequence[float]] = None,
    max_norm: Optional[float] = None,
) -> np.ndarray:
    """Sanitise then aggregate ``updates`` with the chosen robust ``method``."""
    clean = sanitize(updates, max_norm=max_norm)
    if not clean:
        raise AggregationError("no finite updates survived sanitisation")
    if method == "trimmed_mean":
        return trimmed_mean(clean, trim=trim)
    if method == "krum":
        return krum(clean, num_byzantine=num_byzantine)
    if method == "reputation":
        if weights is None:
            raise AggregationError("reputation method requires weights")
        # sanitize may have dropped rows; caller must pass matching weights only
        # when no drops occur. Guard on length to fail loudly instead of silently.
        return reputation_weighted_mean(clean, weights)
    raise AggregationError(f"unknown method: {method}")
