import numpy as np
import pytest

from neurochain.ai.federated import (
    AggregationError,
    aggregate,
    krum,
    reputation_weighted_mean,
    sanitize,
    trimmed_mean,
)


def test_trimmed_mean_ignores_outlier():
    honest = [np.array([1.0, 1.0]) for _ in range(9)]
    poison = [np.array([1000.0, -1000.0])]
    out = trimmed_mean(honest + poison, trim=0.1)
    assert np.allclose(out, [1.0, 1.0], atol=1e-6)


def test_krum_selects_honest_cluster():
    rng = np.random.default_rng(0)
    honest = [np.array([0.0, 0.0]) + 0.01 * rng.standard_normal(2) for _ in range(7)]
    byzantine = [np.array([50.0, -50.0]), np.array([-40.0, 60.0])]
    out = krum(honest + byzantine, num_byzantine=2)
    assert np.linalg.norm(out) < 1.0  # picked a near-origin honest update


def test_krum_requires_enough_updates():
    with pytest.raises(AggregationError):
        krum([np.array([1.0]), np.array([2.0])], num_byzantine=1)


def test_sanitize_drops_nan_and_inf():
    updates = [
        np.array([1.0, 2.0]),
        np.array([np.nan, 1.0]),
        np.array([np.inf, 0.0]),
        np.array([3.0, 4.0]),
    ]
    kept = sanitize(updates)
    assert len(kept) == 2


def test_sanitize_clips_norm():
    kept = sanitize([np.array([3.0, 4.0])], max_norm=1.0)  # norm 5 -> clip to 1
    assert np.isclose(np.linalg.norm(kept[0]), 1.0)


def test_dimension_mismatch_raises():
    with pytest.raises(AggregationError):
        trimmed_mean([np.array([1.0, 2.0]), np.array([1.0])])


def test_empty_input_raises():
    with pytest.raises(AggregationError):
        trimmed_mean([])


def test_reputation_weighted_mean():
    up = [np.array([0.0]), np.array([10.0])]
    out = reputation_weighted_mean(up, weights=[3.0, 1.0])
    assert np.isclose(out[0], 2.5)  # (0*3 + 10*1) / 4


def test_reputation_zero_weights_falls_back_to_uniform():
    up = [np.array([0.0]), np.array([10.0])]
    out = reputation_weighted_mean(up, weights=[0.0, 0.0])
    assert np.isclose(out[0], 5.0)


def test_reputation_rejects_negative_weights():
    with pytest.raises(AggregationError):
        reputation_weighted_mean([np.array([1.0])], weights=[-1.0])


def test_aggregate_all_byzantine_after_sanitise_raises():
    updates = [np.array([np.nan]), np.array([np.inf])]
    with pytest.raises(AggregationError):
        aggregate(updates, method="trimmed_mean")


def test_aggregate_unknown_method():
    with pytest.raises(AggregationError):
        aggregate([np.array([1.0])], method="bogus")
