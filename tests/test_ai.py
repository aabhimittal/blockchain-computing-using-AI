import numpy as np

from neurochain.ai.adaptive_difficulty import RLDifficultyController
from neurochain.ai.fraud_detection import FraudDetector
from neurochain.ai.neural_net import MLP
from neurochain.ai.validator_reputation import ReputationLedger
from neurochain.core.transaction import Transaction


def test_mlp_fits_linear_function():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 3))
    w = np.array([1.5, -2.0, 0.5])
    y = (x @ w).reshape(-1, 1)
    model = MLP([3, 16, 1], seed=1, lr=1e-2)
    history = model.train(x, y, epochs=300, batch_size=32)
    assert history[-1] < history[0]
    assert history[-1] < 0.05


def test_fraud_detector_flags_outlier():
    rng = np.random.default_rng(1)
    normal = [
        Transaction(sender=f"a{i}", recipient="b", amount=float(10 + rng.random()),
                    fee=0.1, nonce=i)
        for i in range(50)
    ]
    det = FraudDetector(threshold_sigma=3.0).fit(normal, epochs=200)
    normal_tx = Transaction(sender="x", recipient="b", amount=10.4, fee=0.1, nonce=1)
    outlier = Transaction(sender="x", recipient="b", amount=500000, fee=9000, nonce=1)
    assert det.anomaly_zscore(outlier) > det.anomaly_zscore(normal_tx)
    assert det.is_anomalous(outlier)


def test_rl_difficulty_increases_when_blocks_too_fast():
    ctrl = RLDifficultyController(epsilon=0.0)  # greedy for determinism
    diff = 5
    # Blocks arriving far faster than target should push difficulty up over time.
    increased = False
    for _ in range(40):
        new = ctrl.next_difficulty(diff, observed_block_time=1.0, target_block_time=10.0)
        if new > diff:
            increased = True
        diff = new
    assert increased


def test_rl_difficulty_respects_bounds():
    ctrl = RLDifficultyController(min_difficulty=1, max_difficulty=3, epsilon=0.0)
    diff = 3
    for _ in range(20):
        diff = ctrl.next_difficulty(diff, observed_block_time=0.1, target_block_time=10.0)
    assert 1 <= diff <= 3


def test_reputation_rewards_and_penalizes():
    ledger = ReputationLedger()
    addr = "0xvalidator"
    for _ in range(5):
        ledger.observe_valid_block(addr, work_quality=0.9, timing_score=1.0)
    good = ledger.reputation_of(addr)
    ledger.observe_invalid_block(addr)
    assert ledger.reputation_of(addr) < good


def test_selection_weights_sum_to_one():
    ledger = ReputationLedger()
    for a in ("x", "y", "z"):
        ledger.observe_valid_block(a, 0.5, 0.5)
    weights = ledger.selection_weights(["x", "y", "z"])
    assert abs(sum(w for _, w in weights) - 1.0) < 1e-9
