"""Validator reputation scoring.

Each validator accrues a reputation from its observed behaviour: how much
useful-work quality it delivers, how reliably it produces valid blocks, and
how close its block timing is to target. Reputations use an exponentially
weighted moving average so recent behaviour dominates, letting the network
demote flaky or adversarial validators and reward consistent ones. The
resulting scores weight leader selection in the PoUI consensus engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class ValidatorStats:
    reputation: float = 0.5           # in [0, 1]
    blocks_proposed: int = 0
    blocks_accepted: int = 0
    invalid_submissions: int = 0
    avg_work_quality: float = 0.0


@dataclass
class ReputationLedger:
    decay: float = 0.2                # EWMA weight on the newest observation
    penalty: float = 0.35             # reputation slash for an invalid block
    floor: float = 0.01
    stats: Dict[str, ValidatorStats] = field(default_factory=dict)

    def _get(self, address: str) -> ValidatorStats:
        return self.stats.setdefault(address, ValidatorStats())

    def observe_valid_block(self, address: str, work_quality: float, timing_score: float) -> float:
        """Record a successfully validated block. Returns the new reputation.

        ``work_quality`` and ``timing_score`` are each in [0, 1] (1 = best).
        """
        s = self._get(address)
        s.blocks_proposed += 1
        s.blocks_accepted += 1
        s.avg_work_quality = (
            (1 - self.decay) * s.avg_work_quality + self.decay * work_quality
        )
        signal = 0.7 * work_quality + 0.3 * timing_score
        s.reputation = (1 - self.decay) * s.reputation + self.decay * signal
        s.reputation = min(1.0, max(self.floor, s.reputation))
        return s.reputation

    def observe_invalid_block(self, address: str) -> float:
        s = self._get(address)
        s.blocks_proposed += 1
        s.invalid_submissions += 1
        s.reputation = max(self.floor, s.reputation - self.penalty)
        return s.reputation

    def reputation_of(self, address: str) -> float:
        return self._get(address).reputation

    def selection_weights(self, candidates: List[str]) -> List[Tuple[str, float]]:
        """Normalised selection weights proportional to reputation."""
        reps = [(c, self.reputation_of(c)) for c in candidates]
        total = sum(r for _, r in reps) or 1.0
        return [(c, r / total) for c, r in reps]

    def acceptance_rate(self, address: str) -> float:
        s = self._get(address)
        return s.blocks_accepted / s.blocks_proposed if s.blocks_proposed else 0.0
