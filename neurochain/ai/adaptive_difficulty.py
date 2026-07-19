"""Reinforcement-learning difficulty controller.

Classical chains (Bitcoin) retarget difficulty with a fixed arithmetic rule
every N blocks. NeuroChain instead learns a retargeting *policy* online with
tabular Q-learning. The agent observes how far the last block time drifted
from the target and chooses to lower, hold, or raise difficulty. The reward
rewards keeping block time near target, so the policy adapts to changing
network hashrate faster than a fixed rule and without hand-tuned constants.

State  : bucketed log2(observed_time / target_time)  -> discrete drift level
Action : -1 (ease), 0 (hold), +1 (harden)
Reward : -|log2(observed/target)|  (0 is perfect, more negative is worse)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Tuple

ACTIONS = (-1, 0, 1)


@dataclass
class RLDifficultyController:
    epsilon: float = 0.1          # exploration rate
    alpha: float = 0.5            # learning rate
    gamma: float = 0.9            # discount
    min_difficulty: int = 1
    max_difficulty: int = 24
    seed: int = 13
    q: Dict[Tuple[int, int], float] = field(default_factory=dict)
    _rng_state: int = field(default=0, init=False)
    _last: Tuple[int, int] = field(default=None, init=False)  # (state, action)

    def __post_init__(self) -> None:
        self._rng_state = self.seed & 0x7FFFFFFF

    # Deterministic LCG so behaviour is reproducible without Math.random-style
    # nondeterminism leaking into consensus.
    def _rand(self) -> float:
        self._rng_state = (1103515245 * self._rng_state + 12345) & 0x7FFFFFFF
        return self._rng_state / 0x7FFFFFFF

    @staticmethod
    def _bucket(observed: float, target: float) -> int:
        ratio = max(observed, 1e-9) / max(target, 1e-9)
        return int(round(max(-4.0, min(4.0, math.log2(ratio)))))

    def _q(self, state: int, action: int) -> float:
        return self.q.get((state, action), 0.0)

    def _greedy_action(self, state: int) -> int:
        return max(ACTIONS, key=lambda a: self._q(state, a))

    def next_difficulty(
        self,
        current_difficulty: int,
        observed_block_time: float,
        target_block_time: float,
    ) -> int:
        state = self._bucket(observed_block_time, target_block_time)
        reward = -abs(math.log2(max(observed_block_time, 1e-9) / max(target_block_time, 1e-9)))

        # Q-learning update for the *previous* decision, now that we see its outcome.
        if self._last is not None:
            ps, pa = self._last
            best_next = max(self._q(state, a) for a in ACTIONS)
            td_target = reward + self.gamma * best_next
            old = self._q(ps, pa)
            self.q[(ps, pa)] = old + self.alpha * (td_target - old)

        # epsilon-greedy action selection
        if self._rand() < self.epsilon:
            action = ACTIONS[int(self._rand() * len(ACTIONS)) % len(ACTIONS)]
        else:
            action = self._greedy_action(state)
        self._last = (state, action)

        new_diff = current_difficulty + action
        return max(self.min_difficulty, min(self.max_difficulty, new_diff))
