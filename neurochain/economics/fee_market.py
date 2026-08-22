"""Congestion-aware fee market (EIP-1559 style base fee with a priority tip).

The base fee is a feedback controller: it rises when blocks run fuller than a
target and falls when they run emptier, nudging demand toward the target
occupancy. Senders add a priority tip to bid for inclusion; the effective fee a
block can charge is ``min(max_fee, base_fee + tip)`` and the block only includes
a transaction whose ``max_fee`` covers the current base fee.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FeeQuote:
    base_fee: int
    tip: int

    @property
    def total(self) -> int:
        return self.base_fee + self.tip


class FeeMarket:
    """Tracks and updates a base fee from realised block occupancy.

    Parameters mirror EIP-1559: the base fee can move by at most
    ``1 / denominator`` per block (12.5% when denominator is 8).
    """

    def __init__(
        self,
        target_gas: int,
        base_fee: int = 1,
        max_gas: int | None = None,
        denominator: int = 8,
        min_base_fee: int = 1,
    ):
        if target_gas <= 0:
            raise ValueError("target_gas must be positive")
        if denominator <= 0:
            raise ValueError("denominator must be positive")
        if min_base_fee < 0:
            raise ValueError("min_base_fee must be non-negative")
        self.target_gas = target_gas
        self.max_gas = max_gas if max_gas is not None else 2 * target_gas
        if self.max_gas < target_gas:
            raise ValueError("max_gas must be >= target_gas")
        self.denominator = denominator
        self.min_base_fee = min_base_fee
        self.base_fee = max(base_fee, min_base_fee)

    def update(self, gas_used: int) -> int:
        """Advance the base fee given the gas the last block actually used.

        Returns the new base fee. Clamps ``gas_used`` into ``[0, max_gas]`` so a
        malformed or oversized block cannot swing the fee arbitrarily.
        """
        gas_used = max(0, min(gas_used, self.max_gas))
        target = self.target_gas
        if gas_used == target:
            return self.base_fee
        if gas_used > target:
            delta = self.base_fee * (gas_used - target) // (target * self.denominator)
            self.base_fee += max(1, delta)  # always move at least one unit up
        else:
            delta = self.base_fee * (target - gas_used) // (target * self.denominator)
            self.base_fee = max(self.min_base_fee, self.base_fee - delta)
        return self.base_fee

    def quote(self, tip: int = 0) -> FeeQuote:
        """Return the fee a sender should attach to land in the next block."""
        if tip < 0:
            raise ValueError("tip must be non-negative")
        return FeeQuote(base_fee=self.base_fee, tip=tip)

    def affordable(self, max_fee: int) -> bool:
        """Whether a transaction bidding ``max_fee`` clears the current base fee."""
        return max_fee >= self.base_fee

    def effective_fee(self, max_fee: int, tip: int) -> int:
        """Fee actually charged, capped by the sender's ``max_fee`` ceiling."""
        if not self.affordable(max_fee):
            raise ValueError("max_fee below base fee; transaction not includable")
        return min(max_fee, self.base_fee + max(0, tip))
