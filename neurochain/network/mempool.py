"""Pending-transaction pool with optional AI fraud gating."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..core.transaction import Transaction


@dataclass
class Mempool:
    fraud_detector: Optional[object] = None    # ai.fraud_detection.FraudDetector
    max_size: int = 5000
    _pool: Dict[str, Transaction] = field(default_factory=dict)
    _flagged: Dict[str, float] = field(default_factory=dict)

    def add(self, tx: Transaction) -> bool:
        if tx.txid in self._pool:
            return False
        if not tx.is_valid():
            return False
        if self.fraud_detector is not None and getattr(self.fraud_detector, "fitted", False):
            try:
                if self.fraud_detector.is_anomalous(tx):
                    self._flagged[tx.txid] = self.fraud_detector.anomaly_zscore(tx)
                    return False
            except Exception:
                pass
        if len(self._pool) >= self.max_size:
            return False
        self._pool[tx.txid] = tx
        return True

    def select(self, limit: int = 100) -> List[Transaction]:
        """Highest-fee-first selection for the next block."""
        ordered = sorted(self._pool.values(), key=lambda t: t.fee, reverse=True)
        return ordered[:limit]

    def remove_confirmed(self, transactions: List[Transaction]) -> None:
        for tx in transactions:
            self._pool.pop(tx.txid, None)

    @property
    def pending(self) -> List[Transaction]:
        return list(self._pool.values())

    @property
    def flagged(self) -> Dict[str, float]:
        return dict(self._flagged)

    def __len__(self) -> int:
        return len(self._pool)
