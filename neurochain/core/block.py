"""Block structure and header hashing."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .crypto import sha256_hex
from .merkle import merkle_root
from .transaction import Transaction


@dataclass
class Block:
    index: int
    previous_hash: str
    transactions: List[Transaction]
    difficulty: int
    miner: str = ""                       # address credited with the block reward
    timestamp: float = field(default_factory=time.time)
    nonce: int = 0
    # Consensus-specific evidence (e.g. the useful-work proof + PoW tie-break).
    proof: Dict[str, Any] = field(default_factory=dict)

    @property
    def merkle_root(self) -> str:
        return merkle_root([tx.txid for tx in self.transactions])

    def header(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "previous_hash": self.previous_hash,
            "merkle_root": self.merkle_root,
            "difficulty": self.difficulty,
            "miner": self.miner,
            "timestamp": self.timestamp,
            "nonce": self.nonce,
            "proof": self.proof,
        }

    @property
    def hash(self) -> str:
        payload = json.dumps(self.header(), sort_keys=True, separators=(",", ":"))
        return sha256_hex(payload.encode())

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.header(),
            "hash": self.hash,
            "transactions": [tx.to_dict() for tx in self.transactions],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Block":
        return cls(
            index=d["index"],
            previous_hash=d["previous_hash"],
            transactions=[Transaction.from_dict(t) for t in d["transactions"]],
            difficulty=d["difficulty"],
            miner=d.get("miner", ""),
            timestamp=d["timestamp"],
            nonce=d.get("nonce", 0),
            proof=d.get("proof", {}),
        )
