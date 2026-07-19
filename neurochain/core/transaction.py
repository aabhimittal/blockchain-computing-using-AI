"""Signed value-transfer transactions."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Optional, Tuple

from .crypto import sha256_hex, verify_hex

COINBASE_SENDER = "COINBASE"


@dataclass
class Transaction:
    sender: str            # address, or COINBASE_SENDER for block rewards
    recipient: str
    amount: float
    fee: float = 0.0
    nonce: int = 0         # per-sender counter, prevents replay
    timestamp: float = field(default_factory=time.time)
    public_key: str = ""   # compressed-hex pubkey of the sender
    signature: Optional[Tuple[int, int]] = None

    # -- serialization -----------------------------------------------------
    def signing_payload(self) -> bytes:
        body = {
            "sender": self.sender,
            "recipient": self.recipient,
            "amount": self.amount,
            "fee": self.fee,
            "nonce": self.nonce,
            "timestamp": self.timestamp,
        }
        return json.dumps(body, sort_keys=True, separators=(",", ":")).encode()

    @property
    def txid(self) -> str:
        return sha256_hex(self.signing_payload())

    def to_dict(self) -> dict:
        d = asdict(self)
        d["txid"] = self.txid
        if self.signature is not None:
            d["signature"] = [str(self.signature[0]), str(self.signature[1])]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Transaction":
        sig = d.get("signature")
        signature = (int(sig[0]), int(sig[1])) if sig else None
        return cls(
            sender=d["sender"],
            recipient=d["recipient"],
            amount=d["amount"],
            fee=d.get("fee", 0.0),
            nonce=d.get("nonce", 0),
            timestamp=d.get("timestamp", 0.0),
            public_key=d.get("public_key", ""),
            signature=signature,
        )

    # -- validation --------------------------------------------------------
    def is_coinbase(self) -> bool:
        return self.sender == COINBASE_SENDER

    def is_valid(self) -> bool:
        if self.amount < 0 or self.fee < 0:
            return False
        if self.is_coinbase():
            return True
        if not self.signature or not self.public_key:
            return False
        # The public key must hash to the claimed sender address.
        from .crypto import address_from_public_hex  # local import

        if address_from_public_hex(self.public_key) != self.sender:
            return False
        return verify_hex(self.public_key, self.signing_payload(), self.signature)
