"""Wallet: keypair management and signed-transaction construction."""
from __future__ import annotations

from .crypto import KeyPair
from .transaction import Transaction


class Wallet:
    def __init__(self, keypair: KeyPair | None = None):
        self.keypair = keypair or KeyPair.generate()

    @property
    def address(self) -> str:
        return self.keypair.address

    @property
    def public_key_hex(self) -> str:
        return self.keypair.public_key_hex

    def create_transaction(
        self, recipient: str, amount: float, nonce: int, fee: float = 0.0
    ) -> Transaction:
        tx = Transaction(
            sender=self.address,
            recipient=recipient,
            amount=amount,
            fee=fee,
            nonce=nonce,
            public_key=self.public_key_hex,
        )
        tx.signature = self.keypair.sign(tx.signing_payload())
        return tx
