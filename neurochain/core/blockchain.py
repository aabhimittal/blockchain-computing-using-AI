"""The NeuroChain ledger: state, validation and block application.

The ledger tracks account balances and per-account nonces, validates every
incoming block (structure, consensus proof, transaction signatures, balances
and nonces), and applies it atomically. Consensus is pluggable via a
``ConsensusEngine``; the default is Proof-of-Useful-Intelligence.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

from ..consensus.base import ConsensusEngine
from .block import Block
from .transaction import COINBASE_SENDER, Transaction

BLOCK_REWARD = 50.0
TARGET_BLOCK_TIME = 10.0  # seconds; used by the AI difficulty controller


class ValidationError(Exception):
    pass


class Blockchain:
    def __init__(
        self,
        consensus: ConsensusEngine,
        initial_difficulty: int = 3,
        difficulty_controller=None,
    ):
        self.consensus = consensus
        self.difficulty = initial_difficulty
        self.difficulty_controller = difficulty_controller
        self.chain: List[Block] = []
        self.balances: Dict[str, float] = {}
        self.nonces: Dict[str, int] = {}
        self._create_genesis()

    # -- genesis -----------------------------------------------------------
    def _create_genesis(self) -> None:
        genesis = Block(
            index=0,
            previous_hash="0" * 64,
            transactions=[],
            difficulty=self.difficulty,
            miner="genesis",
            timestamp=0.0,
            proof={"genesis": True},
        )
        self.chain.append(genesis)

    # -- accessors ---------------------------------------------------------
    @property
    def last_block(self) -> Block:
        return self.chain[-1]

    @property
    def height(self) -> int:
        return len(self.chain) - 1

    def balance_of(self, address: str) -> float:
        return self.balances.get(address, 0.0)

    def nonce_of(self, address: str) -> int:
        return self.nonces.get(address, 0)

    # -- block construction ------------------------------------------------
    def create_candidate(self, transactions: List[Transaction], miner: str) -> Block:
        """Assemble (but do not seal) the next block from a set of transactions."""
        reward = BLOCK_REWARD + sum(tx.fee for tx in transactions)
        coinbase = Transaction(
            sender=COINBASE_SENDER, recipient=miner, amount=reward, nonce=0
        )
        return Block(
            index=self.height + 1,
            previous_hash=self.last_block.hash,
            transactions=[coinbase, *transactions],
            difficulty=self.difficulty,
            miner=miner,
            timestamp=time.time(),
        )

    def mine(self, transactions: List[Transaction], miner: str) -> Block:
        """Create, seal and append a block in one call."""
        candidate = self.create_candidate(transactions, miner)
        sealed = self.consensus.seal(candidate)
        self.add_block(sealed)
        return sealed

    # -- validation --------------------------------------------------------
    def _validate_transactions(self, block: Block) -> None:
        seen_coinbase = False
        working_balances = dict(self.balances)
        working_nonces = dict(self.nonces)
        for tx in block.transactions:
            if tx.is_coinbase():
                if seen_coinbase:
                    raise ValidationError("multiple coinbase transactions")
                seen_coinbase = True
                expected = BLOCK_REWARD + sum(
                    t.fee for t in block.transactions if not t.is_coinbase()
                )
                if round(tx.amount, 8) > round(expected, 8):
                    raise ValidationError("coinbase over-issues reward")
                continue
            if not tx.is_valid():
                raise ValidationError(f"invalid signature on tx {tx.txid[:10]}")
            if tx.nonce != working_nonces.get(tx.sender, 0):
                raise ValidationError(
                    f"bad nonce for {tx.sender[:10]}: got {tx.nonce}"
                )
            total = tx.amount + tx.fee
            if working_balances.get(tx.sender, 0.0) < total:
                raise ValidationError(f"insufficient funds for {tx.sender[:10]}")
            working_balances[tx.sender] = working_balances.get(tx.sender, 0.0) - total
            working_balances[tx.recipient] = (
                working_balances.get(tx.recipient, 0.0) + tx.amount
            )
            working_nonces[tx.sender] = tx.nonce + 1

    def validate_block(self, block: Block, previous: Optional[Block] = None) -> None:
        previous = previous or self.last_block
        if block.index != previous.index + 1:
            raise ValidationError("non-sequential index")
        if block.previous_hash != previous.hash:
            raise ValidationError("previous-hash mismatch")
        if not self.consensus.validate(block, previous):
            raise ValidationError("consensus proof rejected")
        self._validate_transactions(block)

    # -- application -------------------------------------------------------
    def _apply(self, block: Block) -> None:
        for tx in block.transactions:
            if tx.is_coinbase():
                self.balances[tx.recipient] = (
                    self.balances.get(tx.recipient, 0.0) + tx.amount
                )
                continue
            self.balances[tx.sender] = self.balances.get(tx.sender, 0.0) - (
                tx.amount + tx.fee
            )
            self.balances[tx.recipient] = (
                self.balances.get(tx.recipient, 0.0) + tx.amount
            )
            self.nonces[tx.sender] = tx.nonce + 1

    def add_block(self, block: Block) -> None:
        self.validate_block(block)
        self._apply(block)
        self.chain.append(block)
        self._retune_difficulty()

    def _retune_difficulty(self) -> None:
        if self.difficulty_controller is None or self.height < 1:
            return
        prev, cur = self.chain[-2], self.chain[-1]
        elapsed = max(cur.timestamp - prev.timestamp, 1e-6) if prev.timestamp else TARGET_BLOCK_TIME
        self.difficulty = self.difficulty_controller.next_difficulty(
            current_difficulty=self.difficulty,
            observed_block_time=elapsed,
            target_block_time=TARGET_BLOCK_TIME,
        )

    # -- integrity ---------------------------------------------------------
    def is_valid_chain(self) -> bool:
        for i in range(1, len(self.chain)):
            block, prev = self.chain[i], self.chain[i - 1]
            if block.previous_hash != prev.hash:
                return False
            if not self.consensus.validate(block, prev):
                return False
        return True

    def to_dict(self) -> dict:
        return {
            "height": self.height,
            "difficulty": self.difficulty,
            "consensus": self.consensus.name,
            "blocks": [b.to_dict() for b in self.chain],
        }
