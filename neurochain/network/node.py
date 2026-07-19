"""A NeuroChain node wiring together the ledger, mempool and AI subsystems.

This is an in-process node abstraction (no sockets): it exposes the operations
a real P2P peer would — submit a transaction, mine a block, import a block
from a peer, and reconcile against a competing chain using an AI-aware
"useful work" fork-choice rule.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..ai.adaptive_difficulty import RLDifficultyController
from ..ai.validator_reputation import ReputationLedger
from ..consensus.proof_of_useful_intelligence import ProofOfUsefulIntelligence
from ..core.block import Block
from ..core.blockchain import Blockchain, ValidationError
from ..core.transaction import Transaction
from .mempool import Mempool


@dataclass
class Node:
    address: str
    initial_difficulty: int = 2
    reputation_ledger: ReputationLedger = field(default_factory=ReputationLedger)
    fraud_detector: Optional[object] = None
    blockchain: Blockchain = field(init=False)
    mempool: Mempool = field(init=False)
    consensus: ProofOfUsefulIntelligence = field(init=False)

    def __post_init__(self) -> None:
        self.consensus = ProofOfUsefulIntelligence(reputation_ledger=self.reputation_ledger)
        self.blockchain = Blockchain(
            consensus=self.consensus,
            initial_difficulty=self.initial_difficulty,
            difficulty_controller=RLDifficultyController(),
        )
        self.mempool = Mempool(fraud_detector=self.fraud_detector)

    # -- transactions ------------------------------------------------------
    def submit(self, tx: Transaction) -> bool:
        return self.mempool.add(tx)

    # -- mining ------------------------------------------------------------
    def mine_block(self, limit: int = 100) -> Block:
        txs = self.mempool.select(limit)
        block = self.blockchain.mine(txs, miner=self.address)
        self.mempool.remove_confirmed(txs)
        self._reward_reputation(block, accepted=True)
        return block

    # -- peer sync ---------------------------------------------------------
    def import_block(self, block: Block) -> bool:
        try:
            self.blockchain.add_block(block)
        except ValidationError:
            self.reputation_ledger.observe_invalid_block(block.miner)
            return False
        self.mempool.remove_confirmed(block.transactions)
        self._reward_reputation(block, accepted=True)
        return True

    def _reward_reputation(self, block: Block, accepted: bool) -> None:
        if not accepted or not block.miner:
            return
        quality = float(block.proof.get("work_quality", 0.0))
        # Timing score: 1.0 when at/under target block time, decaying after.
        self.reputation_ledger.observe_valid_block(
            block.miner, work_quality=quality, timing_score=1.0
        )

    # -- fork choice -------------------------------------------------------
    def useful_work_score(self, chain: List[Block]) -> float:
        """Cumulative useful work of a chain.

        Each validated block contributes its difficulty (the heaviest-chain
        component, analogous to Nakamoto cumulative work) plus a bonus for the
        *useful-work quality* of the trained model it carries. So a chain wins
        by being both longer/harder and by producing better intelligence:

            score = sum_over_blocks( difficulty * (1 + work_quality) )
        """
        return sum(
            b.difficulty * (1.0 + float(b.proof.get("work_quality", 0.0)))
            for b in chain[1:]
        )

    def reconcile(self, other_chain: List[Block]) -> bool:
        """Adopt ``other_chain`` if it is valid and carries more useful work."""
        candidate = Blockchain(
            consensus=self.consensus, initial_difficulty=self.initial_difficulty
        )
        for blk in other_chain[1:]:
            try:
                candidate.add_block(blk)
            except ValidationError:
                return False
        if self.useful_work_score(candidate.chain) > self.useful_work_score(
            self.blockchain.chain
        ):
            self.blockchain = candidate
            return True
        return False
