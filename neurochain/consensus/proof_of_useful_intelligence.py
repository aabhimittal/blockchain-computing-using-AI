"""Proof-of-Useful-Intelligence (PoUI) consensus.

The novel idea: replace wasteful hash-grinding with *useful* machine-learning
work whose difficulty and verification retain PoW's asymmetry (hard to
produce, cheap to check).

For each block the engine deterministically derives a learning challenge from
the block's own context (``previous_hash`` + ``merkle_root``): a synthetic
regression task with a train and a hold-out validation split. To seal the
block the miner must **train a model** whose validation loss drops below a
difficulty-scaled threshold, then find a small binding nonce so the proof is
also anchored by a light proof-of-work (anti-grinding / tie-break).

Verification is cheap and asymmetric: the verifier re-derives the exact same
challenge from public block data, loads the submitted weights, runs a single
forward pass over the validation split, and checks the loss and the binding
nonce. No re-training required.

Validator reputation (optional) grants trusted validators a small threshold
leniency, coupling the AI reputation subsystem to consensus.

Why "useful": the trained models are real function approximators bound to
on-chain entropy; the challenge generator can be swapped for domain datasets
(protein folding surrogates, hashing-hard ML tasks, etc.) so the energy that
would be burned on SHA grinding instead produces reusable models.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..ai.neural_net import MLP
from ..core.block import Block
from .base import ConsensusEngine

CHALLENGE_INPUT_DIM = 4
CHALLENGE_HIDDEN = [16, 16]
DATASET_SIZE = 64
BASE_THRESHOLD = 0.15
DECAY = 0.7            # threshold shrinks geometrically with difficulty
WEIGHT_ROUND = 6


def _challenge_seed(previous_hash: str, merkle_root: str) -> int:
    digest = hashlib.sha256((previous_hash + merkle_root).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _make_challenge(seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic synthetic regression task derived from ``seed``.

    A non-linear target (``sin`` of a random projection) so a hidden layer is
    genuinely required. The miner must *fit* it below the difficulty threshold,
    so harder blocks demand more gradient-descent compute (the "useful work").
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((DATASET_SIZE, CHALLENGE_INPUT_DIM))
    w_true = rng.standard_normal(CHALLENGE_INPUT_DIM)
    y = np.sin(x @ w_true).reshape(-1, 1)
    return x, y


def _threshold(difficulty: int, reputation: float) -> float:
    base = BASE_THRESHOLD * (DECAY ** max(difficulty, 0))
    # Higher reputation -> up to 15% leniency on the required loss.
    return base * (1.0 + 0.15 * max(0.0, min(1.0, reputation)))


def _serialize_weights(model: MLP) -> List[List[float]]:
    flat = []
    for w in model.weights:
        flat.append([round(float(v), WEIGHT_ROUND) for v in w.ravel().tolist()])
    for b in model.biases:
        flat.append([round(float(v), WEIGHT_ROUND) for v in b.ravel().tolist()])
    return flat


def _load_weights(model: MLP, flat: List[List[float]]) -> None:
    n = len(model.weights)
    for i in range(n):
        model.weights[i] = np.array(flat[i], dtype=float).reshape(model.weights[i].shape)
    for i in range(n):
        model.biases[i] = np.array(flat[n + i], dtype=float).reshape(model.biases[i].shape)


def _weights_digest(flat: List[List[float]]) -> str:
    return hashlib.sha256(str(flat).encode()).hexdigest()


def _dataset_loss(flat: List[List[float]], seed: int) -> float:
    x, y = _make_challenge(seed)
    model = MLP(layer_sizes=[CHALLENGE_INPUT_DIM, *CHALLENGE_HIDDEN, 1], seed=1)
    _load_weights(model, flat)
    pred = model.predict(x)
    return float(np.mean((pred - y) ** 2))


@dataclass
class ProofOfUsefulIntelligence(ConsensusEngine):
    name: str = "proof-of-useful-intelligence"
    reputation_ledger: Optional[object] = None   # ReputationLedger (duck-typed)
    pow_prefix: str = "0"                         # binding-nonce leading hex zeros
    max_epochs: int = 1200

    # -- production --------------------------------------------------------
    def seal(self, block: Block) -> Block:
        seed = _challenge_seed(block.previous_hash, block.merkle_root)
        x, y = _make_challenge(seed)
        reputation = self._reputation(block.miner)
        threshold = _threshold(block.difficulty, reputation)

        model = MLP(
            layer_sizes=[CHALLENGE_INPUT_DIM, *CHALLENGE_HIDDEN, 1],
            seed=(seed % 100000) + 1,
            lr=2e-2,
        )
        epochs_per_round = 50
        rounds = max(1, self.max_epochs // epochs_per_round)
        best_flat = _serialize_weights(model)
        best_loss = _dataset_loss(best_flat, seed)
        for _ in range(rounds):
            model.train(x, y, epochs=epochs_per_round, batch_size=16)
            flat = _serialize_weights(model)
            loss = _dataset_loss(flat, seed)  # loss of the *rounded* weights we ship
            if loss < best_loss:
                best_loss, best_flat = loss, flat
            if best_loss <= threshold:
                break

        digest = _weights_digest(best_flat)
        nonce, binding = self._find_binding_nonce(digest, seed)
        block.nonce = nonce
        block.proof = {
            "algorithm": self.name,
            "challenge_seed": str(seed),
            "val_loss": round(best_loss, WEIGHT_ROUND),
            "threshold": round(threshold, WEIGHT_ROUND),
            "weights": best_flat,
            "weights_digest": digest,
            "binding_hash": binding,
            "work_quality": self._work_quality(best_loss, threshold),
        }
        return block

    def _find_binding_nonce(self, digest: str, seed: int) -> Tuple[int, str]:
        nonce = 0
        prefix = self.pow_prefix
        while True:
            h = hashlib.sha256(f"{digest}:{seed}:{nonce}".encode()).hexdigest()
            if h.startswith(prefix):
                return nonce, h
            nonce += 1

    @staticmethod
    def _work_quality(loss: float, threshold: float) -> float:
        """1.0 when loss is far below threshold, ->0 as it approaches it."""
        if threshold <= 0:
            return 0.0
        return float(max(0.0, min(1.0, 1.0 - loss / (threshold + 1e-9))))

    # -- verification ------------------------------------------------------
    def validate(self, block: Block, previous: Block) -> bool:
        proof = block.proof
        if proof.get("genesis"):
            return True
        if proof.get("algorithm") != self.name:
            return False
        try:
            seed = int(proof["challenge_seed"])
            flat = proof["weights"]
            digest = proof["weights_digest"]
        except (KeyError, ValueError, TypeError):
            return False

        # 1. The challenge must be bound to this exact block context.
        if seed != _challenge_seed(block.previous_hash, block.merkle_root):
            return False
        # 2. Weights digest integrity.
        if _weights_digest(flat) != digest:
            return False
        # 3. Binding proof-of-work anchor.
        binding = hashlib.sha256(f"{digest}:{seed}:{block.nonce}".encode()).hexdigest()
        if binding != proof.get("binding_hash") or not binding.startswith(self.pow_prefix):
            return False
        # 4. Recompute validation loss and enforce the difficulty threshold.
        reputation = self._reputation(block.miner)
        threshold = _threshold(block.difficulty, reputation)
        try:
            loss = _dataset_loss(flat, seed)
        except Exception:
            return False
        return loss <= threshold + 1e-9

    # -- reputation hook ---------------------------------------------------
    def _reputation(self, address: str) -> float:
        if self.reputation_ledger is None or not address:
            return 0.5
        try:
            return float(self.reputation_ledger.reputation_of(address))
        except Exception:
            return 0.5
