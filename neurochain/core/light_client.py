"""Light-client (SPV) transaction inclusion proofs.

A light client stores only block headers, not full blocks. To confirm a payment
it asks a full node for a Merkle inclusion proof and checks it against the
merkle root committed in the trusted header -- no need to download or trust the
node's transaction list. This module builds and verifies those proofs on top of
the existing Merkle tree.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .block import Block
from .merkle import merkle_proof, merkle_root, verify_proof


class InclusionError(Exception):
    pass


@dataclass
class InclusionProof:
    txid: str
    block_index: int
    merkle_root: str
    path: List[Tuple[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "txid": self.txid,
            "block_index": self.block_index,
            "merkle_root": self.merkle_root,
            "path": [list(step) for step in self.path],
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "InclusionProof":
        return cls(
            txid=d["txid"],
            block_index=d["block_index"],
            merkle_root=d["merkle_root"],
            path=[tuple(step) for step in d["path"]],
        )


def build_inclusion_proof(block: Block, txid: str) -> InclusionProof:
    """Produce an SPV proof that ``txid`` is committed by ``block``'s merkle root."""
    txids = [tx.txid for tx in block.transactions]
    try:
        index = txids.index(txid)
    except ValueError:
        raise InclusionError("transaction not present in block")
    path = merkle_proof(txids, index)
    return InclusionProof(
        txid=txid,
        block_index=block.index,
        merkle_root=block.merkle_root,
        path=path,
    )


def verify_inclusion(proof: InclusionProof, trusted_root: str) -> bool:
    """Check an SPV proof against a merkle root taken from a trusted header.

    The proof's own ``merkle_root`` must match the trusted root (else a node
    could hand a proof for a different block), and the sibling path must
    reconstruct that root from the leaf.
    """
    if proof.merkle_root != trusted_root:
        return False
    return verify_proof(proof.txid, proof.path, trusted_root)


def verify_against_header(proof: InclusionProof, header: Dict) -> bool:
    """Verify a proof against a full block header dict (uses its merkle_root)."""
    root = header.get("merkle_root")
    if not root:
        return False
    return verify_inclusion(proof, root)


__all__ = [
    "InclusionProof",
    "InclusionError",
    "build_inclusion_proof",
    "verify_inclusion",
    "verify_against_header",
    "merkle_root",
]
