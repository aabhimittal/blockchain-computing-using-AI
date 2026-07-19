"""Merkle tree over transaction hashes with inclusion proofs."""
from __future__ import annotations

from typing import List, Tuple

from .crypto import double_sha256


def _h(data: bytes) -> str:
    return double_sha256(data).hex()


def merkle_root(tx_hashes: List[str]) -> str:
    if not tx_hashes:
        return _h(b"")
    layer = list(tx_hashes)
    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer.append(layer[-1])  # duplicate last (Bitcoin convention)
        layer = [
            _h(bytes.fromhex(layer[i]) + bytes.fromhex(layer[i + 1]))
            for i in range(0, len(layer), 2)
        ]
    return layer[0]


def merkle_proof(tx_hashes: List[str], index: int) -> List[Tuple[str, str]]:
    """Return the sibling path for ``tx_hashes[index]`` as (side, hash) pairs."""
    if not tx_hashes:
        raise ValueError("empty tx set")
    proof: List[Tuple[str, str]] = []
    layer = list(tx_hashes)
    idx = index
    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer.append(layer[-1])
        sibling = idx ^ 1
        side = "right" if idx % 2 == 0 else "left"
        proof.append((side, layer[sibling]))
        layer = [
            _h(bytes.fromhex(layer[i]) + bytes.fromhex(layer[i + 1]))
            for i in range(0, len(layer), 2)
        ]
        idx //= 2
    return proof


def verify_proof(leaf: str, proof: List[Tuple[str, str]], root: str) -> bool:
    acc = leaf
    for side, sibling in proof:
        if side == "right":
            acc = _h(bytes.fromhex(acc) + bytes.fromhex(sibling))
        else:
            acc = _h(bytes.fromhex(sibling) + bytes.fromhex(acc))
    return acc == root
