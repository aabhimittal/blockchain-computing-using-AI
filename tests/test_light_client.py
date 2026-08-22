import pytest

from neurochain.core.block import Block
from neurochain.core.light_client import (
    InclusionError,
    InclusionProof,
    build_inclusion_proof,
    verify_against_header,
    verify_inclusion,
)
from neurochain.core.transaction import Transaction


def make_block(n):
    txs = [
        Transaction(sender=f"a{i}", recipient=f"b{i}", amount=float(i), nonce=i)
        for i in range(n)
    ]
    return Block(index=1, previous_hash="00", transactions=txs, difficulty=1)


def test_inclusion_proof_verifies():
    block = make_block(5)
    target = block.transactions[2].txid
    proof = build_inclusion_proof(block, target)
    assert verify_inclusion(proof, block.merkle_root)


def test_single_transaction_block():
    block = make_block(1)
    target = block.transactions[0].txid
    proof = build_inclusion_proof(block, target)
    assert verify_inclusion(proof, block.merkle_root)


def test_every_tx_in_block_proves():
    block = make_block(8)
    for tx in block.transactions:
        proof = build_inclusion_proof(block, tx.txid)
        assert verify_inclusion(proof, block.merkle_root)


def test_missing_tx_raises():
    block = make_block(3)
    with pytest.raises(InclusionError):
        build_inclusion_proof(block, "deadbeef")


def test_tampered_root_rejected():
    block = make_block(4)
    proof = build_inclusion_proof(block, block.transactions[1].txid)
    assert not verify_inclusion(proof, "00" * 32)


def test_proof_for_wrong_root_rejected():
    block = make_block(6)
    proof = build_inclusion_proof(block, block.transactions[0].txid)
    # A light client trusts a header root that differs -> reject even if path is valid.
    assert not verify_inclusion(proof, "ff" * 32)


def test_verify_against_header():
    block = make_block(5)
    proof = build_inclusion_proof(block, block.transactions[3].txid)
    assert verify_against_header(proof, block.header())


def test_proof_round_trips_through_dict():
    block = make_block(7)
    proof = build_inclusion_proof(block, block.transactions[5].txid)
    restored = InclusionProof.from_dict(proof.to_dict())
    assert verify_inclusion(restored, block.merkle_root)


def test_odd_leaf_count_duplication():
    # 3 txs forces the Bitcoin odd-node duplication path in the tree.
    block = make_block(3)
    for tx in block.transactions:
        proof = build_inclusion_proof(block, tx.txid)
        assert verify_inclusion(proof, block.merkle_root)
