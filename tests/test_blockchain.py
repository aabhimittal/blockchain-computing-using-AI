import pytest

from neurochain.consensus.proof_of_useful_intelligence import ProofOfUsefulIntelligence
from neurochain.core.blockchain import BLOCK_REWARD, Blockchain, ValidationError
from neurochain.core.merkle import merkle_proof, merkle_root, verify_proof
from neurochain.core.wallet import Wallet


@pytest.fixture
def chain():
    return Blockchain(consensus=ProofOfUsefulIntelligence(), initial_difficulty=2)


def test_genesis(chain):
    assert chain.height == 0
    assert chain.chain[0].previous_hash == "0" * 64


def test_mine_rewards_miner(chain):
    w = Wallet()
    chain.mine([], miner=w.address)
    assert chain.balance_of(w.address) == BLOCK_REWARD
    assert chain.is_valid_chain()


def test_value_transfer_and_balances(chain):
    miner, alice = Wallet(), Wallet()
    chain.mine([], miner=miner.address)
    tx = miner.create_transaction(alice.address, 10, nonce=0, fee=2)
    chain.mine([tx], miner=miner.address)
    # miner: 50 - 10 - 2 (spent) + 50 + 2 (second reward incl fee) = 90
    assert chain.balance_of(alice.address) == 10
    assert chain.balance_of(miner.address) == pytest.approx(90.0)


def test_double_spend_rejected(chain):
    miner, alice = Wallet(), Wallet()
    chain.mine([], miner=miner.address)
    tx = miner.create_transaction(alice.address, 999, nonce=0)
    with pytest.raises(ValidationError):
        chain.mine([tx], miner=miner.address)


def test_bad_nonce_rejected(chain):
    miner, alice = Wallet(), Wallet()
    chain.mine([], miner=miner.address)
    tx = miner.create_transaction(alice.address, 5, nonce=7)
    with pytest.raises(ValidationError):
        chain.mine([tx], miner=miner.address)


def test_tampered_previous_hash_breaks_chain(chain):
    w = Wallet()
    chain.mine([], miner=w.address)
    chain.chain[1].previous_hash = "f" * 64
    assert not chain.is_valid_chain()


def test_merkle_inclusion_proof():
    leaves = [f"{i:064x}" for i in range(5)]
    root = merkle_root(leaves)
    proof = merkle_proof(leaves, 3)
    assert verify_proof(leaves[3], proof, root)
    assert not verify_proof(leaves[0], proof, root)
