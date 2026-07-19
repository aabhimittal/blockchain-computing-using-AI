import copy

from neurochain.consensus.proof_of_useful_intelligence import (
    ProofOfUsefulIntelligence,
    _challenge_seed,
)
from neurochain.core.blockchain import Blockchain
from neurochain.core.wallet import Wallet


def _mine_one(difficulty=2):
    engine = ProofOfUsefulIntelligence()
    chain = Blockchain(consensus=engine, initial_difficulty=difficulty)
    w = Wallet()
    block = chain.mine([], miner=w.address)
    return engine, chain, block


def test_sealed_block_validates():
    engine, chain, block = _mine_one()
    assert engine.validate(block, chain.chain[-2])
    assert block.proof["val_loss"] <= block.proof["threshold"] + 1e-9


def test_challenge_is_bound_to_block_context():
    engine, chain, block = _mine_one()
    prev = chain.chain[-2]
    assert int(block.proof["challenge_seed"]) == _challenge_seed(
        block.previous_hash, block.merkle_root
    )


def test_tampered_weights_rejected():
    engine, chain, block = _mine_one()
    bad = copy.deepcopy(block)
    bad.proof["weights"][0][0] += 5.0
    assert not engine.validate(bad, chain.chain[-2])


def test_tampered_binding_nonce_rejected():
    engine, chain, block = _mine_one()
    bad = copy.deepcopy(block)
    bad.nonce += 1
    assert not engine.validate(bad, chain.chain[-2])


def test_wrong_algorithm_rejected():
    engine, chain, block = _mine_one()
    bad = copy.deepcopy(block)
    bad.proof["algorithm"] = "sha256-pow"
    assert not engine.validate(bad, chain.chain[-2])


def test_binding_hash_has_required_prefix():
    engine, chain, block = _mine_one()
    assert block.proof["binding_hash"].startswith(engine.pow_prefix)


def test_higher_difficulty_lowers_threshold():
    _, _, easy = _mine_one(difficulty=1)
    _, _, hard = _mine_one(difficulty=5)
    assert hard.proof["threshold"] < easy.proof["threshold"]
