from neurochain.core.wallet import Wallet
from neurochain.network.node import Node


def test_node_mines_and_tracks_balance():
    w = Wallet()
    node = Node(address=w.address, initial_difficulty=2)
    node.mine_block()
    assert node.blockchain.balance_of(w.address) > 0


def test_mempool_selects_and_clears():
    miner, alice = Wallet(), Wallet()
    node = Node(address=miner.address, initial_difficulty=2)
    node.mine_block()
    tx = miner.create_transaction(alice.address, 5, nonce=0, fee=1)
    assert node.submit(tx)
    assert len(node.mempool) == 1
    node.mine_block()
    assert len(node.mempool) == 0
    assert node.blockchain.balance_of(alice.address) == 5


def test_reputation_grows_with_mining():
    w = Wallet()
    node = Node(address=w.address, initial_difficulty=2)
    start = node.reputation_ledger.reputation_of(w.address)
    for _ in range(3):
        node.mine_block()
    assert node.reputation_ledger.reputation_of(w.address) >= start


def test_reconcile_prefers_more_useful_work():
    a, b = Wallet(), Wallet()
    node_a = Node(address=a.address, initial_difficulty=2)
    node_b = Node(address=b.address, initial_difficulty=2)
    for _ in range(3):
        node_a.mine_block()
    node_b.mine_block()
    # B should adopt A's longer / higher-useful-work chain.
    assert node_b.reconcile(node_a.blockchain.chain)
    assert node_b.blockchain.height == node_a.blockchain.height


def test_import_invalid_block_penalizes_reputation():
    a, b = Wallet(), Wallet()
    node = Node(address=a.address, initial_difficulty=2)
    node.mine_block()
    # Craft a block that won't validate (wrong previous hash).
    other = Node(address=b.address, initial_difficulty=2)
    bad_block = other.mine_block()
    bad_block.previous_hash = "0" * 64
    assert not node.import_block(bad_block)
