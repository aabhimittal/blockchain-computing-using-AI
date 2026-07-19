"""End-to-end NeuroChain demo.

Creates wallets, funds them via mining, sends value transfers, shows the AI
subsystems (fraud detection, reputation, RL difficulty) reacting, and verifies
chain integrity.

Run with:  python -m examples.demo   (or)  neurochain demo
"""
from __future__ import annotations

import numpy as np

from neurochain.ai.fraud_detection import FraudDetector
from neurochain.core.transaction import Transaction
from neurochain.core.wallet import Wallet
from neurochain.network.node import Node


def _banner(title: str) -> None:
    print("\n" + "=" * 60 + f"\n{title}\n" + "=" * 60)


def main() -> None:
    miner, alice, bob = Wallet(), Wallet(), Wallet()

    # Train a fraud detector on synthetic "normal" traffic before wiring it in.
    rng = np.random.default_rng(0)
    normal = []
    for i in range(120):
        amt = float(rng.uniform(1, 200))
        normal.append(
            Transaction(sender=f"a{i}", recipient="b", amount=amt,
                        fee=amt * 0.01 + rng.uniform(0, 0.2), nonce=i % 8)
        )
    detector = FraudDetector(threshold_sigma=4.0).fit(normal, epochs=250)

    node = Node(address=miner.address, initial_difficulty=2, fraud_detector=detector)

    _banner("1. Mine two blocks to fund the miner")
    for _ in range(2):
        blk = node.mine_block()
        print(f"  mined block {blk.index}: useful-work quality={blk.proof['work_quality']:.2f}, "
              f"loss={blk.proof['val_loss']:.4f} <= thr={blk.proof['threshold']:.4f}")
    print(f"  miner balance: {node.blockchain.balance_of(miner.address)}")

    _banner("2. Miner pays Alice, Alice pays Bob")
    tx1 = miner.create_transaction(alice.address, 30, nonce=node.blockchain.nonce_of(miner.address), fee=1)
    print("  submit miner->alice:", node.submit(tx1))
    node.mine_block()
    tx2 = alice.create_transaction(bob.address, 10, nonce=node.blockchain.nonce_of(alice.address), fee=0.5)
    print("  submit alice->bob :", node.submit(tx2))
    node.mine_block()
    for name, w in [("miner", miner), ("alice", alice), ("bob", bob)]:
        print(f"  {name:5} balance: {node.blockchain.balance_of(w.address):.2f}")

    _banner("3. Fraud detector rejects an anomalous transaction")
    weird = alice.create_transaction(bob.address, 999999, nonce=node.blockchain.nonce_of(alice.address), fee=9999)
    print("  z-score:", round(detector.anomaly_zscore(weird), 2))
    print("  accepted by mempool:", node.submit(weird), "(expected False)")

    _banner("4. AI subsystem state")
    print(f"  RL-tuned difficulty : {node.blockchain.difficulty}")
    print(f"  miner reputation    : {node.reputation_ledger.reputation_of(miner.address):.3f}")
    print(f"  cumulative useful-work score: {node.useful_work_score(node.blockchain.chain):.2f}")

    _banner("5. Integrity")
    print(f"  chain valid: {node.blockchain.is_valid_chain()}")
    print(f"  height     : {node.blockchain.height}")


if __name__ == "__main__":
    main()
