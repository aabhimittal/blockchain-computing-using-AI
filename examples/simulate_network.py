"""Multi-node simulation: mining, block propagation and useful-work fork choice.

Two nodes mine independently, then reconcile. The node adopting the chain with
the greater cumulative *useful work* (not merely the longest) wins — showing
the PoUI fork-choice rule in action.

Run with:  python -m examples.simulate_network
"""
from __future__ import annotations

from neurochain.core.wallet import Wallet
from neurochain.network.node import Node


def main() -> None:
    a, b = Wallet(), Wallet()
    node_a = Node(address=a.address, initial_difficulty=2)
    node_b = Node(address=b.address, initial_difficulty=2)

    print("Node A mines 3 blocks, Node B mines 2 blocks (independently)")
    for _ in range(3):
        node_a.mine_block()
    for _ in range(2):
        node_b.mine_block()

    print(f"  A: height={node_a.blockchain.height} "
          f"useful-work={node_a.useful_work_score(node_a.blockchain.chain):.2f}")
    print(f"  B: height={node_b.blockchain.height} "
          f"useful-work={node_b.useful_work_score(node_b.blockchain.chain):.2f}")

    print("\nNode B reconciles against Node A's chain...")
    switched = node_b.reconcile(node_a.blockchain.chain)
    print(f"  B switched to A's chain: {switched}")
    print(f"  B height now: {node_b.blockchain.height}")
    print(f"  B chain valid: {node_b.blockchain.is_valid_chain()}")


if __name__ == "__main__":
    main()
