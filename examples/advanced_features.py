"""Demonstrate NeuroChain's advanced subsystems end-to-end.

Run: ``python examples/advanced_features.py``
"""
import numpy as np

from neurochain.ai.federated import aggregate
from neurochain.core.block import Block
from neurochain.core.light_client import build_inclusion_proof, verify_inclusion
from neurochain.core.transaction import Transaction
from neurochain.economics.fee_market import FeeMarket
from neurochain.vm import contracts
from neurochain.vm.neurovm import Context, Contract, NeuroVM, assemble


def demo_vm():
    print("== NeuroVM smart contract ==")
    vm = NeuroVM(gas_limit=10_000)
    prog = Contract(code=assemble("PUSH 6 PUSH 7 MUL RETURN"))
    r = vm.run(prog)
    print(f"  6 * 7 = {r.return_value}  (gas used {r.gas_used})")

    lock = contracts.timelock(unlock_ts=1_000)
    print(f"  timelock before unlock: success={vm.run(lock, Context(timestamp=999)).success}")
    print(f"  timelock after  unlock: success={vm.run(lock, Context(timestamp=1_001)).success}")


def demo_fee_market():
    print("\n== EIP-1559 fee market ==")
    fm = FeeMarket(target_gas=1000, base_fee=100)
    for used in (2000, 2000, 0, 0):
        fm.update(used)
        print(f"  gas_used={used:>5}  base_fee -> {fm.base_fee}")


def demo_federated():
    print("\n== Byzantine-robust federated aggregation ==")
    honest = [np.array([1.0, 1.0]) for _ in range(9)]
    poison = [np.array([999.0, -999.0])]
    out = aggregate(honest + poison, method="trimmed_mean", trim=0.1)
    print(f"  aggregate (poison rejected): {out}")


def demo_spv():
    print("\n== Light-client SPV proof ==")
    txs = [Transaction(sender=f"a{i}", recipient=f"b{i}", amount=float(i), nonce=i) for i in range(5)]
    block = Block(index=1, previous_hash="00", transactions=txs, difficulty=1)
    target = txs[2].txid
    proof = build_inclusion_proof(block, target)
    print(f"  tx {target[:12]}... included: {verify_inclusion(proof, block.merkle_root)}")


if __name__ == "__main__":
    demo_vm()
    demo_fee_market()
    demo_federated()
    demo_spv()
