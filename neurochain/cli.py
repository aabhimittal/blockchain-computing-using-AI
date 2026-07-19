"""Command-line interface for NeuroChain.

Usage::

    neurochain wallet                 # generate a fresh keypair
    neurochain demo                   # run an end-to-end demo
    neurochain simulate --blocks 8    # mine N blocks and print chain stats
    neurochain serve --port 8000      # launch the REST API (needs fastapi)
"""
from __future__ import annotations

import argparse
import json
import sys

from .core.wallet import Wallet
from .network.node import Node


def _cmd_wallet(_args: argparse.Namespace) -> int:
    w = Wallet()
    print(json.dumps({"address": w.address, "public_key": w.public_key_hex}, indent=2))
    return 0


def _cmd_simulate(args: argparse.Namespace) -> int:
    miner = Wallet()
    node = Node(address=miner.address, initial_difficulty=args.difficulty)
    print(f"miner: {miner.address}")
    for i in range(args.blocks):
        block = node.mine_block()
        print(
            f"block {block.index:>3} "
            f"loss={block.proof['val_loss']:.4f} "
            f"thr={block.proof['threshold']:.4f} "
            f"wq={block.proof['work_quality']:.2f} "
            f"diff={node.blockchain.difficulty} "
            f"rep={node.reputation_ledger.reputation_of(miner.address):.3f}"
        )
    print(f"\nchain valid: {node.blockchain.is_valid_chain()}")
    print(f"miner balance: {node.blockchain.balance_of(miner.address)}")
    print(f"useful-work score: {node.useful_work_score(node.blockchain.chain):.2f}")
    return 0


def _cmd_demo(_args: argparse.Namespace) -> int:
    from examples.demo import main as demo_main  # type: ignore

    demo_main()
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("Install API extras first:  pip install 'neurochain[api]'", file=sys.stderr)
        return 1
    uvicorn.run("neurochain.api.server:app", host=args.host, port=args.port, reload=False)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neurochain", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("wallet", help="generate a new wallet").set_defaults(func=_cmd_wallet)

    sim = sub.add_parser("simulate", help="mine N blocks locally")
    sim.add_argument("--blocks", type=int, default=5)
    sim.add_argument("--difficulty", type=int, default=2)
    sim.set_defaults(func=_cmd_simulate)

    sub.add_parser("demo", help="run the end-to-end demo").set_defaults(func=_cmd_demo)

    serve = sub.add_parser("serve", help="launch the REST API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=_cmd_serve)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
