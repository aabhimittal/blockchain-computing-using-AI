"""Optional FastAPI server exposing a single-node NeuroChain.

Run with::

    pip install 'neurochain[api]'
    neurochain serve --port 8000

Endpoints:
    GET  /              health + node summary
    GET  /chain         full chain as JSON
    GET  /stats         height, difficulty, useful-work score
    GET  /balance/{a}   balance + nonce for an address
    POST /wallet        create a server-side demo wallet
    POST /tx            submit a signed transaction
    POST /mine          mine the pending mempool into a block
"""
from __future__ import annotations

from typing import List, Optional

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover - only hit without api extras
    raise ImportError(
        "FastAPI/pydantic not installed. Install with: pip install 'neurochain[api]'"
    ) from exc

from ..core.transaction import Transaction
from ..core.wallet import Wallet
from ..network.node import Node

app = FastAPI(title="NeuroChain", version="0.1.0")

_SERVER_MINER = Wallet()
_NODE = Node(address=_SERVER_MINER.address, initial_difficulty=2)


class TxIn(BaseModel):
    sender: str
    recipient: str
    amount: float
    fee: float = 0.0
    nonce: int
    public_key: str
    signature: List[str]  # [r, s] as decimal strings


@app.get("/")
def root() -> dict:
    return {
        "name": "NeuroChain",
        "consensus": _NODE.consensus.name,
        "height": _NODE.blockchain.height,
        "miner": _NODE.address,
    }


@app.get("/chain")
def chain() -> dict:
    return _NODE.blockchain.to_dict()


@app.get("/stats")
def stats() -> dict:
    bc = _NODE.blockchain
    return {
        "height": bc.height,
        "difficulty": bc.difficulty,
        "pending": len(_NODE.mempool),
        "useful_work_score": _NODE.useful_work_score(bc.chain),
    }


@app.get("/balance/{address}")
def balance(address: str) -> dict:
    return {
        "address": address,
        "balance": _NODE.blockchain.balance_of(address),
        "nonce": _NODE.blockchain.nonce_of(address),
    }


@app.post("/wallet")
def wallet() -> dict:
    w = Wallet()
    return {"address": w.address, "public_key": w.public_key_hex}


@app.post("/tx")
def submit_tx(tx_in: TxIn) -> dict:
    tx = Transaction(
        sender=tx_in.sender,
        recipient=tx_in.recipient,
        amount=tx_in.amount,
        fee=tx_in.fee,
        nonce=tx_in.nonce,
        public_key=tx_in.public_key,
        signature=(int(tx_in.signature[0]), int(tx_in.signature[1])),
    )
    if not _NODE.submit(tx):
        raise HTTPException(status_code=400, detail="transaction rejected (invalid or flagged)")
    return {"accepted": True, "txid": tx.txid}


@app.post("/mine")
def mine() -> dict:
    block = _NODE.mine_block()
    return {
        "index": block.index,
        "hash": block.hash,
        "transactions": len(block.transactions),
        "proof": {k: block.proof[k] for k in ("val_loss", "threshold", "work_quality")},
    }
