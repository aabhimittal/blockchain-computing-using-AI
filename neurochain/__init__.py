"""NeuroChain: a blockchain with AI-driven Proof-of-Useful-Intelligence consensus."""
from .consensus.proof_of_useful_intelligence import ProofOfUsefulIntelligence
from .core.block import Block
from .core.blockchain import Blockchain
from .core.crypto import KeyPair
from .core.light_client import (
    InclusionProof,
    build_inclusion_proof,
    verify_inclusion,
)
from .core.transaction import Transaction
from .core.wallet import Wallet
from .economics.fee_market import FeeMarket, FeeQuote
from .network.node import Node
from .vm.neurovm import Contract, NeuroVM

__version__ = "0.2.0"

__all__ = [
    "Blockchain",
    "Block",
    "Transaction",
    "Wallet",
    "KeyPair",
    "ProofOfUsefulIntelligence",
    "Node",
    "NeuroVM",
    "Contract",
    "FeeMarket",
    "FeeQuote",
    "InclusionProof",
    "build_inclusion_proof",
    "verify_inclusion",
    "__version__",
]
