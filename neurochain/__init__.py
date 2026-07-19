"""NeuroChain: a blockchain with AI-driven Proof-of-Useful-Intelligence consensus."""
from .consensus.proof_of_useful_intelligence import ProofOfUsefulIntelligence
from .core.block import Block
from .core.blockchain import Blockchain
from .core.crypto import KeyPair
from .core.transaction import Transaction
from .core.wallet import Wallet
from .network.node import Node

__version__ = "0.1.0"

__all__ = [
    "Blockchain",
    "Block",
    "Transaction",
    "Wallet",
    "KeyPair",
    "ProofOfUsefulIntelligence",
    "Node",
    "__version__",
]
