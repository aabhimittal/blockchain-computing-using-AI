"""Consensus engine interface.

A consensus engine is responsible for producing the ``proof`` attached to a
block (``seal``) and for validating that proof (``validate``). The ledger in
``core.blockchain`` stays agnostic to *how* blocks are sealed.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid a runtime import cycle
    from ..core.block import Block


class ConsensusEngine(ABC):
    name: str = "abstract"

    @abstractmethod
    def seal(self, block: "Block") -> "Block":
        """Populate ``block.proof`` (and ``block.nonce``) so it satisfies the rule."""

    @abstractmethod
    def validate(self, block: "Block", previous: "Block") -> bool:
        """Return True iff ``block``'s proof is valid given its predecessor."""
