"""Standard NeuroVM contract templates.

Each factory returns a :class:`Contract` whose ``code`` encodes a common
financial primitive. Storage slots and the expected call context are documented
per template so a node can drive them deterministically.
"""
from __future__ import annotations

from .neurovm import Contract, assemble


def timelock(unlock_ts: int) -> Contract:
    """Release (RETURN 1) only once the block timestamp reaches ``unlock_ts``.

    Reverts before then. Context: ``timestamp`` supplies the current time.
    """
    # cond = (timestamp < unlock); assert not cond; return 1
    code = assemble(
        f"PUSH {unlock_ts} TIMESTAMP LT ISZERO ASSERT PUSH 1 RETURN"
    )
    return Contract(code=code)


def multisig(threshold: int) -> Contract:
    """Approve (RETURN 1) once storage slot 0 holds at least ``threshold`` signatures.

    Signers increment slot 0 via SSTORE in prior calls; here we gate on the tally.
    """
    # assert approvals(slot0) >= threshold  <=>  not (approvals < threshold)
    code = assemble(
        f"PUSH {threshold} PUSH 0 SLOAD LT ISZERO ASSERT PUSH 1 RETURN"
    )
    return Contract(code=code)


def escrow() -> Contract:
    """Release funds only after the arbiter sets the confirm flag in storage slot 1.

    Reverts while slot 1 is zero (funds held). Use case: buyer/seller settlement.
    """
    code = assemble("PUSH 1 SLOAD ASSERT PUSH 1 RETURN")
    return Contract(code=code)


def htlc(secret: int) -> Contract:
    """Hash-time-locked style claim: release iff the caller supplies ``secret`` as VALUE.

    A mismatched secret reverts, so funds stay locked. Context: ``value`` carries
    the revealed preimage. (In-VM hashing is out of scope; the secret is compared
    directly, mirroring a preimage check performed off-chain.)
    """
    code = assemble(f"PUSH {secret} VALUE EQ ASSERT PUSH 1 RETURN")
    return Contract(code=code)
