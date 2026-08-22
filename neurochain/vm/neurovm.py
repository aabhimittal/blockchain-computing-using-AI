"""NeuroVM: a small deterministic stack machine with gas metering and reverts.

The VM is intentionally EVM-flavoured but far smaller. Every opcode has a fixed
gas cost, arithmetic wraps modulo 2**256 (so results never grow unbounded and
are identical on every node), and any fault -- stack underflow, bad jump,
division by zero, running out of gas -- unwinds as a clean revert rather than a
crash. Determinism is the whole point: given the same bytecode, storage and
context, every honest node must compute byte-identical output.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

WORD = 1 << 256
MASK = WORD - 1
MAX_STACK = 1024
MAX_STEPS = 1_000_000  # hard backstop; gas normally bites first

# opcode -> (arg_count popped is dynamic, gas cost)
GAS: Dict[str, int] = {
    "PUSH": 1, "POP": 1, "DUP": 1, "SWAP": 1,
    "ADD": 3, "SUB": 3, "MUL": 5, "DIV": 5, "MOD": 5,
    "EQ": 3, "LT": 3, "GT": 3, "AND": 3, "OR": 3, "NOT": 3, "ISZERO": 3,
    "JUMP": 8, "JUMPI": 10, "JUMPDEST": 1,
    "SLOAD": 50, "SSTORE": 200,
    "CALLER": 2, "VALUE": 2, "TIMESTAMP": 2, "BALANCE": 20,
    "ASSERT": 3, "LOG": 4, "STOP": 0, "RETURN": 0,
}


class VMError(Exception):
    """Base class for VM faults."""


class OutOfGas(VMError):
    pass


class Revert(VMError):
    """Explicit or fault-induced revert; storage changes are rolled back."""


@dataclass
class Contract:
    """A deployed program plus its persistent key/value storage."""

    code: List[Tuple[str, Optional[int]]]
    storage: Dict[int, int] = field(default_factory=dict)
    address: str = ""


@dataclass
class ExecutionResult:
    success: bool
    gas_used: int
    return_value: Optional[int]
    logs: List[int]
    storage: Dict[int, int]
    error: Optional[str] = None


@dataclass
class Context:
    """Runtime environment a call executes against (never mutated by the VM)."""

    caller: str = ""
    value: int = 0
    timestamp: int = 0
    balances: Dict[str, int] = field(default_factory=dict)


def assemble(source: str) -> List[Tuple[str, Optional[int]]]:
    """Parse whitespace/newline separated assembly into (op, arg) pairs.

    Example: ``"PUSH 2 PUSH 3 ADD RETURN"``. Comments start with ``#``.
    """
    program: List[Tuple[str, Optional[int]]] = []
    tokens = source.replace("\n", " ").split()
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("#"):  # skip to end of line-comment token run
            i += 1
            continue
        op = tok.upper()
        if op not in GAS:
            raise VMError(f"unknown opcode: {tok}")
        if op == "PUSH":
            if i + 1 >= len(tokens):
                raise VMError("PUSH missing immediate")
            program.append(("PUSH", int(tokens[i + 1], 0) & MASK))
            i += 2
        else:
            program.append((op, None))
            i += 1
    return program


class NeuroVM:
    """Executes a :class:`Contract` under a gas budget and returns a result."""

    def __init__(self, gas_limit: int = 100_000):
        if gas_limit <= 0:
            raise ValueError("gas_limit must be positive")
        self.gas_limit = gas_limit

    def run(self, contract: Contract, context: Optional[Context] = None) -> ExecutionResult:
        ctx = context or Context()
        stack: List[int] = []
        # Work on a copy so a revert leaves the caller's storage untouched.
        storage = dict(contract.storage)
        logs: List[int] = []
        gas = self.gas_limit
        code = contract.code
        jumpdests = {i for i, (op, _) in enumerate(code) if op == "JUMPDEST"}
        pc = 0
        steps = 0

        def pop() -> int:
            if not stack:
                raise Revert("stack underflow")
            return stack.pop()

        def push(v: int) -> None:
            if len(stack) >= MAX_STACK:
                raise Revert("stack overflow")
            stack.append(v & MASK)

        try:
            while pc < len(code):
                steps += 1
                if steps > MAX_STEPS:
                    raise OutOfGas("step limit exceeded")
                op, arg = code[pc]
                cost = GAS[op]
                if gas < cost:
                    raise OutOfGas(f"out of gas at pc={pc} ({op})")
                gas -= cost

                if op == "PUSH":
                    push(arg or 0)
                elif op == "POP":
                    pop()
                elif op == "DUP":
                    a = pop(); push(a); push(a)
                elif op == "SWAP":
                    a = pop(); b = pop(); push(a); push(b)
                elif op == "ADD":
                    push(pop() + pop())
                elif op == "MUL":
                    push(pop() * pop())
                elif op == "SUB":
                    a = pop(); b = pop(); push(a - b)
                elif op == "DIV":
                    a = pop(); b = pop()
                    if b == 0:
                        raise Revert("division by zero")
                    push(a // b)
                elif op == "MOD":
                    a = pop(); b = pop()
                    if b == 0:
                        raise Revert("modulo by zero")
                    push(a % b)
                elif op == "EQ":
                    push(1 if pop() == pop() else 0)
                elif op == "LT":
                    a = pop(); b = pop(); push(1 if a < b else 0)
                elif op == "GT":
                    a = pop(); b = pop(); push(1 if a > b else 0)
                elif op == "AND":
                    push(pop() & pop())
                elif op == "OR":
                    push(pop() | pop())
                elif op == "NOT":
                    push(~pop())
                elif op == "ISZERO":
                    push(1 if pop() == 0 else 0)
                elif op == "JUMPDEST":
                    pass
                elif op == "JUMP":
                    dest = pop()
                    if dest not in jumpdests:
                        raise Revert(f"invalid jump destination {dest}")
                    pc = dest
                    continue
                elif op == "JUMPI":
                    dest = pop(); cond = pop()
                    if cond != 0:
                        if dest not in jumpdests:
                            raise Revert(f"invalid jump destination {dest}")
                        pc = dest
                        continue
                elif op == "SLOAD":
                    push(storage.get(pop(), 0))
                elif op == "SSTORE":
                    key = pop(); val = pop()
                    storage[key] = val
                elif op == "CALLER":
                    push(int(ctx.caller, 16) & MASK if ctx.caller else 0)
                elif op == "VALUE":
                    push(ctx.value & MASK)
                elif op == "TIMESTAMP":
                    push(ctx.timestamp & MASK)
                elif op == "BALANCE":
                    addr = pop()
                    push(ctx.balances.get(hex(addr), 0) & MASK)
                elif op == "ASSERT":
                    if pop() == 0:
                        raise Revert("assertion failed")
                elif op == "LOG":
                    logs.append(pop())
                elif op == "STOP":
                    return ExecutionResult(True, self.gas_limit - gas, None, logs, storage)
                elif op == "RETURN":
                    ret = pop() if stack else 0
                    return ExecutionResult(True, self.gas_limit - gas, ret, logs, storage)
                pc += 1

            # Fell off the end without STOP/RETURN: implicit success, no value.
            return ExecutionResult(True, self.gas_limit - gas, None, logs, storage)

        except OutOfGas as exc:
            # All gas is consumed on OOG (EVM semantics); storage rolled back.
            return ExecutionResult(False, self.gas_limit, None, [], dict(contract.storage), str(exc))
        except Revert as exc:
            return ExecutionResult(False, self.gas_limit - gas, None, [], dict(contract.storage), str(exc))
