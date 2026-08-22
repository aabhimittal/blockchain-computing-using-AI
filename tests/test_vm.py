import pytest

from neurochain.vm import contracts
from neurochain.vm.neurovm import Context, Contract, NeuroVM, assemble


def run(source, gas=100_000, ctx=None, storage=None):
    vm = NeuroVM(gas_limit=gas)
    c = Contract(code=assemble(source), storage=storage or {})
    return vm.run(c, ctx)


def test_arithmetic_and_return():
    r = run("PUSH 2 PUSH 3 ADD RETURN")
    assert r.success and r.return_value == 5


def test_arithmetic_wraps_mod_2_256():
    r = run(f"PUSH {(1 << 256) - 1} PUSH 1 ADD RETURN")
    assert r.success and r.return_value == 0  # overflow wraps to 0


def test_division_by_zero_reverts():
    r = run("PUSH 0 PUSH 5 DIV RETURN")
    assert not r.success and "division by zero" in r.error


def test_modulo_by_zero_reverts():
    r = run("PUSH 0 PUSH 5 MOD RETURN")
    assert not r.success and "modulo by zero" in r.error


def test_stack_underflow_reverts():
    r = run("ADD")
    assert not r.success and "underflow" in r.error


def test_out_of_gas_on_infinite_loop_consumes_all_gas():
    # JUMPDEST at pc 0, unconditional JUMP back to 0 -> loops until gas runs out.
    r = run("JUMPDEST PUSH 0 JUMP", gas=500)
    assert not r.success
    assert "gas" in r.error.lower()
    assert r.gas_used == 500  # OOG bills the full budget


def test_invalid_jump_destination_reverts():
    r = run("PUSH 99 JUMP")
    assert not r.success and "invalid jump" in r.error


def test_conditional_jump_taken():
    # JUMPI pops dest (top) then cond; jump to the JUMPDEST at op index 5.
    # ops: PUSH1(0) PUSH5(1) JUMPI(2) PUSH0(3) RETURN(4) JUMPDEST(5) PUSH7(6) RETURN(7)
    r = run("PUSH 1 PUSH 5 JUMPI PUSH 0 RETURN JUMPDEST PUSH 7 RETURN")
    assert r.success and r.return_value == 7


def test_conditional_jump_not_taken():
    r = run("PUSH 0 PUSH 5 JUMPI PUSH 0 RETURN JUMPDEST PUSH 7 RETURN")
    assert r.success and r.return_value == 0  # cond false -> fall through


def test_assert_failure_reverts():
    r = run("PUSH 0 ASSERT")
    assert not r.success and "assertion failed" in r.error


def test_storage_persists_within_run_and_rolls_back_on_revert():
    # Write 42 to slot 1, then force a revert; result storage must be unchanged.
    r = run("PUSH 42 PUSH 1 SSTORE PUSH 0 ASSERT", storage={1: 0})
    assert not r.success
    assert r.storage == {1: 0}  # rolled back


def test_storage_survives_successful_run():
    r = run("PUSH 42 PUSH 1 SSTORE STOP", storage={})
    assert r.success and r.storage[1] == 42


def test_determinism_same_output_twice():
    a = run("PUSH 6 PUSH 9 PUSH 4 MUL SUB RETURN")  # (9*4) - 6 = 30
    b = run("PUSH 6 PUSH 9 PUSH 4 MUL SUB RETURN")
    assert a.return_value == b.return_value == 30


def test_stack_overflow_reverts():
    # Push far past MAX_STACK via a loop.
    prog = "JUMPDEST PUSH 1 PUSH 0 JUMP"
    r = run(prog, gas=1_000_000)
    assert not r.success and ("overflow" in r.error or "gas" in r.error.lower())


def test_unknown_opcode_rejected_at_assembly():
    with pytest.raises(Exception):
        assemble("FOOBAR")


def test_gas_limit_must_be_positive():
    with pytest.raises(ValueError):
        NeuroVM(gas_limit=0)


# ---- standard contract templates ----------------------------------------

def test_timelock_locked_then_unlocks():
    c = contracts.timelock(1000)
    vm = NeuroVM()
    assert not vm.run(c, Context(timestamp=999)).success   # still locked
    assert vm.run(c, Context(timestamp=1000)).success       # unlocked at boundary
    assert vm.run(c, Context(timestamp=2000)).success


def test_multisig_threshold():
    c = contracts.multisig(3)
    vm = NeuroVM()
    c.storage = {0: 2}
    assert not vm.run(c).success            # 2 < 3 signatures
    c.storage = {0: 3}
    assert vm.run(c).success                # threshold met


def test_escrow_gated_on_confirm_flag():
    c = contracts.escrow()
    vm = NeuroVM()
    assert not vm.run(c).success            # slot 1 unset -> held
    c.storage = {1: 1}
    assert vm.run(c).success                # arbiter confirmed


def test_htlc_requires_correct_secret():
    c = contracts.htlc(secret=12345)
    vm = NeuroVM()
    assert not vm.run(c, Context(value=1)).success
    assert vm.run(c, Context(value=12345)).success
