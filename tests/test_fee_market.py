import pytest

from neurochain.economics.fee_market import FeeMarket


def test_full_block_raises_base_fee():
    fm = FeeMarket(target_gas=1000, base_fee=100)
    fm.update(2000)  # double target
    assert fm.base_fee > 100


def test_empty_block_lowers_base_fee_but_not_below_floor():
    fm = FeeMarket(target_gas=1000, base_fee=100, min_base_fee=10)
    for _ in range(100):
        fm.update(0)
    assert fm.base_fee == 10  # decays to the floor, never below


def test_target_occupancy_holds_fee_steady():
    fm = FeeMarket(target_gas=1000, base_fee=100)
    fm.update(1000)
    assert fm.base_fee == 100


def test_oversized_gas_is_clamped():
    fm = FeeMarket(target_gas=1000, base_fee=100, max_gas=2000)
    fm.update(10 ** 9)  # absurd -> clamped to max_gas, bounded rise
    # 12.5% cap at 2x target: 100 -> at most ~112
    assert 100 < fm.base_fee <= 113


def test_fee_moves_up_by_at_least_one_unit():
    fm = FeeMarket(target_gas=1000, base_fee=1)
    fm.update(1001)  # tiny surplus, integer delta would floor to 0
    assert fm.base_fee == 2  # minimum +1 enforced


def test_affordable_and_effective_fee():
    fm = FeeMarket(target_gas=1000, base_fee=50)
    assert fm.affordable(50)
    assert not fm.affordable(49)
    assert fm.effective_fee(max_fee=100, tip=20) == 70
    assert fm.effective_fee(max_fee=55, tip=20) == 55  # capped by max_fee


def test_effective_fee_below_base_rejected():
    fm = FeeMarket(target_gas=1000, base_fee=50)
    with pytest.raises(ValueError):
        fm.effective_fee(max_fee=40, tip=5)


def test_negative_tip_rejected():
    fm = FeeMarket(target_gas=1000, base_fee=50)
    with pytest.raises(ValueError):
        fm.quote(tip=-1)


def test_invalid_construction():
    with pytest.raises(ValueError):
        FeeMarket(target_gas=0)
    with pytest.raises(ValueError):
        FeeMarket(target_gas=1000, max_gas=500)
    with pytest.raises(ValueError):
        FeeMarket(target_gas=1000, denominator=0)
