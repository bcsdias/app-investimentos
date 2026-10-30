import pytest
import pandas as pd
import numpy as np
from engine.irr import calculate_xirr, period_return

D1 = ["2023-01-01", "2024-01-01"]  # 365 dias

def test_irr_empty_input():
    res = calculate_xirr([], [])
    assert res is None

def test_irr_simple_case():
    # Investe R$ 1000 num ano, recebe R$ 1100 exatamente no ano seguinte
    dates = pd.to_datetime(['2020-01-01', '2021-01-01']) # 366 dias (bissexto) = ~1.002 anos
    flows = [-1000, 1100]
    
    res = calculate_xirr(flows, dates)
    # Taxa esperada próxima a 10% 
    assert abs(res - 0.10) < 0.01

def test_irr_multi_flow():
    # Múltiplos aportes e o resgate final
    dates = ['2020-01-01', '2020-07-01', '2021-01-01']
    flows = [-500, -500, 1100]
    
    res = calculate_xirr(flows, dates)
    # O aporte de julho teve menos de 1 ano para render, então a taxa anual (TIR) será > 10%
    assert res > 0.10 
    assert res < 0.15 # Approx 14.8%

def test_irr_extreme_outlier():
    # Resgate absurdo que joga a TIR pra lua, mas não pode dar erro
    dates = ['2020-01-01', '2020-01-02']
    flows = [-1000, 2000] # Dobrou o capital em 1 dia
    
    res = calculate_xirr(flows, dates)
    # Deve convergir matematicamente para 2**365 - 1 (~7.5e109)
    assert res is not None
    assert res > 1000.0
    assert res == pytest.approx(2.0 ** 365 - 1.0, rel=1e-5)

def test_xirr_closed_form():
    assert calculate_xirr([-1000, 1100], D1) == pytest.approx(0.10, abs=1e-9)

def test_xirr_unordered_dates():
    assert calculate_xirr([1100, -1000], D1[::-1]) == pytest.approx(0.10, abs=1e-9)

@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e7])
def test_xirr_scale_invariant(scale):
    assert calculate_xirr([-1000 * scale, 1100 * scale], D1) == pytest.approx(0.10, abs=1e-9)

def test_xirr_negative_rate():
    assert calculate_xirr([-1000, 500], D1) == pytest.approx(-0.5, abs=1e-9)

def test_xirr_large_rate():
    assert calculate_xirr([-1000, 30000], D1) == pytest.approx(29.0, rel=1e-9)

def test_xirr_no_sign_change_is_none():
    assert calculate_xirr([-1000, -10], D1) is None

def test_xirr_single_date_is_none():
    assert calculate_xirr([-1000, 1100], ["2023-01-01"] * 2) is None


def test_period_return_one_month_loss():
    # -1000 em 2024-01-31, +970 em 2024-02-29: XIRR a.a. ≈ -32 %; TIR do período = -3 %
    r = calculate_xirr([-1000, 970], ["2024-01-31", "2024-02-29"])
    assert period_return(r, "2024-01-31", "2024-02-29") == pytest.approx(-0.03, abs=1e-9)


def test_period_return_one_year_equals_annual():
    assert period_return(0.10, "2023-01-01", "2024-01-01") == pytest.approx(0.10)


def test_period_return_zero_days():
    assert period_return(0.10, "2024-01-01", "2024-01-01") == 0.0


def test_period_return_none_or_nan():
    assert period_return(None, "2024-01-01", "2024-02-01") is None
    assert period_return(float("nan"), "2024-01-01", "2024-02-01") is None


