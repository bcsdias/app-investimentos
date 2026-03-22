import pytest
import pandas as pd
import numpy as np
from src.engine.irr import calculate_xirr

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
    # Deve convergir matematicamente para um número astronômico ou retornar None dependendo do cutoff
    # No caso atual, a Newton-Raphson com 50 iterações em dias vai retornar um número bem grande
    if res is not None:
        assert res > 1000.0
