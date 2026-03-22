import pandas as pd
import numpy as np
from src.engine.twr import calculate_twr

def test_twr_empty_input():
    df = pd.DataFrame()
    res = calculate_twr(df)
    assert res.empty

def test_twr_single_period_no_flows():
    data = {
        'date': ['2023-01-01', '2023-01-02'],
        'vlr_mercado': [100.0, 110.0],
        'vlr_investido': [100.0, 100.0],
        'proventos': [0.0, 0.0]
    }
    df = pd.DataFrame(data)
    twr = calculate_twr(df)
    
    assert len(twr) == 2
    # Dia 1 = base 1.0 (hpr = 100/100, mas com init é 1.0)
    # Dia 2 = hpr = 110 / 100 = 1.1 -> twr = 1.1
    np.testing.assert_almost_equal(twr.iloc[1], 1.1)

def test_twr_multi_period_no_flows():
    data = {
        'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'vlr_mercado': [100.0, 110.0, 104.5], # +10%, então -5%
        'vlr_investido': [100.0, 100.0, 100.0],
        'proventos': [0.0, 0.0, 0.0]
    }
    df = pd.DataFrame(data)
    twr = calculate_twr(df)
    
    np.testing.assert_almost_equal(twr.iloc[2], 1.1 * 0.95)

def test_twr_with_aporte():
    data = {
        'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'vlr_mercado': [100.0, 160.0, 176.0], # Aporte de 50 no dia 2, rendimento base do dia 1=0. Dia 2 rendeu 10 (100 -> 110 + 50 aporte = 160). Dia 3 rendeu +10% (160 -> 176).
        'vlr_investido': [100.0, 150.0, 150.0],
        'proventos': [0.0, 0.0, 0.0]
    }
    df = pd.DataFrame(data)
    twr = calculate_twr(df)
    
    # Dia 1: Base = 1.0
    # Dia 2: hpr = 160 / (100 (vlr_inicial) + 50 (fluxo)) = 160 / 150 = 1.06666...
    # Dia 3: hpr = 176 / (160 + 0) = 1.1
    # TWR cumulado = 1 * 1.06666 * 1.1 = 1.17333...
    np.testing.assert_almost_equal(twr.iloc[1], 160/150)
    np.testing.assert_almost_equal(twr.iloc[2], (160/150) * 1.1)

def test_twr_with_resgate_and_dividends():
    data = {
        'date': ['2023-01-01', '2023-01-02'],
        'vlr_mercado': [100.0, 48.0], # Retirou 50, e teve dividendos de 5. Rentabilidade real foi +3% (100 -> 103... e aí -55 fluxo real)
        'vlr_investido': [100.0, 50.0], # Fluxo de -50 (Investido diff)
        'proventos': [0.0, 5.0] # Fluxo final = -50 - 5 = -55
    }
    df = pd.DataFrame(data)
    twr = calculate_twr(df)
    
    # hpr = vlr_mercado / (vlr_inicial + fluxo)
    # vlr_inicial = 100
    # fluxo = -50 - 5 = -55
    # denominador = 100 - 55 = 45
    # hpr = 48 / 45 = 1.06666... (Rendimento implícito positivo de 6.66%!)
    np.testing.assert_almost_equal(twr.iloc[1], 48/45)
