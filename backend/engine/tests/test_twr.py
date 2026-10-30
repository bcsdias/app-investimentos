import pandas as pd
import numpy as np
from engine.twr import calculate_twr

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
    
    # Com a nova regra de fluxo externo e HPR da DLP:
    # M_0 = 100, M_1 = 48, F_1 = -55 (resgate no fim do mês)
    # hpr = (M_1 - min(F_1, 0)) / (M_0 + max(F_1, 0)) = (48 - (-55)) / (100 + 0) = 103 / 100 = 1.03 (+3%)
    np.testing.assert_almost_equal(twr.iloc[1], 1.03)


def test_calculate_twr_skips_regroup_when_grouped(monkeypatch):
    from engine.flows import group_portfolio
    import engine.twr

    df = pd.DataFrame({
        "date": ["2024-01-31", "2024-02-29"],
        "vlr_mercado": [1000.0, 1100.0],
        "vlr_investido": [1000.0, 1000.0],
        "lucro": [0.0, 0.0],
        "proventos": [0.0, 0.0],
    })
    grp_pronto = group_portfolio(df)

    def explode(*args, **kwargs):
        raise AssertionError("group_portfolio should not be called when df is already grouped")

    monkeypatch.setattr(engine.twr, "group_portfolio", explode)
    twr = calculate_twr(grp_pronto)
    assert len(twr) == 2


def test_calculate_twr_same_result_grouped_or_raw():
    from engine.flows import external_flows, group_portfolio

    df_cru = pd.DataFrame({
        "date": ["2024-01-31", "2024-02-29", "2024-02-29", "2024-03-31"],
        "vlr_mercado": [1000.0, 400.0, 300.0, 720.0],
        "vlr_investido": [1000.0, 350.0, 250.0, 600.0],
        "lucro": [0.0, 120.0, 80.0, 0.0],
        "proventos": [0.0, 0.0, 0.0, 10.0],
    })
    twr_raw = calculate_twr(df_cru)

    grp = group_portfolio(df_cru)
    grp["fluxo"] = external_flows(grp)
    twr_grouped = calculate_twr(grp)

    pd.testing.assert_series_equal(twr_raw, twr_grouped, check_exact=True)

