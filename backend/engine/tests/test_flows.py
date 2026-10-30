import pandas as pd
import pytest
from engine import flows

# Carteira-base:
# - Mês 1: Aporta 1000 (custo 1000, mercado 1000). Fluxo = 1000.
# - Mês 2: Vende custo 400 por 600 (lucro 200, custo restante 600), resto da carteira vale 700.
#          Δvlr_investido = -400. Fluxo = -400 - 200 - 0 = -600 (resgate líquido de 600).
# - Mês 3: Recebe provento 10, mercado vale 720.
#          Δvlr_investido = 0. Fluxo = 0 - 0 - 10 = -10 (saída líquida de 10).
BASE = pd.DataFrame({
    "date": ["2024-01-31", "2024-02-29", "2024-03-31"],
    "vlr_mercado": [1000.0, 700.0, 720.0],
    "vlr_investido": [1000.0, 600.0, 600.0],
    "lucro": [0.0, 200.0, 0.0],
    "proventos": [0.0, 0.0, 10.0],
})


def test_flow_columns_constant():
    assert flows.FLOW_COLUMNS == ("vlr_mercado", "vlr_investido", "proventos", "lucro")


def test_external_flows_subtracts_realized_profit_and_dividends():
    assert list(flows.external_flows(flows.group_portfolio(BASE))) == [1000.0, -600.0, -10.0]


def test_hpr_withdrawal_at_end_of_month():
    g = flows.group_portfolio(BASE)
    h = flows.hpr(g["vlr_mercado"], flows.external_flows(g))
    assert h.tolist() == pytest.approx([1.0, 1.30, 730 / 700])


def test_hpr_contribution_at_start_of_month():
    # M0=1000, aporte 500, M1=1600 -> 1600/1500
    df = pd.DataFrame({
        "date": ["2024-01-31", "2024-02-29"],
        "vlr_mercado": [1000.0, 1600.0],
        "vlr_investido": [1000.0, 1500.0],
        "lucro": [0.0, 0.0],
        "proventos": [0.0, 0.0],
    })
    g = flows.group_portfolio(df)
    h = flows.hpr(g["vlr_mercado"], flows.external_flows(g))
    assert h.iloc[1] == pytest.approx(1600 / 1500)


def test_hpr_full_exit_then_reentry():
    # vende tudo (custo 100, lucro 10) -> (0+110)/100 = 1.10; mês parado com M0=0,F=0 -> 1.0;
    # reentra aportando 200 e vale 210 -> 1.05
    df = pd.DataFrame({
        "date": ["2024-01-31", "2024-02-29", "2024-03-31", "2024-04-30"],
        "vlr_mercado": [100.0, 0.0, 0.0, 210.0],
        "vlr_investido": [100.0, 0.0, 0.0, 200.0],
        "lucro": [0.0, 10.0, 0.0, 0.0],
        "proventos": [0.0, 0.0, 0.0, 0.0],
    })
    g = flows.group_portfolio(df)
    h = flows.hpr(g["vlr_mercado"], flows.external_flows(g))
    assert h.tolist() == pytest.approx([1.0, 1.10, 1.0, 1.05])


def test_missing_lucro_column_counts_as_zero():
    g = flows.group_portfolio(BASE.drop(columns=["lucro"]))
    assert (g["lucro"] == 0).all()


def test_same_date_assets_are_summed_before_flows():
    # dois ativos em 2024-02-29 com lucro 120 e 80 -> mesmo fluxo de BASE
    df = pd.DataFrame({
        "date": ["2024-01-31", "2024-02-29", "2024-02-29", "2024-03-31"],
        "vlr_mercado": [1000.0, 400.0, 300.0, 720.0],
        "vlr_investido": [1000.0, 350.0, 250.0, 600.0],
        "lucro": [0.0, 120.0, 80.0, 0.0],
        "proventos": [0.0, 0.0, 0.0, 10.0],
    })
    g = flows.group_portfolio(df)
    assert list(flows.external_flows(g)) == [1000.0, -600.0, -10.0]


def test_group_portfolio_preserves_fluxo_column_if_present():
    df = BASE.copy()
    df["fluxo"] = [1000.0, -600.0, -10.0]
    g = flows.group_portfolio(df)
    assert "fluxo" in g.columns
    assert list(g["fluxo"]) == [1000.0, -600.0, -10.0]


def test_empty_dataframe():
    g = flows.group_portfolio(pd.DataFrame())
    assert g.empty
    flows_series = flows.external_flows(g)
    assert flows_series.empty
    h = flows.hpr(g["vlr_mercado"] if "vlr_mercado" in g else pd.Series(dtype=float), flows_series)
    assert h.empty


def test_is_grouped_true_for_group_portfolio_output():
    df_cru = pd.DataFrame({
        "date": ["2024-01-31", "2024-02-29", "2024-02-29", "2024-03-31"],
        "vlr_mercado": [1000.0, 400.0, 300.0, 720.0],
        "vlr_investido": [1000.0, 350.0, 250.0, 600.0],
        "lucro": [0.0, 120.0, 80.0, 0.0],
        "proventos": [0.0, 0.0, 0.0, 10.0],
    })
    grp = flows.group_portfolio(df_cru)
    assert flows.is_grouped(grp) is True


def test_is_grouped_false_cases():
    # DataFrame vazio
    assert flows.is_grouped(pd.DataFrame()) is False
    assert flows.is_grouped(None) is False

    # date como string
    df_str_date = pd.DataFrame({
        "date": ["2024-01-31", "2024-02-29"],
        "vlr_mercado": [1000.0, 1100.0],
        "vlr_investido": [1000.0, 1000.0],
        "lucro": [0.0, 0.0],
        "proventos": [0.0, 0.0],
    })
    assert flows.is_grouped(df_str_date) is False

    # Datas repetidas
    df_dup = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-31", "2024-01-31"]),
        "vlr_mercado": [500.0, 500.0],
        "vlr_investido": [500.0, 500.0],
        "lucro": [0.0, 0.0],
        "proventos": [0.0, 0.0],
    })
    assert flows.is_grouped(df_dup) is False

    # Datas fora de ordem
    df_unsorted = pd.DataFrame({
        "date": pd.to_datetime(["2024-02-29", "2024-01-31"]),
        "vlr_mercado": [1100.0, 1000.0],
        "vlr_investido": [1000.0, 1000.0],
        "lucro": [0.0, 0.0],
        "proventos": [0.0, 0.0],
    })
    assert flows.is_grouped(df_unsorted) is False

    # Falta coluna 'lucro'
    df_missing_col = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-31", "2024-02-29"]),
        "vlr_mercado": [1000.0, 1100.0],
        "vlr_investido": [1000.0, 1000.0],
        "proventos": [0.0, 0.0],
    })
    assert flows.is_grouped(df_missing_col) is False

