import pandas as pd
import pytest

from parity.cases import Case, Window
from parity.engine_side import METRICS
from parity.regress import build_regression_grid, changed_columns, format_markdown_table


def test_changed_columns_reports_only_changed():
    dates = pd.date_range("2024-01-01", periods=3)
    old = {
        "twr": pd.DataFrame({"Carteira": [100.0, 105.0, 110.0], "CDI": [100.0, 101.0, 102.0]}, index=dates),
        "drawdown": pd.DataFrame({"Carteira": [0.0, -0.01, 0.0], "CDI": [0.0, 0.0, 0.0]}, index=dates),
    }
    new = {
        "twr": pd.DataFrame({"Carteira": [100.0, 106.0, 112.0], "CDI": [100.0, 101.0, 102.0]}, index=dates),
        "drawdown": pd.DataFrame({"Carteira": [0.0, -0.01, 0.0], "CDI": [0.0, 0.0, 0.0]}, index=dates),
    }

    result = changed_columns(old, new)
    assert result == {"twr": ["Carteira"]}


def test_changed_columns_identical_returns_empty():
    dates = pd.date_range("2024-01-01", periods=3)
    data = {
        "twr": pd.DataFrame({"Carteira": [100.0, 105.0, 110.0]}, index=dates),
        "irr": pd.DataFrame({"TIR": [0.12]}, index=[dates[-1]]),
    }
    assert changed_columns(data, data) == {}


def test_changed_columns_multiple_columns():
    dates = pd.date_range("2024-01-01", periods=2)
    old = {
        "twr": pd.DataFrame({"Carteira": [100.0, 101.0], "CDI": [100.0, 100.5]}, index=dates),
        "risk_return": pd.DataFrame({"Retorno (CAGR)": [0.1], "Sharpe": [1.5]}, index=["Carteira"]),
    }
    new = {
        "twr": pd.DataFrame({"Carteira": [100.0, 102.0], "CDI": [100.0, 101.0]}, index=dates),
        "risk_return": pd.DataFrame({"Retorno (CAGR)": [0.1], "Sharpe": [2.0]}, index=["Carteira"]),
    }
    result = changed_columns(old, new)
    assert result == {
        "twr": ["Carteira", "CDI"],
        "risk_return": ["Sharpe"],
    }


def test_changed_columns_added_column():
    dates = pd.date_range("2024-01-01", periods=2)
    old = {"twr": pd.DataFrame({"Carteira": [100.0, 101.0]}, index=dates)}
    new = {"twr": pd.DataFrame({"Carteira": [100.0, 101.0], "IFIX": [100.0, 100.2]}, index=dates)}
    result = changed_columns(old, new)
    assert result == {"twr": ["IFIX"]}


def test_build_regression_grid_formats_cells():
    import datetime as dt

    win = Window("w1", dt.date(2024, 1, 1), dt.date(2024, 1, 31))
    c1 = Case("c1", win)
    c2 = Case("c2", win)
    dates = pd.date_range("2024-01-01", periods=2)

    old_outputs = {
        "c1": {
            "twr": pd.DataFrame({"Carteira": [100.0, 101.0], "CDI": [100.0, 100.5]}, index=dates),
            "drawdown": pd.DataFrame({"Carteira": [0.0, 0.0]}, index=dates),
        }
    }
    new_outputs = {
        "c1": {
            "twr": pd.DataFrame({"Carteira": [100.0, 102.0], "CDI": [100.0, 100.5]}, index=dates),
            "drawdown": pd.DataFrame({"Carteira": [0.0, 0.0]}, index=dates),
        },
        "c2": {
            "twr": pd.DataFrame({"Carteira": [100.0, 101.0]}, index=dates),
        },
    }

    grid = build_regression_grid([c1, c2], old_outputs, new_outputs)
    assert grid.loc["c1", "twr"] == "Carteira"
    assert grid.loc["c1", "drawdown"] == "="
    assert grid.loc["c2", "twr"] == "novo"
    assert grid.loc["c2", "portfolio"] == "novo"

    rendered = format_markdown_table(grid)
    assert "| caso |" in rendered
    assert "| c1   |" in rendered
    assert "| c2   |" in rendered
