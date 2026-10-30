import datetime as dt

import numpy as np
import pandas as pd

from apps.api.serializers import (
    clean,
    frame_to_rows,
    frame_to_series_list,
    series_to_json,
    yearly_to_json,
)


def test_clean_handles_nan_inf_numpy():
    assert clean(None) is None
    assert clean(float("nan")) is None and clean(np.inf) is None and clean(-np.inf) is None
    assert clean(np.nan) is None
    assert clean(np.float64(1.5)) == 1.5 and type(clean(np.float64(1.5))) is float


def test_series_to_json_iso_dates_and_nulls():
    s = pd.Series(
        [1.0, np.nan],
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        name="CDI",
    )
    assert series_to_json(s) == {
        "name": "CDI",
        "data": [["2024-01-02", 1.0], ["2024-01-03", None]],
    }


def test_series_to_json_custom_name_and_date_index():
    s = pd.Series([10.0], index=[dt.date(2024, 5, 10)])
    assert series_to_json(s, name="TIR") == {
        "name": "TIR",
        "data": [["2024-05-10", 10.0]],
    }


def test_frame_to_series_list_preserves_column_order():
    df = pd.DataFrame({"B": [1.0], "A": [2.0]}, index=pd.to_datetime(["2024-01-02"]))
    assert [s["name"] for s in frame_to_series_list(df)] == ["B", "A"]


def test_frame_to_rows_renames():
    df = pd.DataFrame(
        {"Volatilidade": [0.1], "Retorno (CAGR)": [0.2], "Sharpe": [1.0]},
        index=["Carteira"],
    )
    assert frame_to_rows(
        df,
        {
            "Volatilidade": "volatilidade",
            "Retorno (CAGR)": "retorno_cagr",
            "Sharpe": "sharpe",
        },
    ) == [{"name": "Carteira", "volatilidade": 0.1, "retorno_cagr": 0.2, "sharpe": 1.0}]


def test_yearly_to_json_shape():
    df = pd.DataFrame(
        {2023: [0.1], 2024: [np.nan], "Total Acum.": [0.3]},
        index=["Carteira"],
    )
    assert yearly_to_json(df) == {
        "years": [2023, 2024],
        "rows": [
            {
                "name": "Carteira",
                "values": {"2023": 0.1, "2024": None},
                "total": 0.3,
            }
        ],
    }


def test_yearly_to_json_empty():
    assert yearly_to_json(pd.DataFrame()) == {"years": [], "rows": []}
