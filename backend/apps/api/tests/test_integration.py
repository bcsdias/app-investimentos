import datetime as dt
import os

import pandas as pd
import pytest

from apps.market import series
from apps.market.services.sources import dlp
from apps.market.services.sources.base import DLPAuthError

pytestmark = pytest.mark.integration


def test_real_dlp_historico_columns():
    token = os.environ.get("DLP_TEST_TOKEN")
    if not token:
        pytest.skip("DLP_TEST_TOKEN não definido no ambiente")
    df = dlp.buscar_historico(token)
    assert isinstance(df, pd.DataFrame) and not df.empty
    print("Colunas DLP /historico:", sorted(df.columns))


def test_real_dlp_invalid_token_raises_auth_error():
    with pytest.raises(DLPAuthError):
        dlp.buscar_historico("invalido-xyz")


def test_real_yahoo_on_demand_aapl(db):
    today = dt.date.today()
    start = today - dt.timedelta(days=30)
    s = series.get_series("AAPL", start, today)
    assert s is not None and not s.empty


def test_real_dlp_in_memory_filter_matches_server_filter():
    token = os.environ.get("DLP_TEST_TOKEN")
    if not token:
        pytest.skip("DLP_TEST_TOKEN não definido no ambiente")
    full = dlp.buscar_historico(token)
    classe = full["classe"].dropna().iloc[0]
    ativo = full["ativo"].dropna().iloc[0]
    order = ["date", "ativo", "corretora"]
    for params in ({"classe": classe}, {"ativo": ativo}):
        mine = dlp.buscar_historico(token, **params).sort_values(order).reset_index(drop=True)
        server = dlp._norm_date_col(pd.DataFrame(dlp._get("/historico", token, params)["historico"]))
        server = server.sort_values(order).reset_index(drop=True)
        assert sorted(mine.columns) == sorted(server.columns)
        pd.testing.assert_frame_equal(mine[sorted(mine.columns)], server[sorted(server.columns)])
