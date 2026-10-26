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
