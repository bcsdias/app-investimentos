import math

import pandas as pd
import pytest
from django.test import Client

from apps.core.models import UserToken
from engine import dataset


def _make_portfolio_fixture():
    idx = pd.bdate_range("2024-03-01", "2025-06-30")
    n = len(idx)
    twr_vals = [100.0 * ((1.18) ** (i / n)) * (1.0 + 0.015 * math.sin(i / 5.0)) for i in range(n)]
    twr = pd.Series(twr_vals, index=idx, dtype=float)

    investido = [10_000.0 + (1_000.0 * (i // 60)) for i in range(n)]
    mercado = [inv * (twr_vals[i] / 100.0) for i, inv in enumerate(investido)]
    pdf = pd.DataFrame(
        {
            "date": idx,
            "vlr_mercado": mercado,
            "vlr_investido": investido,
            "proventos": [0.0] * n,
        }
    )
    pdf["fluxo"] = pdf["vlr_investido"].diff().fillna(pdf["vlr_investido"].iloc[0]) - pdf["proventos"]
    return twr, pdf


def _make_bench(slope: float) -> pd.Series:
    idx = pd.bdate_range("2024-01-02", "2025-06-30")
    n = len(idx)
    return pd.Series(
        [100.0 * ((1.0 + slope) ** (i / n)) for i in range(n)],
        index=idx,
        dtype=float,
    )


PORTFOLIO_TWR, PORTFOLIO_DF = _make_portfolio_fixture()
FAKE_CDI = _make_bench(0.11)
FAKE_SELIC = _make_bench(0.1125)
FAKE_BOVA11 = _make_bench(0.08).loc["2024-03-15":]



@pytest.fixture(autouse=True)
def _api_locmem_cache(locmem_cache):
    """Testes da API nunca tocam o Redis real."""
    return locmem_cache


@pytest.fixture
def user_with_token(user):
    ut = UserToken(user=user)
    ut.set_token("tok-teste")
    ut.save()
    return user


@pytest.fixture
def user_with_token_client(user_with_token):
    client = Client()
    client.force_login(user_with_token)
    return client


@pytest.fixture
def portfolio_env(user_with_token_client, monkeypatch):
    monkeypatch.setattr(
        "apps.api.analysis.fetch_user_portfolio",
        lambda *a, **k: (PORTFOLIO_TWR.copy(), PORTFOLIO_DF.copy()),
    )

    def fake_get_series(name, s, e):
        src = {"CDI": FAKE_CDI, "SELIC": FAKE_SELIC, "BOVA11": FAKE_BOVA11}.get(name)
        if src is None:
            return None
        return src.loc[str(s) : str(e)].copy()

    def fake_get_many(names, s, e):
        out = {}
        for n in names:
            ser = fake_get_series(n, s, e)
            if ser is not None and not ser.empty:
                out[n] = ser
        return out

    monkeypatch.setattr(dataset.series, "get_series", fake_get_series)
    monkeypatch.setattr(dataset.series, "get_many", fake_get_many)
    return user_with_token_client
