from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def migration_assets_csv(settings):
    from apps.market import assets

    settings.MARKET_ASSETS_CSV = Path(__file__).resolve().parent / "fixtures" / "ativos_migration.csv"
    assets.load_universe.cache_clear()
    yield
    assets.load_universe.cache_clear()


_QUOTES = {"VWRA11.SA": [49.0, 50.0], "BITH11.SA": [20.0], "GDX.L BRL": [7.0]}


@pytest.fixture
def fake_series(monkeypatch, migration_assets_csv):
    """get_series falso: cotações fixas por nome; registra os nomes pedidos."""

    class Fake:
        calls: list[str] = []

        def __call__(self, name, start, end):
            self.calls.append(name)
            vals = _QUOTES.get(name)
            if vals is None:
                return None
            idx = pd.bdate_range(end=end, periods=len(vals))
            return pd.Series(vals, index=idx, dtype=float)

    fake = Fake()
    fake.calls = []
    monkeypatch.setattr("apps.migration_ir.prices.series.get_series", fake)
    return fake
