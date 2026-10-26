from apps.market import assets
from apps.market.catalog import SeriesDef, ToBRL


def test_yahoo_ticker_mapping(assets_csv):
    u = assets.load_universe()
    assert u["PETR4.SA"].mercado == "BR"
    assert u["AAPL"].mercado == "INTL"
    assert "GDX.L" in u and "GDX" in u          # mesma sigla, duas bolsas
    assert "USML.L" in u                         # LSE: -> .L


def test_excluded_classes(assets_csv):
    u = assets.load_universe()
    assert not any(a.classe in {"CRIPTO", "INDICE", "OURO"} for a in u.values())


def test_bom_does_not_break_header(assets_csv):
    assert assets.load_universe()["PETR4.SA"].sigla == "PETR4"


def test_resolve_raw_and_brl(assets_csv):
    assert assets.resolve("PETR4.SA") == SeriesDef("PETR4.SA", "YF", "yf:PETR4.SA", "level")
    assert assets.resolve("AAPL BRL").derivation == ToBRL("AAPL")
    assert assets.resolve("PETR4.SA BRL") is None   # BR não tem variante BRL
    assert assets.resolve("PETR4") is None          # sigla pura não é identificador
    assert assets.resolve("XYZ") is None


def test_search_ranking_and_limit(assets_csv):
    assert [a.yf_ticker for a in assets.search("gdx")][:2] in (["GDX.L", "GDX"], ["GDX", "GDX.L"])
    assert assets.search("logistica")[0].yf_ticker == "HGLG11.SA"
    assert len(assets.search("a", limit=2)) <= 2


def test_real_file_loads():
    assets.load_universe.cache_clear()
    assert len(assets.load_universe()) > 5000


def test_collision_prefers_higher_ranked_exchange(assets_csv):
    meta = assets.load_universe()["META"]
    assert meta.nome == "META PLATFORMS INC"
    assert meta.classe == "STOCK"


def test_collision_tie_keeps_first_row(assets_csv):
    assert assets.load_universe()["TIE"].nome == "TIE FIRST"


def test_real_file_meta_is_meta_platforms():
    assets.load_universe.cache_clear()
    try:
        assert assets.load_universe()["META"].nome == "META PLATFORMS INC."
    finally:
        assets.load_universe.cache_clear()
