import pytest

from apps.market import catalog as cat
from apps.market.synthetics import SyntheticSpec


def test_resolve_known_and_unknown():
    cdi = cat.resolve("CDI")
    assert cdi.source == "BCB" and cdi.series_key == "bcb:12" and cdi.kind == "rate"
    with pytest.raises(KeyError):
        cat.resolve("Bananas")


def test_ipca_is_monthly_rate():
    ipca = cat.resolve("IPCA")
    assert ipca.kind == "rate" and ipca.periodicity == "monthly"


def test_ima_b_has_proxy_splice_derivation():
    ima = cat.resolve("IMA-B")
    assert isinstance(ima.derivation, cat.ProxySplice)
    assert ima.derivation.bcb_key == "bcb:12466"
    assert ima.derivation.etf_key.startswith("yf:")


def test_ingestible_excludes_pure_derivations_and_includes_selic():
    keys = {d.series_key for d in cat.ingestible()}
    assert "bcb:11" in keys           # SELIC
    assert "bcb:12" in keys           # CDI
    assert "ptax:USDBRL" in keys
    # séries puramente derivadas (Composite/Synthetic sem series_key) ficam de fora
    assert all(d.series_key for d in cat.ingestible())


def test_every_seriesdef_is_well_formed():
    for name in cat.all_names():
        d = cat.resolve(name)
        assert d.kind in ("rate", "level")
        assert d.periodicity in ("daily", "monthly")
        if d.derivation is None:
            assert d.series_key and d.source


def test_legacy_csv_name_mapping():
    assert cat.legacy_csv_name(cat.resolve("CDI")) == "BCB_12.csv"
    assert cat.legacy_csv_name(cat.resolve("S&P 500")) == "YF_GSPC.csv"
    assert cat.legacy_csv_name(cat.resolve("IFIX")) == "IFIX_all.csv"
    assert cat.legacy_csv_name(cat.resolve("USD/BRL")) is None


def test_default_benchmarks_are_resolvable():
    for name in cat.DEFAULT_BENCHMARKS:
        assert cat.resolve(name)
