import pytest

from apps.migration_ir.tests.fixtures import RESUMO
from apps.migration_ir.wallet import normalize_wallet


def test_normalize_maps_fx_fields_and_sorts_by_saldo():
    hs, warns = normalize_wallet(RESUMO)
    assert [h.ativo for h in hs] == ["BTC", "CDB-XP-2027", "PETR4", "HGLG11", "IMAB11"]
    petr = hs[2]
    assert (petr.price, petr.pm, petr.saldo, petr.saldo_source) == (38.10, 30.0, pytest.approx(11430.0), "market")
    cdb = hs[1]
    assert (cdb.price, cdb.saldo, cdb.saldo_source) == (None, 12000.0, "cost")
    assert warns == []


def test_normalize_coerces_strings_and_zero_price():
    hs, _ = normalize_wallet({"wallet": [
        {"ativo": "X", "classe": "ACAO", "quantidade": "10", "price_fx": "0", "pm_fx": None, "vlr_investido": "50"},
        {"ativo": "Z", "classe": "ACAO", "qtd": 0, "price_fx": 10, "vlr_investido": 0},
    ]})
    assert len(hs) == 1 and hs[0].qtd == 10.0 and hs[0].price is None and hs[0].saldo == 50.0


def test_normalize_aggregates_duplicate_ativo():
    hs, warns = normalize_wallet({"wallet": [
        {"ativo": "PETR4", "classe": "ACAO", "qtd": 100, "price_fx": None, "pm_fx": 20.0, "vlr_investido": 2000},
        {"ativo": "PETR4", "classe": "BDR", "qtd": 300, "price_fx": 40.0, "pm_fx": 40.0, "vlr_investido": 12000},
    ]})
    (h,) = hs
    assert (h.qtd, h.vlr_investido, h.price, h.classe) == (400, 14000, 40.0, "BDR")
    assert h.pm == pytest.approx(35.0)  # (100*20 + 300*40) / 400
    assert warns == ["mixed_class:PETR4"]


def test_normalize_without_wallet_key():
    assert normalize_wallet({}) == ([], [])
