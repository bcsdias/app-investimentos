import pytest

from apps.migration_ir.tax import estimate_ir, estimate_profit, tax_treatment


@pytest.mark.parametrize("classe,expected", [
    ("ACAO", ("exempt_20k", None)), (" acao ", ("exempt_20k", None)),
    ("FII", ("taxed", 0.20)), ("ETF", ("taxed", 0.15)), ("BDR", ("taxed", 0.15)), ("BDR ETF", ("taxed", 0.15)),
    ("RF", ("not_estimated", None)), ("ETF RF", ("not_estimated", None)), ("CRIPTO", ("not_estimated", None)),
    ("STOCK", ("not_estimated", None)), ("ETF_GB", ("not_estimated", None)), ("QUALQUER", ("not_estimated", None)),
])
def test_tax_treatment_table(classe, expected):
    assert tax_treatment(classe) == expected


def test_profit_and_ir():
    assert estimate_profit(160.0, 150.0, 50) == 500.0
    assert estimate_ir(500.0, 0.20) == 100.0
    assert estimate_ir(-300.0, 0.15) == 0.0
    assert estimate_profit(10.0, None, 5) is None and estimate_ir(None, 0.15) is None
