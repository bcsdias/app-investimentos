import pytest

from apps.market.synthetics import SyntheticSpec, parse_legacy_string


@pytest.mark.parametrize(
    "text,expected",
    [
        ("IPCA + 6%", SyntheticSpec("IPCA", "+", 6.0)),
        ("IPCA + 6.5%", SyntheticSpec("IPCA", "+", 6.5)),
        ("CDI + 2%", SyntheticSpec("CDI", "+", 2.0)),
        ("110% do CDI", SyntheticSpec("CDI", "%", 110.0)),
        ("95.5% do CDI", SyntheticSpec("CDI", "%", 95.5)),
    ],
)
def test_parse_legacy_string_known_forms(text, expected):
    assert parse_legacy_string(text) == expected


@pytest.mark.parametrize("text", ["", "IBOV", "IPCA menos 2%", "banana", "6% do SELIC"])
def test_parse_legacy_string_garbage_returns_none(text):
    assert parse_legacy_string(text) is None


def test_spec_is_frozen():
    with pytest.raises(Exception):
        SyntheticSpec("IPCA", "+", 6.0).valor = 7.0
