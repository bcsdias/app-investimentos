import datetime as dt

import pandas as pd

from parity.cases import BENCHMARKS, Case, Window, build_cases, build_windows, pick_classe

TODAY = dt.date(2025, 6, 30)


def _dates():
    days = pd.bdate_range("2023-03-01", "2025-06-27")
    both = list(days) + list(days)
    return pd.Series(both).sample(frac=1.0, random_state=7).tolist(), days


def test_windows_from_unsorted_duplicated_dates():
    shuffled, days = _dates()
    w1, w2, w3 = build_windows(shuffled, TODAY)
    assert (w1.id, w1.start, w1.end) == ("w1-completo", dt.date(2023, 3, 1), dt.date(2025, 6, 27))
    assert (w2.id, w2.start, w2.end) == ("w2-12m", dt.date(2024, 6, 27), dt.date(2025, 6, 27))
    assert w3.id == "w3-meio"
    assert w3.start == days[len(days) // 2].date()
    assert w3.end == TODAY


def _hist(rows):
    return pd.DataFrame(rows, columns=["date", "ativo", "classe"])


def test_pick_classe_most_dates_then_alphabetical():
    hist = _hist(
        [
            ("2024-01-01", "A", "Renda Fixa"),
            ("2024-01-02", "A", "Renda Fixa"),
            ("2024-01-02", "B", "Ações"),
            ("2024-01-03", "B", "Ações"),
            ("2024-01-03", "C", "FII"),
        ]
    )
    assert pick_classe(hist) == "Ações"  # empate 2×2 com Renda Fixa → alfabética
    more = pd.concat([hist, _hist([("2024-01-04", "A", "Renda Fixa")])])
    assert pick_classe(more) == "Renda Fixa"


def test_pick_classe_accepts_tipo_column_case_insensitive():
    hist = pd.DataFrame({"date": ["2024-01-01"], "Tipo": ["FII"]})
    assert pick_classe(hist) == "FII"


def test_pick_classe_without_column_returns_none():
    assert pick_classe(pd.DataFrame({"date": ["2024-01-01"], "ativo": ["A"]})) is None
    assert pick_classe(pd.DataFrame({"date": ["2024-01-01"], "classe": [None]})) is None


def test_build_cases_has_four_cases_with_classe_only_on_w1():
    shuffled, _ = _dates()
    hist = pd.DataFrame(
        {
            "date": shuffled,
            "classe": ["Ações", "Renda Fixa"] * (len(shuffled) // 2),
        }
    )
    cases = build_cases(hist, TODAY)
    assert [c.id for c in cases] == ["w1-completo", "w2-12m", "w3-meio", "w1-completo-classe"]
    assert [c.classe for c in cases[:3]] == [None, None, None]
    assert cases[3].classe in {"Ações", "Renda Fixa"}
    assert cases[3].window == cases[0].window
    assert all(c.benchmarks == BENCHMARKS for c in cases)


def test_api_payload_shape():
    case = Case("x", Window("x", dt.date(2024, 1, 2), dt.date(2024, 12, 31)), classe="FII")
    assert case.api_payload() == {
        "start": "2024-01-02",
        "end": "2024-12-31",
        "ativo": None,
        "classe": "FII",
        "benchmarks": ["CDI", "IPCA + 6%", "S&P 500 BRL", "IFIX"],
        "synthetics": [],
    }
