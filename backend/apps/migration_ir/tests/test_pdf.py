import datetime as dt

from apps.migration_ir.pdf import render_plan_pdf
from apps.migration_ir.schemas import PlanRequest
from apps.migration_ir.tests.fixtures import PLAN_BODY


def test_pdf_has_dynamic_targets_ir_and_footer(plan_envelope):
    raw = render_plan_pdf(plan_envelope, PlanRequest(**PLAN_BODY),
                          generated_at=dt.datetime(2026, 10, 26, 9, 0), compress=False)
    assert raw.startswith(b"%PDF")
    for needle in (b"VWRA11", b"BITH11", b"IR estimado", b"Confira a apura", b"app-investimentos"):
        assert needle in raw
    assert b"Antigravity" not in raw


def test_pdf_survives_non_latin1_text(plan_envelope):
    plan_envelope["meta"]["warnings"].append("fx_approx:ÜB€R→")
    raw = render_plan_pdf(plan_envelope, PlanRequest(**PLAN_BODY), generated_at=dt.datetime(2026, 10, 26))
    assert raw.startswith(b"%PDF")


def test_month_blocks_never_split(monkeypatch):
    import apps.migration_ir.pdf as pdf_mod

    recorded = []
    real_month = pdf_mod._month

    def wrap(pdf, m, data):
        pages_added = []
        real_add_page = pdf.add_page

        def track_add_page(*args, **kwargs):
            res = real_add_page(*args, **kwargs)
            pages_added.append(pdf.page_no())
            return res

        pdf.add_page = track_add_page
        try:
            real_month(pdf, m, data)
        finally:
            pdf.add_page = real_add_page

        start_page = pages_added[0] if pages_added else pdf.page_no()
        end_page = pdf.page_no()
        recorded.append((start_page, end_page))

    monkeypatch.setattr(pdf_mod, "_month", wrap)

    months = []
    for m in range(1, 31):
        n_sales = 2 if m % 2 == 0 else 3
        n_buys = 3 if m % 2 == 0 else 4
        sales = [
            {"ativo": f"A{i}", "qtd": 10, "valor": 1000.0, "ir_estimado": 50.0}
            for i in range(1, n_sales + 1)
        ]
        purchases = [
            {"ticker": f"T{i}", "price": 100.0, "qtd_sugerida": 10, "valor": 1000.0,
             "saldo_final": 5000.0, "participacao_final": 0.25}
            for i in range(1, n_buys + 1)
        ]
        months.append({
            "month": m,
            "exempt_sales_total": 2000.0,
            "taxed_sales_total": 1000.0,
            "purchase_total": 3000.0,
            "sales": sales,
            "purchases": purchases,
            "ir_estimado_total": 150.0,
            "warnings": [],
        })

    envelope = {
        "data": {
            "initial_balances": [{"ticker": "IMAB11", "direct": 5000.0, "counted": [], "total": 5000.0}],
            "months": months,
            "totals": {"months": 30, "ir_estimado": 4500.0, "ir_not_estimated": []},
        },
        "meta": {"monthly_cap": 20000.0, "warnings": []},
    }

    raw = render_plan_pdf(envelope, PlanRequest(**PLAN_BODY), generated_at=dt.datetime(2026, 10, 26))
    assert raw.startswith(b"%PDF")
    assert len(recorded) == 30
    assert all(before == after for before, after in recorded)
    distinct_pages = {p for before, after in recorded for p in (before, after)}
    assert len(distinct_pages) > 1


def test_month_height_calculation():
    from apps.migration_ir.pdf import _month_height

    # Empty sales and purchases (falls back to 1 row)
    # 6 (title) + 9 (headers) + 4 (1 row) + 1 (gap) + 4 (ir) + 3 (spacing) = 27
    h_min = _month_height({"month": 1, "sales": [], "purchases": []})
    assert h_min == 27.0

    # 3 sales, 2 purchases -> max is 3 rows
    # 23 + 3 * 4 = 35.0
    h_3 = _month_height({
        "month": 1,
        "sales": [{}, {}, {}],
        "purchases": [{}, {}],
    })
    assert h_3 == 35.0

    # 1 sale, 5 purchases -> max is 5 rows
    # 23 + 5 * 4 = 43.0
    h_5 = _month_height({
        "month": 1,
        "sales": [{}],
        "purchases": [{}, {}, {}, {}, {}],
    })
    assert h_5 == 43.0

    # With warning: +4
    h_warn = _month_height({
        "month": 1,
        "sales": [],
        "purchases": [],
        "warnings": ["Warning 1"],
    })
    assert h_warn == 31.0


def test_initial_balances_repeats_header_on_page_break():
    initial_balances = [
        {"ticker": f"TICK{i}", "direct": 1000.0 * i, "counted": [{"ativo": f"OLD{i}", "saldo": 500.0}], "total": 1500.0 * i}
        for i in range(1, 61)
    ]
    envelope = {
        "data": {
            "initial_balances": initial_balances,
            "months": [],
            "totals": {"months": 0, "ir_estimado": 0.0, "ir_not_estimated": []},
        },
        "meta": {"monthly_cap": 20000.0, "warnings": []},
    }
    raw = render_plan_pdf(envelope, PlanRequest(**PLAN_BODY), generated_at=dt.datetime(2026, 10, 26), compress=False)
    assert raw.startswith(b"%PDF")
    assert raw.count(b"Ativo alvo") >= 2

