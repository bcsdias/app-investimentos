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
