from parity.cases import build_cases
from parity.perf import time_case
from parity.tests.synthetic import TODAY, make_hist, make_snapshot


def test_time_case_reports_positive_medians():
    hist = make_hist()
    snap = make_snapshot()
    case = build_cases(hist, TODAY)[0]

    res = time_case(case, snap, hist, repeat=2)

    expected_keys = {"fetch_user_portfolio", "build_dataset", "compute_irr_evolution", "total"}
    assert set(res.keys()) == expected_keys
    for k in expected_keys:
        assert isinstance(res[k], float)
        assert res[k] > 0
    assert res["total"] >= res["fetch_user_portfolio"]
