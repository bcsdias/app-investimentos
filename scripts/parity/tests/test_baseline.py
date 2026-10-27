import json
import subprocess
from pathlib import Path

import pandas as pd

from parity.baseline import load_baseline, save_baseline
from parity.cases import build_cases
from parity.compare import OK, compare_frames
from parity.engine_side import METRICS, run_engine
from parity.tests.synthetic import TODAY, make_hist, make_snapshot

PARITY_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PARITY_DIR.parents[1]
BACKEND_DIR = REPO_ROOT / "backend"


def _materials():
    hist = make_hist()
    snap = make_snapshot()
    cases = build_cases(hist, TODAY)
    hists = {
        c.id: hist if c.classe is None else hist[hist["classe"] == c.classe].reset_index(drop=True)
        for c in cases
    }
    outputs = {c.id: run_engine(c, snap, hists[c.id]) for c in cases}
    return cases, snap, hists, outputs


def test_baseline_round_trip(tmp_path):
    cases, snap, hists, outputs = _materials()
    path = save_baseline(tmp_path, cases, snap, hists, outputs)
    assert path == tmp_path / "baseline"
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["version"] == 1
    assert manifest["cases"][0]["start"] == cases[0].window.start.isoformat()

    b = load_baseline(path)
    assert b.cases == cases
    assert set(b.snapshot.series) == set(snap.series)
    for key, s in snap.series.items():
        pd.testing.assert_series_equal(b.snapshot.series[key], s)
    for cid, h in hists.items():
        pd.testing.assert_frame_equal(b.hists[cid], h)
    for cid, out in outputs.items():
        assert set(b.outputs[cid]) == set(METRICS)
        for metric, df in out.items():
            pd.testing.assert_frame_equal(b.outputs[cid][metric], df)


def test_rerun_engine_from_baseline_reproduces_outputs(tmp_path):
    cases, snap, hists, outputs = _materials()
    b = load_baseline(save_baseline(tmp_path, cases, snap, hists, outputs))
    for case in b.cases:
        again = run_engine(case, b.snapshot, b.hists[case.id])
        for metric in METRICS:
            diffs = compare_frames(
                metric, b.outputs[case.id][metric], again[metric], rel_tol=0.0,
                fail_status="FALHA", abs_floor=0.0,
            )
            assert all(d.status == OK for d in diffs), (case.id, metric, diffs)


def test_src_free_modules_import_without_src():
    code = (
        "import sys, os\n"
        "sys.modules['src'] = None\n"
        f"sys.path[:0] = [{str(BACKEND_DIR)!r}, {str(PARITY_DIR.parent)!r}]\n"
        "os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')\n"
        "import django; django.setup()\n"
        "import parity.compare, parity.cases, parity.frozen, parity.kpis, "
        "parity.engine_side, parity.baseline\n"
    )
    proc = subprocess.run(
        [str(BACKEND_DIR / ".venv/bin/python"), "-c", code],
        cwd=str(PARITY_DIR.parent),  # scripts/: fora de REPO_ROOT, src não é importável por cwd
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
