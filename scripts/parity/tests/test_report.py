import datetime as dt
import os
import subprocess
from pathlib import Path

import pandas as pd

from parity.cases import Case, Window
from parity.compare import DESVIO, FALHA, INVESTIGAR, OK, REL_TOL_L1, SeriesDiff, compare_frames
from parity.report import CaseResult, exit_code, write_details, write_summary

PARITY_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PARITY_DIR.parents[1]

CASE = Case("w1-completo", Window("w1-completo", dt.date(2023, 3, 1), dt.date(2025, 6, 27)))


def _meta(**extra):
    return {
        "timestamp": "20261027-120000",
        "commit": "abc1234",
        "cases": [CASE],
        "missing": [],
        "integration": None,
        "layers": [1, 2],
    } | extra


def _diff(metric="twr", column="CDI", status=OK):
    return SeriesDiff(metric, column, status, 0.0, 0 if status == OK else 3, None if status == OK else "2024-01-02", 0, 0)


def test_summary_never_contains_raw_values(tmp_path):
    leg = pd.DataFrame({"valor": [123456.78, 0.1]}, index=["patrimonio", "cagr"])
    new = pd.DataFrame({"valor": [123999.00, 0.1]}, index=["patrimonio", "cagr"])
    diffs = compare_frames("kpis", leg, new, REL_TOL_L1, FALHA)
    assert diffs[0].status == FALHA
    path = write_summary(tmp_path, [CaseResult(CASE, 1, diffs, [])], _meta())
    text = path.read_text()
    assert path == tmp_path / "summary.md"
    assert "123456" not in text
    assert "123999" not in text
    assert "123.456" not in text
    assert "4.373e-03" in text  # razão relativa, não valor
    assert "patrimonio" in text


def test_summary_lists_not_covered_and_missing_snapshot_keys(tmp_path):
    text = write_summary(tmp_path, [], _meta(missing=["b3:IFIX"])).read_text()
    assert "Tesouro" in text
    assert "Carteiras compostas" in text
    assert "b3:IFIX" in text
    assert "2023-03-01" in text and "2025-06-27" in text


def test_summary_collapses_ok_rows_and_counts_statuses(tmp_path):
    diffs = [_diff("twr", "CDI"), _diff("twr", "IFIX"), _diff("drawdown", "CDI", INVESTIGAR)]
    results = [CaseResult(CASE, 2, diffs, ["legado: sem carteira"])]
    text = write_summary(tmp_path, results, _meta()).read_text()
    assert "twr: 2" in text
    assert "| drawdown | CDI | INVESTIGAR |" in text
    assert "legado: sem carteira" in text
    assert "INVESTIGAR: 1" in text


def test_summary_has_integration_line(tmp_path):
    text = write_summary(tmp_path, [], _meta(integration="rc=0; 1 passed in 4.10s")).read_text()
    assert "Teste integration (Fase 3c)" in text
    assert "rc=0; 1 passed in 4.10s" in text
    text = write_summary(tmp_path, [], _meta()).read_text()
    assert "não executado" in text


def test_write_details(tmp_path):
    df = pd.DataFrame({"A": [1.0]}, index=pd.to_datetime(["2024-01-02"]))
    write_details(tmp_path, "w1", 1, {"twr": df}, {"twr": df, "irr": df})
    base = tmp_path / "details" / "w1" / "L1"
    assert (base / "twr.legacy.csv").exists()
    assert (base / "twr.new.csv").exists()
    assert (base / "irr.new.csv").exists()
    assert not (base / "irr.legacy.csv").exists()


def test_exit_code_rules():
    assert exit_code([CaseResult(CASE, 2, [_diff(status=INVESTIGAR)], [])]) == 0
    assert exit_code([CaseResult(CASE, 1, [_diff(status=FALHA)], [])]) == 1
    assert exit_code([CaseResult(CASE, 1, [_diff(status=DESVIO)], [])]) == 1
    assert exit_code([CaseResult(CASE, 1, [_diff()], [])]) == 0
    assert exit_code([CaseResult(CASE, 2, [_diff(status=FALHA)], [])]) == 0


def test_cli_without_token_exits_2(tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in ("DLP_TOKEN", "PARITY_USER", "PARITY_PASSWORD")}
    out = tmp_path / "out"
    proc = subprocess.run(
        [str(REPO_ROOT / "backend/.venv/bin/python"), str(REPO_ROOT / "scripts/parity_check.py"),
         "--out", str(out)],
        env=env, capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 2, proc.stderr[-2000:]
    assert "DLP_TOKEN" in proc.stderr
    assert not out.exists()


def test_cli_layer2_without_credentials_exits_2(tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in ("PARITY_USER", "PARITY_PASSWORD")}
    env["DLP_TOKEN"] = "tok-falso"
    out = tmp_path / "out"
    proc = subprocess.run(
        [str(REPO_ROOT / "backend/.venv/bin/python"), str(REPO_ROOT / "scripts/parity_check.py"),
         "--out", str(out), "--layers", "1,2"],
        env=env, capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 2
    assert "PARITY_USER" in proc.stderr
    assert "tok-falso" not in proc.stderr + proc.stdout
    assert not out.exists()
