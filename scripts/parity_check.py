#!/usr/bin/env python
"""Harness de paridade da Fase 6: legado Streamlit (src/) × engine/API novos.

Uso (no host, com backend/.venv):
    export DLP_TOKEN=... PARITY_USER=... PARITY_PASSWORD=...   # nunca na linha de comando
    backend/.venv/bin/python scripts/parity_check.py --layers 1,2 --with-integration

Camada 1: os dois lados sobre o mesmo mercado congelado do Postgres (tolerância 1e-6).
Camada 2: legado ao vivo × API HTTP em PARITY_API_URL (tolerância 1e-4, INVESTIGAR).
Relatório em reports/parity/<timestamp>/ (ignorado pelo git). Código de saída:
0 = camada 1 OK; 1 = FALHA/DESVIO na camada 1; 2 = configuração ausente; 3 = erro.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
INTEGRATION_TEST = (
    "apps/api/tests/test_integration.py::test_real_dlp_in_memory_filter_matches_server_filter"
)


def _parse(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--layers", default="1,2", help="camadas a rodar, p.ex. 1 ou 1,2")
    p.add_argument("--out", default=str(REPO_ROOT / "reports" / "parity"))
    p.add_argument("--today", type=dt.date.fromisoformat, default=None, help="YYYY-MM-DD")
    p.add_argument("--with-integration", action="store_true")
    return p.parse_args(argv)


def _fail(msg: str) -> int:
    print(msg, file=sys.stderr)
    return 2


def _commit() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
    )
    return proc.stdout.strip() or "—"


def _run_integration(token: str) -> str:
    from parity.bootstrap import BACKEND_DIR

    proc = subprocess.run(
        [str(BACKEND_DIR / ".venv/bin/pytest"), "-m", "integration", INTEGRATION_TEST,
         "-q", "-p", "no:cacheprovider"],
        cwd=BACKEND_DIR,
        env=os.environ | {"DLP_TEST_TOKEN": token},
        capture_output=True,
        text=True,
    )
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
    return f"rc={proc.returncode}; {lines[-1] if lines else 'sem saída'}"


def _error_diff(note: str):
    from parity.compare import FALHA, SeriesDiff

    return SeriesDiff("caso", "-", FALHA, float("nan"), 0, None, 0, 0, note=note)


def main(argv=None) -> int:
    args = _parse(argv)
    try:
        layers = sorted({int(x) for x in args.layers.split(",") if x.strip()})
    except ValueError:
        return _fail("--layers deve ser uma lista como 1 ou 1,2")
    if not layers or not set(layers) <= {1, 2}:
        return _fail("--layers aceita só 1 e 2")

    token = os.environ.get("DLP_TOKEN")
    if not token:
        return _fail("DLP_TOKEN não definido no ambiente (exporte-o no shell; nunca por argumento).")
    user, password = os.environ.get("PARITY_USER"), os.environ.get("PARITY_PASSWORD")
    if 2 in layers and not (user and password):
        return _fail("Camada 2 exige PARITY_USER e PARITY_PASSWORD no ambiente.")

    sys.path.insert(0, str(SCRIPTS_DIR))
    from parity import bootstrap

    bootstrap.configure()

    from parity.baseline import save_baseline
    from parity.cases import build_cases
    from parity.frozen import MarketSnapshot
    from parity.legacy_side import freeze_history, run_layer1, run_layer2
    from parity.live import ApiClient, ApiError
    from parity.report import CaseResult, exit_code, write_details, write_summary

    today = args.today or dt.date.today()
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out) / timestamp

    print("Lendo mercado do Postgres e histórico da DLP...")
    snapshot = MarketSnapshot.from_repository(end=today)
    hist_full = freeze_history(token)
    cases = build_cases(hist_full, today)
    hists = {c.id: hist_full if c.classe is None else freeze_history(token, c.classe) for c in cases}
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[CaseResult] = []
    if 1 in layers:
        new_outputs = {}
        for case in cases:
            print(f"Camada 1: {case.id}")
            try:
                diffs, legacy_out, new_out = run_layer1(case, snapshot, hists[case.id])
            except Exception as exc:  # noqa: BLE001 — vira FALHA no relatório
                print(f"  erro: {type(exc).__name__}", file=sys.stderr)
                results.append(CaseResult(case, 1, [_error_diff(f"erro: {type(exc).__name__}")], []))
                continue
            results.append(CaseResult(case, 1, diffs, []))
            write_details(out_dir, case.id, 1, legacy_out, new_out)
            if new_out:
                new_outputs[case.id] = new_out
        save_baseline(out_dir, cases, snapshot, hists, new_outputs)

    if 2 in layers:
        client = ApiClient(os.environ.get("PARITY_API_URL", "http://127.0.0.1:8080"))
        login_error = None
        try:
            client.login(user, password)
        except ApiError as exc:
            login_error = f"login: {exc}"
        except Exception as exc:  # noqa: BLE001 — API fora do ar etc.
            login_error = f"login: {type(exc).__name__}"
        for case in cases:
            print(f"Camada 2: {case.id}")
            if login_error:
                results.append(CaseResult(case, 2, [], [login_error]))
                continue
            try:
                diffs, notes, legacy_out, new_out = run_layer2(case, token, client)
            except Exception as exc:  # noqa: BLE001 — camada 2 não derruba o relatório
                print(f"  erro: {type(exc).__name__}", file=sys.stderr)
                results.append(CaseResult(case, 2, [], [f"erro: {type(exc).__name__}"]))
                continue
            results.append(CaseResult(case, 2, diffs, notes))
            write_details(out_dir, case.id, 2, legacy_out, new_out)

    meta = {
        "timestamp": timestamp,
        "commit": _commit(),
        "cases": cases,
        "missing": snapshot.missing(),
        "integration": _run_integration(token) if args.with_integration else None,
        "layers": layers,
    }
    path = write_summary(out_dir, results, meta)
    counts = Counter(d.status for r in results for d in r.diffs)
    print(f"Relatório: {path}")
    print("Status: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return exit_code(results)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 — nunca repr de argumentos (token)
        print(f"erro: {type(exc).__name__}", file=sys.stderr)
        sys.exit(3)
