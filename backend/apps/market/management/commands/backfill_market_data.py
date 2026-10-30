"""Bootstrap OFFLINE de core.MarketSeries a partir dos CSVs do repo legado.

Ferramenta de migração ÚNICA e descartável (não é rotina). Percorre o catálogo,
para cada série ingestível procura o CSV correspondente em --raw-dir / --static-dir,
faz o parse por família e grava via repository.upsert_points (idempotente).
Arquivos que não mapeiam para nenhuma série do catálogo são pulados com log.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from django.core.management.base import BaseCommand

from apps.market import catalog, repository
from apps.market.services.sources.base import RawPoint

_REPO_ROOT = Path(__file__).resolve().parents[5]


def _read_csv(path: Path, prefix: str) -> list[RawPoint]:
    if prefix == "bcb":
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        col = df.columns[0]
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        return [RawPoint(ts.date(), float(v)) for ts, v in s.items()]
    if prefix == "yf":
        df = pd.read_csv(path, skiprows=3, names=["Date", "Close"], parse_dates=["Date"], index_col="Date")
        s = pd.to_numeric(df["Close"], errors="coerce").dropna()
        return [RawPoint(ts.date(), float(v)) for ts, v in s.items()]
    if prefix == "b3":
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        s = pd.to_numeric(df["Close"], errors="coerce").dropna()
        return [RawPoint(ts.date(), float(v)) for ts, v in s.items()]
    if prefix == "td":
        df = pd.read_csv(path, sep=";", decimal=",", index_col=0, parse_dates=True)
        s = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
        return [RawPoint(ts.date(), float(v)) for ts, v in s.items()]
    raise ValueError(f"prefixo sem parser: {prefix}")


class Command(BaseCommand):
    help = "Semeia MarketSeries a partir dos CSVs do repo legado (bootstrap único)."

    def add_arguments(self, parser):
        parser.add_argument("--raw-dir", default=str(_REPO_ROOT / "data" / "raw"))
        parser.add_argument("--static-dir", default=str(_REPO_ROOT / "data" / "static"))
        parser.add_argument("--only", default="")
        parser.add_argument("--dry-run", action="store_true")

    def handle(self, *args, **opts):
        raw_dir = Path(opts["raw_dir"])
        static_dir = Path(opts["static_dir"])
        only = {k.strip() for k in opts["only"].split(",") if k.strip()}
        dry = opts["dry_run"]

        seen_files: set[str] = set()
        n_series = n_points = n_failed = 0

        for sdef in catalog.ingestible():
            if only and sdef.series_key not in only:
                continue
            fname = catalog.legacy_csv_name(sdef)
            if not fname:
                continue
            prefix = sdef.series_key.split(":", 1)[0]
            search = static_dir if prefix == "b3" else raw_dir
            path = search / fname
            if not path.exists() or path.stat().st_size == 0:
                self.stdout.write(f"  - sem CSV para {sdef.series_key} ({fname})")
                continue
            seen_files.add(fname)
            try:
                points = _read_csv(path, prefix)
            except Exception as exc:  # noqa: BLE001
                n_failed += 1
                self.stderr.write(f"  ! falha ao ler {fname}: {exc}")
                continue
            if not dry:
                repository.upsert_points(sdef.series_key, sdef.source, sdef.kind, points)
            n_series += 1
            n_points += len(points)
            self.stdout.write(f"  ok {sdef.series_key}: {len(points)} pontos")

        # arquivos órfãos (não mapeados)
        for d in (raw_dir, static_dir):
            if not d.exists():
                continue
            for p in sorted(d.glob("*.csv")):
                if p.name not in seen_files:
                    self.stdout.write(f"  pulado (não mapeado): {p.name}")

        verb = "simularia" if dry else "gravou"
        self.stdout.write(
            self.style.SUCCESS(
                f">> backfill {verb} {n_series} série(s), {n_points} pontos; {n_failed} falha(s)."
            )
        )
        if n_failed:
            sys.exit(1)
