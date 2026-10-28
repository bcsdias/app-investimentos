import calendar
import html
import os
from pathlib import Path
import re
import pandas as pd
import pytest

from engine.twr import calculate_twr

try:
    from engine import flows
except ImportError:
    flows = None

pytestmark = pytest.mark.integration

PT_MONTHS = {
    "jan": 1,
    "fev": 2,
    "mar": 3,
    "abr": 4,
    "mai": 5,
    "jun": 6,
    "jul": 7,
    "ago": 8,
    "set": 9,
    "out": 10,
    "nov": 11,
    "dez": 12,
}


def resolve_dlp_html_path() -> Path | None:
    candidates: list[Path] = []

    if "DLP_SHEET_DIR" in os.environ:
        candidates.append(Path(os.environ["DLP_SHEET_DIR"]))

    # Relativo ao arquivo de teste dentro do repositório
    repo_root = Path(__file__).resolve().parents[3]
    candidates.append(repo_root / "dlp_sheet")

    # Django settings (se configurado)
    try:
        from django.conf import settings
        if settings.configured or os.environ.get("DJANGO_SETTINGS_MODULE"):
            candidates.append(settings.BASE_DIR.parent / "dlp_sheet")
    except Exception:
        pass

    # Caminho canônico do workspace
    candidates.append(Path("/data/projetos/app-investimentos/dlp_sheet"))

    for c in candidates:
        target = c / "google_sheets_DLP_html" / "dashboard..html"
        if target.exists():
            return target
        target2 = c / "dashboard..html"
        if target2.exists():
            return target2

    return None


def _parse_num(val: str) -> float:
    if not val:
        return 0.0
    cleaned = val.strip().replace(".", "").replace(",", ".")
    if cleaned.endswith("%"):
        cleaned = cleaned[:-1]
    return float(cleaned)


def _parse_date(val: str) -> pd.Timestamp | None:
    m = re.match(r"^(\d{4})/([a-z]{3})\.?$", val.strip().lower())
    if not m:
        return None
    year = int(m.group(1))
    month_str = m.group(2)
    if month_str not in PT_MONTHS:
        return None
    month = PT_MONTHS[month_str]
    last_day = calendar.monthrange(year, month)[1]
    return pd.Timestamp(year=year, month=month, day=last_day)


def parse_dlp_dashboard_html(path: Path) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", text, re.DOTALL)
    header_idx = -1
    header_cells: list[str] = []
    for i, r in enumerate(rows):
        cells = [
            html.unescape(re.sub(r"<[^>]+>", "", c)).strip()
            for c in re.findall(r"<td[^>]*>(.*?)</td>", r, re.DOTALL)
        ]
        if "sum vlrInvest" in cells:
            header_idx = i
            header_cells = cells
            break

    if header_idx == -1:
        raise ValueError(f"Linha de cabeçalho com sum vlrInvest não encontrada em {path}")

    col_map = {
        "date": header_cells.index("date"),
        "vlr_investido": header_cells.index("sum vlrInvest"),
        "vlr_mercado": header_cells.index("sum vlrMercado"),
        "proventos": header_cells.index("sum proventos"),
        "lucro": header_cells.index("sum lucro_trades"),
        "fluxo_caixa": header_cells.index("fluxo_caixa"),
        "TWR % a.m": header_cells.index("TWR % a.m"),
        "TWR_acc": header_cells.index("TWR_acc"),
    }

    records: list[dict] = []
    for j in range(header_idx + 1, len(rows)):
        cells = [
            html.unescape(re.sub(r"<[^>]+>", "", c)).strip()
            for c in re.findall(r"<td[^>]*>(.*?)</td>", rows[j], re.DOTALL)
        ]
        if not cells or not cells[0]:
            continue
        dt = _parse_date(cells[col_map["date"]])
        if dt is None:
            break
        records.append({
            "date": dt,
            "vlr_investido": _parse_num(cells[col_map["vlr_investido"]]),
            "vlr_mercado": _parse_num(cells[col_map["vlr_mercado"]]),
            "proventos": _parse_num(cells[col_map["proventos"]]),
            "lucro": _parse_num(cells[col_map["lucro"]]),
            "fluxo_caixa": _parse_num(cells[col_map["fluxo_caixa"]]),
            "TWR % a.m": _parse_num(cells[col_map["TWR % a.m"]]),
            "TWR_acc": _parse_num(cells[col_map["TWR_acc"]]),
        })

    df = pd.DataFrame(records)
    df.index = df["date"]
    df.index.name = None
    return df


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    path = resolve_dlp_html_path()
    if path is None or not path.exists():
        pytest.skip("Export HTML da planilha DLP não encontrado")
    return parse_dlp_dashboard_html(path)


def test_dlp_monthly_flow_matches(frame):
    if flows is None:
        pytest.fail("engine.flows não existe nesta versão do engine")
    g = flows.group_portfolio(frame)
    assert flows.external_flows(g).values == pytest.approx(frame["fluxo_caixa"].values, abs=0.05)


def test_dlp_monthly_twr_matches(frame):
    ret = calculate_twr(frame).pct_change().iloc[1:]
    assert (ret - frame["TWR % a.m"].iloc[1:] / 100).abs().max() < 0.0001  # 0,01 pp


def test_dlp_accumulated_twr_matches(frame):
    assert (calculate_twr(frame).iloc[-1] - 1) * 100 == pytest.approx(frame["TWR_acc"].iloc[-1], abs=0.05)


# Janela jan/26–set/26 filtrada no dashboard da DLP em 2026-10-28: XTIR 16,72 % a.a.
WINDOW_START, WINDOW_END, WINDOW_DLP_XTIR = "2026-01-01", "2026-09-30", 0.1672


@pytest.fixture
def window(frame, monkeypatch):
    from engine import dataset

    if frame["date"].max() < pd.Timestamp(WINDOW_END):
        pytest.skip("Export DLP não cobre a janela de referência")
    hist = frame[["date", "vlr_investido", "vlr_mercado", "proventos", "lucro"]].reset_index(drop=True)
    monkeypatch.setattr(dataset.dlp, "buscar_historico", lambda *a, **k: hist.copy())
    return dataset.fetch_user_portfolio("t", start_date=WINDOW_START, end_date=WINDOW_END)


def test_dlp_window_twr_includes_first_month(frame, window):
    twr, _ = window
    in_window = frame[(frame["date"] >= WINDOW_START) & (frame["date"] <= WINDOW_END)]
    expected = (1 + in_window["TWR % a.m"] / 100).prod() - 1
    assert twr.iloc[-1] - 1 == pytest.approx(expected, abs=0.0005)


def test_dlp_window_xirr_matches_dashboard(window):
    from engine.compute import compute_irr_evolution

    _, pdf = window
    assert compute_irr_evolution(pdf).iloc[-1] / 100 == pytest.approx(WINDOW_DLP_XTIR, abs=0.0005)

