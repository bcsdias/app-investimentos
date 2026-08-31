# Fase 2.5 — Refatorar o Engine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **PRÉ-REQUISITO:** a **Fase 2** (`docs/superpowers/plans/2026-08-31-fase-2-servicos-de-mercado.md`) precisa estar **fechada e revisada**. Este plano consome `apps.market.series`, `apps.market.transforms` e `apps.market.synthetics`.

**Goal:** Copiar o engine puro (`twr`/`irr`/`metrics`) sem alteração para `backend/engine/`, extrair as 7 funções `compute_*` de dentro dos `plot_*` de `financial_report.py`, adaptar `fetch_user_portfolio` / `build_dataset` para consumir `apps.market.series` (não mais `processar_benchmarks`), preservar os `plot_*` Matplotlib num `report_legacy.py` só-CLI, e corrigir o bug do `self.logger`.

**Architecture:** `backend/engine/` é pacote Python puro, **irmão de `apps/`** (não é Django app — sem modelos). `twr.py`/`irr.py`/`metrics.py` são cópia literal de `src/engine/` (preserva a validade da paridade da Fase 6). `compute.py` são funções puras (pandas dentro, pandas/dict fora; zero matplotlib, zero I/O) — a serialização para o formato do ECharts é da Fase 3. `dataset.py` monta o `df_combined` chamando `series.get_many`. `report_legacy.py` chama `compute.*` + desenha com Matplotlib, só para export CLI — a API nunca o importa.

**Tech Stack:** `pandas~=2.2`, `numpy~=2.1` (já da Fase 2), `matplotlib~=3.9` (novo — só `report_legacy.py`), `pytest-django`.

**Spec:** `docs/superpowers/specs/2026-08-31-fase-2-e-2-5-servicos-e-engine-design.md`

## Global Constraints

Todo task herda implicitamente esta seção.

- **Branch:** `migracao-django`. Não commitar em `main`/`dev`.
- **Python:** sempre `.venv/bin/python` e `.venv/bin/pytest` a partir de `backend/`.
- **`backend/engine/` é pacote puro**, irmão de `apps/` — **não** entra em `INSTALLED_APPS`, **não** tem `models.py`. Importável como `import engine.twr` (o `pythonpath = .` do `pytest.ini` e o `manage.py` rodando de `backend/` já resolvem).
- **Cópia literal de `twr.py`/`irr.py`/`metrics.py`:** o código de cálculo é copiado **sem alteração de lógica**. Única exceção consciente (spec §5.4): `pandas.pct_change()` ganha `fill_method=None` explícito em `metrics.py` — e isso é registrado no ADR como pendência a espelhar no `src/` legado antes da paridade da Fase 6. Nada mais muda.
- **Direção das dependências:** `engine.twr/irr/metrics/compute` são puros (não importam `apps`). `engine.dataset` importa `apps.market.series`, `apps.market.transforms`, `apps.market.synthetics`, `engine.twr`. `engine.report_legacy` importa `engine.compute`, `engine.dataset`, `matplotlib`. **`apps/` nunca importa `engine`** nesta fase (a API da Fase 3 é quem liga os dois).
- **A API não importa Matplotlib.** `import matplotlib` vive **só** em `engine/report_legacy.py`. `grep -rn "matplotlib" backend/apps backend/config backend/engine/compute.py backend/engine/dataset.py` não retorna nada.
- **Bug `self.logger`:** `financial_report.build_dataset` (linha 191) chama `self.logger.warning(...)` sem `self.logger` existir → quebra no caminho de dados vazios. Em `dataset.py` vira `logger = logging.getLogger(__name__)` no módulo, com teste de regressão.
- **Sem `self.` mutável:** `dataset.py` são funções que devolvem um `DatasetResult` (dataclass), não uma classe que muta `self.df_combined`.
- **`pct_change` sempre `fill_method=None`** em qualquer código novo (`compute.py`, `dataset.py`).
- **Testes portados passam SEM alteração de lógica:** `test_twr.py`, `test_irr.py`, `test_metrics.py` só trocam o import (`src.engine` → `engine`). Se um assert de `test_metrics.py` precisar mudar por causa do `fill_method`, documentar o porquê no commit.
- **Commits:** Conventional Commits + Gitmoji + imperativo em português. **Nunca** `Co-Authored-By`. Um commit por task.
- **TDD obrigatório:** escrever o teste, **rodar e ver falhar**, só então implementar, rodar e ver passar.

---

## Estrutura de arquivos

Tudo em `backend/`. Criar salvo indicação `MODIFICAR`.

```
engine/
├── __init__.py
├── twr.py                  # cópia literal de src/engine/twr.py
├── irr.py                  # cópia literal de src/engine/irr.py
├── metrics.py              # cópia literal + pct_change(fill_method=None)  (spec §5.4)
├── compute.py              # NOVO — 7 funções puras extraídas dos plot_*
├── dataset.py              # NOVO — fetch_user_portfolio + build_dataset (consome apps.market.series)
├── report_legacy.py        # NOVO — os plot_* Matplotlib, só-CLI
└── tests/
    ├── __init__.py
    ├── test_twr.py         # portado de tests/test_twr.py (troca de import)
    ├── test_irr.py         # portado
    ├── test_metrics.py     # portado
    ├── test_compute.py     # NOVO
    └── test_dataset.py     # NOVO

pytest.ini                  # MODIFICAR — testpaths = apps engine
requirements.txt            # MODIFICAR — matplotlib~=3.9
docs/adr/0012-engine-copia-literal-fase-2-5.md   # NOVO
docs/adr/README.md          # MODIFICAR — linha do 0012
docs/consolidacao_arquitetura_e_migracao.md      # MODIFICAR — §2.0, §2.2, §2.3, checklist §9
```

> Nota sobre o número do ADR: a Fase 2 já reserva o **0011** (`0011-marketseries-valor-cru-kind.md`). O ADR do engine é o **0012** (o spec §5.7 dizia "0011" antes da Fase 2 também ganhar um ADR).

---

## Ferramentas recomendadas nesta fase

Ver `docs/ferramentas-de-desenvolvimento.md` §"Fase 2 / 2.5":

- **`pr-review-toolkit` → `pr-test-analyzer`** — cobertura dos testes das `compute_*` (Task 5).
- **`superpowers:systematic-debugging`** — qualquer divergência numérica ao portar/extrair (TWR/TIR/Sharpe/Shadow).
- **`context7`** — API atual de `matplotlib` 3.9 (só se necessário no `report_legacy.py`).
- **`pyright-lsp`** — contínuo.
- **`code-review`** + **`code-simplifier`** — ao fechar a Task 5, sobre o diff de `backend/engine/`.

---

## Task 1: Cópia literal do engine puro + testes portados

**Files:**
- Create: `backend/engine/__init__.py`
- Create: `backend/engine/twr.py` (cópia de `src/engine/twr.py`)
- Create: `backend/engine/irr.py` (cópia de `src/engine/irr.py`)
- Create: `backend/engine/metrics.py` (cópia de `src/engine/metrics.py` + `fill_method=None`)
- Create: `backend/engine/tests/__init__.py`
- Create: `backend/engine/tests/test_twr.py` (portado)
- Create: `backend/engine/tests/test_irr.py` (portado)
- Create: `backend/engine/tests/test_metrics.py` (portado)
- Modify: `backend/pytest.ini`

**Interfaces:**
- Consumes: `src/engine/{twr,irr,metrics}.py` (fonte da cópia); `tests/test_{twr,irr,metrics}.py` (fonte dos testes portados).
- Produces:
  - `engine.twr.calculate_twr(df, start_date=None, end_date=None) -> pd.Series` — idêntico ao legado.
  - `engine.irr.calculate_xirr(cash_flows, dates) -> float | None` — idêntico.
  - `engine.metrics.calculate_drawdown(df) -> pd.DataFrame`, `calculate_rolling_volatility(df, window=252) -> pd.DataFrame`, `calculate_rolling_sharpe(df, rf_series=None, rf_constant=0.10, window=252) -> pd.DataFrame` — idênticos, exceto `pct_change(fill_method=None)`.

- [ ] **Step 1: Copiar os três módulos**

Run:
```bash
cd /data/projetos/app-investimentos
mkdir -p backend/engine/tests
touch backend/engine/__init__.py backend/engine/tests/__init__.py
cp src/engine/twr.py backend/engine/twr.py
cp src/engine/irr.py backend/engine/irr.py
cp src/engine/metrics.py backend/engine/metrics.py
```

- [ ] **Step 2: Aplicar `fill_method=None` em `engine/metrics.py`** (única exceção — spec §5.4)

Em `backend/engine/metrics.py`, trocar as duas ocorrências de `df.pct_change()` por `df.pct_change(fill_method=None)`:
- em `calculate_rolling_volatility`: `daily_ret = df.pct_change(fill_method=None)`
- em `calculate_rolling_sharpe`: `daily_ret = df.pct_change(fill_method=None)` e `selic_daily = rf_series.pct_change(fill_method=None).fillna(0)`

Adicionar um comentário no topo do arquivo:
```python
# NOTA (Fase 2.5, spec §5.4): pct_change usa fill_method=None explícito para casar
# com o comportamento do pandas 2.2 e não divergir na paridade da Fase 6. Este é o
# ÚNICO desvio da cópia literal de src/engine/metrics.py. Ver ADR 0012.
```

- [ ] **Step 3: Portar os três testes (trocar só o import)**

Run:
```bash
cd /data/projetos/app-investimentos
sed 's/from src\.engine\./from engine./' tests/test_twr.py     > backend/engine/tests/test_twr.py
sed 's/from src\.engine\./from engine./' tests/test_irr.py     > backend/engine/tests/test_irr.py
sed 's/from src\.engine\./from engine./' tests/test_metrics.py > backend/engine/tests/test_metrics.py
grep -n "^from engine\." backend/engine/tests/test_*.py
```
Expected: cada arquivo tem `from engine.twr import ...` / `from engine.irr import ...` / `from engine.metrics import ...`.

- [ ] **Step 4: Apontar o `pytest.ini` para `engine`**

Em `backend/pytest.ini`, trocar `testpaths = apps` por:
```ini
testpaths = apps engine
```

- [ ] **Step 5: Rodar os testes portados e ver passar**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/pytest engine -q
```
Expected: PASSA — `test_twr.py` (5), `test_irr.py` (4), `test_metrics.py` (3). Os testes de `metrics` usam séries sintéticas de retorno constante (sem NaN a preencher), então o `fill_method=None` não muda os asserts. Se **algum** assert de `test_metrics.py` falhar, aplicar `superpowers:systematic-debugging`, entender a diferença numérica, e **documentar no commit** por que o assert mudou (não "consertar" o número sem entender).

- [ ] **Step 6: Rodar a suíte inteira — nada quebrou**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/pytest -q
```
Expected: PASSA — Fases 0+1 (15) + Fase 2 (~60) + engine portado (12).

- [ ] **Step 7: Commit**

```bash
cd /data/projetos/app-investimentos
git add backend/engine/__init__.py backend/engine/twr.py backend/engine/irr.py backend/engine/metrics.py \
        backend/engine/tests/__init__.py backend/engine/tests/test_twr.py backend/engine/tests/test_irr.py \
        backend/engine/tests/test_metrics.py backend/pytest.ini
git commit -m "✨ feat(engine): cópia literal de twr/irr/metrics + testes portados

backend/engine/ nasce como pacote puro (irmão de apps/). twr.py e irr.py são
cópia literal de src/engine/. metrics.py idem, com o ÚNICO desvio consciente:
pct_change(fill_method=None) (spec §5.4, ADR 0012). Os 3 testes portados só
trocam o import (src.engine -> engine) e passam sem alteração. pytest.ini:
testpaths = apps engine."
```

---

## Task 2: `engine/compute.py` — extrair cálculo dos `plot_*`

**Files:**
- Create: `backend/engine/compute.py`
- Create: `backend/engine/tests/test_compute.py`

**Interfaces:**
- Consumes: `engine.metrics.calculate_drawdown` / `calculate_rolling_volatility` / `calculate_rolling_sharpe` (Task 1); `engine.irr.calculate_xirr` (Task 1); `pandas`, `numpy`.
- Produces (todas puras — sem I/O, sem matplotlib):
  - `compute.compute_twr_evolution(df_combined: pd.DataFrame) -> dict` — `{"series": DataFrame (colunas ordenadas pelo último valor, desc), "final_returns": dict[str, float]}` (`final_returns[col] = df[col].iloc[-1] - 100`).
  - `compute.compute_drawdown(df_combined: pd.DataFrame) -> pd.DataFrame` — `calculate_drawdown(df_combined)`.
  - `compute.compute_risk_return(df_combined: pd.DataFrame) -> pd.DataFrame` — colunas `Volatilidade`, `Retorno (CAGR)`, `Sharpe`; `pct_change(fill_method=None)`; `Sharpe = (cagr - 0.10) / volatility`.
  - `compute.compute_rolling_metrics(df_combined: pd.DataFrame, rf_series: pd.Series | None, window: int = 252) -> dict` — `{"volatility": DataFrame, "sharpe": DataFrame}`.
  - `compute.compute_irr_evolution(portfolio_df: pd.DataFrame) -> pd.Series` — caminhada XIRR mensal (`* 100`); `Series` vazia se não convergir nada.
  - `compute.compute_shadow(portfolio_df: pd.DataFrame, df_combined: pd.DataFrame) -> pd.DataFrame` — colunas: `Carteira Real`, `Total Investido`, e uma por benchmark.
  - `compute.compute_yearly_summary(df_combined: pd.DataFrame) -> pd.DataFrame` — retorno ano a ano (colunas = anos) + coluna `Total Acum.` (valores numéricos, **não** formatados).

- [ ] **Step 1: Escrever os testes (devem falhar)**

`backend/engine/tests/test_compute.py`:
```python
import numpy as np
import pandas as pd
import pytest

from engine import compute


def _combined():
    idx = pd.bdate_range("2022-01-03", periods=520)  # ~2 anos úteis
    carteira = pd.Series(np.linspace(100, 130, len(idx)), index=idx, name="Carteira")
    cdi = pd.Series(np.linspace(100, 120, len(idx)), index=idx, name="CDI")
    return pd.concat([carteira, cdi], axis=1)


def test_twr_evolution_sorts_and_reports_final_returns():
    df = _combined()
    out = compute.compute_twr_evolution(df)
    assert list(out["series"].columns)[0] == "Carteira"  # maior último valor primeiro
    assert round(out["final_returns"]["Carteira"], 1) == 30.0
    assert round(out["final_returns"]["CDI"], 1) == 20.0


def test_drawdown_is_nonpositive_and_starts_at_zero():
    df = _combined()
    dd = compute.compute_drawdown(df)
    assert (dd.fillna(0) <= 1e-9).all().all()
    assert abs(dd["CDI"].iloc[0]) < 1e-9


def test_risk_return_columns_and_sharpe_formula():
    df = _combined()
    rr = compute.compute_risk_return(df)
    assert list(rr.columns) == ["Volatilidade", "Retorno (CAGR)", "Sharpe"]
    row = rr.loc["Carteira"]
    np.testing.assert_allclose(
        row["Sharpe"], (row["Retorno (CAGR)"] - 0.10) / row["Volatilidade"], rtol=1e-9
    )


def test_rolling_metrics_returns_both_frames():
    df = _combined()
    out = compute.compute_rolling_metrics(df, rf_series=None, window=252)
    assert set(out) == {"volatility", "sharpe"}
    assert isinstance(out["volatility"], pd.DataFrame)
    assert isinstance(out["sharpe"], pd.DataFrame)


def test_irr_evolution_positive_for_growing_portfolio():
    dates = pd.to_datetime(["2020-01-01", "2020-07-01", "2021-01-01", "2021-07-01"])
    pdf = pd.DataFrame(
        {
            "date": dates,
            "vlr_mercado": [1000.0, 1600.0, 2200.0, 2900.0],
            "vlr_investido": [1000.0, 1500.0, 2000.0, 2500.0],
            "proventos": [0.0, 0.0, 0.0, 0.0],
        }
    )
    pdf["fluxo"] = pdf["vlr_investido"].diff().fillna(pdf["vlr_investido"].iloc[0]) - pdf["proventos"]
    s = compute.compute_irr_evolution(pdf)
    assert isinstance(s, pd.Series)
    assert not s.empty
    assert s.iloc[-1] > 0


def test_shadow_has_real_and_invested_columns():
    df = _combined()
    dates = df.index[:: len(df) // 4][:4]
    pdf = pd.DataFrame(
        {
            "date": dates,
            "vlr_mercado": [1000.0, 1200.0, 1500.0, 1800.0],
            "vlr_investido": [1000.0, 1100.0, 1300.0, 1500.0],
            "proventos": [0.0, 0.0, 0.0, 0.0],
        }
    )
    pdf["fluxo"] = pdf["vlr_investido"].diff().fillna(pdf["vlr_investido"].iloc[0]) - pdf["proventos"]
    sw = compute.compute_shadow(pdf, df)
    assert "Carteira Real" in sw.columns
    assert "Total Investido" in sw.columns
    assert "CDI" in sw.columns


def test_yearly_summary_has_years_and_total():
    df = _combined()
    ys = compute.compute_yearly_summary(df)
    assert "Total Acum." in ys.columns
    assert any(isinstance(c, int) or str(c).isdigit() for c in ys.columns)
    assert pd.api.types.is_numeric_dtype(ys["Total Acum."])
```

- [ ] **Step 2: Rodar e ver falhar**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/pytest engine/tests/test_compute.py -q
```
Expected: FALHA com `ModuleNotFoundError: No module named 'engine.compute'`.

- [ ] **Step 3: Implementar `compute.py`**

`backend/engine/compute.py`:
```python
"""Funções de cálculo PURAS extraídas dos plot_* de src/engine/financial_report.py.

Cada função recebe um DataFrame/Series já pronto e devolve DataFrame/Series/dict
JSON-serializável. Zero matplotlib, zero I/O. A serialização para o formato do
ECharts é da Fase 3 (apps/api/serializers.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.irr import calculate_xirr
from engine.metrics import (
    calculate_drawdown,
    calculate_rolling_sharpe,
    calculate_rolling_volatility,
)


def compute_twr_evolution(df_combined: pd.DataFrame) -> dict:
    if df_combined.empty:
        return {"series": df_combined, "final_returns": {}}
    order = df_combined.iloc[-1].sort_values(ascending=False).index
    series = df_combined[order]
    final_returns = {col: float(series[col].iloc[-1] - 100) for col in series.columns}
    return {"series": series, "final_returns": final_returns}


def compute_drawdown(df_combined: pd.DataFrame) -> pd.DataFrame:
    return calculate_drawdown(df_combined)


def compute_risk_return(df_combined: pd.DataFrame) -> pd.DataFrame:
    if df_combined.empty:
        return pd.DataFrame(columns=["Volatilidade", "Retorno (CAGR)", "Sharpe"])
    daily_ret = df_combined.pct_change(fill_method=None)
    volatility = daily_ret.std() * np.sqrt(252)
    days = (df_combined.index[-1] - df_combined.index[0]).days
    total_ret = df_combined.iloc[-1] / df_combined.iloc[0]
    cagr = (total_ret ** (365.25 / days)) - 1 if days > 0 else total_ret * 0
    return pd.DataFrame(
        {
            "Volatilidade": volatility,
            "Retorno (CAGR)": cagr,
            "Sharpe": (cagr - 0.10) / volatility,
        }
    ).dropna()


def compute_rolling_metrics(
    df_combined: pd.DataFrame, rf_series: pd.Series | None, window: int = 252
) -> dict:
    return {
        "volatility": calculate_rolling_volatility(df_combined, window),
        "sharpe": calculate_rolling_sharpe(
            df_combined, rf_series, rf_constant=0.10, window=window
        ),
    }


def compute_irr_evolution(portfolio_df: pd.DataFrame) -> pd.Series:
    if portfolio_df is None or portfolio_df.empty:
        return pd.Series(dtype=float)
    df = portfolio_df.sort_values("date")
    dates_to_calc = df.set_index("date").resample("ME").last().index
    if df["date"].iloc[-1] not in dates_to_calc:
        dates_to_calc = dates_to_calc.union([df["date"].iloc[-1]])
    dates_to_calc = dates_to_calc[dates_to_calc >= df["date"].iloc[0]]

    all_dates = df["date"].values
    all_flows = -1 * df["fluxo"].values
    all_markets = df["vlr_mercado"].values

    irr_history, valid_dates = [], []
    for target in dates_to_calc:
        mask = all_dates <= target
        if not np.any(mask):
            continue
        last_idx = np.where(mask)[0][-1]
        if all_markets[last_idx] == 0 and np.sum(np.abs(all_flows[mask])) == 0:
            continue
        calc_flows = np.append(all_flows[mask], all_markets[last_idx])
        calc_dates = np.append(all_dates[mask], all_dates[last_idx])
        try:
            res = calculate_xirr(calc_flows, pd.to_datetime(calc_dates))
        except Exception:  # noqa: BLE001
            res = None
        if res is not None and -0.99 < res < 10.0:
            irr_history.append(res)
            valid_dates.append(target)

    if not irr_history:
        return pd.Series(dtype=float)
    return pd.Series(irr_history, index=valid_dates) * 100


def compute_shadow(portfolio_df: pd.DataFrame, df_combined: pd.DataFrame) -> pd.DataFrame:
    if df_combined.empty or portfolio_df is None or portfolio_df.empty:
        return pd.DataFrame()

    flows = portfolio_df.set_index("date")[["fluxo", "vlr_mercado", "vlr_investido"]]
    flows.index = pd.to_datetime(flows.index)
    prices = df_combined.copy()
    idx = prices.index

    flows_aligned = flows["fluxo"].groupby(flows.index).sum().reindex(idx, fill_value=0.0)
    sw = pd.DataFrame(index=idx)
    sw["Carteira Real"] = flows["vlr_mercado"].reindex(idx).ffill()
    sw["Total Investido"] = flows_aligned.cumsum() + (
        flows["vlr_investido"].iloc[0] if not flows.empty else 0
    )

    initial_balance = flows["vlr_mercado"].iloc[0] if not flows.empty else 0
    for col in prices.columns:
        if col == "Carteira":
            continue
        price = prices[col]
        shares_flow = flows_aligned.div(price).fillna(0)
        initial_shares = initial_balance / price.iloc[0] if price.iloc[0] != 0 else 0
        sw[col] = (shares_flow.cumsum() + initial_shares) * price
    return sw


def compute_yearly_summary(df_combined: pd.DataFrame) -> pd.DataFrame:
    if df_combined.empty:
        return pd.DataFrame()
    df = df_combined
    total_ret = (df.iloc[-1] / df.iloc[0]) - 1
    yearly = df.resample("YE").last()
    yearly_ret = yearly.pct_change(fill_method=None)
    yearly_ret.iloc[0] = (yearly.iloc[0] / df.iloc[0]) - 1
    summary = yearly_ret.T
    summary.columns = [c.year for c in summary.columns]
    summary["Total Acum."] = total_ret
    return summary
```

- [ ] **Step 4: Rodar e ver passar**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/pytest engine/tests/test_compute.py -q
```
Expected: PASSA (7 testes).

- [ ] **Step 5: Commit**

```bash
cd /data/projetos/app-investimentos
git add backend/engine/compute.py backend/engine/tests/test_compute.py
git commit -m "✨ feat(engine): compute.py — cálculo extraído dos plot_*

7 funções puras (twr_evolution, drawdown, risk_return, rolling_metrics,
irr_evolution, shadow, yearly_summary) com a lógica de dentro dos plot_* de
financial_report.py. Pandas/dict entra e sai; zero matplotlib, zero I/O.
pct_change(fill_method=None) em todo lugar."
```

---

## Task 3: `engine/dataset.py` — montagem do dataset consumindo `apps.market.series`

**Files:**
- Create: `backend/engine/dataset.py`
- Create: `backend/engine/tests/test_dataset.py`

**Interfaces:**
- Consumes: `apps.market.series.get_series` / `get_many` (Fase 2 Task 10); `apps.market.transforms.synthetic_from_spec` / `rebase_100` (Fase 2 Task 8); `apps.market.synthetics.SyntheticSpec` (Fase 2 Task 7); `apps.market.services.sources.dlp.buscar_historico` (Fase 2 Task 6); `engine.twr.calculate_twr` (Task 1).
- Produces:
  - `dataset.DatasetResult` — `@dataclass` com `df_combined: pd.DataFrame` (base 100), `selic_series: pd.Series | None`, `portfolio_df: pd.DataFrame | None`.
  - `dataset.fetch_user_portfolio(token: str, ativo=None, classe=None, start_date=None, end_date=None) -> tuple[pd.Series | None, pd.DataFrame | None]` — `(twr_series, portfolio_df)`; porta de `FinancialReport.fetch_user_portfolio`, chamando `dlp.buscar_historico` e `twr.calculate_twr`.
  - `dataset.build_dataset(user_series: pd.Series | None = None, benchmarks: list[str] | None = None, extra_specs: list[SyntheticSpec] | None = None, start_date: str | None = None, end_date: str | None = None, portfolio_df: pd.DataFrame | None = None) -> DatasetResult` — monta `df_combined` (`Carteira` + benchmarks, tudo rebased a 100); `selic_series` vem de `get_series("SELIC", ...)`; **não** usa `processar_benchmarks`; **corrige o bug `self.logger`**.

- [ ] **Step 1: Escrever os testes (devem falhar)**

`backend/engine/tests/test_dataset.py`:
```python
import datetime as dt

import pandas as pd
import pytest

from apps.market.synthetics import SyntheticSpec
from engine import dataset

pytestmark = pytest.mark.django_db


def _bench_series(start="2022-01-03", n=260, slope=0.03):
    idx = pd.bdate_range(start, periods=n)
    return pd.Series(
        [100 * (1 + slope) ** (i / n) for i in range(n)], index=idx, dtype=float
    )


def test_build_dataset_combines_user_and_benchmarks(monkeypatch):
    cdi = _bench_series(slope=0.10)
    selic = _bench_series(slope=0.11)

    def fake_get_series(name, s, e):
        return {"CDI": cdi, "SELIC": selic}.get(name)

    def fake_get_many(names, s, e):
        return {n: fake_get_series(n, s, e) for n in names if fake_get_series(n, s, e) is not None}

    monkeypatch.setattr(dataset.series, "get_series", fake_get_series)
    monkeypatch.setattr(dataset.series, "get_many", fake_get_many)

    user = _bench_series(slope=0.20)
    res = dataset.build_dataset(user_series=user, benchmarks=["CDI"])
    assert "Carteira" in res.df_combined.columns
    assert "CDI" in res.df_combined.columns
    # tudo rebased a 100 no início
    assert abs(res.df_combined["Carteira"].iloc[0] - 100) < 1e-6
    assert abs(res.df_combined["CDI"].iloc[0] - 100) < 1e-6
    assert res.selic_series is not None


def test_build_dataset_empty_does_not_raise_self_logger(monkeypatch, caplog):
    # regressão do bug financial_report.py:191 (self.logger inexistente)
    monkeypatch.setattr(dataset.series, "get_series", lambda *a, **k: None)
    monkeypatch.setattr(dataset.series, "get_many", lambda *a, **k: {})
    res = dataset.build_dataset(user_series=None, benchmarks=["CDI"],
                                start_date="2022-01-01", end_date="2022-12-31")
    assert res.df_combined.empty
    assert "Nenhum dado" in caplog.text  # logou via logging.getLogger, não quebrou


def test_build_dataset_extra_synthetic_spec(monkeypatch):
    ipca = _bench_series(slope=0.06)
    monkeypatch.setattr(dataset.series, "get_series",
                        lambda name, s, e: ipca if name == "IPCA" else None)
    monkeypatch.setattr(dataset.series, "get_many", lambda names, s, e: {})
    user = _bench_series(slope=0.15)
    spec = SyntheticSpec("IPCA", "+", 6.0)
    res = dataset.build_dataset(user_series=user, benchmarks=[], extra_specs=[spec])
    assert any("IPCA + 6" in c for c in res.df_combined.columns)


def test_fetch_user_portfolio_uses_dlp_and_twr(monkeypatch):
    hist = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-02", "2023-01-03", "2023-01-04"]),
            "vlr_mercado": [100.0, 110.0, 104.5],
            "vlr_investido": [100.0, 100.0, 100.0],
            "proventos": [0.0, 0.0, 0.0],
        }
    )
    monkeypatch.setattr(dataset.dlp, "buscar_historico", lambda *a, **k: hist)
    twr, pdf = dataset.fetch_user_portfolio("DLP-tok")
    assert isinstance(twr, pd.Series) and not twr.empty
    assert isinstance(pdf, pd.DataFrame) and "fluxo" in pdf.columns


def test_fetch_user_portfolio_none_history_returns_none(monkeypatch):
    monkeypatch.setattr(dataset.dlp, "buscar_historico", lambda *a, **k: None)
    assert dataset.fetch_user_portfolio("DLP-tok") == (None, None)
```

- [ ] **Step 2: Rodar e ver falhar**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/pytest engine/tests/test_dataset.py -q
```
Expected: FALHA com `ModuleNotFoundError: No module named 'engine.dataset'`.

- [ ] **Step 3: Implementar `dataset.py`**

`backend/engine/dataset.py`:
```python
"""Montagem do dataset de análise — porta funcional de FinancialReport.

fetch_user_portfolio: DLP -> TWR da carteira.
build_dataset: Carteira + benchmarks -> df_combined (base 100), consumindo
apps.market.series (NÃO mais processar_benchmarks). Corrige o bug self.logger.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from apps.market import series, transforms
from apps.market.services.sources import dlp
from apps.market.synthetics import SyntheticSpec
from engine.twr import calculate_twr

logger = logging.getLogger(__name__)


@dataclass
class DatasetResult:
    df_combined: pd.DataFrame
    selic_series: pd.Series | None = None
    portfolio_df: pd.DataFrame | None = None


def fetch_user_portfolio(
    token: str, ativo=None, classe=None, start_date=None, end_date=None
) -> tuple[pd.Series | None, pd.DataFrame | None]:
    df = dlp.buscar_historico(token, ativo=ativo, classe=classe)
    if df is None or df.empty:
        logger.warning("DLP não retornou histórico para a carteira.")
        return None, None

    if "date" not in df.columns:
        for col in df.columns:
            if col.lower() in ("date", "data"):
                df = df.rename(columns={col: "date"})
                break
    if "date" not in df.columns:
        logger.error("Histórico DLP sem coluna 'date'.")
        return None, None

    df["date"] = pd.to_datetime(df["date"])
    grp = (
        df.groupby("date")[["vlr_mercado", "vlr_investido", "proventos"]]
        .sum()
        .reset_index()
    )
    grp["fluxo"] = (
        grp["vlr_investido"].diff().fillna(grp["vlr_investido"].iloc[0])
        - grp["proventos"]
    )

    if start_date:
        filt = grp[grp["date"] >= pd.to_datetime(start_date)].copy()
        if not filt.empty:
            gap = filt["vlr_mercado"].iloc[0] - filt["vlr_investido"].iloc[0]
            filt["vlr_investido"] = filt["vlr_investido"] + gap
        grp = filt

    twr = calculate_twr(grp, start_date, end_date)
    if twr.empty:
        return None, None
    return twr, grp


def build_dataset(
    user_series: pd.Series | None = None,
    benchmarks: list[str] | None = None,
    extra_specs: list[SyntheticSpec] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    portfolio_df: pd.DataFrame | None = None,
) -> DatasetResult:
    if user_series is not None and not user_series.empty:
        start = user_series.index.min().date()
        end = user_series.index.max().date()
    else:
        start = pd.to_datetime(start_date).date() if start_date else (pd.Timestamp.today() - pd.DateOffset(years=1)).date()
        end = pd.to_datetime(end_date).date() if end_date else pd.Timestamp.today().date()

    names = list(benchmarks or [])
    bench = series.get_many(names, start, end)
    selic = series.get_series("SELIC", start, end)

    frames: list[pd.Series] = []
    if user_series is not None and not user_series.empty:
        us = user_series.copy()
        us.name = "Carteira"
        us.index = pd.to_datetime(us.index)
        frames.append(us)

    for name in names:
        s = bench.get(name)
        if s is None or s.empty:
            continue
        s = s.copy()
        s.index = pd.to_datetime(s.index)
        s.name = name
        frames.append(pd.to_numeric(s, errors="coerce"))

    for spec in extra_specs or []:
        base = series.get_series(spec.base, start, end)
        if base is None or base.empty:
            continue
        syn = transforms.synthetic_from_spec(spec, base)
        syn.name = f"{spec.base} + {spec.valor:g}%" if spec.operador == "+" else f"{spec.valor:g}% do {spec.base}"
        syn.index = pd.to_datetime(syn.index)
        frames.append(syn)

    if not frames:
        logger.warning("Nenhum dado disponível para análise.")
        return DatasetResult(df_combined=pd.DataFrame(), selic_series=selic, portfolio_df=portfolio_df)

    raw = pd.concat(frames, axis=1).sort_index()
    if start_date and end_date:
        raw = raw.loc[str(start_date):str(end_date)]
    raw = raw.ffill()
    raw = raw.dropna(subset=["Carteira"]) if "Carteira" in raw.columns else raw.dropna(how="all")

    for col in raw.columns:
        raw[col] = transforms.rebase_100(raw[col])

    return DatasetResult(df_combined=raw, selic_series=selic, portfolio_df=portfolio_df)
```

- [ ] **Step 4: Rodar e ver passar**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/pytest engine/tests/test_dataset.py -q
```
Expected: PASSA (5 testes).

- [ ] **Step 5: Rodar a suíte inteira**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/pytest -q
```
Expected: PASSA — Fases 0+1 + Fase 2 + engine (portado 12 + compute 7 + dataset 5).

- [ ] **Step 6: Commit**

```bash
cd /data/projetos/app-investimentos
git add backend/engine/dataset.py backend/engine/tests/test_dataset.py
git commit -m "✨ feat(engine): dataset.py — build_dataset consumindo apps.market.series

fetch_user_portfolio (DLP -> TWR) e build_dataset (Carteira + benchmarks -> df
base 100) portados de FinancialReport, funcionais (DatasetResult, sem self.
mutável). Usa series.get_many no lugar de processar_benchmarks; extra_specs para
sintéticos ad-hoc. Corrige o bug self.logger (regressão coberta por teste)."
```

---

## Task 4: `engine/report_legacy.py` — os `plot_*` Matplotlib (só-CLI)

**Files:**
- Modify: `backend/requirements.txt`
- Create: `backend/engine/report_legacy.py`
- Create: `backend/engine/tests/test_report_legacy.py`

**Interfaces:**
- Consumes: `engine.compute.*` (Task 2); `engine.dataset.DatasetResult` (Task 3); `matplotlib`.
- Produces:
  - `report_legacy.plot_twr_evolution(df_combined, title_suffix="") -> matplotlib.figure.Figure`
  - `report_legacy.plot_drawdown(df_combined, title_suffix="") -> Figure`
  - `report_legacy.plot_risk_return(df_combined, title_suffix="") -> Figure`
  - `report_legacy.plot_rolling_sharpe(df_combined, rf_series=None, window=252, title_suffix="") -> Figure`
  - `report_legacy.plot_shadow(portfolio_df, df_combined, title_suffix="") -> Figure`
  - `report_legacy.export_all(result: DatasetResult, out_dir: str) -> list[str]` — salva os PNGs, devolve os caminhos.
  - **Nada aqui é importado por `apps/` nem por `engine/compute.py` / `engine/dataset.py`.**

- [ ] **Step 1: Adicionar `matplotlib` ao `requirements.txt`**

Em `backend/requirements.txt`, na seção runtime (após as deps da Fase 2):
```
# --- Fase 2.5: export CLI de relatório (NÃO usado pela API) ---
matplotlib~=3.9
```

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/pip install -r requirements.txt
.venv/bin/python -c "import matplotlib; print('matplotlib', matplotlib.__version__)"
```
Expected: imprime a versão.

- [ ] **Step 2: Escrever os testes (devem falhar)**

`backend/engine/tests/test_report_legacy.py`:
```python
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # backend sem display

from engine import report_legacy  # noqa: E402


def _combined():
    idx = pd.bdate_range("2022-01-03", periods=300)
    return pd.DataFrame(
        {
            "Carteira": np.linspace(100, 130, len(idx)),
            "CDI": np.linspace(100, 118, len(idx)),
        },
        index=idx,
    )


def test_plot_twr_evolution_returns_figure():
    fig = report_legacy.plot_twr_evolution(_combined(), title_suffix="teste")
    assert fig.__class__.__name__ == "Figure"
    assert len(fig.axes) >= 1


def test_plot_drawdown_returns_figure():
    assert report_legacy.plot_drawdown(_combined()).__class__.__name__ == "Figure"


def test_plot_risk_return_returns_figure():
    assert report_legacy.plot_risk_return(_combined()).__class__.__name__ == "Figure"


def test_export_all_writes_pngs(tmp_path):
    from engine.dataset import DatasetResult

    res = DatasetResult(df_combined=_combined(), selic_series=None, portfolio_df=None)
    paths = report_legacy.export_all(res, str(tmp_path))
    assert paths
    for p in paths:
        assert p.endswith(".png")
        assert (tmp_path / p.split("/")[-1]).exists()
```

- [ ] **Step 3: Rodar e ver falhar**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/pytest engine/tests/test_report_legacy.py -q
```
Expected: FALHA com `ModuleNotFoundError: No module named 'engine.report_legacy'`.

- [ ] **Step 4: Implementar `report_legacy.py`**

`backend/engine/report_legacy.py`:
```python
"""Export CLI dos gráficos históricos (Matplotlib). NÃO é importado pela API.

Cada plot_* chama a compute_* correspondente e desenha. Preservado para gerar os
PNGs antigos durante a coexistência e como conferência visual na paridade (Fase 6).
`import matplotlib` vive SÓ aqui.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from engine import compute
from engine.dataset import DatasetResult


def plot_twr_evolution(df_combined, title_suffix: str = ""):
    out = compute.compute_twr_evolution(df_combined)
    df = out["series"]
    fig, ax = plt.subplots(figsize=(12, 7))
    for col in df.columns:
        lw = 3 if col == "Carteira" else 1.5
        ax.plot(df.index, df[col], label=f"{col} ({out['final_returns'][col]:.1f}%)", linewidth=lw)
    ax.set_title(f"Evolução TWR (Base 100) - {title_suffix}")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    return fig


def plot_drawdown(df_combined, title_suffix: str = ""):
    dd = compute.compute_drawdown(df_combined)
    fig, ax = plt.subplots(figsize=(12, 6))
    for col in dd.columns:
        ax.plot(dd.index, dd[col], label=col)
    ax.set_title(f"Drawdown - {title_suffix}")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    return fig


def plot_risk_return(df_combined, title_suffix: str = ""):
    rr = compute.compute_risk_return(df_combined)
    fig, ax = plt.subplots(figsize=(10, 8))
    for name, row in rr.iterrows():
        ax.scatter(row["Volatilidade"], row["Retorno (CAGR)"], s=120 if name == "Carteira" else 70)
        ax.text(row["Volatilidade"], row["Retorno (CAGR)"], f"  {name}", fontsize=9, va="center")
    ax.set_title(f"Risco x Retorno - {title_suffix}")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(True, linestyle="--", alpha=0.5)
    return fig


def plot_rolling_sharpe(df_combined, rf_series=None, window: int = 252, title_suffix: str = ""):
    sharpe = compute.compute_rolling_metrics(df_combined, rf_series, window)["sharpe"]
    fig, ax = plt.subplots(figsize=(12, 6))
    for col in sharpe.columns:
        ax.plot(sharpe.index, sharpe[col], label=col)
    ax.set_title(f"Sharpe Ratio Móvel ({window}d) - {title_suffix}")
    ax.grid(True)
    ax.legend()
    return fig


def plot_shadow(portfolio_df, df_combined, title_suffix: str = ""):
    sw = compute.compute_shadow(portfolio_df, df_combined)
    fig, ax = plt.subplots(figsize=(12, 7))
    for col in sw.columns:
        ax.plot(sw.index, sw[col], label=col, linewidth=3 if col == "Carteira Real" else 1.5)
    ax.set_title(f"Simulação de Aportes - {title_suffix}")
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    return fig


def export_all(result: DatasetResult, out_dir: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    jobs = [
        ("twr_evolucao.png", plot_twr_evolution(result.df_combined)),
        ("drawdown.png", plot_drawdown(result.df_combined)),
        ("risco_retorno.png", plot_risk_return(result.df_combined)),
        ("sharpe_movel.png", plot_rolling_sharpe(result.df_combined, result.selic_series)),
    ]
    if result.portfolio_df is not None and not result.portfolio_df.empty:
        jobs.append(("simulacao_aportes.png", plot_shadow(result.portfolio_df, result.df_combined)))

    paths: list[str] = []
    for name, fig in jobs:
        path = os.path.join(out_dir, name)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths
```

- [ ] **Step 5: Rodar e ver passar**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/pytest engine/tests/test_report_legacy.py -q
```
Expected: PASSA (4 testes).

- [ ] **Step 6: Confirmar que a API não importa Matplotlib**

Run:
```bash
cd /data/projetos/app-investimentos/backend
grep -rn "matplotlib" apps config engine/compute.py engine/dataset.py || echo "OK — sem matplotlib fora de report_legacy"
```
Expected: imprime `OK — sem matplotlib fora de report_legacy`.

- [ ] **Step 7: Commit**

```bash
cd /data/projetos/app-investimentos
git add backend/requirements.txt backend/engine/report_legacy.py backend/engine/tests/test_report_legacy.py
git commit -m "✨ feat(engine): report_legacy.py — export CLI dos gráficos (Matplotlib)

plot_* que chamam as compute_* e desenham; export_all salva os PNGs. NÃO
importado pela API — matplotlib vive só aqui. matplotlib~=3.9 no requirements."
```

---

## Task 5: ADR do débito do engine + documentação viva + verificação final

**Files:**
- Create: `docs/adr/0012-engine-copia-literal-fase-2-5.md`
- Modify: `docs/adr/README.md`
- Modify: `docs/consolidacao_arquitetura_e_migracao.md` (§2.0, §2.2, §2.3, checklist §9)
- Modify: `backend/README.md` (estrutura + engine)
- Modify: `backend/CLAUDE.md` (se surgirem convenções de `engine/`)

**Interfaces:**
- Consumes: tudo das Tasks 1-4.
- Produces: DoD da Fase 2.5 (spec §3.3) verificada; ADR do débito técnico; docs atualizadas.

- [ ] **Step 1: Suíte completa + `check` + `makemigrations --check`**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/python manage.py check
.venv/bin/python manage.py makemigrations --check --dry-run
.venv/bin/pytest -q
```
Expected: `check` limpo; "No changes detected" (a Fase 2.5 não mexe em modelos); **toda** a suíte verde.

- [ ] **Step 2: Confirmar os 3 testes portados sem alteração de lógica**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/pytest engine/tests/test_twr.py engine/tests/test_irr.py engine/tests/test_metrics.py -q -v
git diff --no-index tests/test_twr.py engine/tests/test_twr.py | grep '^[+-]' | grep -v 'src.engine\|from engine\|^[+-][+-][+-]' || echo "test_twr: só o import mudou"
```
Expected: 12 testes verdes; o `diff` mostra que **só o import** mudou em cada teste portado (ou, se `test_metrics.py` mudou um assert por causa do `fill_method`, que isso está explicado no commit da Task 1).

- [ ] **Step 3: `grep` de Matplotlib fora do `report_legacy`**

Run:
```bash
cd /data/projetos/app-investimentos/backend
grep -rn "import matplotlib\|from matplotlib" apps config engine/compute.py engine/dataset.py engine/twr.py engine/irr.py engine/metrics.py
```
Expected: **nenhuma saída**.

- [ ] **Step 4: Smoke ponta a ponta — dataset real + export CLI**

Run (usa o `MarketSeries` semeado pelo `backfill` da Fase 2):
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/python manage.py shell -c "
import datetime as dt
from engine import dataset, report_legacy
res = dataset.build_dataset(benchmarks=['CDI', 'IMA-B'], start_date='2020-01-01', end_date='2024-12-31')
print('df_combined shape:', res.df_combined.shape, '| cols:', list(res.df_combined.columns))
paths = report_legacy.export_all(res, '/tmp/relatorio_fase25')
print('PNGs:', paths)
"
ls -la /tmp/relatorio_fase25/
```
Expected: `df_combined` não-vazio com colunas `CDI`/`IMA-B` (base 100); PNGs gerados em `/tmp/relatorio_fase25/`.

- [ ] **Step 5: Criar `docs/adr/0012-engine-copia-literal-fase-2-5.md`**

Conteúdo (formato dos ADRs existentes — Status/Data/Fase, Contexto, Decisão, Consequências):

- **Contexto:** o engine puro (`twr`/`irr`/`metrics`) é NumPy/pandas testado; a paridade da Fase 6 compara o backend novo contra o Streamlit rodando **este mesmo código**. Refatorar no port invalidaria o teste de paridade.
- **Decisão:** copiar `twr.py`/`irr.py`/`metrics.py` **sem alteração de lógica** na Fase 2.5. Única exceção: `pct_change(fill_method=None)` em `metrics.py` (pandas 2.2). Débito de refatoração fica registrado, revisão **pós-Fase 6** (quando há o harness de paridade como rede de regressão, o `src/` é deletado, e o contrato da API está estável).
- **Pontos concretos a revisar pós-Fase 6:**
  - `irr.py` — Newton-Raphson sem fallback (bisseção/Brent); cutoff mágico `abs(npv) < 0.1`; chute `0.1` fixo; `365.0` fixo (ignora bissexto); clamp `r <= -1.0 → -0.99` que pode oscilar; não guarda `dates` fora de ordem. Candidato a `scipy.optimize.brentq` / `numpy_financial.xirr`.
  - `twr.py` — `np.where(denominador != 0, ...)` sem epsilon; normalização de coluna duplicada com `dataset.fetch_user_portfolio`; `fluxo = vlr_investido.diff().fillna(iloc[0])` assume 1ª linha como capital limpo.
  - `metrics.py` — `252` e `rf_constant=0.10` mágicos espalhados.
- **Pendência de paridade:** o `src/engine/metrics.py` legado precisa do mesmo `fill_method=None` (ou pin de pandas alinhado nos dois lados) **antes** da suíte de paridade da Fase 6.
- **Consequências:** `+` paridade da Fase 6 permanece maçã-com-maçã; `−` o débito fica em aberto até a Fase 6.

Atualizar `docs/adr/README.md` com a linha do `0012`.

- [ ] **Step 6: Atualizar `docs/consolidacao_arquitetura_e_migracao.md`**

- **§2.0** (tabela "Já implementado"): adicionar linha **Fase 2.5** — `backend/engine/` (cópia literal `twr`/`irr`/`metrics` + `compute.py` + `dataset.py` + `report_legacy.py`), bug `self.logger` corrigido, ~28 testes; `pytest.ini` com `testpaths = apps engine`.
- **§2.2** ("O que não existe ainda"): ajustar/remover a menção a `financial_report.py` misturando cálculo + plotagem (agora separado em `compute.py` / `report_legacy.py`).
- **§2.3** (dívida técnica): adicionar o item do engine (`irr.py`/`twr.py`/`metrics.py` — cópia literal na 2.5, revisão pós-Fase 6, ver ADR 0012); marcar o bug `self.logger` como **corrigido** (mantendo o registro histórico).
- Marcar item **4** do checklist §9 como concluído.

- [ ] **Step 7: Atualizar `backend/README.md`**

Na "Estrutura", acrescentar o bloco `engine/` (com uma linha por módulo). Na seção "Rodar", mencionar:
```bash
.venv/bin/python manage.py shell -c "from engine import dataset, report_legacy; ..."  # export CLI dos gráficos
```

- [ ] **Step 8: `code-review` + `code-simplifier` no diff de `backend/engine/`**

Rodar `/code-review` sobre `backend/engine/` (Tasks 1-4). Correções de correção → commit `🐛 fix(engine): ...`; limpezas → `♻️ refactor(engine): ...`. Rodar `.venv/bin/pytest engine -q` de novo — verde. **Não** aceitar sugestões que alterem a lógica de `twr.py`/`irr.py`/`metrics.py` (cópia literal — qualquer mudança ali vira item do ADR 0012, não um fix agora).

- [ ] **Step 9: Commit da documentação**

```bash
cd /data/projetos/app-investimentos
git add docs/adr/0012-engine-copia-literal-fase-2-5.md docs/adr/README.md \
        docs/consolidacao_arquitetura_e_migracao.md backend/README.md backend/CLAUDE.md
git commit -m "📝 docs: fecha a Fase 2.5 — ADR 0012, §2 do doc mestre, README

ADR do débito técnico do engine (cópia literal na 2.5, revisão pós-Fase 6, com
os pontos concretos de irr/twr/metrics e a pendência do pct_change no legado).
Documentação viva: engine/ no estado atual, bug self.logger marcado corrigido."
```

---

## Self-Review

**1. Cobertura do spec (§5 — Fase 2.5)**

| Item do spec | Task |
|---|---|
| §5.1 estrutura `backend/engine/` | Tasks 1-4 |
| §5.1 `pytest.ini` `testpaths = apps engine` | Task 1 Step 4 |
| §5.2 `compute.py` — 7 funções puras extraídas dos `plot_*` (tabela de mapeamento) | Task 2 |
| §5.3 `dataset.py` — consome `series.get_many`, `fetch_user_portfolio` via `dlp`, sem `self.`, corrige `self.logger` | Task 3 |
| §5.3 `extra_specs: list[SyntheticSpec]` para sintéticos ad-hoc | Task 3 Step 3 + teste `test_build_dataset_extra_synthetic_spec` |
| §5.4 `pct_change(fill_method=None)` em `metrics.py` + código novo; pendência no legado registrada | Task 1 Step 2 + Task 2/3 (código novo) + Task 5 Step 5 (ADR) |
| §5.5 `report_legacy.py` — `plot_*` Matplotlib, só-CLI, `matplotlib` só aqui | Task 4 |
| §5.6 direção das dependências (`engine` → `apps.market`, nunca o contrário) | Task 3 (imports) + Task 4 Step 6 / Task 5 Step 3 (grep) |
| §5.7 débito técnico do engine — ADR + §2.3 | Task 5 Steps 5-6 |
| §6.2 testes: 3 portados + `test_compute.py` + `test_dataset.py` | Tasks 1, 2, 3 |
| §6.2 `report_legacy` — teste só "importa e produz Figure" | Task 4 Step 2 |
| §3.3 Definition of Done (Fase 2.5), itens 1-5 | Task 5 Steps 1-4 + ADR |
| §6.3 `pytest.ini` (marker `integration` já vem da Fase 2) | herdado da Fase 2 |

Sem lacunas para o escopo da Fase 2.5.

**2. Placeholders**

Nenhum "TBD/TODO". Pontos deixados ao implementador, todos marcados e não bloqueantes:
- Task 1 Step 5 — se `test_metrics.py` mudar um assert por causa do `fill_method`, o passo manda documentar o porquê (com `systematic-debugging`), não "consertar" cego.
- Task 5 Step 8 — `code-review` não pode alterar a lógica de `twr/irr/metrics` (cópia literal).
- Conteúdo textual do ADR 0012 (Task 5 Step 5) — o passo lista todos os pontos concretos a incluir; não é um "escreva algo aqui".

**3. Consistência de tipos**

- `engine.twr.calculate_twr(df, start_date=None, end_date=None) -> pd.Series` — Task 1 (cópia), usado em `dataset.fetch_user_portfolio` (Task 3).
- `engine.metrics.calculate_drawdown/…_rolling_volatility/…_rolling_sharpe` — Task 1 (cópia), usados em `compute.compute_drawdown` / `compute_rolling_metrics` (Task 2).
- `engine.irr.calculate_xirr(cash_flows, dates) -> float | None` — Task 1, usado em `compute.compute_irr_evolution` (Task 2).
- `compute.compute_*` assinaturas — fixadas nos "Interfaces" da Task 2, consumidas por `report_legacy.*` (Task 4) com os mesmos nomes (`compute_twr_evolution`, `compute_drawdown`, `compute_risk_return`, `compute_rolling_metrics`, `compute_shadow`).
- `dataset.DatasetResult(df_combined, selic_series, portfolio_df)` — Task 3 (def), consumido por `report_legacy.export_all` (Task 4) — campos `df_combined`, `selic_series`, `portfolio_df` batem.
- `dataset.build_dataset(...)` / `dataset.fetch_user_portfolio(...)` — assinaturas idênticas nos "Interfaces" da Task 3 e nos testes (`test_dataset.py`).
- `apps.market.series.get_series` / `get_many`, `apps.market.transforms.synthetic_from_spec` / `rebase_100`, `apps.market.synthetics.SyntheticSpec`, `apps.market.services.sources.dlp.buscar_historico` — todas produzidas pela **Fase 2** (plano 1) e consumidas aqui com as mesmas assinaturas; `dataset.py` faz `from apps.market import series, transforms` e `from apps.market.services.sources import dlp`, e os testes monkeypatcham exatamente esses atributos (`dataset.series.get_series`, `dataset.dlp.buscar_historico`).

Sem inconsistências.

---

## Execution Handoff

Plano completo e salvo em `docs/superpowers/plans/2026-08-31-fase-2-5-refatorar-engine.md`.

**Só executar depois da Fase 2 (plano 1) fechada e revisada** — este plano importa `apps.market.*`.

Duas opções de execução:

1. **Subagent-Driven (recomendado)** — um subagente novo por task, revisão em dois estágios entre tasks.
2. **Inline Execution** — executar as tasks nesta sessão com `executing-plans`, em lote com checkpoints.

Qual abordagem? (E confirme que a Fase 2 já está concluída.)
