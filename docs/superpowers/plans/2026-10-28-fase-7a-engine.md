# Fase 7a — Correção do engine (fluxo, recortes, XIRR, CDI) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fazer o engine calcular TWR, TIR, KPIs e shadow corretamente, reproduzindo a metodologia da DLP (fluxo com lucro realizado; aporte no início e resgate no fim do mês).

**Architecture:** Um módulo novo `engine/flows.py` passa a ser o único dono do agrupamento por data, do fluxo externo e do HPR. `twr.py`, `dataset.py`, `compute.py`, o router `/summary` e o harness de paridade consomem esse módulo. A correção é provada por um gabarito de casos calculados à mão, por um teste de oráculo contra o export da planilha DLP e por uma regressão contra a linha de base da Fase 6, que só pode mudar o que a fase promete.

**Tech Stack:** Python 3.12, pandas, numpy, Django 5 + django-ninja, pytest (`backend/.venv/bin/pytest`).

**Spec:** [`docs/superpowers/specs/2026-10-28-fase-7-confiabilidade-design.md`](../specs/2026-10-28-fase-7-confiabilidade-design.md) — §3.2, §5.2, §5.2.1, §5.3, §5.3.1.

**Escopo deste plano:** só a frente do engine (spec §5.1–§5.3 e §5.6, 1º ponto de espera). Planos seguintes, um por frente: **7b** dados de mercado (§5.4), **7c** backlog SPA/API (§5.5), **7d** otimização (§5.7).

## Global Constraints

- TDD conforme `backend/CLAUDE.md`: escrever o teste, rodar e **ver falhar**, implementar, ver passar.
- Um commit por task; Conventional Commits + Gitmoji, imperativo em português; **nunca** `Co-Authored-By`.
- Fluxo externo: `fluxo = Δvlr_investido − lucro − proventos`; na 1ª linha `Δvlr_investido = vlr_investido`.
- HPR mensal: `r = (M_t − min(F,0)) / (M_{t−1} + max(F,0))`; 1ª linha do recorte e denominador 0 → `1.0`.
- XIRR: Actual/365 contado do fluxo mais antigo; datas ordenadas; não converge → `None` (API `null`).
- Taxa livre de risco = série `"CDI"` do período; sem série → Sharpe `NaN`/`null`, **nunca** 10 % fixo.
- Contrato da API v1 só com mudanças aditivas; nomes de campos do `/summary` mantidos.
- Relatórios, ADR e mensagens de commit **sem valores em R$** da carteira real (só %, razões, contagens).
- `reports/` e `dlp_sheet/` nunca versionados.
- Rodar testes a partir de `backend/`: `.venv/bin/pytest -q`; paridade: `cd scripts/parity && ../../backend/.venv/bin/pytest -q`.

## Review Focus

1. Histórico sem coluna `lucro` (filtro/payload antigo da DLP) → tratado como `lucro = 0`, sem `KeyError`. Teste na Task 1.
2. Mês após zeragem total sem novo aporte (`M_{t−1} = 0`, `F ≤ 0`) → HPR `1.0`, sem divisão por zero. Teste na Task 1.
3. Dois ativos na mesma data com lucro/provento (filtro de classe) → somados antes do fluxo. Teste na Task 1.
4. `end_date` antes do 1º dado ou `start_date` depois do último → `(None, None)`, sem exceção. Teste na Task 2.
5. XIRR com todos os fluxos do mesmo sinal ou uma data só → `None`. Teste na Task 4.

---

### Task 1: `engine/flows.py` — agrupamento, fluxo externo e HPR da DLP

**Files:**
- Create: `backend/engine/flows.py`
- Create: `backend/engine/tests/test_flows.py`
- Modify: `backend/engine/twr.py` (corpo de `calculate_twr`, assinatura mantida)
- Modify: `backend/engine/tests/test_twr.py:54-69` (`test_twr_with_resgate_and_dividends`)

**Interfaces:**
- Produces:
  - `FLOW_COLUMNS: tuple[str, ...] = ("vlr_mercado", "vlr_investido", "proventos", "lucro")`
  - `group_portfolio(df: pd.DataFrame) -> pd.DataFrame` — `date` (datetime64) + `FLOW_COLUMNS` (+ `fluxo` se vier na entrada), somados por data, ordenados; colunas ausentes de `FLOW_COLUMNS` viram `0.0`.
  - `external_flows(grp: pd.DataFrame) -> pd.Series` — fórmula das Global Constraints, mesmo índice de `grp`.
  - `hpr(vlr_mercado: pd.Series, fluxo: pd.Series) -> pd.Series` — regra das Global Constraints, posição 0 = `1.0`.
  - `calculate_twr(df, start_date=None, end_date=None) -> pd.Series` (em `twr.py`): usa `df["fluxo"]` se existir, senão `external_flows`; índice `date`, `cumprod` de `hpr`, cortes de período como hoje.

- [ ] **Step 1: Escrever o gabarito do fluxo e do HPR em `test_flows.py`**

Carteira-base (comentar o raciocínio no teste): mês 1 aporta 1000; mês 2 vende custo 400 por 600 (lucro 200), resto vale 700; mês 3 recebe provento 10, vale 720.

```python
BASE = pd.DataFrame({
    "date": ["2024-01-31", "2024-02-29", "2024-03-31"],
    "vlr_mercado": [1000.0, 700.0, 720.0],
    "vlr_investido": [1000.0, 600.0, 600.0],
    "lucro": [0.0, 200.0, 0.0],
    "proventos": [0.0, 0.0, 10.0],
})

def test_external_flows_subtracts_realized_profit_and_dividends():
    assert list(flows.external_flows(flows.group_portfolio(BASE))) == [1000.0, -600.0, -10.0]

def test_hpr_withdrawal_at_end_of_month():
    g = flows.group_portfolio(BASE)
    h = flows.hpr(g["vlr_mercado"], flows.external_flows(g))
    assert h.tolist() == pytest.approx([1.0, 1.30, 730 / 700])

def test_hpr_contribution_at_start_of_month():
    # M0=1000, aporte 500, M1=1600 -> 1600/1500
    ...  assert h.iloc[1] == pytest.approx(1600 / 1500)

def test_hpr_full_exit_then_reentry():
    # vende tudo (custo 100, lucro 10) -> (0+110)/100 = 1.10; mês parado com M0=0,F=0 -> 1.0;
    # reentra aportando 200 e vale 210 -> 1.05
    ...  assert h.tolist() == pytest.approx([1.0, 1.10, 1.0, 1.05])

def test_missing_lucro_column_counts_as_zero():
    g = flows.group_portfolio(BASE.drop(columns=["lucro"]))
    assert (g["lucro"] == 0).all()

def test_same_date_assets_are_summed_before_flows():
    # dois ativos em 2024-02-29 com lucro 120 e 80 -> mesmo fluxo de BASE
    ...  assert list(flows.external_flows(g)) == [1000.0, -600.0, -10.0]
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cd backend && .venv/bin/pytest engine/tests/test_flows.py -q`
Expected: FAIL com `ModuleNotFoundError: No module named 'engine.flows'`

- [ ] **Step 3: Implementar `engine/flows.py` e reescrever o miolo de `calculate_twr` usando `group_portfolio` + `hpr`**

Manter em `calculate_twr` a normalização da coluna de data que já existe (rename `Data`/índice). Atualizar `test_twr_with_resgate_and_dividends` para o valor correto pela regra nova: `twr.iloc[1] == 1.03` (o comentário original do teste já dizia "+3 %").

- [ ] **Step 4: Rodar e ver passar**

Run: `cd backend && .venv/bin/pytest engine/tests/test_flows.py engine/tests/test_twr.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/engine/flows.py backend/engine/twr.py backend/engine/tests/test_flows.py backend/engine/tests/test_twr.py
git commit -m "🐛 fix(engine): fluxo externo com lucro realizado e resgate no fim do mês"
```

---

### Task 2: `fetch_user_portfolio` — `lucro`, corte por `end_date` e fluxo após o `gap`

**Files:**
- Modify: `backend/engine/dataset.py:29-66`
- Test: `backend/engine/tests/test_dataset.py`

**Interfaces:**
- Consumes: `flows.group_portfolio`, `flows.external_flows`, `calculate_twr` (Task 1).
- Produces: `fetch_user_portfolio(...) -> (twr: pd.Series | None, portfolio_df: pd.DataFrame | None)`. `portfolio_df` tem `date` + `FLOW_COLUMNS` + `fluxo`, recortado por `start_date` **e** `end_date`. Com `start_date` no meio do histórico: `vlr_investido += gap` e `fluxo.iloc[0] = vlr_mercado.iloc[0]` (aporte inicial fictício = patrimônio de abertura).

- [ ] **Step 1: Escrever os testes**

Histórico com `monkeypatch` em `dataset.dlp.buscar_historico`: datas `2024-01-31..2024-04-30`, `vlr_investido [1000,1000,1500,1500]`, `vlr_mercado [1000,1100,1650,1700]`, `lucro`/`proventos` 0.

```python
def test_end_date_cuts_portfolio_df(...):
    twr, pdf = dataset.fetch_user_portfolio("t", end_date="2024-02-29")
    assert pdf["date"].max() == pd.Timestamp("2024-02-29") and len(pdf) == 2

def test_start_date_gap_sets_opening_flow_to_market_value(...):
    twr, pdf = dataset.fetch_user_portfolio("t", start_date="2024-02-29")
    assert pdf["vlr_investido"].tolist() == [1100.0, 1600.0, 1600.0]
    assert pdf["fluxo"].tolist() == [1100.0, 500.0, 0.0]
    assert twr.tolist() == pytest.approx([1.0, 1650 / 1600, 1650 / 1600 * 1700 / 1650])

def test_portfolio_df_carries_lucro(...):
    assert "lucro" in pdf.columns

def test_window_outside_history_returns_none(...):
    assert dataset.fetch_user_portfolio("t", start_date="2025-01-01") == (None, None)
    assert dataset.fetch_user_portfolio("t", end_date="2023-12-31") == (None, None)
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cd backend && .venv/bin/pytest engine/tests/test_dataset.py -q`
Expected: FAIL (`len(pdf) == 4` no corte por `end_date`; `fluxo == [1000, 500, 0]` no gap; `lucro` ausente)

- [ ] **Step 3: Implementar em `fetch_user_portfolio`**

Agrupar com `group_portfolio`, calcular `fluxo` com `external_flows`, recortar por `start_date` e `end_date`, aplicar o `gap` e sobrescrever `fluxo.iloc[0]`. Passar o `grp` já com `fluxo` para `calculate_twr`. Recorte vazio → `(None, None)`.

- [ ] **Step 4: Rodar e ver passar**

Run: `cd backend && .venv/bin/pytest engine -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/engine/dataset.py backend/engine/tests/test_dataset.py
git commit -m "🐛 fix(engine): recorta a carteira por end_date e recalcula o fluxo após o gap"
```

---

### Task 3: KPIs do `/summary` no engine e `/shadow` sem dupla contagem

**Files:**
- Modify: `backend/engine/compute.py` (nova `compute_summary_kpis`; `compute_shadow`)
- Modify: `backend/apps/api/routers/portfolio.py:101-140` (router `summary` passa a chamar `compute_summary_kpis`)
- Modify: `scripts/parity/kpis.py` (`derive_kpis` usa `compute_summary_kpis`)
- Test: `backend/engine/tests/test_compute.py`, `backend/apps/api/tests/test_portfolio_misc.py:15-30`

**Interfaces:**
- Consumes: `portfolio_df` da Task 2 (com `fluxo`).
- Produces: `compute_summary_kpis(portfolio_df: pd.DataFrame) -> dict[str, float]` com chaves `patrimonio_inicial`, `aportes_periodo`, `patrimonio`, `ganho_periodo`, em que `aportes_periodo = fluxo.iloc[1:].sum()` (aportes líquidos de resgates, proventos e lucro realizado) e `ganho_periodo = patrimonio − patrimonio_inicial − aportes_periodo` (retorno total).
- `compute_shadow`: `Total Investido = vlr_mercado.iloc[0] + cumsum(fluxo[1:])`; benchmarks compram cotas com o saldo inicial e depois só com `fluxo[1:]`.

- [ ] **Step 1: Escrever os testes**

```python
PDF = pd.DataFrame({  # carteira-base da Task 1, com fluxo
    "date": pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"]),
    "vlr_mercado": [1000.0, 700.0, 720.0], "vlr_investido": [1000.0, 600.0, 600.0],
    "lucro": [0.0, 200.0, 0.0], "proventos": [0.0, 0.0, 10.0], "fluxo": [1000.0, -600.0, -10.0],
})

def test_summary_kpis_total_return():
    k = compute.compute_summary_kpis(PDF)
    # ganho = lucro 200 + valorização aberta 100 + 20 + provento 10
    assert k == pytest.approx({"patrimonio_inicial": 1000.0, "aportes_periodo": -610.0,
                               "patrimonio": 720.0, "ganho_periodo": 330.0})

def test_shadow_does_not_double_count_opening_balance():
    combined = pd.DataFrame({"Carteira": [100.0, 130.0, 135.0], "CDI": [100.0] * 3}, index=PDF["date"])
    sw = compute.compute_shadow(PDF, combined)
    assert sw["Total Investido"].tolist() == [1000.0, 400.0, 390.0]
    assert sw["CDI"].tolist() == pytest.approx([1000.0, 400.0, 390.0])
```

No teste da API, trocar a asserção de `aportes_periodo` para `== PORTFOLIO_DF["fluxo"].iloc[1:].sum()` (derivando `fluxo` com `flows.external_flows` se a fixture não o tiver) e manter a identidade `inicial + aportes + ganho == patrimônio`.

- [ ] **Step 2: Rodar e ver falhar**

Run: `cd backend && .venv/bin/pytest engine/tests/test_compute.py apps/api/tests/test_portfolio_misc.py -q`
Expected: FAIL (`AttributeError: compute_summary_kpis`; `Total Investido` começando em 2000)

- [ ] **Step 3: Implementar `compute_summary_kpis`, corrigir `compute_shadow`, ligar no router e em `derive_kpis`**

- [ ] **Step 4: Rodar e ver passar**

Run: `cd backend && .venv/bin/pytest -q && cd ../scripts/parity && ../../backend/.venv/bin/pytest -q`
Expected: PASS nas duas suítes

- [ ] **Step 5: Commit**

```bash
git add backend/engine/compute.py backend/engine/tests/test_compute.py backend/apps/api/routers/portfolio.py backend/apps/api/tests/test_portfolio_misc.py scripts/parity/kpis.py
git commit -m "🐛 fix(engine): KPIs de retorno total e shadow sem dupla contagem do saldo inicial"
```

---

### Task 4: XIRR robusto

**Files:**
- Modify: `backend/engine/irr.py`
- Test: `backend/engine/tests/test_irr.py`

**Interfaces:**
- Produces: `calculate_xirr(cash_flows, dates) -> float | None` (mesma assinatura). Ordena por data; anos = dias desde a data mais antiga / 365; Newton a partir de 0.1 com convergência `|npv| ≤ 1e-9 · Σ|fluxos|`; se Newton falhar, bisseção num intervalo com troca de sinal entre `-0.999999` e `1e3` (expandindo o teto); sem troca de sinal → `None`.

- [ ] **Step 1: Escrever os testes**

```python
D1 = ["2023-01-01", "2024-01-01"]  # 365 dias
def test_xirr_closed_form(): assert calculate_xirr([-1000, 1100], D1) == pytest.approx(0.10, abs=1e-9)
def test_xirr_unordered_dates(): assert calculate_xirr([1100, -1000], D1[::-1]) == pytest.approx(0.10, abs=1e-9)
@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e7])
def test_xirr_scale_invariant(scale): assert calculate_xirr([-1000 * scale, 1100 * scale], D1) == pytest.approx(0.10, abs=1e-9)
def test_xirr_negative_rate(): assert calculate_xirr([-1000, 500], D1) == pytest.approx(-0.5, abs=1e-9)
def test_xirr_large_rate(): assert calculate_xirr([-1000, 30000], D1) == pytest.approx(29.0, rel=1e-9)
def test_xirr_no_sign_change_is_none(): assert calculate_xirr([-1000, -10], D1) is None
def test_xirr_single_date_is_none(): assert calculate_xirr([-1000, 1100], ["2023-01-01"] * 2) is None
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cd backend && .venv/bin/pytest engine/tests/test_irr.py -q`
Expected: FAIL (datas fora de ordem, escala 1e-3 e/ou sem troca de sinal)

- [ ] **Step 3: Implementar**

Revisar os testes antigos de `test_irr.py` (`test_irr_extreme_outlier`): se algum codifica o comportamento antigo (clamp `-0.99`, tolerância absoluta 0.1), corrigir para o valor certo e explicar no commit.

- [ ] **Step 4: Rodar e ver passar**

Run: `cd backend && .venv/bin/pytest engine -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/engine/irr.py backend/engine/tests/test_irr.py
git commit -m "🐛 fix(engine): XIRR ordenado, invariante à escala e com fallback por bisseção"
```

---

### Task 5: Taxa livre de risco = CDI do período

**Files:**
- Modify: `backend/engine/dataset.py` (`DatasetResult.selic_series` → `rf_series`, buscado como `series.get_series("CDI", start, end)`)
- Modify: `backend/engine/metrics.py` (`calculate_rolling_sharpe(df, rf_series=None, window=252)` — remove `rf_constant`)
- Modify: `backend/engine/compute.py` (`compute_risk_return(df_combined, rf_series=None)`; `compute_rolling_metrics` sem `rf_constant`)
- Modify: `backend/apps/api/analysis.py:190` (`AnalysisContext.selic` → `rf`), `backend/apps/api/routers/portfolio.py` (`/summary`, `/risk-return`, `/rolling-metrics` passam `ctx.rf`), `backend/engine/report_legacy.py:84`, `scripts/parity/engine_side.py`
- Test: `backend/engine/tests/test_compute.py`, `backend/engine/tests/test_metrics.py`

**Interfaces:**
- Produces: `compute_risk_return(df_combined, rf_series=None) -> DataFrame`, com `Sharpe = (CAGR − CAGR_rf) / Volatilidade`, em que `CAGR_rf` é o CAGR da série `rf_series` recortada ao índice de `df_combined` (mesma fórmula `365.25/dias`). Sem `rf_series` → coluna `Sharpe` NaN, mas a linha **permanece**. Volatilidade zero continua derrubando a linha. `calculate_rolling_sharpe` sem `rf_series` → `DataFrame()` vazio.

- [ ] **Step 1: Escrever os testes**

```python
def test_risk_return_uses_rf_cagr():
    # carteira e CDI com CAGR conhecidos -> Sharpe == (cagr - cagr_cdi) / vol
def test_risk_return_without_rf_keeps_row_with_nan_sharpe():
    rr = compute.compute_risk_return(df, rf_series=None)
    assert "Carteira" in rr.index and np.isnan(rr.loc["Carteira", "Sharpe"])
def test_rolling_sharpe_without_rf_is_empty():
    assert metrics.calculate_rolling_sharpe(df, rf_series=None).empty
def test_build_dataset_fetches_cdi_as_rf(monkeypatch):
    # fake get_series registra os nomes pedidos; espera "CDI" e não "SELIC"; res.rf_series is not None
```

Atualizar `test_risk_return_columns_and_sharpe_formula` e `test_sharpe_sign`, que hoje fixam 0.10.

- [ ] **Step 2: Rodar e ver falhar**

Run: `cd backend && .venv/bin/pytest engine -q`
Expected: FAIL (`TypeError` no novo parâmetro / `AttributeError: rf_series`)

- [ ] **Step 3: Implementar e renomear em todos os consumidores listados**

Run: `grep -rn "selic_series\|ctx.selic\|rf_constant" backend scripts/parity` — Expected: nenhuma ocorrência.

- [ ] **Step 4: Rodar e ver passar**

Run: `cd backend && .venv/bin/pytest -q && cd ../scripts/parity && ../../backend/.venv/bin/pytest -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A backend/engine backend/apps/api scripts/parity
git commit -m "🐛 fix(engine): Sharpe contra o CDI do período, sem taxa livre de risco fixa"
```

---

### Task 6: Oráculo DLP (teste `integration` contra o export da planilha)

**Files:**
- Create: `backend/engine/tests/test_oracle_dlp.py`

**Interfaces:**
- Consumes: `calculate_twr` (Task 1), `calculate_xirr` (Task 4), `flows.external_flows`.
- Fonte: `<repo>/dlp_sheet/google_sheets_DLP_html/dashboard..html` (aba oculta: tabela cuja linha de cabeçalho contém `date`, `sum vlrInvest`, `sum vlrMercado`, `sum proventos`, `sum lucro_trades`, `fluxo_caixa`, `TWR % a.m`, `TWR_acc`; datas no formato `2016/jul.`; números `1.234,56` e `2,09%`).

- [ ] **Step 1: Escrever o teste**

`pytestmark = pytest.mark.integration`; `pytest.skip` se o arquivo não existir. Parser com `re` + `html.unescape` da stdlib (sem dependência nova), mapeando `sum vlrInvest→vlr_investido`, `sum vlrMercado→vlr_mercado`, `sum proventos→proventos`, `sum lucro_trades→lucro`; data = fim do mês.

```python
def test_dlp_monthly_flow_matches(frame):
    assert flows.external_flows(g) == pytest.approx(frame["fluxo_caixa"], abs=0.05)
def test_dlp_monthly_twr_matches(frame):
    ret = calculate_twr(frame).pct_change().iloc[1:]
    assert (ret - frame["TWR % a.m"].iloc[1:] / 100).abs().max() < 0.0001   # 0,01 pp
def test_dlp_accumulated_twr_matches(frame):
    assert (calculate_twr(frame).iloc[-1] - 1) * 100 == pytest.approx(frame["TWR_acc"].iloc[-1], abs=0.05)
```

- [ ] **Step 2: Ver o oráculo falhar no engine antigo**

Run: `git worktree add ../_oraculo-antigo <sha do commit anterior à Task 1> && cp backend/engine/tests/test_oracle_dlp.py ../_oraculo-antigo/backend/engine/tests/ && (cd ../_oraculo-antigo/backend && /data/projetos/app-investimentos/backend/.venv/bin/pytest -m integration engine/tests/test_oracle_dlp.py -q); git worktree remove --force ../_oraculo-antigo`
Expected: FAIL em `test_dlp_monthly_twr_matches` e `test_dlp_accumulated_twr_matches` (o export da planilha é lido pelo caminho absoluto do repo principal; o parser deve resolver o caminho a partir de `settings.BASE_DIR.parent` ou de uma env `DLP_SHEET_DIR`, para funcionar no worktree).

- [ ] **Step 3: Rodar com o engine corrigido**

Run: `cd backend && .venv/bin/pytest -m integration engine/tests/test_oracle_dlp.py -q`
Expected: PASS (3 testes)

- [ ] **Step 4: Comparar a TIR (pendência da spec §3.2)**

Calcular `calculate_xirr` com `-fluxo` por mês + `vlr_mercado` final na última data e comparar com a TIR total da aba `dashboard` (`dashboard.html`, linha "TOTAL", formato `0,9% am11,1% aa`). Registrar as duas taxas a.a. para o ADR (Task 8). Diferença > 0,5 pp a.a.: **parar e reportar ao usuário** antes de seguir (pode ser outra convenção de dias ou de fluxo).

- [ ] **Step 5: Commit**

```bash
git add backend/engine/tests/test_oracle_dlp.py
git commit -m "✅ test(engine): oráculo do TWR mensal contra o export da planilha DLP"
```

---

### Task 7: Regressão contra a linha de base e nova linha de base da Fase 7

**Files:**
- Modify: `scripts/parity/cases.py` (`build_windows` ganha `Window("w4-fim-passado", first, <data no índice 3/4 das datas únicas>)`)
- Create: `scripts/parity/regress.py`
- Test: `scripts/parity/tests/test_cases.py`, `scripts/parity/tests/test_regress.py`

**Interfaces:**
- Consumes: `load_baseline`, `save_baseline`, `run_engine`, `compare_frames`, `METRICS` (existentes em `scripts/parity/`).
- Produces: `python -m parity.regress --baseline <dir> [--save <root>]` (rodado de `scripts/parity/`). Para cada caso da linha de base, mais `w4-fim-passado` (montado com o `hist` de `w1-completo` quando ausente), roda o engine, compara com tolerância 0 e imprime uma tabela `caso × métrica → colunas que mudaram` (ou `=`; `novo` para caso sem saída antiga). `--save` grava a nova linha de base com `save_baseline` em `<root>/baseline` reutilizando o **mesmo** snapshot e os mesmos `hist`.
- Função testável: `changed_columns(old: dict[str, DataFrame], new: dict[str, DataFrame]) -> dict[str, list[str]]`.

- [ ] **Step 1: Escrever os testes**

`test_build_windows_has_past_end_window`: com 8 datas mensais, `w4-fim-passado.end` é a 7ª data (índice `len*3//4`) e é `< last`. `test_changed_columns_reports_only_changed`: dois dicts sintéticos, só a coluna `Carteira` de `twr` difere → `{"twr": ["Carteira"]}` e as demais métricas ausentes.

- [ ] **Step 2: Rodar e ver falhar**

Run: `cd scripts/parity && ../../backend/.venv/bin/pytest -q`
Expected: FAIL (`w4-fim-passado` ausente; `ModuleNotFoundError: parity.regress`)

- [ ] **Step 3: Implementar e ver passar**

Run: `cd scripts/parity && ../../backend/.venv/bin/pytest -q`
Expected: PASS

- [ ] **Step 4: Rodar a regressão contra a linha de base da Fase 6 e conferir a promessa**

Run: `cd scripts/parity && ../../backend/.venv/bin/python -m parity.regress --baseline ../../reports/parity/20261027-182533/baseline`
Expected (a fase só pode mudar isto):
- `twr`, `drawdown`, `rolling_vol`, `yearly`: só a coluna `Carteira`.
- `rolling_sharpe`, `risk_return`: `Carteira` e as linhas/colunas de benchmarks (Sharpe passou a usar o CDI).
- `irr`, `shadow`, `kpis`: mudam (fluxo novo).
- `portfolio`: só a coluna `fluxo` (o ajuste de `vlr_investido` pelo `gap` já existia).
- Qualquer **coluna de benchmark** mudando em `twr`/`drawdown`/`rolling_vol`/`yearly` → parar e investigar.

- [ ] **Step 5: Gravar a nova linha de base**

Run: `cd scripts/parity && ../../backend/.venv/bin/python -m parity.regress --baseline ../../reports/parity/20261027-182533/baseline --save ../../reports/parity/$(date +%Y%m%d-%H%M%S)-fase7`
Expected: diretório `baseline/` com `manifest.json` incluindo `w4-fim-passado`. Rodar de novo apontando `--baseline` para ele → tudo `=`.

- [ ] **Step 6: Commit**

```bash
git add scripts/parity/cases.py scripts/parity/regress.py scripts/parity/tests
git commit -m "✅ test(parity): regressão por coluna e janela com fim no passado para a Fase 7"
```

---

### Task 8: ADR, documentação, rebuild e ponto de espera do usuário

**Files:**
- Create: `docs/adr/0018-metodologia-de-retorno-dlp.md`
- Modify: `docs/adr/README.md` (ou índice equivalente), `docs/consolidacao_arquitetura_e_migracao.md` (§2.2, §2.3, §5 Fase 7, §9 item 12), `backend/CLAUDE.md` (linha de base da Fase 7 nova), `backend/README.md` (contagem de testes)

- [ ] **Step 1: Escrever o ADR 0018**

Contexto (bug do lucro realizado herdado do legado); decisão (fórmula do fluxo, regra aporte-início/resgate-fim, XIRR Actual/365 com fallback, rf = CDI, KPIs de retorno total, shadow); evidência sem R$ (123/123 meses no fluxo, 122/122 no TWR mensal, acumulado idêntico, TIR da Task 6); consequências (todos os números da carteira mudam; nova linha de base em `reports/parity/<ts>-fase7/`).

- [ ] **Step 2: Rodar o checklist de `docs/README.md` (fechamento parcial: Fase 7a)**

- [ ] **Step 3: Verificação completa**

Run: `cd backend && .venv/bin/pytest -q && .venv/bin/python manage.py check && cd ../scripts/parity && ../../backend/.venv/bin/pytest -q && cd ../../backend && .venv/bin/pytest -m integration engine/tests/test_oracle_dlp.py -q`
Expected: tudo verde.

- [ ] **Step 4: Commit**

```bash
git add docs backend/CLAUDE.md backend/README.md
git commit -m "📝 docs: ADR 0018 (metodologia de retorno da DLP) e fechamento da Fase 7a"
```

- [ ] **Step 5: Rebuild do container no LAB**

Run: `cd /data/projetos/app-investimentos && docker compose up -d --build && docker compose ps`
Expected: serviços `Up (healthy)`; `curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/api/v1/auth/csrf` → `200`. Esse rebuild também leva `fad1dc8` (spec §5.1).

- [ ] **Step 6: PARAR e pedir ao usuário para testar**

Mensagem ao usuário: os números da carteira mudaram (TWR, TIR, ganho, shadow, Sharpe); comparar com a DLP; cada estranhamento vira teste antes da correção. **Não iniciar o plano 7b sem o retorno do usuário.**
