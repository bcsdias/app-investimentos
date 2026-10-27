# Fase 6 — Validação de paridade e remoção do legado — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provar, com um harness de duas camadas, que o engine/API novos entregam os mesmos números que o `FinancialReport` legado, congelar a linha de base para a Fase 7 e, após o "ok" do usuário, remover `src/` e as libs legadas.

**Architecture:** Um pacote `scripts/parity/` (módulos pequenos, testáveis com dados sintéticos) e um CLI fino `scripts/parity_check.py`. Camada 1: um `MarketSnapshot` lido **uma vez** do Postgres alimenta os dois lados — no legado via monkeypatch das funções `buscar_*` (formato bruto de cada uma), no novo via patch de `apps.market.repository.load_series` — mais um histórico DLP congelado. Camada 2: legado ao vivo × API HTTP real dos containers. Os módulos sem `src` (snapshot, engine, comparação, baseline) sobrevivem à remoção do legado e servem à Fase 7.

**Tech Stack:** Python 3.12 (`backend/.venv`), pandas, Django 5.2 (settings do backend), pytest + pytest-django, `requests`, matplotlib `Agg`.

**Spec:** [`docs/superpowers/specs/2026-10-27-fase-6-paridade-design.md`](../specs/2026-10-27-fase-6-paridade-design.md)

## Global Constraints

- **Nenhuma mudança** em `backend/engine/` nem em `backend/apps/market/` (nem para "fazer bater"). Divergência na camada 1 = parar e reportar ao usuário.
- Tolerâncias: camada 1 `1e-6` relativo, piso absoluto `1e-9`, NaN casa só com NaN; camada 2 `1e-4` relativo (acima disso: `INVESTIGAR`, não reprova).
- Token DLP **só** em `DLP_TOKEN` (env). Camada 2: `PARITY_API_URL` (default `http://127.0.0.1:8080`), `PARITY_USER`, `PARITY_PASSWORD`. Nunca impressos, logados ou gravados.
- Relatório em `reports/parity/<timestamp>/` (já no `.gitignore`). `summary.md` **sem valores em R$** (só razões, contagens e datas). Arquivos de detalhe e baseline podem ter valores da carteira e nunca são versionados.
- O harness não grava no Postgres, não limpa nem lê o Redis do LAB (cache Django = `DummyCache` no processo; `_ensure_fresh` = no-op; `upsert_points` levanta erro).
- Benchmarks fixos: `CDI`, `IPCA + 6%`, `S&P 500 BRL`, `IFIX`. Tesouro: fora (o `CATALOGO_TD` do legado está vazio e o catálogo novo não tem TD) — registrado no relatório. Sem carteiras compostas (decisão 4 da spec).
- Remoção do legado (Task 9) **só** após aprovação explícita do relatório pelo usuário; commits separados do harness.
- Sempre `backend/.venv/bin/python` / `backend/.venv/bin/pytest`. Testes do harness: `cd scripts/parity && ../../backend/.venv/bin/pytest -q`. Suíte do backend: `cd backend && .venv/bin/pytest -q`.
- Commits: gitmoji + Conventional Commits em português, imperativo; **sem** linha `Co-Authored-By`.

## Review Focus

1. **Ingestão rodando durante a execução** (cron 06:00/19:00 do `lab-appinvest-ingestion`) → os dois lados da camada 1 ainda veem os mesmos números, porque ambos leem do snapshot em memória e não do Postgres. Pinado em `test_engine_patches_never_touch_repository` (Task 3).
2. **Cache Redis do app já contém a série `market:series:CDI:...`** → o lado novo da camada 1 ignora esse cache e calcula do snapshot. Pinado em `test_engine_patches_use_dummy_cache` (Task 3).
3. **Selenium da B3 falha no host (ARM)** → `IFIX` some só do lado legado na camada 2 e aparece como `INVESTIGAR` com nota "só no novo", sem derrubar o resto do caso. Pinado em `test_column_only_on_one_side_gets_fail_status_and_note` (Task 1) e `test_run_case_layer2_legacy_missing_column` (Task 6).
4. **Valores da carteira vazando para `summary.md`** (p.ex. um KPI `patrimonio` de 123456.78 divergente) → o texto só mostra razão relativa, contagem e data. Pinado em `test_summary_never_contains_raw_values` (Task 7).
5. **Token rejeitado / API fora do ar na camada 2** → a camada 2 do caso vira uma nota de erro com o código HTTP e o `code` da API, sem o corpo nem o token, e a camada 1 continua valendo para o código de saída. Pinado em `test_api_client_error_has_no_body_or_token` (Task 6).

---

## File Structure

| Arquivo | Responsabilidade | Importa `src`? | Sobrevive à Task 9? |
|---|---|---|---|
| `scripts/parity/__init__.py` | Pacote. | não | sim |
| `scripts/parity/pytest.ini` | Config de testes do harness (Django settings + `pythonpath`). | — | sim |
| `scripts/parity/compare.py` | `SeriesDiff`, `compare_frames`, `classify_origin`, `apply_adr0012_recheck`. | não | sim |
| `scripts/parity/cases.py` | `Window`, `Case`, `BENCHMARKS`, `build_windows`, `pick_classe`, `build_cases`. | não | sim |
| `scripts/parity/frozen.py` | `MarketSnapshot`, `engine_patches`. | não | sim |
| `scripts/parity/engine_side.py` | `run_engine`, `engine_outputs`, `METRICS`. | não | sim |
| `scripts/parity/kpis.py` | `derive_kpis` (mesma regra do router `/summary`). | não | sim |
| `scripts/parity/baseline.py` | `save_baseline` / `load_baseline` (linha de base da Fase 7). | não | sim |
| `scripts/parity/legacy_side.py` | Fakes legados, `legacy_patches`, `run_legacy`, `legacy_outputs`, `legacy_yearly`, `pct_change_fill_none`, `freeze_history`. | **sim** | não |
| `scripts/parity/live.py` | `ApiClient`, conversores JSON → DataFrame, `api_outputs`. | não | não (sem legado não há camada 2) |
| `scripts/parity/report.py` | `CaseResult`, `write_summary`, `write_details`, `exit_code`. | não | não |
| `scripts/parity/bootstrap.py` | `sys.path`, Django, `Agg`, isolamento do legado (Upstash, log em arquivo). | **sim** | não |
| `scripts/parity_check.py` | CLI: orquestra camadas, relatório, baseline, teste integration. | **sim** | não |
| `scripts/parity/tests/` | `conftest.py`, `synthetic.py` (dados sintéticos), `test_*.py`. | parcial | só os src-free |
| `docs/adr/0017-paridade-legado.md` + docs | ADR e atualização do mestre (Task 10). | — | — |

Contrato comum de saída (os dois lados, as duas camadas): `dict[str, pd.DataFrame]` com as chaves de `engine_side.METRICS`:
`"twr"`, `"drawdown"`, `"rolling_vol"`, `"rolling_sharpe"`, `"risk_return"`, `"irr"`, `"shadow"`, `"yearly"`, `"kpis"`, `"portfolio"`.
Normalização: séries viram DataFrame de uma coluna (`irr` → coluna `"TIR"`); `yearly` com colunas `str` (`"2024"`, ..., `"Total Acum."`); `kpis` com índice = nome do KPI e coluna `"valor"`; `portfolio` = `portfolio_df.set_index("date")[["vlr_mercado","vlr_investido","proventos","fluxo"]]` (só camada 1); `risk_return` com índice = nome da série e colunas `"Volatilidade"`, `"Retorno (CAGR)"`, `"Sharpe"`.

---

### Task 1: Comparador numérico (`compare.py`) e infraestrutura de testes do harness

**Files:**
- Create: `scripts/parity/__init__.py`, `scripts/parity/pytest.ini`, `scripts/parity/compare.py`
- Create: `scripts/parity/tests/__init__.py`, `scripts/parity/tests/test_compare.py`

**Interfaces:**
- Produces:
  - `REL_TOL_L1 = 1e-6`, `REL_TOL_L2 = 1e-4`, `ABS_FLOOR = 1e-9`
  - `FALHA = "FALHA"`, `INVESTIGAR = "INVESTIGAR"`, `OK = "OK"`, `DESVIO = "DESVIO CONHECIDO (ADR 0012)"`
  - `@dataclass(frozen=True) class SeriesDiff: metric: str; column: str; status: str; max_rel: float; n_out: int; first_date: str | None; only_legacy: int; only_new: int; note: str = ""; origin: str = ""`
  - `compare_frames(metric: str, legacy: pd.DataFrame, new: pd.DataFrame, rel_tol: float, fail_status: str, abs_floor: float = ABS_FLOOR) -> list[SeriesDiff]`
  - `classify_origin(diff: SeriesDiff) -> str`
  - `ADR0012_METRICS = frozenset({"rolling_vol", "rolling_sharpe", "risk_return", "yearly"})`
  - `apply_adr0012_recheck(diffs: list[SeriesDiff], legacy_fill_none: dict[str, pd.DataFrame], new: dict[str, pd.DataFrame]) -> list[SeriesDiff]`

Regras de `compare_frames` (uma `SeriesDiff` por coluna da união de colunas):
- Coluna só de um lado → `status=fail_status`, `note="só no legado"` / `"só no novo"`, `max_rel=nan`, `n_out` = nº de linhas do lado presente.
- Coluna comum: rótulos do índice normalizados (`Timestamp` → `YYYY-MM-DD`, demais → `str`). Rótulos só de um lado contam em `only_legacy`/`only_new` **e** em `n_out`. Na interseção, ponto diverge se exatamente um é NaN, ou se `|a-b| > max(rel_tol * max(|a|,|b|), abs_floor)`. Dois NaN casam.
- `max_rel` = máximo de `|a-b| / max(|a|,|b|)` nos pares finitos com `max(|a|,|b|) > 0` (0.0 se nenhum). `first_date` = menor rótulo divergente (incluindo os de um lado só) ou `None`. `status = OK if n_out == 0 else fail_status`.

`classify_origin` (só camada 2): `only_legacy or only_new` → `"calendário/datas"`; coluna termina com `" BRL"` → `"conversão de câmbio"`; `metric in {"irr","kpis","portfolio"}` ou coluna em `{"Carteira","Carteira Real","Total Investido","TIR"}` → `"cálculo"`; senão `"revisão de cotação na fonte"`.

`apply_adr0012_recheck`: se **todas** as diffs não-OK têm `metric in ADR0012_METRICS`, recompara essas métricas com `legacy_fill_none` (tolerância da camada 1); se a recomparação ficar toda OK, as diffs originalmente não-OK viram `status=DESVIO` com `note` = `f"some com fill_method=None; max_rel original {max_rel:.3e}"` (mantém `max_rel`/`n_out` originais). Caso contrário devolve `diffs` intacto.

- [ ] **Step 1: Criar `scripts/parity/pytest.ini`**

```ini
[pytest]
DJANGO_SETTINGS_MODULE = config.settings
pythonpath = ../../backend ../.. ..
testpaths = tests
addopts = -p no:cacheprovider
```

(`../../backend` → `config`, `apps`, `engine`; `../..` → `src`; `..` → pacote `parity`.)

- [ ] **Step 2: Escrever os testes que falham em `tests/test_compare.py`**

```python
def test_identical_frames_are_ok():  # 2 colunas, 5 datas -> todas OK, max_rel == 0.0, first_date is None
def test_relative_diff_above_tol_fails():
    # legacy 100.0, new 100.0002 numa data (rel 2e-6) -> FALHA, n_out == 1, first_date == "2024-01-03"
def test_relative_diff_below_tol_is_ok():  # rel 5e-7 -> OK
def test_abs_floor_for_values_near_zero():  # 0.0 vs 5e-10 -> OK; 0.0 vs 5e-9 -> FALHA
def test_nan_matches_nan_only():  # NaN/NaN OK; NaN/1.0 FALHA
def test_index_mismatch_counts_as_divergence():
    # novo sem a data 2024-01-04 -> only_legacy == 1, n_out == 1, first_date == "2024-01-04"
def test_column_only_on_one_side_gets_fail_status_and_note():
    # "IFIX" só no novo, fail_status=INVESTIGAR -> status INVESTIGAR, note "só no novo"
def test_non_date_index_is_compared_by_label():  # risk_return com índice ["Carteira","CDI"]
def test_classify_origin_rules():  # um assert por regra, na ordem de precedência acima
def test_adr0012_recheck_marks_known_deviation():
    # diffs: rolling_vol FALHA; recheck frames iguais ao novo -> status DESVIO, max_rel original preservado
def test_adr0012_recheck_keeps_failure_when_other_metric_fails():  # twr FALHA também -> intacto
def test_adr0012_recheck_keeps_failure_when_recheck_still_differs():
```

- [ ] **Step 3: Rodar e ver falhar**

Run: `cd scripts/parity && ../../backend/.venv/bin/pytest tests/test_compare.py -q`
Expected: FAIL com `ModuleNotFoundError: No module named 'parity.compare'`.

- [ ] **Step 4: Implementar `compare.py` conforme as regras acima** (puro pandas/numpy; sem Django).

- [ ] **Step 5: Rodar e ver passar**

Run: `cd scripts/parity && ../../backend/.venv/bin/pytest tests/test_compare.py -q`
Expected: todos PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/parity/__init__.py scripts/parity/pytest.ini scripts/parity/compare.py scripts/parity/tests/
git commit -m "✅ test(parity): comparador numérico com tolerâncias da Fase 6"
```

---

### Task 2: Matriz de casos (`cases.py`)

**Files:**
- Create: `scripts/parity/cases.py`, `scripts/parity/tests/test_cases.py`

**Interfaces:**
- Produces:
  - `BENCHMARKS: tuple[str, ...] = ("CDI", "IPCA + 6%", "S&P 500 BRL", "IFIX")`
  - `NOT_COVERED: dict[str, str] = {"Tesouro": "CATALOGO_TD do legado vazio e catálogo novo sem TD", "Carteiras compostas": "perda aceita (decisão 4 da spec)"}`
  - `@dataclass(frozen=True) class Window: id: str; start: dt.date; end: dt.date`
  - `@dataclass(frozen=True) class Case: id: str; window: Window; classe: str | None = None; benchmarks: tuple[str, ...] = BENCHMARKS` com `api_payload(self) -> dict` = `{"start": iso, "end": iso, "ativo": None, "classe": self.classe, "benchmarks": list(self.benchmarks), "synthetics": []}`
  - `build_windows(dates: Iterable, today: dt.date) -> list[Window]`
  - `pick_classe(hist: pd.DataFrame) -> str | None`
  - `build_cases(hist: pd.DataFrame, today: dt.date) -> list[Case]`

Regras: `dates` → datas únicas ordenadas (`pd.to_datetime(...).normalize()`).
- `Window("w1-completo", first, last)`
- `Window("w2-12m", (pd.Timestamp(last) - pd.DateOffset(years=1)).date(), last)`
- `Window("w3-meio", uniq[len(uniq) // 2], today)`

`pick_classe`: escolhe a partir do histórico DLP congelado, a mesma fonte que o `/filters` lê (a spec cita o `/filters`, mas assim a camada 1 roda sem a API no ar): coluna `classe` (ou `tipo`, case-insensitive, como `CLASS_COLUMNS` do router) → classe com mais datas distintas; empate → ordem alfabética; sem coluna/sem valores → `None`.
`build_cases`: um `Case(window.id, window)` por janela + `Case("w1-completo-classe", w1, classe=pick_classe(hist))` se houver classe.

- [ ] **Step 1: Testes que falham** (`tests/test_cases.py`)

```python
def test_windows_from_unsorted_duplicated_dates():
    # datas 2023-03-01..2025-06-27 repetidas por 2 ativos, embaralhadas; today=2025-06-30
    # w1 == (2023-03-01, 2025-06-27); w2 == (2024-06-27, 2025-06-27); w3.start == mediana; w3.end == today
def test_pick_classe_most_dates_then_alphabetical():
def test_pick_classe_without_column_returns_none():
def test_build_cases_has_four_cases_with_classe_only_on_w1():
def test_api_payload_shape():  # chaves e ISO exatos; synthetics == []
```

- [ ] **Step 2: Rodar e ver falhar** — `cd scripts/parity && ../../backend/.venv/bin/pytest tests/test_cases.py -q` → `ModuleNotFoundError`.
- [ ] **Step 3: Implementar `cases.py`.**
- [ ] **Step 4: Rodar e ver passar** — mesmo comando, PASS.
- [ ] **Step 5: Commit**

```bash
git add scripts/parity/cases.py scripts/parity/tests/test_cases.py
git commit -m "✨ feat(parity): matriz de casos (janelas, filtro de classe, benchmarks)"
```

---

### Task 3: Snapshot de mercado e lado novo da camada 1 (`frozen.py`, `kpis.py`, `engine_side.py`)

**Files:**
- Create: `scripts/parity/frozen.py`, `scripts/parity/kpis.py`, `scripts/parity/engine_side.py`
- Create: `scripts/parity/tests/synthetic.py`, `scripts/parity/tests/test_engine_side.py`

**Interfaces:**
- Consumes: `Case` (Task 2).
- Produces:
  - `SNAPSHOT_KEYS: tuple[str, ...] = ("bcb:11", "bcb:12", "bcb:433", "yf:^GSPC", "ptax:USDBRL", "b3:IFIX")`
  - `@dataclass class MarketSnapshot: series: dict[str, pd.Series]` com
    - `@classmethod from_repository(cls, end: dt.date, keys: Iterable[str] = SNAPSHOT_KEYS) -> MarketSnapshot` — `repository.load_series(key, catalog.EPOCH, end)` para cada chave (chamada **antes** de qualquer patch).
    - `load(self, series_key: str, start: dt.date, end: dt.date) -> pd.Series` — mesmo contrato de `repository.load_series` (fatia inclusiva, `name=series_key`, série vazia com `DatetimeIndex([])` se não houver pontos); `KeyError` se a chave não está no snapshot.
    - `missing(self) -> list[str]` — chaves com série vazia.
  - `engine_patches(snapshot: MarketSnapshot, hist: pd.DataFrame)` (context manager): `apps.market.repository.load_series` → `snapshot.load`; `apps.market.repository.upsert_points` → levanta `RuntimeError("parity: escrita no Postgres proibida")`; `apps.market.series._ensure_fresh` → no-op; `apps.market.services.sources.dlp.buscar_historico` → `lambda *a, **k: hist.copy(deep=True)`; `override_settings(CACHES={"default": {"BACKEND": "django.core.cache.backends.dummy.DummyCache"}})`.
  - `derive_kpis(portfolio_df, twr: pd.DataFrame, risk_return: pd.DataFrame, drawdown: pd.DataFrame, irr: pd.Series) -> pd.DataFrame` — índice `["patrimonio_inicial","aportes_periodo","patrimonio","ganho_periodo","twr_total","cagr","volatilidade","sharpe","max_drawdown","xirr"]`, coluna `"valor"`, mesma derivação de `backend/apps/api/routers/portfolio.py:104-127` (`twr_total = twr["Carteira"].iloc[-1] - 100`; ausências → NaN).
  - `METRICS: tuple[str, ...] = ("twr","drawdown","rolling_vol","rolling_sharpe","risk_return","irr","shadow","yearly","kpis","portfolio")`
  - `engine_outputs(res: DatasetResult) -> dict[str, pd.DataFrame]` — via `engine.compute.compute_*` (rolling com `res.selic_series`, `window=252`), normalizado conforme o contrato do File Structure.
  - `run_engine(case: Case, snapshot: MarketSnapshot, hist: pd.DataFrame) -> dict[str, pd.DataFrame] | None` — dentro de `engine_patches`, reproduz `apps.api.analysis.run_analysis`: `fetch_user_portfolio("parity", classe=case.classe, start_date=iso, end_date=iso)` → `build_dataset(user_series, benchmarks=list(case.benchmarks), extra_specs=[], start_date=iso, end_date=iso, portfolio_df=...)` → `engine_outputs`. `None` se a carteira vier vazia.
  - `tests/synthetic.py`: `make_snapshot() -> MarketSnapshot` (dias úteis 2023-01-02..2025-06-30: `bcb:11`/`bcb:12` taxas diárias ~0.04 %, `bcb:433` taxa mensal no dia 1, `yf:^GSPC`/`b3:IFIX` níveis com passeio aleatório de semente fixa, `ptax:USDBRL` ~5.0) e `make_hist() -> pd.DataFrame` (colunas `date, ativo, corretora, classe, vlr_mercado, vlr_investido, proventos`; 2 ativos em classes `"Ações"` e `"Renda Fixa"`, dias úteis 2023-03-01..2025-06-27, aportes em 3 datas e proventos em 2). `TODAY = dt.date(2025, 6, 30)`.

- [ ] **Step 1: Testes que falham** (`tests/test_engine_side.py`)

```python
def test_snapshot_load_matches_repository_contract():  # fatia inclusiva, name, vazio com DatetimeIndex
def test_snapshot_load_unknown_key_raises():
def test_engine_patches_never_touch_repository(monkeypatch):
    # repository.load_series original trocado por um que levanta antes de entrar em engine_patches;
    # run_engine(case w1) devolve as 10 métricas sem chamá-lo
def test_engine_patches_use_dummy_cache():
    # django cache.set("market:x", 1) dentro do patch -> cache.get("market:x") is None
def test_engine_patches_block_writes():  # repository.upsert_points(...) -> RuntimeError
def test_run_engine_outputs_contract():
    # chaves == set(METRICS); twr tem colunas {"Carteira","CDI","IPCA + 6%","S&P 500 BRL","IFIX"};
    # irr.columns == ["TIR"]; kpis.columns == ["valor"]; yearly tem "Total Acum."
def test_derive_kpis_matches_summary_rule():  # valores calculados à mão num portfolio_df de 3 linhas
```

- [ ] **Step 2: Rodar e ver falhar** — `cd scripts/parity && ../../backend/.venv/bin/pytest tests/test_engine_side.py -q` → `ModuleNotFoundError`.
- [ ] **Step 3: Implementar `frozen.py`, `kpis.py`, `engine_side.py`, `tests/synthetic.py`.** Patches com `unittest.mock.patch` num `contextlib.ExitStack`.
- [ ] **Step 4: Rodar e ver passar** — mesmo comando, PASS.
- [ ] **Step 5: Commit**

```bash
git add scripts/parity/frozen.py scripts/parity/kpis.py scripts/parity/engine_side.py scripts/parity/tests/
git commit -m "✨ feat(parity): snapshot de mercado e lado novo da camada 1 isolado do LAB"
```

---

### Task 4: Linha de base da Fase 7 (`baseline.py`)

**Files:**
- Create: `scripts/parity/baseline.py`, `scripts/parity/tests/test_baseline.py`

**Interfaces:**
- Consumes: `Case`, `Window` (Task 2); `MarketSnapshot`, `run_engine`, `METRICS` (Task 3); `compare_frames` (Task 1).
- Produces:
  - `save_baseline(root: Path, cases: list[Case], snapshot: MarketSnapshot, hists: dict[str, pd.DataFrame], outputs: dict[str, dict[str, pd.DataFrame]]) -> Path` — grava em `root/"baseline"`: `manifest.json` (casos com datas ISO, `classe`, `benchmarks`, chaves de mercado → nome de arquivo, versão `1`), `market/<quote(key, safe="")>.pkl`, `hist/<case_id>.pkl`, `outputs/<case_id>/<metric>.pkl`. Formato pickle do pandas (sem `pyarrow`, que sai com o legado).
  - `@dataclass class Baseline: cases: list[Case]; snapshot: MarketSnapshot; hists: dict[str, pd.DataFrame]; outputs: dict[str, dict[str, pd.DataFrame]]`
  - `load_baseline(path: Path) -> Baseline` — `path` é o diretório `baseline/`.

- [ ] **Step 1: Testes que falham**

```python
def test_baseline_round_trip(tmp_path):  # casos, snapshot e outputs idênticos após load
def test_rerun_engine_from_baseline_reproduces_outputs(tmp_path):
    # run_engine sobre load_baseline(...) -> compare_frames(..., rel_tol=0.0, abs_floor=0.0) todo OK
def test_src_free_modules_import_without_src():
    # subprocess com backend/.venv/bin/python: sys.modules["src"] = None; django.setup();
    # import parity.compare, parity.cases, parity.frozen, parity.kpis, parity.engine_side, parity.baseline -> rc == 0
```

- [ ] **Step 2: Rodar e ver falhar** — `cd scripts/parity && ../../backend/.venv/bin/pytest tests/test_baseline.py -q` → `ModuleNotFoundError`.
- [ ] **Step 3: Implementar `baseline.py`.**
- [ ] **Step 4: Rodar e ver passar.**
- [ ] **Step 5: Commit**

```bash
git add scripts/parity/baseline.py scripts/parity/tests/test_baseline.py
git commit -m "✨ feat(parity): linha de base recarregável sem src para a Fase 7"
```

---

### Task 5: Lado legado e paridade da camada 1 sobre dados sintéticos (`legacy_side.py`)

**Files:**
- Create: `scripts/parity/legacy_side.py`, `scripts/parity/bootstrap.py`, `scripts/parity/tests/conftest.py`, `scripts/parity/tests/test_legacy_side.py`

**Interfaces:**
- Consumes: `MarketSnapshot`, `run_engine`, `engine_outputs`, `METRICS`, `derive_kpis` (Task 3); `Case`, `BENCHMARKS`, `build_cases` (Task 2); `compare_frames`, `apply_adr0012_recheck`, `REL_TOL_L1`, `FALHA` (Task 1).
- Produces:
  - `bootstrap.REPO_ROOT`, `bootstrap.BACKEND_DIR`; `bootstrap.configure() -> None` (insere `BACKEND_DIR`, `REPO_ROOT`, `REPO_ROOT/"scripts"` no `sys.path`; `matplotlib.use("Agg")`; `DJANGO_SETTINGS_MODULE=config.settings`; `django.setup()`; chama `isolate_legacy()`); `bootstrap.isolate_legacy() -> None` (importa `src.data.cache` — que roda `load_dotenv()` — e **depois** remove `UPSTASH_REDIS_REST_URL`/`UPSTASH_REDIS_REST_TOKEN` do `os.environ`; `src.data.cache._get_redis_client = lambda: None`; remove todo `RotatingFileHandler` de `src.utils.logger.logger.handlers` para não persistir log em `log/`).
  - `LEGACY_INDEX_CODES = (12466, 12467, 12468, 12461, 12463, 12464, 12469, 12462)` — cópia de `src/data/sources/market_data.py:397` (comentário apontando a origem).
  - `legacy_patches(snapshot: MarketSnapshot, hist: pd.DataFrame)` (context manager) que troca, em `src.data.sources.market_data`, as funções abaixo, e `src.engine.financial_report.buscar_historico`:

    | Função legada | Fake devolve |
    |---|---|
    | `buscar_historico(token, ativo=None, classe=None, corretora=None)` | `hist.copy(deep=True)` |
    | `buscar_dados_bcb(codigo, start, end, nome=None)` | `s = snapshot.load(f"bcb:{codigo}", EPOCH, max)`; se `codigo in LEGACY_INDEX_CODES`: `s.ffill()`; senão `((s / 100) + 1).cumprod().fillna(1)`; `index.name = "Date"`; `.loc[start:end]`; vazio → `None` |
    | `buscar_dados_benchmark(ticker, start, end, nome=None)` | `snapshot.load(f"yf:{ticker}", ...)` com máscara `start <= idx <= end`; vazio → `None` |
    | `buscar_dados_b3(indice, start, end)` | idem com `b3:{indice}` |
    | `buscar_dolar_bcb(start, end)` | `snapshot.load("ptax:USDBRL", start, end)`; vazio → `None` |
    | `buscar_dados_tesouro(...)` | `None` |

    (Fakes reproduzem o caminho de *cache hit* do legado: série cheia desde 1995, depois fatiada.)
  - `run_legacy(token: str, case: Case) -> tuple[FinancialReport, dict[str, pd.DataFrame]] | None` — exatamente a sequência de `src/ui/pages/4_configuracoes.py:123-134`: `FinancialReport()`, `fetch_user_portfolio(token, classe=case.classe, start_date=iso, end_date=iso)`, `build_dataset(user_series=..., active_benchmarks=list(case.benchmarks), start_date=iso, end_date=iso)`; `None` se `user_series is None`.
  - `legacy_outputs(report) -> dict[str, pd.DataFrame]` — oráculos dos `plot_*(return_fig=True)` (`plot_rolling_volatility(252)`, `plot_rolling_sharpe(252)`), cada figura fechada com `plt.close(fig)`; `yearly = legacy_yearly(report.df_combined)`; `kpis = derive_kpis(...)` sobre os oráculos; mesma normalização do contrato.
  - `legacy_yearly(df_combined: pd.DataFrame) -> pd.DataFrame` — cópia literal de `src/engine/financial_report.py:406-413` sem o `.map` nem o `export_csv`:

    ```python
    # Cópia literal de src/engine/financial_report.py:406-413 (generate_summary_table),
    # parando antes do .map de formatação — o legado não devolve esses números.
    df = df_combined
    total_ret = (df.iloc[-1] / df.iloc[0]) - 1
    yearly = df.resample('YE').last()
    yearly_ret = yearly.pct_change()
    yearly_ret.iloc[0] = (yearly.iloc[0] / df.iloc[0]) - 1
    summary = yearly_ret.T
    summary.columns = [c.year for c in summary.columns]
    summary['Total Acum.'] = total_ret
    ```
  - `pct_change_fill_none()` (context manager): troca `pd.DataFrame.pct_change` e `pd.Series.pct_change` por wrappers que forçam `fill_method=None`.
  - `legacy_metrics_fill_none(report) -> dict[str, pd.DataFrame]` — dentro de `pct_change_fill_none()`, recalcula só as chaves de `ADR0012_METRICS`.
  - `freeze_history(token: str, classe: str | None = None) -> pd.DataFrame` — `src.data.sources.market_data.buscar_historico(token, classe=classe)` **original** (filtro no servidor DLP, como o legado fazia); levanta `RuntimeError("DLP sem histórico")` se vazio.
  - `run_layer1(case: Case, snapshot: MarketSnapshot, hist: pd.DataFrame) -> tuple[list[SeriesDiff], dict, dict]` — `legacy_patches` + `run_legacy("parity", case)` × `run_engine(case, snapshot, hist)`; `compare_frames(..., REL_TOL_L1, FALHA)` por métrica; `apply_adr0012_recheck` com `legacy_metrics_fill_none`. Devolve `(diffs, legacy_out, new_out)`.
  - `tests/conftest.py`: chama `bootstrap.isolate_legacy()` uma vez (sessão).

- [ ] **Step 1: Testes que falham** (`tests/test_legacy_side.py`)

```python
def test_benchmarks_exist_in_both_catalogs():
    # catalog.resolve(n) para cada BENCHMARKS; legado: "CDI"/"IPCA" em CATALOGO_BCB, "IFIX" em CATALOGO_B3, "S&P 500" em CATALOGO_YF
def test_isolate_legacy_disables_upstash_and_file_log():
    # _get_redis_client() is None; nenhum RotatingFileHandler em src.utils.logger.logger.handlers
def test_fake_bcb_rate_matches_legacy_accumulation():  # bcb:12 -> cumprod de (taxa/100+1), index.name == "Date"
def test_fake_bcb_index_code_returns_level():  # código 12466 com chave no snapshot -> nível ffill
def test_legacy_yearly_matches_engine_on_ffilled_frame():
def test_layer1_parity_on_synthetic_data():
    # para cada caso de build_cases(make_hist(), TODAY) (hist filtrado por classe no caso -classe):
    # run_layer1 -> todas as diffs com status == "OK"
def test_layer1_detects_injected_difference():
    # run_legacy com legacy_patches(make_snapshot()) × run_engine com uma cópia do snapshot em que
    # bcb:12 tem taxa * 1.001 a partir de 2024-01-02 -> compare_frames de twr marca "CDI" como FALHA
```

- [ ] **Step 2: Rodar e ver falhar** — `cd scripts/parity && ../../backend/.venv/bin/pytest tests/test_legacy_side.py -q` → `ModuleNotFoundError: parity.legacy_side`.
- [ ] **Step 3: Implementar `bootstrap.py`, `legacy_side.py`, `tests/conftest.py`.**
- [ ] **Step 4: Rodar e ver passar.**

Run: `cd scripts/parity && ../../backend/.venv/bin/pytest -q`
Expected: todos PASS. **Se `test_layer1_parity_on_synthetic_data` falhar por diferença real de cálculo (não de harness), parar e reportar ao usuário com métrica, coluna, `max_rel` e `first_date` — não alterar `backend/engine/` nem `apps/market/` (Global Constraints).**

- [ ] **Step 5: Commit**

```bash
git add scripts/parity/legacy_side.py scripts/parity/bootstrap.py scripts/parity/tests/
git commit -m "✨ feat(parity): lado legado e camada 1 com mercado congelado"
```

---

### Task 6: Camada 2 — API HTTP real × legado ao vivo (`live.py`)

**Files:**
- Create: `scripts/parity/live.py`, `scripts/parity/tests/test_live.py`
- Modify: `scripts/parity/legacy_side.py` (adiciona `run_layer2`)

**Interfaces:**
- Consumes: `Case.api_payload` (Task 2); `run_legacy`, `legacy_outputs` (Task 5); `compare_frames`, `classify_origin`, `REL_TOL_L2`, `INVESTIGAR` (Task 1).
- Produces:
  - `class ApiError(RuntimeError)` com `status: int`, `code: str | None`; `str()` = `f"HTTP {status} ({code})"` — nunca o corpo nem headers.
  - `class ApiClient: __init__(self, base_url: str, session: requests.Session | None = None)`; `login(self, username: str, password: str) -> None` (`GET /api/v1/auth/csrf`, depois `POST /api/v1/auth/login` com header `X-CSRFToken` = cookie `csrftoken`); `post(self, path: str, payload: dict) -> dict` (mesmo header; `!= 200` → `ApiError` com `code` lido do envelope `{"code": ...}` se JSON).
  - `ENDPOINTS = {"twr": "/api/v1/portfolio/twr-evolution", "drawdown": "/api/v1/portfolio/drawdown", "rolling": "/api/v1/portfolio/rolling-metrics", "risk_return": "/api/v1/portfolio/risk-return", "irr": "/api/v1/portfolio/irr-evolution", "shadow": "/api/v1/portfolio/shadow", "yearly": "/api/v1/portfolio/yearly-summary", "summary": "/api/v1/portfolio/summary"}`
  - Conversores: `series_list_to_frame(items: list[dict]) -> pd.DataFrame` (`{"name", "data": [[iso, v|null]]}` → colunas por `name`, `DatetimeIndex`, `null` → NaN); `rows_to_frame(rows) -> pd.DataFrame` (reverte `_RISK_RETURN_COLUMNS`: `volatilidade`→`"Volatilidade"`, `retorno_cagr`→`"Retorno (CAGR)"`, `sharpe`→`"Sharpe"`, índice = `name`); `yearly_to_frame(data) -> pd.DataFrame` (colunas `str(ano)` + `"Total Acum."`); `summary_to_kpis(data) -> pd.DataFrame` (mesmo índice de `derive_kpis`).
  - `api_outputs(client: ApiClient, case: Case) -> dict[str, pd.DataFrame]` — todas as chaves de `METRICS` exceto `"portfolio"`; `rolling` com `payload | {"window": 252}` → `rolling_vol`/`rolling_sharpe`; `irr` coluna `"TIR"`.
  - `run_layer2(case: Case, token: str, client: ApiClient) -> tuple[list[SeriesDiff], list[str], dict, dict]` (em `legacy_side.py`) — `run_legacy(token, case)` sem patches × `api_outputs`; `compare_frames(..., REL_TOL_L2, INVESTIGAR)`; cada diff não-OK recebe `origin=classify_origin(d)` (`dataclasses.replace`). Erros (`ApiError`, legado `None`) viram notas (`list[str]`) e diffs vazias.

- [ ] **Step 1: Testes que falham** (`tests/test_live.py`, com uma `FakeSession` que grava requisições e devolve respostas montadas)

```python
def test_login_sends_csrf_header_from_cookie():
def test_post_sends_csrf_header_and_returns_json():
def test_api_client_error_has_no_body_or_token():
    # resposta 422 {"code": "dlp_token_rejected", "message": "...", "details": {"x": "DLP-segredo"}}
    # -> ApiError.status == 422, .code == "dlp_token_rejected", "DLP-segredo" not in str(exc)
def test_series_list_to_frame_handles_nulls():
def test_rows_to_frame_restores_legacy_column_names():
def test_yearly_to_frame_columns():  # ["2024", "2025", "Total Acum."]
def test_summary_to_kpis_index_matches_derive_kpis():
def test_run_case_layer2_legacy_missing_column(monkeypatch):
    # run_legacy devolve twr sem "IFIX"; api devolve com "IFIX" -> diff INVESTIGAR, note "só no novo", origin preenchida
```

- [ ] **Step 2: Rodar e ver falhar** — `cd scripts/parity && ../../backend/.venv/bin/pytest tests/test_live.py -q` → `ModuleNotFoundError: parity.live`.
- [ ] **Step 3: Implementar `live.py` e `run_layer2`.**
- [ ] **Step 4: Rodar e ver passar** — `cd scripts/parity && ../../backend/.venv/bin/pytest -q`, tudo PASS.
- [ ] **Step 5: Commit**

```bash
git add scripts/parity/live.py scripts/parity/legacy_side.py scripts/parity/tests/test_live.py
git commit -m "✨ feat(parity): camada 2 contra a API em execução e o legado ao vivo"
```

---

### Task 7: Relatório e CLI (`report.py`, `scripts/parity_check.py`)

**Files:**
- Create: `scripts/parity/report.py`, `scripts/parity_check.py`, `scripts/parity/tests/test_report.py`

**Interfaces:**
- Consumes: tudo das Tasks 1–6.
- Produces:
  - `@dataclass class CaseResult: case: Case; layer: int; diffs: list[SeriesDiff]; notes: list[str]`
  - `write_summary(out_dir: Path, results: list[CaseResult], meta: dict) -> Path` — `summary.md` com: cabeçalho (timestamp, `git rev-parse --short HEAD`, tolerâncias), tabela de casos (id, janela ISO, classe, benchmarks), `NOT_COVERED`, `snapshot.missing()`, uma tabela por caso × camada (`métrica | coluna | status | max_rel (%.3e) | pontos fora | 1ª data divergente | origem | nota`), seção "Teste integration (Fase 3c)" com `meta["integration"]`, e total de FALHA/DESVIO/INVESTIGAR. Linhas OK colapsadas numa contagem por métrica.
  - `write_details(out_dir: Path, case_id: str, layer: int, legacy: dict, new: dict) -> None` — `details/<case_id>/L<layer>/<metric>.legacy.csv` e `.new.csv`.
  - `exit_code(results: list[CaseResult]) -> int` — `1` se alguma diff de camada 1 tem status `FALHA` ou `DESVIO`; senão `0`.
  - CLI `scripts/parity_check.py [--layers 1,2] [--out reports/parity] [--today YYYY-MM-DD] [--with-integration]`:
    1. `bootstrap.configure()`; `DLP_TOKEN` ausente → mensagem e `sys.exit(2)`; camada 2 sem `PARITY_USER`/`PARITY_PASSWORD` → mensagem e `sys.exit(2)`.
    2. `snapshot = MarketSnapshot.from_repository(end=today)`; `hist_full = freeze_history(token)`; `cases = build_cases(hist_full, today)`; hist por caso = `hist_full` ou `freeze_history(token, case.classe)`.
    3. Camada 1: `run_layer1` por caso → `CaseResult(layer=1)` + `write_details`; `save_baseline(out_dir, cases, snapshot, hists, new_outputs)`.
    4. Camada 2: `ApiClient(PARITY_API_URL).login(...)`; `run_layer2` por caso.
    5. `--with-integration`: `subprocess.run([BACKEND_DIR/".venv/bin/pytest", "-m", "integration", "apps/api/tests/test_integration.py::test_real_dlp_in_memory_filter_matches_server_filter", "-q", "-p", "no:cacheprovider"], cwd=BACKEND_DIR, env=os.environ | {"DLP_TEST_TOKEN": token}, capture_output=True, text=True)`; grava em `meta["integration"]` só o código de retorno e a **última linha** do stdout (ex.: `1 passed in 4.10s`) — nunca o resto da saída.
    6. `write_summary`; imprime o caminho do relatório e a contagem por status; `sys.exit(exit_code(results))`.
  - Exceções no CLI: imprimir `type(exc).__name__` e o código/caso; nunca `repr` de argumentos.

- [ ] **Step 1: Testes que falham** (`tests/test_report.py`)

```python
def test_summary_never_contains_raw_values(tmp_path):
    # diffs de kpis/patrimonio com dados de 123456.78 vs 123999.00 -> "123456" e "123999" ausentes do summary.md
def test_summary_lists_not_covered_and_missing_snapshot_keys(tmp_path):
def test_summary_has_integration_line(tmp_path):
def test_exit_code_rules():
    # só INVESTIGAR na camada 2 -> 0; FALHA na camada 1 -> 1; DESVIO na camada 1 -> 1
def test_cli_without_token_exits_2(tmp_path):
    # subprocess backend/.venv/bin/python scripts/parity_check.py com env sem DLP_TOKEN -> rc == 2,
    # nenhum diretório criado em --out
```

- [ ] **Step 2: Rodar e ver falhar** — `cd scripts/parity && ../../backend/.venv/bin/pytest tests/test_report.py -q`.
- [ ] **Step 3: Implementar `report.py` e `scripts/parity_check.py`.**
- [ ] **Step 4: Rodar e ver passar** — `cd scripts/parity && ../../backend/.venv/bin/pytest -q`, tudo PASS; e `cd backend && .venv/bin/pytest -q` continua verde (nada do backend mudou).
- [ ] **Step 5: Commit**

```bash
git add scripts/parity/report.py scripts/parity_check.py scripts/parity/tests/test_report.py
git commit -m "✨ feat(parity): relatório e CLI do harness de paridade"
```

---

### Task 8: Execução real e checkpoint com o usuário

Sem código novo. Esta task produz o relatório e **para** para aprovação.

- [ ] **Step 1: Pré-condições.** `docker compose ps` mostra `lab-appinvest-web`/`backend` saudáveis; `curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/api/v1/auth/csrf` → `204`. O usuário confirma que a conta `PARITY_USER` tem o token DLP cadastrado. Se a camada 2 não puder rodar, seguir com `--layers 1` e registrar.

- [ ] **Step 2: Rodar o harness** (o usuário exporta as variáveis no próprio shell; o agente não escreve o token em nenhum comando nem arquivo):

```bash
cd /data/projetos/app-investimentos
# export DLP_TOKEN=... PARITY_USER=... PARITY_PASSWORD=...   (feito pelo usuário)
backend/.venv/bin/python scripts/parity_check.py --layers 1,2 --with-integration
```

Expected: imprime `reports/parity/<timestamp>/summary.md`; código de saída `0` (camada 1 toda OK).

- [ ] **Step 3: Conferir vazamento.** `grep -rn "DLP-" reports/parity/<timestamp>/summary.md` → nada; `git status --short` → nada de `reports/` nem `log/`.

- [ ] **Step 4: Triagem.** Camada 1 com `FALHA`: **parar** e reportar ao usuário (métrica, coluna, `max_rel`, 1ª data), sem tocar em `engine/`/`apps/market/`. `DESVIO CONHECIDO`: apresentar a magnitude para o usuário decidir. Camada 2 com `INVESTIGAR`: para cada linha, abrir os CSVs de `details/` e confirmar ou corrigir a origem provável; anotar a conclusão (sem R$) para a ADR.

- [ ] **Step 5: Checkpoint.** Apresentar ao usuário o `summary.md` e a lista de divergências com a explicação de cada uma. **Aguardar "ok" explícito** antes da Task 9. Registrar no fim do `summary.md` (seção "Aprovação") a data e as divergências aceitas.

---

### Task 9: Remoção do legado (só após o "ok" da Task 8)

**Files:**
- Delete: `src/`, `redis.py`, `requirements.txt` (raiz), `.streamlit/`, `scripts/sync_to_redis.py`, `scripts/update_b3_cache.py`, `scripts/verify_paths.py`, `scripts/verify_fix.py`, `.github/workflows/update_b3.yml` (roda `scripts/update_b3_cache.py`), `scripts/parity_check.py`, `scripts/parity/legacy_side.py`, `scripts/parity/live.py`, `scripts/parity/report.py`, `scripts/parity/bootstrap.py`, `scripts/parity/tests/{conftest.py,test_legacy_side.py,test_live.py,test_report.py}`
- Delete (após verificação): `tests/` (raiz), `ativos.csv` (raiz), `.gemini/antigravity/**/scratch/`, `tree.txt`, `deployment_guide.md`
- Não tocar: `data/` (o `backfill_market_data` lê os CSVs legados via `catalog.legacy_csv_name`).

- [ ] **Step 1: Verificar antes de apagar.**
  - `grep -rn "from src\|import src\|src\." backend frontend/src --include=*.py --include=*.ts --include=*.tsx` → nada.
  - `grep -rn "ativos.csv\|tree.txt\|deployment_guide\|update_b3_cache\|sync_to_redis" backend frontend docker-compose.yml .dockerignore` → só `backend/apps/market/data/ativos.csv` (`settings.MARKET_ASSETS_CSV`), que fica.
  - Testes da raiz portados: `grep -h "^def test" tests/*.py | sort` × `grep -h "^def test" backend/engine/tests/test_{irr,metrics,twr}.py | sort` → mesma lista (12 testes). Se faltar algum, portar antes de apagar `tests/`.
  - `tree.txt` e `deployment_guide.md`: `git log --format='%h %an %s' -- <arquivo>`; se não for claramente scratch/legado, perguntar ao usuário antes.

- [ ] **Step 2: Commit — remover o legado e o harness dependente de `src`**

```bash
git rm -r src redis.py requirements.txt .streamlit scripts/sync_to_redis.py scripts/update_b3_cache.py \
  scripts/verify_paths.py scripts/verify_fix.py .github/workflows/update_b3.yml scripts/parity_check.py \
  scripts/parity/legacy_side.py scripts/parity/live.py scripts/parity/report.py scripts/parity/bootstrap.py \
  scripts/parity/tests/conftest.py scripts/parity/tests/test_legacy_side.py scripts/parity/tests/test_live.py \
  scripts/parity/tests/test_report.py
git commit -m "🔥 chore: remove o legado Streamlit e o harness dependente de src após a paridade"
```

- [ ] **Step 3: Rodar o que sobrou do harness sem `src`.** `cd scripts/parity && ../../backend/.venv/bin/pytest -q` → PASS (inclui `test_src_free_modules_import_without_src` e a releitura da baseline). Remover de `.dockerignore` as linhas `src`, `tests`, `.streamlit` se os diretórios não existem mais.

- [ ] **Step 4: Commit — resíduos verificados no Step 1**

```bash
git rm -r tests ativos.csv .gemini/antigravity tree.txt deployment_guide.md   # só os confirmados
git add .dockerignore
git commit -m "🔥 chore: remove testes portados e resíduos do legado"
```

- [ ] **Step 5: Desinstalar libs só do legado em `backend/.venv` (não versionado).**

```bash
cd backend
.venv/bin/pip uninstall -y streamlit altair pydeck supabase supabase-auth supabase-functions postgrest storage3 realtime upstash-redis webdriver-manager
.venv/bin/pip install -r requirements-dev.txt     # repõe qualquer dependência compartilhada
.venv/bin/pip check
.venv/bin/python manage.py check
.venv/bin/pytest -q
```

Expected: `pip check` sem conflitos; `System check identified no issues`; suíte verde com a mesma contagem de antes. (`pyarrow` pode ficar se outra lib o exigir; o harness não depende dele.)

---

### Task 10: Documentação e fechamento da fase

**Files:**
- Create: `docs/adr/0017-paridade-legado.md`
- Modify: `docs/adr/README.md`, `docs/consolidacao_arquitetura_e_migracao.md` (§2.0, §2.1, §2.2, §2.3, §5 Fase 6, §8.1, §8.2, §9 item 11), `docs/README.md`, `backend/README.md`, `backend/CLAUDE.md`, `README.md` (raiz)

- [ ] **Step 1: ADR 0017** — método (duas camadas; snapshot em memória para os dois lados, em vez de cada lado ler o Postgres, para imunizar contra a ingestão concorrente; tolerâncias `1e-6`/`1e-4`, piso `1e-9`), matriz (janelas com datas, classe, benchmarks, `NOT_COVERED`), resultado resumido **sem R$**, divergências aceitas com a origem, perda aceita das carteiras compostas (decisão 4), desvio ADR 0012 (se apareceu), onde fica a baseline e como a Fase 7 a usa (`load_baseline` + `run_engine` + `compare_frames`). Linha no índice `docs/adr/README.md`.
- [ ] **Step 2: Mestre** — §2.0 (Fase 6 concluída em `<data>`), §2.1 (legado removido), §2.2/§2.3 (itens resolvidos), §5 Fase 6 (resultado + link para a ADR), §8.1 (comando com `DLP_TOKEN`/`PARITY_*` no lugar de `--token`), §8.2 (marcar o item de paridade), §9 item 11 `[x]`.
- [ ] **Step 3: Checklist de `docs/README.md`** ("Checklist de atualização ao fechar uma fase"), `backend/README.md` (Estado, Estrutura, contagem de testes) e `backend/CLAUDE.md` (remover "o `src/` coexiste e permanece funcional até a Fase 6"; citar `scripts/parity/` como baseline da Fase 7). README da raiz sem instruções de Streamlit.
- [ ] **Step 4: Verificação final** — `cd backend && .venv/bin/pytest -q && .venv/bin/python manage.py check`; `cd scripts/parity && ../../backend/.venv/bin/pytest -q`; `grep -rni "streamlit" README.md backend/README.md backend/CLAUDE.md` só em contexto histórico.
- [ ] **Step 5: Commit**

```bash
git add docs backend/README.md backend/CLAUDE.md README.md
git commit -m "📝 docs: fecha a Fase 6 (ADR 0017, doc mestre, READMEs)"
```
