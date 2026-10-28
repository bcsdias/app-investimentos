# Fase 7b — Integridade dos dados de mercado Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dados de mercado sem cache defasado e sem buracos silenciosos: a ingestão invalida o cache, re-busca lacunas no meio das séries, e o buraco que a fonte não tem aparece para o usuário (nunca interpolado).

**Architecture:**
- **Cache:** uma geração global de séries no Redis entra em todas as chaves `market:series:*`. `repository.upsert_points`, o único ponto de escrita, incrementa essa geração. Com isso, séries cruas, derivadas (`S&P 500 BRL` ← S&P × PTAX, `IMA-S` splice), sintéticas (`IPCA + 6%`, `CDI + 2%`) e sob demanda deixam de ser servidas do cache antigo assim que um ponto novo é gravado.
- **Lacunas:** um módulo puro `apps/market/gaps.py` detecta lacunas pela cadência da série. Ele alimenta três consumidores:
  - o comando de diagnóstico;
  - a re-busca na ingestão;
  - o aviso aditivo `meta.gaps` da API, que a SPA mostra no `PeriodNotice`.
- **Abertura dos benchmarks:** passam a abrir no último valor conhecido até a data de abertura da carteira, pendência da Fase 7a.

**Tech Stack:** Python 3.12, Django 5 + django-ninja, django-redis, pandas, pytest; React + Vitest no `frontend/`.

**Spec:** [`docs/superpowers/specs/2026-10-28-fase-7-confiabilidade-design.md`](../specs/2026-10-28-fase-7-confiabilidade-design.md): §3.3 (cache e ingestão), §5.3.1 (linha "Buraco de dado de mercado"), §5.4, §5.6, §5.8. Fase anterior: [plano 7a](2026-10-28-fase-7a-engine.md), [ADR 0018](../../adr/0018-metodologia-de-retorno-dlp.md).

## Global Constraints

- TDD conforme `backend/CLAUDE.md`: escrever o teste, rodar e **ver falhar**, implementar, ver passar.
- Commits: um por task, Conventional Commits + Gitmoji, imperativo em português, **nunca** `Co-Authored-By`.
- Buraco de dado de mercado: `ffill` (último valor conhecido) **mais** aviso aditivo em `meta` com série e intervalo; **nunca** interpolação.
- Contrato da API v1 ([ADR 0013](../../adr/0013-contrato-da-api-v1.md)) só com mudanças aditivas: `meta.start`, `meta.end` e `meta.missing` intocados.
- O cache é otimização: toda falha de Redis é logada e engolida (padrão de `apps/market/cache.py`). Falha de cache **nunca** pode fazer servir série antiga.
- Tolerância de lacuna por cadência: `daily` → intervalo > **10** dias corridos entre pontos consecutivos; `monthly` → > **45** dias.
- Uma série falhar ou ficar com lacuna **não** aborta a ingestão. Lacuna sem dado na fonte **não** é falha (não entra no `exit 1`).
- `reports/` e `dlp_sheet/` nunca versionados; relatórios, ADR e commits sem valores em R$ da carteira.
- Rodar a partir de `backend/`: `.venv/bin/pytest -q`. Paridade: `cd scripts/parity && ../../backend/.venv/bin/pytest -q`. SPA: `cd frontend && npm test -- --run`.
- Linha de base vigente: `reports/parity/20261028-180012-fase7-abertura/baseline`.

## Review Focus

1. A chave de geração foi despejada pelo LRU ou o Redis reiniciou: a geração nova não pode coincidir com uma antiga, para nada antigo voltar a ser servido. O valor inicial é `time.time_ns()`. Teste na Task 1.
2. Feriados longos (Natal/Ano-Novo, Carnaval, feriados de Londres no `IMID.L`) **não** viram lacuna numa série diária: 9 dias sem ponto não é lacuna. Teste na Task 2.
3. Série vazia ou com 1 ponto: sem lacuna e sem exceção, tanto no diagnóstico quanto no `meta`. Teste na Task 2.
4. A re-busca de uma lacuna levanta `FetchError` ou volta vazia: a série não é marcada como falha, o código de saída não muda e o buraco aparece no log como "sem dado na fonte". Teste na Task 3.
5. Lacuna que começa antes da abertura e termina dentro da janela aparece no `meta` com as datas reais. Lacuna inteira fora da janela não aparece. Teste na Task 5.

---

### Task 1: Geração de séries no cache, invalidada a cada escrita

**Files:**
- Modify: `backend/apps/market/cache.py` (geração), `backend/apps/market/series.py:27-40` (`get_series`), `backend/apps/market/repository.py:20-61` (`upsert_points`)
- Test: `backend/apps/market/tests/test_cache.py`, `backend/apps/market/tests/test_series.py`, `backend/apps/market/tests/test_repository.py`

**Interfaces:**
- Produces:
  - `cache.GEN_KEY = "market:series:gen"`
  - `cache.series_generation() -> int`: lê a geração. Se ela não existir, cria com `cache.add(GEN_KEY, time.time_ns(), timeout=None)` e relê. Se o Redis falhar, devolve `0`.
  - `cache.bump_series_generation() -> None`: `cache.incr(GEN_KEY)`. Chave ausente (`ValueError`) → `cache.add(GEN_KEY, time.time_ns(), None)`. Outras falhas são logadas e engolidas.
  - Chave das séries: `cache.make_key("series", f"g{gen}", name, start, end)`.
  - `repository.upsert_points(...)` chama `cache.bump_series_generation()` quando grava ≥ 1 linha.

- [ ] **Step 1: Escrever os testes**

Usar a fixture `_locmem_cache` já existente em `test_series.py`. Em `test_cache.py`, usar a fixture `locmem_cache` do `conftest.py`.

```python
# test_cache.py
def test_generation_starts_from_time_ns_when_missing(locmem_cache, monkeypatch):
    monkeypatch.setattr(mc.time, "time_ns", lambda: 1_000)
    assert mc.series_generation() == 1_000

def test_bump_changes_generation(locmem_cache):
    g0 = mc.series_generation(); mc.bump_series_generation()
    assert mc.series_generation() == g0 + 1

def test_generation_after_eviction_does_not_reuse_old_value(locmem_cache, monkeypatch):
    monkeypatch.setattr(mc.time, "time_ns", lambda: 1_000); g0 = mc.series_generation()
    locmem_cache.delete(mc.GEN_KEY)
    monkeypatch.setattr(mc.time, "time_ns", lambda: 2_000)
    assert mc.series_generation() == 2_000 != g0

# test_series.py
def test_write_invalidates_cached_series(monkeypatch):
    # 1ª leitura cacheia [100, 110]; upsert de um ponto (repository real, db) e load passa a
    # devolver [100, 110, 120] -> 2ª leitura devolve 3 pontos
def test_write_to_base_invalidates_derived_series(monkeypatch):
    # "S&P 500 BRL" cacheada; upsert em "ptax:USDBRL" -> próxima leitura recalcula (load chamado de novo)

# test_repository.py
def test_upsert_without_rows_does_not_bump(monkeypatch):
    bumps = []; monkeypatch.setattr(repository.cache, "bump_series_generation", lambda: bumps.append(1))
    repository.upsert_points("bcb:12", "BCB", "rate", [RawPoint(dt.date(2024,1,2), float("nan"))])
    assert bumps == []
```

Ajustar `test_cache_key_includes_window` ao novo formato de chave.

- [ ] **Step 2: Rodar e ver falhar**

Run: `cd backend && .venv/bin/pytest apps/market -q`
Expected: FAIL (`AttributeError: series_generation` / `GEN_KEY`; série antiga servida após o upsert)

- [ ] **Step 3: Implementar**

Em `get_series`:
1. Ler a geração e consultar o cache.
2. No miss, rodar `_ensure_fresh`, que pode gravar e fazer bump.
3. **Reler a geração** antes do `set_df`, para não gravar na chave da geração antiga.

- [ ] **Step 4: Rodar e ver passar**

Run: `cd backend && .venv/bin/pytest -q && cd ../scripts/parity && ../../backend/.venv/bin/pytest -q`
Expected: PASS nas duas suítes. O harness usa `DummyCache` e proíbe escrita; nada muda.

- [ ] **Step 5: Commit**

```bash
git add backend/apps/market
git commit -m "🐛 fix(market): invalida o cache de séries a cada escrita por geração"
```

---

### Task 2: Detecção de lacunas e comando de diagnóstico

**Files:**
- Create: `backend/apps/market/gaps.py`, `backend/apps/market/management/commands/market_gaps.py`
- Modify: `backend/apps/market/repository.py` (`load_dates`)
- Test: `backend/apps/market/tests/test_gaps.py`, `backend/apps/market/tests/test_commands.py`

**Interfaces:**
- Produces (`apps/market/gaps.py`, sem Django):
  - `TOLERANCE_DAYS: dict[str, int] = {"daily": 10, "monthly": 45}`
  - `@dataclass(frozen=True) class Gap: start: dt.date; end: dt.date`. `start` é o último dia com dado antes do buraco; `end` é o primeiro dia com dado depois dele.
  - `@dataclass(frozen=True) class SeriesGap: series: str; start: dt.date; end: dt.date`
  - `find_gaps(dates: Iterable, periodicity: str) -> list[Gap]`: ordena, deduplica e devolve os pares consecutivos com diferença `> TOLERANCE_DAYS[periodicity]`. Aceita `dt.date`, `pd.Timestamp` ou `DatetimeIndex`.
  - `infer_periodicity(index: pd.Index) -> str`: `"monthly"` se a mediana dos passos for ≥ 20 dias, senão `"daily"` (mesmo critério de `transforms._steps_per_year`).
- `repository.load_dates(series_key: str) -> list[dt.date]`: todas as datas da série, ordenadas.
- `python manage.py market_gaps [--only k1,k2]`: percorre `catalog.ingestible()` e as chaves `yf:*` sob demanda (reusar `ingest_market_data._on_demand_defs`). Imprime por série `"<key> (<periodicity>) última=<data> lacunas=<n>"` e, para cada lacuna, `"  <start> → <end> (<dias> dias)"`. Sempre `exit 0`.

- [ ] **Step 1: Escrever os testes**

```python
D = dt.date
def test_daily_gap_beyond_tolerance():
    assert find_gaps([D(2024,1,2), D(2024,1,3), D(2024,1,20)], "daily") == [Gap(D(2024,1,3), D(2024,1,20))]
def test_holiday_cluster_is_not_a_gap():
    assert find_gaps([D(2024,12,23), D(2025,1,1)], "daily") == []          # 9 dias
def test_monthly_tolerance():
    assert find_gaps([D(2024,1,1), D(2024,2,1), D(2024,4,1)], "monthly") == [Gap(D(2024,2,1), D(2024,4,1))]
def test_empty_and_single_point_have_no_gaps():
    assert find_gaps([], "daily") == [] and find_gaps([D(2024,1,2)], "daily") == []
def test_unsorted_duplicated_timestamps():
    idx = pd.to_datetime(["2024-01-20", "2024-01-02", "2024-01-02"])
    assert find_gaps(idx, "daily") == [Gap(D(2024,1,2), D(2024,1,20))]
def test_infer_periodicity():
    assert infer_periodicity(pd.date_range("2024-01-01", periods=6, freq="MS")) == "monthly"
    assert infer_periodicity(pd.bdate_range("2024-01-01", periods=6)) == "daily"

# test_commands.py
def test_market_gaps_lists_gap(db, capsys):
    # upsert bcb:12 em 2024-01-02, 2024-01-03, 2024-02-01 -> saída contém "bcb:12" e "2024-01-03 → 2024-02-01"
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cd backend && .venv/bin/pytest apps/market/tests/test_gaps.py apps/market/tests/test_commands.py -q`
Expected: FAIL (`ModuleNotFoundError: apps.market.gaps`; `Unknown command: 'market_gaps'`)

- [ ] **Step 3: Implementar `gaps.py`, `load_dates` e o comando**

- [ ] **Step 4: Rodar e ver passar**

Run: `cd backend && .venv/bin/pytest apps/market -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/apps/market
git commit -m "✨ feat(market): detecção de lacunas por cadência e comando market_gaps"
```

---

### Task 3: Ingestão re-busca lacunas no meio das séries

**Files:**
- Modify: `backend/apps/market/management/commands/ingest_market_data.py`
- Test: `backend/apps/market/tests/test_commands.py`

**Interfaces:**
- Consumes: `gaps.find_gaps`, `gaps.Gap`, `repository.load_dates` (Task 2).
- Produces: `_refill_gaps(sdef: catalog.SeriesDef) -> tuple[int, list[Gap]]`, que devolve (pontos gravados, lacunas que continuam). Para cada lacuna:
  - chama `_fetch(sdef, gap.start + 1 dia, gap.end − 1 dia)` e grava com `upsert_points`;
  - `FetchError` ou resultado vazio → a lacuna continua, sem propagar;
  - lacuna que continua → `logger.warning("lacuna sem dado na fonte em %s: %s..%s", ...)`.
- `handle` chama `_refill_gaps` para cada série depois do fetch incremental, inclusive quando o incremental respondeu "já atualizado". A cadência vem de `sdef.periodicity`. Saída por série com lacunas: `"  <key>: <n> lacuna(s) re-buscada(s), <m> sem dado na fonte"`. Uma lacuna que continua **não** incrementa `failed`.
- `--only` também filtra a re-busca.

- [ ] **Step 1: Escrever os testes**

```python
def test_ingest_refills_middle_gap(monkeypatch, db):
    # bcb:12 com 2024-01-02 e 2024-02-01 gravados; fake _fetch registra (start, end) e devolve
    # um ponto em 2024-01-15 quando start == 2024-01-03 -> chamada com (2024-01-03, 2024-01-31)
    # e 3 pontos no banco
def test_gap_without_source_data_is_not_a_failure(monkeypatch, db, caplog):
    # fake _fetch: incremental devolve [], re-busca levanta FetchError
    # -> call_command não levanta SystemExit; caplog contém "lacuna sem dado na fonte em bcb:12"
def test_only_filters_gap_refill(monkeypatch, db):
    # lacunas em bcb:12 e bcb:11, --only bcb:12 -> _fetch nunca chamado com bcb:11
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cd backend && .venv/bin/pytest apps/market/tests/test_commands.py -q`
Expected: FAIL (lacuna não re-buscada)

- [ ] **Step 3: Implementar**

- [ ] **Step 4: Rodar e ver passar**

Run: `cd backend && .venv/bin/pytest apps/market -q`
Expected: PASS, incluindo os testes antigos de ingestão.

- [ ] **Step 5: Commit**

```bash
git add backend/apps/market
git commit -m "🐛 fix(market): ingestão re-busca lacunas no meio das séries"
```

---

### Task 4: Benchmarks abrem no último valor conhecido até a abertura da carteira

Pendência da Fase 7a (ADR 0018, "Revisão pós-teste"): a carteira abre num fechamento mensal (ex.: 31/12/2025), mas cada benchmark começa no 1º ponto que tem **a partir** dessa data (02/01 para diários, 01/01 para o IPCA mensal). Pela convenção da §5.3.1 (`ffill`), o valor de um benchmark na abertura é o último valor conhecido até ela.

**Files:**
- Modify: `backend/engine/dataset.py` (`build_dataset`)
- Test: `backend/engine/tests/test_dataset.py`

**Interfaces:**
- Produces:
  - `dataset.BENCH_LOOKBACK_DAYS = 40`.
  - `build_dataset` busca benchmarks e sintéticos (`get_many`, `get_series` dos `extra_specs`) a partir de `start − BENCH_LOOKBACK_DAYS`.
  - Em seguida faz `concat`, depois `ffill` e só então recorta em `[crop_start, end]`. O `crop_start` é o da Fase 7a (o menor entre `start_date` e a 1ª data da carteira).
  - `rf = get_series("CDI", start, end)` não muda.

- [ ] **Step 1: Escrever o teste**

```python
def test_benchmark_opens_at_last_value_before_portfolio_opening(monkeypatch):
    # CDI com pontos em 2023-12-29 (100.0), 2024-01-02 (101.0), 2024-01-31 (102.0);
    # carteira em 2023-12-31 e 2024-01-31; start_date="2024-01-01", end_date="2024-01-31"
    res = dataset.build_dataset(user_series=user, benchmarks=["CDI"], start_date="2024-01-01", end_date="2024-01-31")
    df = res.df_combined
    assert df.index[0] == pd.Timestamp("2023-12-31")
    assert df["CDI"].iloc[0] == pytest.approx(100.0)
    assert df["CDI"].iloc[-1] == pytest.approx(102.0)      # 102/100, não 102/101
```

O fake de `get_many` precisa respeitar o `start` recebido. Com isso, o teste falha se o lookback não existir.

- [ ] **Step 2: Rodar e ver falhar**

Run: `cd backend && .venv/bin/pytest engine/tests/test_dataset.py -q`
Expected: FAIL (`df["CDI"].iloc[-1] == 100.99…`, rebase em 02/01)

- [ ] **Step 3: Implementar**

- [ ] **Step 4: Rodar suítes e regressão**

Run: `cd backend && .venv/bin/pytest -q && cd ../scripts/parity && ../../backend/.venv/bin/pytest -q && ../../backend/.venv/bin/python -m parity.regress --baseline ../../reports/parity/20261028-180012-fase7-abertura/baseline`

Expected:
- `irr`, `kpis` e `portfolio`: `=` em todos os casos.
- Coluna/linha `Carteira`: `=` em todas as métricas.
- Só podem mudar colunas de benchmark (`CDI`, `IPCA + 6%`, `S&P 500 BRL`, `IFIX`) em `twr`, `drawdown`, `rolling_vol`, `rolling_sharpe`, `risk_return`, `yearly` e `shadow`:
  - `IPCA + 6%` (mensal, datado no dia 1º) deve mudar em todos os casos;
  - um benchmark diário só pode mudar em caso cuja data de abertura não tenha ponto desse benchmark.
- Qualquer outra mudança → **parar e investigar**. Registrar a tabela para o ADR.

- [ ] **Step 5: Gravar a nova linha de base e re-verificar**

Run: `cd scripts/parity && ../../backend/.venv/bin/python -m parity.regress --baseline ../../reports/parity/20261028-180012-fase7-abertura/baseline --save ../../reports/parity/$(date +%Y%m%d-%H%M%S)-fase7b`, e de novo com `--baseline` na nova.
Expected: tudo `=`.

- [ ] **Step 6: Commit**

```bash
git add backend/engine/dataset.py backend/engine/tests/test_dataset.py
git commit -m "🐛 fix(engine): benchmarks abrem no último valor conhecido até a abertura"
```

---

### Task 5: `meta.gaps` na API

**Files:**
- Modify: `backend/engine/dataset.py` (`DatasetResult.gaps`), `backend/apps/api/analysis.py` (`AnalysisContext.gaps`), `backend/apps/api/schemas.py` (`AnalysisMetaOut`), `backend/apps/api/routers/portfolio.py:54-62` (`_envelope`)
- Test: `backend/engine/tests/test_dataset.py`, `backend/apps/api/tests/test_analysis.py`, `backend/apps/api/tests/test_portfolio_misc.py`

**Interfaces:**
- Consumes: `gaps.find_gaps`, `gaps.infer_periodicity`, `gaps.SeriesGap` (Task 2); lookback da Task 4.
- Produces:
  - `DatasetResult.gaps: list[SeriesGap]` (default `[]`). Para cada benchmark e sintético servido, aplica `find_gaps(serie.index, infer_periodicity(serie.index))` **antes** do `ffill` e mantém as lacunas com `gap.end > crop_start` e `gap.start < end`. A ordem segue a das colunas, e a carteira não entra.
  - `AnalysisContext.gaps: list[SeriesGap]`.
  - Schema `GapOut(series: str, start: str, end: str)`; `AnalysisMetaOut.gaps: list[GapOut] = []`.
  - JSON: `"meta": {"start", "end", "missing", "gaps": [{"series": "IMA-S", "start": "2024-01-02", "end": "2024-10-15"}]}`, com datas ISO.

- [ ] **Step 1: Escrever os testes**

```python
# test_dataset.py
def test_dataset_reports_benchmark_gap_inside_window(monkeypatch):
    # CDI diário com buraco 2024-03-01 -> 2024-04-15; janela 2024-01-01..2024-06-30
    assert res.gaps == [SeriesGap("CDI", D(2024,3,1), D(2024,4,15))]
def test_dataset_ignores_gap_outside_window(monkeypatch):
    # mesmo CDI, janela 2024-05-01..2024-06-30 -> []
def test_dataset_reports_gap_crossing_opening(monkeypatch):
    # janela 2024-04-01..2024-06-30 -> [SeriesGap("CDI", D(2024,3,1), D(2024,4,15))]

# test_portfolio_misc.py (contrato)
def test_summary_meta_has_gaps_list(...):
    assert body["meta"]["gaps"] == [] and {"start", "end", "missing"} <= body["meta"].keys()
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cd backend && .venv/bin/pytest engine/tests/test_dataset.py apps/api -q`
Expected: FAIL (`AttributeError: gaps`; `KeyError: 'gaps'`)

- [ ] **Step 3: Implementar**

- [ ] **Step 4: Rodar e ver passar**

Run: `cd backend && .venv/bin/pytest -q && cd ../scripts/parity && ../../backend/.venv/bin/pytest -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/engine backend/apps/api
git commit -m "✨ feat(api): aviso aditivo meta.gaps com lacunas dos benchmarks no período"
```

---

### Task 6: `PeriodNotice` mostra as lacunas

**Files:**
- Modify: `frontend/src/components/portfolio/PeriodNotice.jsx`
- Test: `frontend/src/test/period-notice.test.jsx`

**Interfaces:**
- Consumes: `meta.gaps` (Task 5). Campo ausente ou que não seja array → tratado como `[]`, para compatibilidade com respostas antigas em cache no navegador.
- Produces: um item âmbar por lacuna, no mesmo estilo de "Sem dados no período", com o texto exato `Lacuna nos dados de <series> de <dd/mm/aaaa> a <dd/mm/aaaa> (último valor repetido)`.

- [ ] **Step 1: Escrever os testes**

```jsx
it('PeriodNotice lista lacunas de benchmark', () => {
  render(<PeriodNotice meta={{ start: '2024-01-01', end: '2024-06-30', missing: [],
    gaps: [{ series: 'IMA-S', start: '2024-03-01', end: '2024-04-15' }] }} />);
  expect(screen.getByText('Lacuna nos dados de IMA-S de 01/03/2024 a 15/04/2024 (último valor repetido)')).toBeInTheDocument();
});
it('PeriodNotice sem gaps não quebra', () => {
  render(<PeriodNotice meta={{ start: '2024-01-01', end: '2024-06-30', missing: [] }} />);
  expect(screen.queryByText(/Lacuna nos dados/)).toBeNull();
});
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cd frontend && npm test -- --run src/test/period-notice.test.jsx`
Expected: FAIL (texto não encontrado)

- [ ] **Step 3: Implementar**

- [ ] **Step 4: Rodar e ver passar**

Run: `cd frontend && npm test -- --run && npm run build`
Expected: PASS e build OK

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/portfolio/PeriodNotice.jsx frontend/src/test/period-notice.test.jsx
git commit -m "✨ feat(spa): PeriodNotice mostra lacunas dos dados de mercado"
```

---

### Task 7: Diagnóstico no LAB, ADR 0019, documentação e ponto de espera

**Files:**
- Create: `docs/adr/0019-integridade-dos-dados-de-mercado.md`
- Modify: `docs/adr/README.md`, `docs/consolidacao_arquitetura_e_migracao.md` (§2.0, §2.2, §2.3, §5 Fase 7b, §9 item 13), `docs/README.md`, `backend/README.md` (contagem de testes, comando `market_gaps`), `backend/CLAUDE.md` (linha de base nova), `docs/superpowers/plans/2026-10-28-fase-7a-engine.md` (Task 9: marcar o último checkbox, validado pelo usuário em 2026-10-28)

- [ ] **Step 1: Rebuild e diagnóstico antes da re-busca**

Run: `docker compose up -d --build && docker compose exec -T backend python manage.py market_gaps`
Expected: a lista inclui `bcb:12469` (2024) e `yf:LFTS11.SA` (2025), os buracos de fonte da spec §3.3. Guardar a saída em `reports/fase7b/gaps-antes.txt`.

- [ ] **Step 2: Ingestão com re-busca e diagnóstico depois**

Run: `docker compose exec -T backend python manage.py ingest_market_data; docker compose exec -T backend python manage.py market_gaps > reports/fase7b/gaps-depois.txt`
Expected: lacunas preenchidas onde a fonte tem o dado; as que continuam aparecem como "sem dado na fonte" no log da ingestão. Se uma lacuna que a fonte **tem** continuar aberta, **parar e investigar** antes de seguir.

- [ ] **Step 3: Escrever o ADR 0019**

O ADR registra:
- **Contexto:** cache sem invalidação; ingestão incremental cega a lacunas; buracos de fonte.
- **Decisão:**
  - geração global em vez de geração por série, porque derivadas e sintéticas dependem de várias bases e a ingestão é diária;
  - bump em `upsert_points`, o único ponto de escrita;
  - tolerâncias 10/45 dias;
  - re-busca na ingestão;
  - `ffill` + `meta.gaps`, sem interpolação;
  - abertura dos benchmarks com lookback de 40 dias.
- **Evidência:**
  - contagem de lacunas antes e depois por série (Steps 1–2);
  - tabela de regressão da Task 4;
  - caminho da nova linha de base.
- **Consequência:** qualquer escrita descarta o cache de todas as séries (aceitável: a ingestão roda uma vez por dia e aquece o cache em seguida).

- [ ] **Step 4: Checklist de `docs/README.md` e verificação completa**

Run: `cd backend && .venv/bin/pytest -q && .venv/bin/python manage.py check && .venv/bin/pytest -m integration engine/tests/test_oracle_dlp.py -q && cd ../scripts/parity && ../../backend/.venv/bin/pytest -q && cd ../../frontend && npm test -- --run`
Expected: tudo verde.

- [ ] **Step 5: Commit**

```bash
git add docs backend/README.md backend/CLAUDE.md
git commit -m "📝 docs: ADR 0019 (integridade dos dados de mercado) e fechamento da Fase 7b"
```

- [ ] **Step 6: Rebuild final e PARAR para o usuário testar**

Run: `docker compose up -d --build && docker compose ps`
Expected: `backend` e `web` healthy.

Mensagem ao usuário:
- benchmarks mudam levemente no início dos períodos (abertura no último valor conhecido; o `IPCA + 6%` sempre);
- lacunas de mercado agora aparecem no "Período analisado";
- pedir para conferir um período que cubra 2024 com IMA-S.

**Não iniciar a 7c sem o retorno do usuário.**
