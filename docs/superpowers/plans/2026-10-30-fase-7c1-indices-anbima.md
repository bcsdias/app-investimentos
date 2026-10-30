# Fase 7c.1 — Índices IMA pela ANBIMA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Trocar a fonte dos 6 índices da família IMA, do BCB (parado desde 05/2023 e com projeção por ETF do legado) para o histórico oficial da ANBIMA, com fallback, ingestão automática e catálogo sem nomes duplicados ou trocados.

**Architecture:**
- Nova fonte `apps/market/services/sources/anbima.py`, chamada pela ingestão existente (`ingest_market_data`, cron 06h/19h de seg a sex). Ela tem duas camadas, ambas da ANBIMA:
  1. arquivo de histórico completo (XLS estático no S3);
  2. consulta por data no site antigo (`ima-sh-down.asp`), só para janelas curtas.
- Se as duas falharem, vale o mecanismo da 7b: `ffill` + aviso em `meta.gaps`. **Nunca** projeção por ETF.
- O catálogo passa a ter um nome por índice, com fonte `ANBIMA`. As emendas BCB + ETF (`ProxySplice`) e as entradas "(BCB)"/"(proxy)" saem. Uma migration de dados apaga as séries `bcb:1246x`.

**Tech Stack:** Python 3.12, Django 5, pandas + openpyxl (leitura do XLS), requests, pytest.

**Spec:** não há spec formal. As decisões vêm da validação da Fase 7c e da conversa com o usuário em 2026-10-30, registradas em [ADR 0019 §4](../../adr/0019-integridade-dos-dados-de-mercado.md) ("Diagnóstico de `bcb:12469`") e no doc mestre §2.3 ("Séries IMA (BCB) com projeção do legado"). Plano anterior: [7c](2026-10-29-fase-7c-spa-api.md).

## Fatos verificados (2026-10-30)

| Índice | Código ANBIMA (arquivo) | Valor da coluna `Índice` | SGS antigo | Histórico no arquivo |
|---|---|---|---|---|
| IMA-S | `IMAS` | `IMA-S` | 12462 (fora do catálogo) | 03/12/2001 → D-1 |
| IMA-B | `IMAB` | `IMA-B` | 12466 | 16/09/2003 → D-1 |
| IMA-B 5 | `IMAB5` | `IMA-B 5` | 12467 | 16/09/2003 → D-1 |
| IMA-B 5+ | `IMAB5MAIS` | `IMA-B 5+` | 12468 | 16/09/2003 → D-1 |
| IRF-M | `IRFM` | `IRF-M` | 12461 | 01/02/2000 → D-1 |
| IMA-Geral | `IMAGERAL` | `IMA-GERAL` | **12469** (o catálogo o chamava de "IMA-S") | 04/12/2001 → D-1 |

- **Histórico (camada 1):** `GET https://s3-data-prd-use1-precos.s3.us-east-1.amazonaws.com/arquivos/indices-historico/<COD>-HISTORICO.xls`. Público, sem login. É xlsx apesar da extensão, e `pd.read_excel` lê com `openpyxl`. Colunas: `Índice`, `Data de Referência` (datetime), `Número Índice` (float), além das variações, `Duration (d.u.)` e `PMR`.
- **Por data (camada 2):** `POST https://www.anbima.com.br/informacoes/ima/ima-sh-down.asp` com o formulário `Pai=ima`, `escolha=2`, `Idioma=PT`, `saida=csv`, `Dt_Ref=DD/MM/AAAA`, `Dt_Ref_Ver=AAAAMMDD`. Devolve um CSV em cp1252, separado por `;`, com uma linha por índice da família. A coluna `Número Índice` vem com número no formato brasileiro (`9.086,067150`). Funciona para datas antigas: 22/05/2023 e 02/01/2024 testadas.
- Em 22/05/2023, ANBIMA e SGS são idênticos nas 6 casas para os 6 índices (mesma base). Não precisa de ajuste de nível.
- Entre 05/2023 e 02/2024, o arquivo não tem buraco maior que 5 dias corridos.
- A matriz de paridade (`scripts/parity/cases.py`, `BENCHMARKS = ("CDI", "IPCA + 6%", "S&P 500 BRL", "IFIX")`) não usa nenhum índice IMA.

**Decisões do usuário (2026-10-30):**
1. A fonte oficial é obrigatória ("o dado é importante, precisamos ter").
2. Nomes "(BCB)" removidos **sem** apelido. Não há consumidor externo, então fica como exceção registrada ao ADR 0013 (contrato só aditivo).
3. Apagar os registros `bcb:12461`, `bcb:12466`, `bcb:12467`, `bcb:12468` e `bcb:12469`.

## Global Constraints

- TDD conforme `backend/CLAUDE.md`: escrever o teste, rodar e **ver falhar**, implementar, ver passar. Testes sem rede: `monkeypatch` em `anbima.requests`, como em `test_sources_ptax.py`.
- Contrato de fonte (`services/sources/base.py`): falha de rede, HTTP ou formato → `FetchError`. `[]` / `None` só significam "a fonte não tem esse dado".
- `MarketSeries.source = "ANBIMA"`, `kind = "level"`, `periodicity = "daily"`. Chaves: `anbima:<COD>` com o código da tabela acima.
- Nomes do catálogo, exatos: `IMA-S`, `IMA-B`, `IMA-B 5`, `IMA-B 5+`, `IRF-M`, `IMA-Geral`.
- Fallback por data só para janelas de até `FALLBACK_MAX_DAYS = 30` dias corridos. Janela maior com o arquivo indisponível → `FetchError`. A próxima rodada do cron recupera tudo, porque o arquivo traz a série inteira.
- Nenhuma projeção, interpolação ou emenda por ETF.
- Commits: um por task, Conventional Commits + Gitmoji, imperativo em português, **nunca** `Co-Authored-By`. `git add` só com caminhos explícitos.
- `reports/` e `dlp_sheet/` nunca versionados; nada de valores em R$ da carteira em docs ou commits.
- Rodar a partir de `backend/`: `.venv/bin/pytest -q`. Paridade: `cd scripts/parity && ../../backend/.venv/bin/pytest -q && ../../backend/.venv/bin/python -m parity.regress --baseline ../../reports/parity/20261029-211840-fase7c/baseline`. SPA: `cd frontend && npm test -- --run`.

## Review Focus

1. **Arquivo da ANBIMA com formato novo** (coluna renomeada, índice com outro rótulo): a fonte tem que levantar `FetchError` e cair no fallback ou falhar alto. Nunca pode gravar zero pontos em silêncio como se a série não tivesse dado. Teste na Task 1.
2. **Feriado ou fim de semana no fallback por data:** dia sem linha do índice → `None` e segue, sem erro. Teste na Task 1.
3. **Número no formato brasileiro** (`9.086,067150`, milhar com ponto): parse exato, sem virar `9.086`. Teste na Task 1.
4. **Período longo pedido com o arquivo fora do ar** (ex.: `--full`): não pode disparar milhares de POSTs. `FetchError` sem chamar o fallback. Teste na Task 1.
5. **Benchmark "IMA-S" no período 01/01–31/12/2024:** sem aviso de lacuna 05/2023–02/2024, e retorno de 2024 igual ao do arquivo da ANBIMA. Verificação no LAB, Task 4.

---

### Task 1: Fonte `anbima` com histórico completo e fallback por data

**Files:**
- Create: `backend/apps/market/services/sources/anbima.py`
- Modify: `backend/requirements.txt` (bloco "Fase 2: serviços de mercado": `openpyxl~=3.1`; hoje está só no venv do host, não no container)
- Test: `backend/apps/market/tests/test_sources_anbima.py`

**Interfaces:**
- Produces:
  - `HIST_URL = "https://s3-data-prd-use1-precos.s3.us-east-1.amazonaws.com/arquivos/indices-historico/{code}-HISTORICO.xls"`
  - `DAILY_URL = "https://www.anbima.com.br/informacoes/ima/ima-sh-down.asp"`
  - `INDEX_LABELS: dict[str, str] = {"IMAS": "IMA-S", "IMAB": "IMA-B", "IMAB5": "IMA-B 5", "IMAB5MAIS": "IMA-B 5+", "IRFM": "IRF-M", "IMAGERAL": "IMA-GERAL"}` (valor da coluna `Índice`)
  - `FALLBACK_MAX_DAYS = 30`
  - `fetch_history(code: str, start: dt.date, end: dt.date) -> list[RawPoint]`: `requests.get(HIST_URL, timeout=60)`; `pd.read_excel(io.BytesIO(content))`; filtra `Índice == INDEX_LABELS[code]` e a data em `[start, end]`. Código desconhecido, HTTP ≥ 400, erro de parse ou coluna ausente (`Índice`, `Data de Referência`, `Número Índice`) → `FetchError`. O arquivo sem nenhuma linha do índice, em qualquer data, também é `FetchError` (formato mudou). Janela sem pontos → `[]`.
  - `fetch_daily(code: str, day: dt.date) -> RawPoint | None`: `requests.post(DAILY_URL, data={...}, timeout=30)` com o formulário dos "Fatos verificados"; decodifica cp1252 e lê `;`. Linha do índice com `Data de Referência == day` → `RawPoint`; sem linha → `None`. HTTP ≥ 400 ou cabeçalho sem `Número Índice` → `FetchError`.
  - `fetch_anbima(code: str, start: dt.date, end: dt.date) -> list[RawPoint]`: tenta `fetch_history`. Num `FetchError`, se `(end - start).days <= FALLBACK_MAX_DAYS`, faz `logger.warning` e chama `fetch_daily` para cada dia de semana da janela, descartando `None`; senão, relança o erro.

- [ ] **Step 1: Escrever os testes**

Monte o XLS do teste em memória com `DataFrame.to_excel(BytesIO)` (colunas reais, incluindo uma linha de outro índice), e o CSV como `str.encode("cp1252")`.

```python
def test_fetch_history_filters_index_and_window(monkeypatch):
    # linhas IMA-S em 2024-01-02/03/04 + uma linha "IMA-B" -> janela 03..04
    assert anbima.fetch_history("IMAS", D(2024,1,3), D(2024,1,4)) == [RawPoint(D(2024,1,3), 6396.27), RawPoint(D(2024,1,4), 6399.20)]
def test_fetch_history_empty_window_returns_empty(monkeypatch): ...      # janela sem pontos -> []
def test_fetch_history_http_error_raises(monkeypatch): ...               # status 403 -> FetchError
def test_fetch_history_renamed_column_raises(monkeypatch): ...           # "Numero Indice" -> FetchError
def test_fetch_history_without_index_rows_raises(monkeypatch): ...       # só linhas "IMA-B" pedindo IMAS -> FetchError
def test_fetch_history_unknown_code_raises(): ...                        # "XPTO" -> FetchError, sem chamar requests
def test_fetch_daily_parses_brazilian_number(monkeypatch):
    # "IMA-S;29/10/2026;9.086,067150;..." -> RawPoint(D(2026,10,29), 9086.06715)
def test_fetch_daily_without_row_returns_none(monkeypatch): ...          # CSV só com cabeçalho -> None
def test_fetch_daily_posts_expected_form(monkeypatch):
    # captura data= -> Dt_Ref == "02/01/2024", Dt_Ref_Ver == "20240102", saida == "csv", Pai == "ima"
def test_fetch_anbima_falls_back_to_daily_for_short_window(monkeypatch):
    # fetch_history levanta FetchError; janela 2024-01-01(seg)..2024-01-07 -> fetch_daily chamado 5x (seg–sex), pontos não-None devolvidos
def test_fetch_anbima_long_window_does_not_fall_back(monkeypatch):
    # fetch_history levanta FetchError; janela de 400 dias -> FetchError e fetch_daily nunca chamado
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cd backend && .venv/bin/pytest apps/market/tests/test_sources_anbima.py -q`
Expected: FAIL (`ImportError: cannot import name 'anbima'`)

- [ ] **Step 3: Implementar** `anbima.py` e acrescentar `openpyxl~=3.1` ao `requirements.txt`

- [ ] **Step 4: Rodar e ver passar**

Run: `cd backend && .venv/bin/pytest apps/market -q`
Expected: PASS

- [ ] **Step 5: Conferir contra a fonte real** (fora do pytest, rede)

Run: `cd backend && .venv/bin/python -c "import datetime as dt; from apps.market.services.sources import anbima as a; d=dt.date(2023,5,22); print(a.fetch_history('IMAGERAL', d, d), a.fetch_daily('IMAS', d))"`
Expected: `IMAGERAL` 7347.910854 e `IMAS` 5926.726013 (os valores do SGS 12469 e 12462).

- [ ] **Step 6: Commit**

```bash
git add backend/apps/market/services/sources/anbima.py backend/apps/market/tests/test_sources_anbima.py backend/requirements.txt
git commit -m "✨ feat(market): fonte ANBIMA para os índices IMA com fallback por data"
```

---

### Task 2: Catálogo com um nome por índice e ingestão pela ANBIMA

**Files:**
- Modify:
  - `backend/apps/market/catalog.py`: em `_RAW`, saem as 5 entradas "(BCB)" e as 5 "(proxy)" (`IMAB11`, `IRFM11`, `B5P211`, `IB5M11`, `LFTS11`) e entram as 6 ANBIMA. Em `_DERIVED`, saem as 5 entradas `ProxySplice`. A classe `ProxySplice` é removida.
  - `backend/apps/market/series.py`: sai o ramo `ProxySplice` de `_materialize`.
  - `backend/apps/market/transforms.py`: sai `splice_proxy`, que só a emenda usava.
  - `backend/apps/market/management/commands/ingest_market_data.py`: `_fetch` passa a tratar o prefixo `anbima`.
  - `backend/apps/api/routers/benchmarks.py`: `_group_label` usa `source == "ANBIMA"` → `"ANBIMA / Tesouro"`, no lugar de `ProxySplice`.
  - `backend/apps/market/services/sources/bcb.py`: docstring sem o exemplo `12466`.
- Test:
  - `backend/apps/market/tests/test_catalog.py`: `test_ima_b_has_proxy_splice_derivation` é substituído.
  - `backend/apps/market/tests/test_series.py`: sai `test_get_series_proxy_splice` e o trecho de comentário sobre ProxySplice no teste seguinte.
  - `backend/apps/market/tests/test_transforms.py`: sai `test_splice_proxy_continuous_at_seam`.
  - `backend/apps/market/tests/test_commands.py`
  - `backend/apps/api/tests/test_benchmarks_assets.py`

**Interfaces:**
- Consumes: `anbima.fetch_anbima(code, start, end)` (Task 1).
- Produces:
  - `catalog.resolve("IMA-S")` → `SeriesDef("IMA-S", "ANBIMA", "anbima:IMAS", "level")`. O mesmo vale para `IMA-B`/`anbima:IMAB`, `IMA-B 5`/`anbima:IMAB5`, `IMA-B 5+`/`anbima:IMAB5MAIS`, `IRF-M`/`anbima:IRFM` e `IMA-Geral`/`anbima:IMAGERAL`.
  - `catalog.DEFAULT_BENCHMARKS` não muda (os nomes `IMA-B`, `IMA-B 5` e `IMA-B 5+` continuam existindo).
  - As chaves `yf:IMAB11.SA` etc. continuam no banco. Fora do catálogo, passam a ser atualizadas por `_on_demand_defs` e encontradas pela busca de ticker.

- [ ] **Step 1: Escrever os testes**

```python
# test_catalog.py
@pytest.mark.parametrize("name,key", [("IMA-S","anbima:IMAS"), ("IMA-B","anbima:IMAB"), ("IMA-B 5","anbima:IMAB5"),
                                      ("IMA-B 5+","anbima:IMAB5MAIS"), ("IRF-M","anbima:IRFM"), ("IMA-Geral","anbima:IMAGERAL")])
def test_ima_family_comes_from_anbima(name, key):
    s = cat.resolve(name); assert (s.source, s.series_key, s.kind, s.derivation) == ("ANBIMA", key, "level", None)
def test_no_bcb_or_proxy_names_left():
    assert not [n for n in cat.all_names() if "(BCB)" in n or "(proxy)" in n]
    assert not [d for d in cat.ingestible() if d.series_key.startswith("bcb:1246")]
def test_ima_family_is_ingestible():
    assert {"anbima:IMAS","anbima:IMAGERAL"} <= {d.series_key for d in cat.ingestible()}

# test_commands.py
def test_ingest_dispatches_anbima(monkeypatch, db):
    # monkeypatch anbima.fetch_anbima registrando (code, start, end) -> 1 ponto; ingest_market_data --only anbima:IMAS
    # -> chamado com "IMAS"; repository.last_date("anbima:IMAS") == data do ponto; MarketSeries.source == "ANBIMA"

# test_benchmarks_assets.py
def test_catalog_groups_ima_under_anbima(auth_client):
    # grupo "ANBIMA / Tesouro" == ["IMA-B","IMA-B 5","IMA-B 5+","IRF-M","IMA-S","IMA-Geral"] (ordem do catálogo); nenhum nome com "(BCB)"
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cd backend && .venv/bin/pytest apps/market apps/api/tests/test_benchmarks_assets.py -q`
Expected: FAIL (`IMA-S` com `derivation=ProxySplice`; nomes "(BCB)" presentes; prefixo `anbima` → `FetchError("prefixo sem fonte")`)

- [ ] **Step 3: Implementar**

`grep -rn "ProxySplice\|splice_proxy\|(BCB)\|(proxy)" backend frontend/src --include=*.py --include=*.js --include=*.jsx` tem que voltar vazio, exceto `.venv` e `node_modules`.

- [ ] **Step 4: Rodar e ver passar**

Run: `cd backend && .venv/bin/pytest -q && cd ../frontend && npm test -- --run`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/apps/market/catalog.py backend/apps/market/series.py backend/apps/market/transforms.py backend/apps/market/management/commands/ingest_market_data.py backend/apps/market/services/sources/bcb.py backend/apps/api/routers/benchmarks.py backend/apps/market/tests/test_catalog.py backend/apps/market/tests/test_series.py backend/apps/market/tests/test_transforms.py backend/apps/market/tests/test_commands.py backend/apps/api/tests/test_benchmarks_assets.py
git commit -m "♻️ refactor(market): índices IMA com um nome cada e fonte ANBIMA, sem emenda por ETF"
```

---

### Task 3: Remover as séries `bcb:1246x` do banco

**Files:**
- Create: `backend/apps/core/migrations/0003_remove_bcb_ima_series.py`
- Test: `backend/apps/core/tests/test_migration_remove_bcb_ima.py` (criar `apps/core/tests/__init__.py` se não existir)

**Interfaces:**
- Produces:
  - `REMOVED_KEYS = ("bcb:12461", "bcb:12466", "bcb:12467", "bcb:12468", "bcb:12469")`
  - `forwards(apps, schema_editor) -> None`: apaga `MarketSeries` com `series_key__in=REMOVED_KEYS` e chama `apps.market.cache.bump_series_generation()`. A invalidação é necessária porque um `delete` fora do `repository` não invalida o cache de séries (ADR 0019).
  - `RunPython(forwards, migrations.RunPython.noop)`. Desfazer não traz os dados de volta; eles estão no histórico da ANBIMA.

- [ ] **Step 1: Escrever o teste**

```python
@pytest.mark.django_db
def test_forwards_removes_only_bcb_ima_keys(monkeypatch):
    # cria pontos em bcb:12469, bcb:12461, bcb:12 e anbima:IMAS; conta chamadas a bump_series_generation
    mod = importlib.import_module("apps.core.migrations.0003_remove_bcb_ima_series")
    mod.forwards(django.apps.apps, None)
    assert set(MarketSeries.objects.values_list("series_key", flat=True)) == {"bcb:12", "anbima:IMAS"}
    assert bumps == 1
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cd backend && .venv/bin/pytest apps/core -q`
Expected: FAIL (`ModuleNotFoundError: ...0003_remove_bcb_ima_series`)

- [ ] **Step 3: Implementar** (dependência: `("core", "0002_marketseries_kind")`)

- [ ] **Step 4: Rodar e ver passar**

Run: `cd backend && .venv/bin/pytest -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/apps/core/migrations/0003_remove_bcb_ima_series.py backend/apps/core/tests/__init__.py backend/apps/core/tests/test_migration_remove_bcb_ima.py
git commit -m "🗃️ chore(core): remove as séries IMA do BCB substituídas pela ANBIMA"
```

---

### Task 4: Carga no LAB, verificação, documentação e ponto de espera

Ordem obrigatória: a carga da ANBIMA vem **antes** da migration que apaga o BCB. Assim os índices IMA não ficam vazios entre um passo e outro.

**Files:**
- Create: `docs/adr/0021-indices-ima-pela-anbima.md`
- Modify:
  - `docs/adr/0019-integridade-dos-dados-de-mercado.md` §4: "Decisão pendente" → resolvida, com link para o 0021;
  - `docs/adr/0013-contrato-da-api-v1.md`: nota da exceção, com link para o 0021;
  - `docs/adr/README.md`;
  - `docs/consolidacao_arquitetura_e_migracao.md`: §2.0 (linha "Serviços de Mercado"), §2.3 (riscar o item "Séries IMA (BCB)"), §5 (Fase 7c.1) e §9;
  - `docs/README.md`;
  - `backend/README.md` (contagem de testes e fontes);
  - `backend/CLAUDE.md`, se a linha de base mudar (não deve).

- [ ] **Step 1: Regressão de paridade**

Run: `cd scripts/parity && ../../backend/.venv/bin/pytest -q && ../../backend/.venv/bin/python -m parity.regress --baseline ../../reports/parity/20261029-211840-fase7c/baseline`
Expected: 41 passed e tudo `=` (a matriz não usa IMA). Qualquer diferença → **parar e investigar**.

- [ ] **Step 2: Carga da ANBIMA no banco, antes da migration**

Run: `cd backend && .venv/bin/python manage.py ingest_market_data --only anbima:IMAS,anbima:IMAB,anbima:IMAB5,anbima:IMAB5MAIS,anbima:IRFM,anbima:IMAGERAL --full`
Expected: `ok anbima:<COD>: +N pontos` para os 6 (N ≈ 5.800–6.700), sem falha.

- [ ] **Step 3: Migration e rebuild do LAB**

Run: `cd backend && .venv/bin/python manage.py migrate && cd .. && docker compose up -d --build && docker compose ps && docker compose exec -T backend python -c "import openpyxl" && docker compose exec -T backend python manage.py market_gaps`
Expected: `backend`, `web` e `ingestion` de pé (os dois primeiros healthy), e `openpyxl` importável no container. O `market_gaps` lista os 6 `anbima:*` com `lacunas=0` e sem "defasada", e não lista mais `bcb:1246x`.

- [ ] **Step 4: Cenário do teste da 7b**, pela API com a conta do usuário (script no scratchpad, sem gravar valores em R$)

Período 01/01/2024–31/12/2024, benchmarks `CDI`, `IMA-S` e `IMA-Geral`:
- `meta.gaps` sem nenhuma entrada de `IMA-S` nem de `IMA-Geral`;
- na tabela anual, `IMA-S` 2024 == `Número Índice` de 31/12/2024 ÷ o de 29/12/2023 − 1 no arquivo `IMAS-HISTORICO.xls` (tolerância 1e-6);
- `/benchmarks/catalog` sem nomes "(BCB)" e "(proxy)".

- [ ] **Step 5: ADR 0021 e checklist de documentação**

ADR 0021, seções:
- **Contexto:** SGS parado em 05/2023; projeção por ETF do legado com `source=BCB` (IMA-Geral −4,54 % e IMA-B −0,48 % em 10/04/2026; IMA-S ~0 % entre 05/2023 e 02/2024 contra +9,49 % real); `bcb:12469` era o IMA-Geral rotulado como IMA-S.
- **Decisão:** as 2 camadas ANBIMA; sem ETF; nomes do catálogo; exceção ao ADR 0013; remoção dos dados; `openpyxl`.
- **Consequências:** benchmarks IMA mudam de número (corretos agora); paridade inalterada; ingestão automática pelo cron existente; a redistribuição dos dados da ANBIMA depende dos termos dela (uso pessoal hoje).

Depois, o checklist de `docs/README.md`.

Run: `cd backend && .venv/bin/pytest -q && .venv/bin/python manage.py check && cd ../frontend && npm test -- --run && npm run build`
Expected: tudo verde.

- [ ] **Step 6: Commit**

```bash
git add docs/adr/0021-indices-ima-pela-anbima.md docs/adr/0019-integridade-dos-dados-de-mercado.md docs/adr/0013-contrato-da-api-v1.md docs/adr/README.md docs/consolidacao_arquitetura_e_migracao.md docs/README.md backend/README.md
git commit -m "📝 docs: ADR 0021 (índices IMA pela ANBIMA) e fechamento da Fase 7c.1"
```

- [ ] **Step 7: PARAR para o usuário testar**

Mensagem ao usuário:
- repetir 01/01/2024–31/12/2024 com `IMA-S`: sem aviso de lacuna, com rendimento próximo ao do CDI;
- o catálogo de comparadores mostra IMA-S, IMA-B, IMA-B 5, IMA-B 5+, IRF-M e IMA-Geral, sem "(BCB)" e sem "(proxy)";
- a ingestão automática roda às 06h e 19h. No dia seguinte, `market_gaps` deve mostrar a data de D-1 nos 6 índices.

**Não iniciar a 7d sem o retorno do usuário.**
