# Fase 3c — Otimizações DLP sem Impacto no Engine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Um único DataFrame de `/historico` cacheado por usuário, com filtros `ativo`/`classe`/`corretora` aplicados em memória; `_extract_distinct` deduplicando antes de normalizar; URL base da DLP vinda de `settings.DLP_BASE_URL`. Nenhum número muda.

**Architecture:** Toda a mudança de cache fica dentro de `apps/market/services/sources/dlp.py` (assinatura pública de `buscar_historico` intacta, então `engine/dataset.py` e os routers não mudam). Um helper puro `_filter_historico` recorta o DataFrame completo. `portfolio._extract_distinct` troca só a ordem dedup → normalização.

**Tech Stack:** Django 5.2, pandas 2.2, `LocMemCache` nos testes, `pytest-django`.

**Spec:** `docs/superpowers/specs/2026-10-26-fase-3c-otimizacoes-dlp-design.md`

## Global Constraints

Todo task herda implicitamente esta seção.

- **Branch:** `migracao-django`. Rodar tudo a partir de `backend/` com `.venv/bin/python` / `.venv/bin/pytest`.
- **`engine/` intocado** (ADR 0012): nenhum arquivo em `engine/` é modificado neste plano.
- **Assinatura pública fixa:** `dlp.buscar_historico(token, ativo=None, classe=None, corretora=None)`; contratos HTTP da Fase 3 inalterados.
- **Chave de cache do histórico:** `mc.make_key("dlp", "hist", _token_hash(token), _generation(token))` — sem hash de params.
- **TTL:** `mc.TTL_DLP` (600 s). Payload vazio → não cacheia, retorna `None`.
- **Default da URL:** `https://users.dlpinvest.com.br` (sem barra final).
- **Testes:** rede sempre mockada; chamadas reais só sob `@pytest.mark.integration`.
- **Commits:** Conventional Commits + Gitmoji + imperativo em português. **Nunca** `Co-Authored-By` (regra do `backend/CLAUDE.md`). Um commit por task, com a mensagem indicada.
- **TDD obrigatório:** teste → rodar e ver falhar → implementar → ver passar.

## Review Focus

1. **Coluna de filtro ausente no payload** (ex.: DLP muda `corretora` de nome) → o spec manda ignorar o filtro; o usuário receberia a carteira inteira como se fosse o recorte. Esperado: filtro ignorado **com `logger.warning`** nomeando a coluna, para ser visível (Task 2, `test_filter_missing_column_is_ignored_and_logged`).
2. **Mutação do DataFrame devolvido** (o `engine` faz `df[...] = ...` no resultado) → a próxima chamada, filtrada ou não, devolve os dados originais (Task 2, `test_filtered_result_mutation_does_not_leak`).
3. **Filtro com espaços nas bordas ou string vazia** (`classe=" ACAO "`, `ativo=""`) → espaços ignorados na comparação; `""`/só espaços = sem filtro (Task 2, `test_filter_strips_and_blank_means_no_filter`).
4. **Valores não-string na coluna** (`ativo` numérico, `NaN`) → comparação por `astype(str).str.strip()` não quebra e `NaN` nunca casa (Task 2, mesmo teste do item 3).
5. **Equivalência com o filtro da DLP** (o filtro em memória só é correto se `/historico` sem filtro devolve as mesmas linhas por papel) → teste `integration` com token real é o portão de saída; se divergir, **parar e reportar**, não ajustar o filtro (Task 4).

---

## Estrutura de arquivos

Tudo em `backend/`. `M` = modificar.

```
config/settings.py                              # M — DLP_BASE_URL
.env.example                                    # M — DLP_BASE_URL comentado
apps/market/services/sources/dlp.py             # M — _get lê settings; cache único; _filter_historico
apps/market/tests/test_sources_dlp.py           # M — testes novos
apps/api/routers/portfolio.py                   # M — _extract_distinct com pd.unique
apps/api/tests/test_portfolio_misc.py           # M — teste de _extract_distinct
apps/api/tests/test_integration.py              # M — equivalência filtro em memória × DLP
```

> O spec cita `apps/api/tests/test_portfolio_filters_refresh.py`, que não existe; os testes de `/filters` vivem em `test_portfolio_misc.py` — usar esse arquivo.

---

### Task 1: `DLP_BASE_URL` em settings

**Files:**
- Modify: `config/settings.py` (junto das outras variáveis lidas por `_env`)
- Modify: `.env.example`
- Modify: `apps/market/services/sources/dlp.py:23,58-64`
- Test: `apps/market/tests/test_sources_dlp.py`

**Interfaces:**
- Produces: `settings.DLP_BASE_URL: str` (sem barra final). `_BASE` deixa de existir.

- [x] **Step 1: Write the failing tests**

```python
def test_get_uses_settings_base_url(monkeypatch, settings):
    settings.DLP_BASE_URL = "https://dlp.exemplo.test"
    seen = []
    monkeypatch.setattr(dlp.requests, "get",
        lambda url, **k: seen.append(url) or _Resp({"historico": [{"data": "2024-01-02"}]}))
    dlp.buscar_historico("tok")
    assert seen == ["https://dlp.exemplo.test/historico"]


def test_default_base_url_is_dlpinvest():
    from django.conf import settings as dj
    assert dj.DLP_BASE_URL == "https://users.dlpinvest.com.br"
```

(O segundo teste assume que `backend/.env` não define `DLP_BASE_URL`; se definir, o teste vale para o default de `_env` — usar `monkeypatch.delenv` + reimport não vale o custo, basta não definir no `.env`.)

- [x] **Step 2: Run to verify they fail**

Run: `.venv/bin/pytest apps/market/tests/test_sources_dlp.py -k "base_url" -v`
Expected: FAIL — `AttributeError: 'Settings' object has no attribute 'DLP_BASE_URL'` / URL `dlombelloplanilhas`.

- [x] **Step 3: Implement**

`config/settings.py`: `DLP_BASE_URL = _env("DLP_BASE_URL", "https://users.dlpinvest.com.br").rstrip("/")`.
`dlp._get`: montar a URL com `settings.DLP_BASE_URL` lido **dentro** de `_get` (`from django.conf import settings`), remover `_BASE`. `.env.example`: linha `# DLP_BASE_URL=https://users.dlpinvest.com.br`.

- [x] **Step 4: Run to verify they pass**

Run: `.venv/bin/pytest apps/market/tests/test_sources_dlp.py -v`
Expected: PASS (todos, incluindo os antigos).

- [x] **Step 5: Commit**

```bash
git add config/settings.py .env.example apps/market/services/sources/dlp.py apps/market/tests/test_sources_dlp.py
git commit -m "🔧 chore(market): URL base da DLP vem de settings.DLP_BASE_URL (dlpinvest.com.br)"
```

---

### Task 2: Cache único de `/historico` + filtro em memória

**Files:**
- Modify: `apps/market/services/sources/dlp.py` (`buscar_historico`, novo `_filter_historico`)
- Test: `apps/market/tests/test_sources_dlp.py`

**Interfaces:**
- Consumes: `_get`, `_norm_date_col`, `_token_hash`, `_generation`, `mc.single_flight`, `mc.get_df`, `mc.set_df`.
- Produces: `_filter_historico(df: pd.DataFrame, *, ativo=None, classe=None, corretora=None) -> pd.DataFrame | None` (puro). `buscar_historico` com a mesma assinatura e o mesmo contrato de retorno (`DataFrame` com coluna `date`, ou `None`).

Fixture de teste sugerida (usar em todos os testes do task):

```python
_ROWS = [
    {"data": "2024-01-02", "ativo": "PETR4", "classe": "ACAO", "corretora": "XP", "vlr_mercado": 100},
    {"data": "2024-01-02", "ativo": "HGLG11", "classe": "FII", "corretora": "XP", "vlr_mercado": 50},
    {"data": "2024-01-02", "ativo": "PETR4", "classe": "ACAO", "corretora": "BTG", "vlr_mercado": 10},
]
```

- [x] **Step 1: Write the failing tests**

```python
def test_filtered_calls_share_one_http_request(monkeypatch):
    calls = []
    monkeypatch.setattr(dlp.requests, "get",
        lambda url, headers=None, params=None, timeout=None: calls.append(params) or _Resp({"historico": _ROWS}))
    full = dlp.buscar_historico("tok")
    acoes = dlp.buscar_historico("tok", classe="ACAO")
    petr_xp = dlp.buscar_historico("tok", ativo="PETR4", corretora="XP")
    assert calls == [{}]
    assert len(full) == 3
    assert set(acoes["ativo"]) == {"PETR4"} and len(acoes) == 2
    assert len(petr_xp) == 1 and petr_xp["vlr_mercado"].iloc[0] == 100
    assert "date" in acoes.columns


def test_filter_without_match_returns_none(monkeypatch):
    monkeypatch.setattr(dlp.requests, "get", lambda *a, **k: _Resp({"historico": _ROWS}))
    assert dlp.buscar_historico("tok", classe="CRIPTO") is None


def test_filtered_first_call_still_caches_full_frame(monkeypatch):
    calls = []
    monkeypatch.setattr(dlp.requests, "get",
        lambda url, params=None, **k: calls.append(params) or _Resp({"historico": _ROWS}))
    dlp.buscar_historico("tok", ativo="PETR4")
    assert len(dlp.buscar_historico("tok")) == 3
    assert calls == [{}]


def test_invalidate_drops_single_key(monkeypatch):
    calls = []
    monkeypatch.setattr(dlp.requests, "get",
        lambda *a, **k: calls.append(1) or _Resp({"historico": _ROWS}))
    dlp.buscar_historico("tok", classe="ACAO")
    dlp.invalidate_user_cache("tok")
    dlp.buscar_historico("tok", classe="FII")
    assert len(calls) == 2


def test_filtered_result_mutation_does_not_leak(monkeypatch):
    monkeypatch.setattr(dlp.requests, "get", lambda *a, **k: _Resp({"historico": _ROWS}))
    df = dlp.buscar_historico("tok", classe="ACAO")
    df["vlr_mercado"] = 0
    assert dlp.buscar_historico("tok", classe="ACAO")["vlr_mercado"].tolist() == [100, 10]


def test_filter_strips_and_blank_means_no_filter():
    df = pd.DataFrame(_ROWS + [{"data": "2024-01-03", "ativo": None, "classe": "ACAO", "corretora": "XP", "vlr_mercado": 1}])
    assert len(dlp._filter_historico(df, classe=" ACAO ")) == 3
    assert len(dlp._filter_historico(df, ativo="", classe="   ")) == 4
    assert dlp._filter_historico(df, ativo="nan") is None  # NaN/None nunca casa


def test_filter_column_lookup_is_case_insensitive():
    df = pd.DataFrame([{"Ativo": "PETR4"}, {"Ativo": "VALE3"}])
    assert dlp._filter_historico(df, ativo="VALE3")["Ativo"].tolist() == ["VALE3"]


def test_filter_missing_column_is_ignored_and_logged(caplog):
    df = pd.DataFrame([{"ativo": "PETR4"}])
    out = dlp._filter_historico(df, corretora="XP")
    assert len(out) == 1
    assert "corretora" in caplog.text
```

- [x] **Step 2: Run to verify they fail**

Run: `.venv/bin/pytest apps/market/tests/test_sources_dlp.py -v`
Expected: FAIL — `calls == [{"classe": "ACAO"}, ...]` (múltiplas chamadas) e `AttributeError: module ... has no attribute '_filter_historico'`.

- [x] **Step 3: Implement**

`_filter_historico`: para cada filtro com `str(val).strip() != ""`, achar a coluna por `lower()`; se ausente, `logger.warning("coluna %s ausente no /historico; filtro ignorado", nome)`; senão máscara `col.notna() & (col.astype(str).str.strip() == str(val).strip())`. Resultado vazio → `None`; senão `.reset_index(drop=True)` (já é cópia).

`buscar_historico`: chave única (Global Constraints); dentro de `single_flight` sempre `_get("/historico", token, {})`; `df.empty` → `return None` sem cachear; `_norm_date_col` → `mc.set_df(key, df, ttl=mc.TTL_DLP)`. Nos três caminhos de retorno (hit fora do lock, hit dentro do lock, fetch) devolver `_filter_historico(df, ativo=..., classe=..., corretora=...)`. Remover o import de `json` se ficar sem uso.

- [x] **Step 4: Run to verify they pass**

Run: `.venv/bin/pytest apps/market/tests/test_sources_dlp.py apps/api engine -q`
Expected: PASS — inclusive os testes da API e do engine (que chamam `buscar_historico` via mocks).

- [x] **Step 5: Commit**

```bash
git add apps/market/services/sources/dlp.py apps/market/tests/test_sources_dlp.py
git commit -m "⚡ perf(market): cache único do /historico por usuário com filtro em memória"
```

---

### Task 3: `_extract_distinct` com `pd.unique`

**Files:**
- Modify: `apps/api/routers/portfolio.py:64-77`
- Test: `apps/api/tests/test_portfolio_misc.py`

**Interfaces:**
- Produces: `_extract_distinct(df, candidates, label) -> list[str]` — mesma assinatura e mesmo retorno.

- [x] **Step 1: Write the test (characterization)**

```python
def test_extract_distinct_dedups_strips_and_drops_blanks():
    from apps.api.routers.portfolio import _extract_distinct
    df = pd.DataFrame({"Classe": ["ACAO", " ACAO", "FII ", None, float("nan"), "", "  ", "ETF", "ACAO"]})
    assert _extract_distinct(df, ("classe",), "classe") == ["ACAO", "ETF", "FII"]
```

- [x] **Step 2: Run it**

Run: `.venv/bin/pytest apps/api/tests/test_portfolio_misc.py::test_extract_distinct_dedups_strips_and_drops_blanks -v`
Expected: PASS com o código atual — é teste de caracterização; ele fixa o comportamento que a refatoração precisa preservar. (Exceção consciente ao "ver falhar": refatoração de performance sem mudança de comportamento.)

- [x] **Step 3: Implement**

Trocar o corpo do ramo encontrado pelo trecho do spec §4.2 (`pd.unique(df[col].dropna())` → `str(v).strip()` → `sorted({v for v in vals if v})`).

- [x] **Step 4: Run to verify it still passes**

Run: `.venv/bin/pytest apps/api/tests/test_portfolio_misc.py -v`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add apps/api/routers/portfolio.py apps/api/tests/test_portfolio_misc.py
git commit -m "⚡ perf(api): deduplica antes de normalizar em _extract_distinct"
```

---

### Task 4: Integração (equivalência com a DLP) + fechamento da fase

**Files:**
- Modify: `apps/api/tests/test_integration.py`
- Modify: `../docs/consolidacao_arquitetura_e_migracao.md`, `../docs/README.md`, `README.md` (backend), conforme checklist

**Interfaces:**
- Consumes: `dlp.buscar_historico`, `dlp._get` (Task 2).

- [x] **Step 1: Write the integration test**

```python
def test_real_dlp_in_memory_filter_matches_server_filter():
    token = os.environ.get("DLP_TEST_TOKEN")
    if not token:
        pytest.skip("DLP_TEST_TOKEN não definido no ambiente")
    full = dlp.buscar_historico(token)
    classe = full["classe"].dropna().iloc[0]
    ativo = full["ativo"].dropna().iloc[0]
    for params in ({"classe": classe}, {"ativo": ativo}):
        mine = dlp.buscar_historico(token, **params).sort_values(["date", "ativo", "corretora"]).reset_index(drop=True)
        server = dlp._norm_date_col(pd.DataFrame(dlp._get("/historico", token, params)["historico"]))
        server = server.sort_values(["date", "ativo", "corretora"]).reset_index(drop=True)
        assert sorted(mine.columns) == sorted(server.columns)
        pd.testing.assert_frame_equal(mine[sorted(mine.columns)], server[sorted(server.columns)])
```

- [x] **Step 2: Run the integration test with the real token**

Run: `DLP_TEST_TOKEN=... .venv/bin/pytest apps/api/tests/test_integration.py -m integration -k in_memory -v` (o Bruno fornece o token).
Expected: PASS. **Se falhar** (linhas ou colunas diferentes): parar, reportar o diff ao Bruno; a Task 2 não pode ir para produção nesse caso.

- [x] **Step 3: Full suite + check**

Run: `.venv/bin/pytest -q && .venv/bin/python manage.py check`
Expected: todos verdes; `System check identified no issues`.

- [x] **Step 4: Checklist de documentação** ([`../docs/README.md`](../../README.md#checklist-de-atualização-ao-fechar-uma-fase))

- Doc mestre: marcar Fase 3c `[x]` em §9 e anotar em §Fase 3c "concluída" com a data; registrar em §2.x o domínio `users.dlpinvest.com.br` e `DLP_BASE_URL`.
- `backend/README.md`: `DLP_BASE_URL` na tabela de variáveis; contagem de testes atualizada.
- ADR: nenhum novo (sem decisão nova além do spec); se o ADR 0014 descreve a chave `dlp:hist:...:<params_hash>`, acrescentar nota "substituída na Fase 3c por chave única + filtro em memória".

- [x] **Step 5: Commit**

```bash
git add apps/api/tests/test_integration.py ../docs README.md
git commit -m "📝 docs: fecha a Fase 3c — cache único da DLP, DLP_BASE_URL e READMEs"
```
