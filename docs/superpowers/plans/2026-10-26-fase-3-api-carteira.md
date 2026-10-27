# Fase 3 — API da Carteira (Django Ninja) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expor como API REST JSON (Django Ninja, sessão + CSRF) o fluxo do Streamlit — token DLP, análises da carteira (8 endpoints), catálogo de benchmarks e comparação contra qualquer ticker do `ativos.csv` — sem alterar nenhum cálculo do engine.

**Architecture:** `config/api.py` tem a única `NinjaAPI` (auth padrão `django_auth`, handler de `ApiError`). Routers finos em `apps/accounts/api.py` e `apps/api/routers/*`; toda orquestração (token → DLP → `build_dataset`) em `apps/api/analysis.py`; pandas → JSON em `apps/api/serializers.py`. Em `apps/market`: erros DLP tipados, invalidação por **geração de cache**, **single-flight** via `cache.add`, universo de tickers (`assets.py`) e fetch-on-read de tickers em `series.py`.

**Tech Stack:** Django 5.2, `django-ninja` 1.6.3 (já instalado), pydantic 2, pandas 2.2, `django-redis` (prod) / `LocMemCache` (testes), `pytest-django`.

**Spec:** `docs/superpowers/specs/2026-10-26-fase-3-api-carteira-design.md`

## Global Constraints

Todo task herda implicitamente esta seção.

- **Branch:** `migracao-django`. Rodar tudo a partir de `backend/` com `.venv/bin/python` / `.venv/bin/pytest` (não há `python` no PATH).
- **Engine intocado nos números** (ADR 0012): nenhuma mudança em `engine/{twr,irr,metrics,compute}.py`. Única mudança permitida em `engine/dataset.py`: usar `synthetics.display_name` (Task 9) — mesmo texto gerado.
- **Prefixo de rotas:** `/api/v1/`; Swagger em `/api/v1/docs`.
- **Auth:** toda rota exige sessão (`django_auth`) exceto `GET /auth/csrf` e `POST /auth/login`. Toda escrita exige CSRF (inclusive o login). Usuário é sempre `request.user` — nenhuma rota recebe id/e-mail de usuário.
- **Corpo de erro de negócio:** `{"code": str, "message": str, "details": object|null}`, `message` em português.
- **Status/códigos (spec §6):** `invalid_credentials` 401 · `dlp_token_missing` 409 · `dlp_token_rejected` **422 (nunca 401)** · `dlp_unavailable` 502 · `portfolio_empty` 404 · `invalid_period` / `unknown_benchmark` / `invalid_synthetic` / `too_many_benchmarks` / `invalid_window` / `invalid_token` / `invalid_query` 422.
- **Sintéticos:** `base ∈ {"IPCA","CDI"}`, `operador ∈ {"+","%"}`; `"+"` → `0 < valor <= 50`; `"%"` → `0 < valor <= 500`. Limite **15** itens somando `benchmarks` + `synthetics`.
- **Defaults do `AnalysisRequest`:** `start` = hoje − 1 ano; `end` = hoje; `benchmarks` omitido → `catalog.DEFAULT_BENCHMARKS`; `[]` explícito é válido.
- **Ticker:** identificador público = ticker Yahoo (`PETR4.SA`, `GDX.L`, `GDX`); variante BRL = `"<ticker yahoo> BRL"`, só para mercado `INTL`. Nome do catálogo vence colisão.
- **Série JSON:** `{"name": str, "data": [["YYYY-MM-DD", float|null], ...]}`; `NaN`/`±inf` → `null`. A API **não converte unidades**.
- **Envelope das análises:** `{"data": ..., "meta": {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD", "missing": [str]}}`.
- **Segredos:** nenhum token (plano ou cifrado) em resposta, log ou mensagem de exceção; identificador de log = `dlp._token_hash(token)`.
- **Proibido:** `django-cors-headers`, log em arquivo, token editável no admin (ver `backend/CLAUDE.md`).
- **Testes:** cache = `LocMemCache` (fixture `locmem_cache`); rede sempre mockada; chamadas reais só sob `@pytest.mark.integration`.
- **Desvio consciente do spec §6:** a invalidação do cache DLP usa **geração por usuário** (contador na chave), não `cache.delete_pattern`, e o single-flight usa `cache.add`, não `cache.lock`. Os dois primeiros só existem no `django-redis`, e os testes rodam em `LocMemCache`. O efeito é o mesmo; registrar no ADR 0014.
- **Commits:** Conventional Commits + Gitmoji + imperativo em português. **Nunca** `Co-Authored-By`. Um commit por task, com a mensagem indicada.
- **TDD obrigatório:** escrever o teste → **rodar e ver falhar** com o erro esperado → implementar → ver passar.

## Review Focus

Entradas que o spec implica, que um usuário vai encontrar e que nenhum teste "óbvio" cobre — cada uma tem teste no task dono:

1. **Métricas móveis com janela maior que o período** (os primeiros 252 pontos são `NaN`) → resposta é JSON estrito com `null`, nunca `NaN` literal (Task 10, `test_rolling_metrics_is_strict_json`).
2. **Benchmark válido sem dados no período** (ex.: ticker deslistado, índice sem pontos) → vai para `meta.missing` com `200`, nunca `500` (Task 10, `test_missing_benchmark_reported_in_meta`).
3. **Token colado com espaços / quebra de linha** → gravado sem eles; token só de espaços → `422 invalid_token` (Task 4).
4. **Histórico da carteira começa depois do `start` pedido** → `meta.start` reflete o período efetivo do dataset, não o pedido (Task 10, `test_meta_reflects_effective_period`).
5. **Escrita com sessão válida mas sem CSRF** → `403`, e leitura sem sessão → `401` (a SPA distingue os dois) (Tasks 2 e 4).

---

## Estrutura de arquivos

Tudo em `backend/`. `M` = modificar.

```
conftest.py                         # NOVO — fixtures compartilhadas (locmem_cache, user, auth_client)
config/
├── api.py                          # NOVO — api = NinjaAPI(...); handle_api_error
├── urls.py                         # M — path("api/v1/", api.urls)
└── settings.py                     # M — "apps.api", CSRF_COOKIE_HTTPONLY, MARKET_ASSETS_CSV
apps/accounts/
├── api.py                          # NOVO — router /auth
└── tests/test_auth_api.py
apps/api/                           # NOVO app (sem models.py)
├── __init__.py  apps.py
├── errors.py                       # ApiError
├── schemas.py                      # schemas Ninja
├── serializers.py                  # pandas -> JSON
├── analysis.py                     # validate_request / get_token / run_analysis
├── routers/{__init__,dlp,portfolio,benchmarks,assets}.py
└── tests/{__init__,conftest,test_api_base,test_dlp_token_api,test_serializers,
           test_analysis,test_portfolio_charts,test_portfolio_misc,test_benchmarks_assets,
           test_integration}.py
apps/market/
├── assets.py                       # NOVO — universo de tickers
├── data/ativos.csv                 # NOVO — cópia de ../data/ativos.csv
├── cache.py                        # M — single_flight
├── catalog.py                      # M — EPOCH
├── synthetics.py                   # M — display_name
├── series.py                       # M — resolução via assets + fetch-on-read
├── services/sources/base.py        # M — DLPAuthError, DLPUnavailableError
├── services/sources/dlp.py         # M — erros tipados, geração, single-flight, invalidate_user_cache
├── management/commands/ingest_market_data.py   # M — tickers sob demanda (D9)
└── tests/  (test_cache, test_sources_dlp, test_assets, test_series, test_commands, test_synthetics)
engine/dataset.py                   # M — usa synthetics.display_name
```

---

## Task 1: Fundação da API — app `apps.api`, `NinjaAPI`, `ApiError`, fixtures

**Files:**
- Create: `conftest.py`, `config/api.py`, `apps/api/{__init__,apps,errors}.py`, `apps/api/routers/__init__.py`, `apps/api/tests/{__init__,test_api_base}.py`
- Modify: `config/urls.py`, `config/settings.py`

**Interfaces:**
- Produces: `apps.api.errors.ApiError(status: int, code: str, message: str, details: dict | list | None = None)` (atributos homônimos); `config.api.api` (`NinjaAPI`, `auth=django_auth`, `urls_namespace="api-v1"`, `docs_url="/docs"`); `config.api.handle_api_error(request, exc: ApiError) -> HttpResponse`; fixtures `locmem_cache`, `user` (username `"bruno"`, senha `"s3nha-forte!"`), `other_user` (`"ana"`), `auth_client` (Django `Client` com `force_login(user)`).

- [x] **Step 1: Escrever `conftest.py` (raiz de `backend/`) com as fixtures acima.** `locmem_cache` troca `settings.CACHES` por `LocMemCache` e faz `cache.clear()`. `user`/`other_user` via `User.objects.create_user`.

- [x] **Step 2: Escrever os testes que falham** — `apps/api/tests/test_api_base.py`:

```python
def test_docs_served(client):
    assert client.get("/api/v1/docs").status_code == 200

def test_openapi_title(client):
    schema = client.get("/api/v1/openapi.json").json()
    assert schema["info"]["title"] == "app-investimentos API"

def test_handle_api_error_renders_contract(rf):
    from config.api import handle_api_error
    from apps.api.errors import ApiError
    resp = handle_api_error(rf.get("/"), ApiError(409, "dlp_token_missing", "Cadastre seu token DLP.", None))
    assert resp.status_code == 409
    assert json.loads(resp.content) == {"code": "dlp_token_missing", "message": "Cadastre seu token DLP.", "details": None}
```

- [x] **Step 3: Rodar e ver falhar** — `.venv/bin/pytest apps/api -q` → FAIL (`ModuleNotFoundError: apps.api` / 404 em `/api/v1/docs`).

- [x] **Step 4: Implementar.** `apps/api/apps.py` (`name = "apps.api"`, `BigAutoField`); `ApiError` em `errors.py`; `config/api.py` com `NinjaAPI(title="app-investimentos API", version="1", auth=django_auth, urls_namespace="api-v1")` e `api.add_exception_handler(ApiError, handle_api_error)` (via `api.create_response`). `urls.py`: `path("api/v1/", api.urls)`. `settings.py`: `"apps.api"` em `INSTALLED_APPS`, `CSRF_COOKIE_HTTPONLY = False`, `MARKET_ASSETS_CSV = BASE_DIR / "apps" / "market" / "data" / "ativos.csv"`.

- [x] **Step 5: Rodar e ver passar** — `.venv/bin/pytest -q` → todos verdes (125 + 3); `.venv/bin/python manage.py check` limpo.

- [x] **Step 6: Commit**

```bash
git add conftest.py config/ apps/api/
git commit -m "✨ feat(api): fundação da API Ninja em /api/v1 com ApiError"
```

---

## Task 2: Auth de sessão — `/auth/csrf`, `login`, `logout`, `me`

**Files:**
- Create: `apps/accounts/api.py`, `apps/accounts/tests/test_auth_api.py`
- Modify: `config/api.py` (`api.add_router("/auth/", accounts_router)`)

**Interfaces:**
- Consumes: `ApiError`, `api`, fixtures da Task 1.
- Produces: `apps.accounts.api.router`; schemas `LoginIn{username, password}`, `UserOut{id, username, email}`, `MeOut(UserOut){has_dlp_token: bool}`.

- [x] **Step 1: Escrever os testes que falham:**

```python
URL = "/api/v1/auth"

def test_csrf_sets_cookie(client):
    r = client.get(f"{URL}/csrf")
    assert r.status_code == 204 and "csrftoken" in r.cookies

def test_login_ok_then_me(client, user):
    r = client.post(f"{URL}/login", {"username": "bruno", "password": "s3nha-forte!"}, content_type="application/json")
    assert r.status_code == 200 and r.json()["username"] == "bruno"
    assert client.get(f"{URL}/me").json() == {"id": user.id, "username": "bruno", "email": user.email, "has_dlp_token": False}

def test_login_invalid(client, user):
    r = client.post(f"{URL}/login", {"username": "bruno", "password": "errada"}, content_type="application/json")
    assert r.status_code == 401 and r.json()["code"] == "invalid_credentials"

def test_login_requires_csrf(user):
    c = Client(enforce_csrf_checks=True)
    r = c.post(f"{URL}/login", {"username": "bruno", "password": "s3nha-forte!"}, content_type="application/json")
    assert r.status_code == 403

def test_login_with_csrf_header_ok(user):
    c = Client(enforce_csrf_checks=True)
    token = c.get(f"{URL}/csrf").cookies["csrftoken"].value
    r = c.post(f"{URL}/login", {"username": "bruno", "password": "s3nha-forte!"},
               content_type="application/json", HTTP_X_CSRFTOKEN=token)
    assert r.status_code == 200

def test_me_unauthenticated_401(client, db):
    assert client.get(f"{URL}/me").status_code == 401

def test_me_has_dlp_token_true(auth_client, user):
    ut = UserToken(user=user); ut.set_token("tok"); ut.save()
    assert auth_client.get(f"{URL}/me").json()["has_dlp_token"] is True

def test_logout_ends_session(auth_client):
    assert auth_client.post(f"{URL}/logout").status_code == 204
    assert auth_client.get(f"{URL}/me").status_code == 401

def test_logout_without_csrf_is_403(user):
    c = Client(enforce_csrf_checks=True); c.force_login(user)
    assert c.post(f"{URL}/logout").status_code == 403
```

- [x] **Step 2: Rodar e ver falhar** — `.venv/bin/pytest apps/accounts/tests/test_auth_api.py -q` → FAIL (404).

- [x] **Step 3: Implementar `router = Router(tags=["auth"])`.** `csrf` e `login` com `auth=None`; `csrf` decorado com `ensure_csrf_cookie` e `login` com `csrf_protect` (via `ninja.decorators.decorate_view`). Login: `authenticate` + `django.contrib.auth.login`; falha → `ApiError(401, "invalid_credentials", "Usuário ou senha inválidos.")`. `has_dlp_token` = `UserToken.objects.filter(user=request.user).exists()`.

- [x] **Step 4: Rodar e ver passar** — `.venv/bin/pytest -q` → verde.

- [x] **Step 5: Commit**

```bash
git add apps/accounts/ config/api.py
git commit -m "✨ feat(accounts): endpoints de sessão /auth (csrf, login, logout, me)"
```

---

## Task 3: DLP — erros tipados, geração de cache, single-flight

**Files:**
- Modify: `apps/market/services/sources/base.py`, `apps/market/services/sources/dlp.py`, `apps/market/cache.py`
- Test: `apps/market/tests/test_sources_dlp.py`, `apps/market/tests/test_cache.py`

**Interfaces:**
- Produces:
  - `base.DLPAuthError(DLPError)`, `base.DLPUnavailableError(DLPError)`.
  - `cache.single_flight(key: str, timeout: float = 90, poll: float = 0.1)` — context manager; lock = `cache.add(key + ":lock", 1, timeout)`; quem não obtém espera (`poll`) até obter ou estourar `timeout` (aí segue com `logger.warning`); libera com `cache.delete`.
  - `dlp.invalidate_user_cache(token: str) -> None` — incrementa a geração `market:dlp:gen:<hash>` (timeout `None`); as chaves `hist`/`resumo` passam a incluir a geração.
  - `buscar_historico` / `buscar_resumo_carteira`: assinatura inalterada; em cache miss entram em `single_flight(chave)`, **re-checam o cache** e só então chamam a DLP.
  - Mapeamento HTTP (helper interno `_get(path, token, params) -> dict`): `requests.Timeout`/`ConnectionError` → `DLPUnavailableError`; status 401/403 → `DLPAuthError`; ≥500 → `DLPUnavailableError`; outro ≥400 → `DLPError`; `resp.json()` inválido → `DLPUnavailableError`. Mensagens só com o path e o status, nunca o token.

- [x] **Step 1: Escrever os testes que falham.** Em `test_sources_dlp.py`, o `_Resp` passa a não depender de `raise_for_status` (o código novo lê `status_code`). Novos testes:

```python
@pytest.mark.parametrize("status", [401, 403])
def test_auth_status_raises_dlpautherror(monkeypatch, status):
    monkeypatch.setattr(dlp.requests, "get", lambda *a, **k: _Resp({}, status=status))
    with pytest.raises(DLPAuthError) as ei:
        dlp.buscar_historico("SEGREDO-123")
    assert "SEGREDO-123" not in str(ei.value)

@pytest.mark.parametrize("exc", [requests.Timeout("t"), requests.ConnectionError("c")])
def test_network_errors_raise_unavailable(monkeypatch, exc):
    def boom(*a, **k): raise exc
    monkeypatch.setattr(dlp.requests, "get", boom)
    with pytest.raises(DLPUnavailableError):
        dlp.buscar_historico("t")

def test_5xx_raises_unavailable(monkeypatch): ...      # status=503 -> DLPUnavailableError
def test_invalid_json_raises_unavailable(monkeypatch): ...  # json() levanta ValueError
def test_other_4xx_raises_plain_dlperror(monkeypatch):
    monkeypatch.setattr(dlp.requests, "get", lambda *a, **k: _Resp({}, status=404))
    with pytest.raises(DLPError) as ei:
        dlp.buscar_historico("t")
    assert type(ei.value) is DLPError

def test_invalidate_user_cache_forces_refetch(monkeypatch):
    calls = []
    monkeypatch.setattr(dlp.requests, "get", lambda *a, **k: calls.append(1) or _Resp({"historico": [{"data": "2024-01-02", "vlr_mercado": 1}]}))
    dlp.buscar_historico("A"); dlp.buscar_historico("A")
    assert len(calls) == 1
    dlp.invalidate_user_cache("A")
    dlp.buscar_historico("A")
    assert len(calls) == 2

def test_invalidate_is_per_user(monkeypatch): ...  # invalidar "A" não refaz a chamada de "B"

def test_concurrent_calls_hit_dlp_once(monkeypatch):
    calls = []
    def slow_get(*a, **k):
        calls.append(1); time.sleep(0.3)
        return _Resp({"historico": [{"data": "2024-01-02", "vlr_mercado": 1}]})
    monkeypatch.setattr(dlp.requests, "get", slow_get)
    threads = [threading.Thread(target=dlp.buscar_historico, args=("A",)) for _ in range(4)]
    [t.start() for t in threads]; [t.join() for t in threads]
    assert len(calls) == 1
```

Em `test_cache.py` (com o `LocMemCache` já usado ali):

```python
def test_single_flight_blocks_second_holder():
    order = []
    def worker(tag):
        with cache_mod.single_flight("k", timeout=5, poll=0.01):
            order.append(f"{tag}-in"); time.sleep(0.1); order.append(f"{tag}-out")
    ts = [threading.Thread(target=worker, args=(t,)) for t in "ab"]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert order[1].endswith("-out")  # sem intercalação

def test_single_flight_times_out_and_proceeds():
    django_cache.add("k:lock", 1, 30)  # lock "preso"
    t0 = time.monotonic()
    with cache_mod.single_flight("k", timeout=0.2, poll=0.01):
        pass
    assert time.monotonic() - t0 < 1
```

- [x] **Step 2: Rodar e ver falhar** — `.venv/bin/pytest apps/market/tests/test_sources_dlp.py apps/market/tests/test_cache.py -q` → FAIL (`ImportError: DLPAuthError`, `AttributeError: single_flight`).

- [x] **Step 3: Implementar** conforme o bloco Interfaces.

- [x] **Step 4: Rodar e ver passar** — `.venv/bin/pytest -q` → verde (inclusive os testes DLP antigos e os de `engine/tests/test_dataset.py`).

- [x] **Step 5: Commit**

```bash
git add apps/market/
git commit -m "✨ feat(market): erros DLP tipados, invalidação por geração e single-flight"
```

---

## Task 4: Token DLP — `POST/DELETE /dlp/token`, `GET /dlp/token-status`

**Files:**
- Create: `apps/api/routers/dlp.py`, `apps/api/schemas.py` (começa aqui), `apps/api/tests/test_dlp_token_api.py`
- Modify: `config/api.py` (`add_router("/dlp/", ...)`)

**Interfaces:**
- Consumes: `UserToken.set_token/get_token`, `dlp.invalidate_user_cache`.
- Produces: schemas `TokenIn{token: str}`, `TokenStatusOut{configured: bool, updated_at: datetime | None}`.

- [x] **Step 1: Escrever os testes que falham:**

```python
URL = "/api/v1/dlp"

def _post(c, tok): return c.post(f"{URL}/token", {"token": tok}, content_type="application/json")

def test_save_token_encrypts_and_reports_status(auth_client, user):
    r = _post(auth_client, "  abc123\n")
    assert r.status_code == 200 and r.json()["configured"] is True
    ut = UserToken.objects.get(user=user)
    assert ut.get_token() == "abc123" and "abc123" not in ut.encrypted_token
    assert "abc123" not in r.content.decode()

def test_blank_token_is_422(auth_client):
    r = _post(auth_client, "   ")
    assert r.status_code == 422 and r.json()["code"] == "invalid_token"

def test_replace_invalidates_old_token_cache(auth_client, monkeypatch):
    seen = []
    monkeypatch.setattr("apps.api.routers.dlp.dlp.invalidate_user_cache", seen.append)
    _post(auth_client, "old"); _post(auth_client, "new")
    assert seen == ["old"]

def test_status_without_token(auth_client):
    assert auth_client.get(f"{URL}/token-status").json() == {"configured": False, "updated_at": None}

def test_delete_is_idempotent(auth_client):
    _post(auth_client, "x")
    assert auth_client.delete(f"{URL}/token").status_code == 204
    assert auth_client.delete(f"{URL}/token").status_code == 204
    assert auth_client.get(f"{URL}/token-status").json()["configured"] is False

def test_isolation_between_users(auth_client, other_user):
    _post(auth_client, "x")
    c = Client(); c.force_login(other_user)
    assert c.get(f"{URL}/token-status").json()["configured"] is False

def test_unauthenticated_401(client, db):
    assert client.get(f"{URL}/token-status").status_code == 401

def test_save_without_csrf_is_403(user):
    c = Client(enforce_csrf_checks=True); c.force_login(user)
    assert _post(c, "x").status_code == 403
```

- [x] **Step 2: Rodar e ver falhar** — `.venv/bin/pytest apps/api/tests/test_dlp_token_api.py -q` → FAIL (404).

- [x] **Step 3: Implementar o router** (`tags=["dlp"]`). Token: `strip()`; vazio → `ApiError(422, "invalid_token", "Informe o token DLP.")`. Ao substituir ou remover, chamar `dlp.invalidate_user_cache(<token antigo>)`.

- [x] **Step 4: Rodar e ver passar** — `.venv/bin/pytest -q`.

- [x] **Step 5: Commit**

```bash
git add apps/api/ config/api.py
git commit -m "✨ feat(api): endpoints do token DLP (salvar, status, remover)"
```

---

## Task 5: `assets.py` — universo de tickers do `ativos.csv`

**Files:**
- Create: `apps/market/assets.py`, `apps/market/data/ativos.csv` (`cp ../data/ativos.csv apps/market/data/ativos.csv`), `apps/market/tests/test_assets.py`, fixture `apps/market/tests/fixtures/ativos_mini.csv`
- Modify: `apps/market/tests/conftest.py` (criar se não existir) — fixture `assets_csv` que aponta `settings.MARKET_ASSETS_CSV` para o mini CSV e chama `assets.load_universe.cache_clear()` antes/depois.

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class Asset: sigla: str; nome: str; classe: str; mercado: str  # "BR" | "INTL"; yf_ticker: str` + propriedade `brl_variant -> bool` (`mercado == "INTL"`).
  - `load_universe() -> dict[str, Asset]` — chave = `yf_ticker`; lê `settings.MARKET_ASSETS_CSV` com `sep=";"`, `encoding="utf-8-sig"`; `functools.lru_cache(maxsize=1)`.
  - `resolve(name: str) -> SeriesDef | None` — `"<yf>"` → `SeriesDef(name, "YF", f"yf:{yf}", "level")`; `"<yf> BRL"` de ativo INTL → `SeriesDef(name, None, None, "level", derivation=ToBRL(yf))`; qualquer outro → `None`.
  - `search(q: str, limit: int = 20) -> list[Asset]` — ordem: sigla exata, depois prefixo de sigla, depois substring do nome; case-insensitive.
- Classes aceitas — BR: `ACAO, ETF, ETF RF, FI_INFRA, FI_SET, FIDC, FII, FIP, FIP_IE, BDR, BDR ETF`; INTL: `ETF_GB, ETF_US, REIT, STOCK`. Yahoo: `BVMF:` → `.SA`; `LON:`/`LSE:` → `.L`; demais → sigla pura.

- [x] **Step 1: Criar `ativos_mini.csv`** (com BOM) contendo: `PETR4;BVMF:PETR4;ACAO;PETROLEO BRASILEIRO`, `HGLG11;BVMF:HGLG11;FII;CSHG LOGISTICA`, `AAPL;NASDAQ:AAPL;STOCK;APPLE INC`, `GDX;LON:GDX;ETF_GB;VANECK GOLD MINERS UCITS`, `GDX;NYSEARCA:GDX;ETF_US;VANECK GOLD MINERS ETF`, `USML;LSE:USML;ETF_GB;X USML`, `BTC;CURRENCY:BTC;CRIPTO;BITCOIN`, `IBOV;INDEX:IBOV;INDICE;IBOVESPA`, `IVVB11;BVMF:IVVB11;ETF;ISHARES S&P 500`.

- [x] **Step 2: Escrever os testes que falham:**

```python
def test_yahoo_ticker_mapping(assets_csv):
    u = assets.load_universe()
    assert u["PETR4.SA"].mercado == "BR"
    assert u["AAPL"].mercado == "INTL"
    assert "GDX.L" in u and "GDX" in u          # mesma sigla, duas bolsas
    assert "USML.L" in u                         # LSE: -> .L

def test_excluded_classes(assets_csv):
    u = assets.load_universe()
    assert not any(a.classe in {"CRIPTO", "INDICE", "OURO"} for a in u.values())

def test_bom_does_not_break_header(assets_csv):
    assert assets.load_universe()["PETR4.SA"].sigla == "PETR4"

def test_resolve_raw_and_brl(assets_csv):
    assert assets.resolve("PETR4.SA") == SeriesDef("PETR4.SA", "YF", "yf:PETR4.SA", "level")
    assert assets.resolve("AAPL BRL").derivation == ToBRL("AAPL")
    assert assets.resolve("PETR4.SA BRL") is None   # BR não tem variante BRL
    assert assets.resolve("PETR4") is None          # sigla pura não é identificador
    assert assets.resolve("XYZ") is None

def test_search_ranking_and_limit(assets_csv):
    assert [a.yf_ticker for a in assets.search("gdx")][:2] in (["GDX.L", "GDX"], ["GDX", "GDX.L"])
    assert assets.search("logistica")[0].yf_ticker == "HGLG11.SA"
    assert len(assets.search("a", limit=2)) <= 2

def test_real_file_loads():
    assets.load_universe.cache_clear()
    assert len(assets.load_universe()) > 5000
```

- [x] **Step 3: Rodar e ver falhar** — `.venv/bin/pytest apps/market/tests/test_assets.py -q` → FAIL (`ModuleNotFoundError`).

- [x] **Step 4: Implementar `assets.py`** conforme Interfaces.

- [x] **Step 5: Rodar e ver passar** — `.venv/bin/pytest -q`.

- [x] **Step 6: Commit**

```bash
git add apps/market/assets.py apps/market/data/ativos.csv apps/market/tests/
git commit -m "✨ feat(market): universo de tickers do ativos.csv (assets.py)"
```

---

## Task 6: `series.py` — tickers sob demanda (fetch-on-read)

**Files:**
- Modify: `apps/market/series.py`, `apps/market/catalog.py` (`EPOCH = dt.date(1995, 1, 1)`), `apps/market/management/commands/ingest_market_data.py` (`_EPOCH = catalog.EPOCH`)
- Test: `apps/market/tests/test_series.py`

**Interfaces:**
- Consumes: `assets.resolve`, `cache.single_flight`, `yahoo.fetch_yahoo`, `repository.{last_date, upsert_points, load_series}`.
- Produces: `get_series(name, start, end)` inalterado na assinatura, agora aceitando nomes do universo de tickers; helpers internos `_resolve(name) -> tuple[SeriesDef, bool]` (`bool` = sob demanda; catálogo primeiro, depois `assets`; nenhum → `KeyError`) e `_ensure_fresh(sdef: SeriesDef, today: dt.date) -> None`; constante `STALE_DAYS = 3`.
- Regras (spec §5.6, D11, D12): só para série crua sob demanda, **após** cache miss e **antes** de materializar; sob `single_flight(f"ondemand:{series_key}")`, re-checar `last_date`; `None` → `fetch_yahoo(ticker, catalog.EPOCH, today)`; `last_date < today − STALE_DAYS` → `fetch_yahoo(ticker, last_date + 1, today)`; `upsert_points(key, "YF", "level", pts)`. `FetchError` → `logger.warning` e segue (materializa o que o banco tiver; vazio → `None`). Variante BRL não precisa de lógica nova: `ToBRL` chama `get_series(base)`, que resolve sob demanda.

- [x] **Step 1: Escrever os testes que falham** (db + `locmem_cache` + `assets_csv`; `fetch_yahoo` mockado em `series.yahoo.fetch_yahoo`; `today` fixado com monkeypatch de `series._today`, função que devolve `dt.date.today()`):

```python
def test_first_use_fetches_full_history(...):
    calls = []
    monkeypatch.setattr(series.yahoo, "fetch_yahoo", lambda t, s, e: calls.append((t, s, e)) or _pts("2024-01-02", 5))
    s = series.get_series("AAPL", dt.date(2024, 1, 1), TODAY)
    assert calls == [("AAPL", catalog.EPOCH, TODAY)]
    assert repository.last_date("yf:AAPL") is not None and not s.empty

def test_fresh_data_does_not_refetch(...):      # last_date = TODAY - 1 -> nenhuma chamada
def test_stale_data_tops_up_from_last_date(...):  # last_date = TODAY - 10 -> chamada com start = last_date + 1
def test_fetch_error_falls_back_to_stored(...):   # dados gravados + FetchError -> série devolvida, sem exceção
def test_fetch_error_without_data_returns_none(...):
def test_brl_variant_uses_ptax(...):            # grava ptax:USDBRL; "AAPL BRL" == AAPL * PTAX alinhado
def test_catalog_name_wins_over_asset(...):     # "IVVB11" resolve pelo catálogo; fetch_yahoo nunca chamado
def test_unknown_name_raises_keyerror(...):
    with pytest.raises(KeyError):
        series.get_series("NAOEXISTE", dt.date(2024, 1, 1), TODAY)
```

- [x] **Step 2: Rodar e ver falhar** — `.venv/bin/pytest apps/market/tests/test_series.py -q` → FAIL (`KeyError: série desconhecida no catálogo: 'AAPL'`).

- [x] **Step 3: Implementar** conforme Interfaces.

- [x] **Step 4: Rodar e ver passar** — `.venv/bin/pytest -q`.

- [x] **Step 5: Commit**

```bash
git add apps/market/
git commit -m "✨ feat(market): tickers sob demanda com histórico completo e single-flight"
```

---

## Task 7: `ingest_market_data` atualiza tickers sob demanda (D9)

**Files:**
- Modify: `apps/market/management/commands/ingest_market_data.py`
- Test: `apps/market/tests/test_commands.py`

**Interfaces:**
- Produces: helper `_on_demand_defs() -> list[SeriesDef]` — chaves `yf:*` de `repository.available_keys()` fora de `{d.series_key for d in catalog.ingestible()}`, cada uma como `SeriesDef(key[3:], "YF", key, "level")`. O loop de `handle` itera `catalog.ingestible() + _on_demand_defs()`, com a mesma política (`--only`, `--full`, `--since`, `failed_keys`, exit 1).

- [x] **Step 1: Escrever os testes que falham:**

```python
def test_ingest_updates_on_demand_ticker(monkeypatch, db):
    repository.upsert_points("yf:AAPL", "YF", "level", [RawPoint(dt.date(2024, 1, 2), 1.0)])
    fetched = []
    monkeypatch.setattr(cmd_mod, "_fetch", lambda sdef, s, e: fetched.append(sdef.series_key) or [])
    call_command("ingest_market_data", "--only", "yf:AAPL")
    assert fetched == ["yf:AAPL"]

def test_on_demand_failure_marks_failed(monkeypatch, db):
    # _fetch levanta FetchError para yf:AAPL -> SystemExit(1) e "yf:AAPL" no stderr

def test_catalog_yf_key_not_duplicated(db):
    repository.upsert_points("yf:IVVB11.SA", "YF", "level", [RawPoint(dt.date(2024, 1, 2), 1.0)])
    assert "yf:IVVB11.SA" not in {d.series_key for d in cmd_mod._on_demand_defs()}
```

- [x] **Step 2: Rodar e ver falhar** — `.venv/bin/pytest apps/market/tests/test_commands.py -q` → FAIL.

- [x] **Step 3: Implementar.**

- [x] **Step 4: Rodar e ver passar** — `.venv/bin/pytest -q`.

- [x] **Step 5: Commit**

```bash
git add apps/market/
git commit -m "✨ feat(market): ingestão também atualiza tickers buscados sob demanda"
```

---

## Task 8: `serializers.py` — pandas → JSON

**Files:**
- Create: `apps/api/serializers.py`, `apps/api/tests/test_serializers.py`

**Interfaces:**
- Produces (funções puras):
  - `clean(v) -> float | None` — `NaN`/`±inf`/`None` → `None`; numpy → `float`.
  - `series_to_json(s: pd.Series, name: str | None = None) -> dict` — `{"name": name or str(s.name), "data": [[date_iso, clean(v)], ...]}`.
  - `frame_to_series_list(df: pd.DataFrame) -> list[dict]` — uma entrada por coluna, na ordem do DataFrame.
  - `frame_to_rows(df: pd.DataFrame, columns: dict[str, str]) -> list[dict]` — cada linha = `{"name": str(index), <novo>: clean(df[<orig>])}`.
  - `yearly_to_json(df: pd.DataFrame) -> dict` — `{"years": [int...], "rows": [{"name", "values": {str(ano): float|None}, "total": float|None}]}` (coluna `"Total Acum."` vira `total`).

- [x] **Step 1: Escrever os testes que falham:**

```python
def test_clean_handles_nan_inf_numpy():
    assert clean(float("nan")) is None and clean(np.inf) is None and clean(-np.inf) is None
    assert clean(np.float64(1.5)) == 1.5 and type(clean(np.float64(1.5))) is float

def test_series_to_json_iso_dates_and_nulls():
    s = pd.Series([1.0, np.nan], index=pd.to_datetime(["2024-01-02", "2024-01-03"]), name="CDI")
    assert series_to_json(s) == {"name": "CDI", "data": [["2024-01-02", 1.0], ["2024-01-03", None]]}

def test_frame_to_series_list_preserves_column_order():
    df = pd.DataFrame({"B": [1.0], "A": [2.0]}, index=pd.to_datetime(["2024-01-02"]))
    assert [s["name"] for s in frame_to_series_list(df)] == ["B", "A"]

def test_frame_to_rows_renames():
    df = pd.DataFrame({"Volatilidade": [0.1], "Retorno (CAGR)": [0.2], "Sharpe": [1.0]}, index=["Carteira"])
    assert frame_to_rows(df, {"Volatilidade": "volatilidade", "Retorno (CAGR)": "retorno_cagr", "Sharpe": "sharpe"}) == \
        [{"name": "Carteira", "volatilidade": 0.1, "retorno_cagr": 0.2, "sharpe": 1.0}]

def test_yearly_to_json_shape():
    df = pd.DataFrame({2023: [0.1], 2024: [np.nan], "Total Acum.": [0.3]}, index=["Carteira"])
    assert yearly_to_json(df) == {"years": [2023, 2024],
                                  "rows": [{"name": "Carteira", "values": {"2023": 0.1, "2024": None}, "total": 0.3}]}
```

- [x] **Step 2: Rodar e ver falhar** — `.venv/bin/pytest apps/api/tests/test_serializers.py -q` → FAIL.
- [x] **Step 3: Implementar.**
- [x] **Step 4: Rodar e ver passar.**
- [x] **Step 5: Commit**

```bash
git add apps/api/
git commit -m "✨ feat(api): serializers pandas -> JSON no formato do ECharts"
```

---

## Task 9: `analysis.py` — validação, token e orquestração

**Files:**
- Create: `apps/api/analysis.py`, `apps/api/tests/test_analysis.py`
- Modify: `apps/api/schemas.py`, `apps/market/synthetics.py`, `engine/dataset.py`, `apps/market/tests/test_synthetics.py`

**Interfaces:**
- Consumes: `fetch_user_portfolio`, `build_dataset` (`engine.dataset`), `catalog`, `assets.resolve`, erros DLP (Task 3), `ApiError`.
- Produces:
  - `synthetics.display_name(spec: SyntheticSpec) -> str` — `"+"` → `f"{base} + {valor:g}%"`; `"%"` → `f"{valor:g}% do {base}"` (texto idêntico ao de `build_dataset`, que passa a usá-la).
  - Schemas: `SyntheticIn{base: str, operador: str, valor: float}`; `AnalysisRequest{start: date|None, end: date|None, ativo: str|None, classe: str|None, benchmarks: list[str]|None, synthetics: list[SyntheticIn] = []}`; `RollingRequest(AnalysisRequest){window: int = 252}`.
  - `MAX_ITEMS = 15`.
  - `@dataclass(frozen=True) class ValidRequest: start: date; end: date; benchmarks: list[str]; specs: list[SyntheticSpec]`.
  - `validate_request(req: AnalysisRequest, today: date | None = None) -> ValidRequest` — levanta `ApiError` 422 com os códigos do Global Constraints; `unknown_benchmark` com `details={"invalid": [...]}`. Benchmark é válido se `catalog.resolve` aceita **ou** `assets.resolve(name) is not None`.
  - `get_token(user) -> str` — sem `UserToken` → `ApiError(409, "dlp_token_missing", ...)`.
  - `@dataclass class AnalysisContext: df_combined: pd.DataFrame; portfolio_df: pd.DataFrame; selic: pd.Series | None; start: date; end: date; missing: list[str]`.
  - `run_analysis(user, req: AnalysisRequest) -> AnalysisContext` — `validate_request` → `get_token` → `fetch_user_portfolio(token, ativo=, classe=, start_date=iso, end_date=iso)` (`DLPAuthError` → 422 `dlp_token_rejected`; demais `DLPError` → 502 `dlp_unavailable`; `None` → 404 `portfolio_empty`) → `build_dataset(...)`. `start`/`end` do contexto = primeiro/último índice de `df_combined`; `missing` = benchmarks + `display_name(spec)` ausentes das colunas.

- [x] **Step 1: Escrever os testes que falham.** Em `test_synthetics.py`: `display_name` para `("IPCA","+",6.0)` → `"IPCA + 6%"` e `("CDI","%",110.0)` → `"110% do CDI"`. Em `test_analysis.py` (validação é pura; `run_analysis` com `analysis.fetch_user_portfolio` e `analysis.build_dataset` monkeypatchados):

```python
TODAY = dt.date(2026, 10, 26)

def _req(**kw): return AnalysisRequest(**kw)

def _code(fn):
    with pytest.raises(ApiError) as ei: fn()
    return ei.value.status, ei.value.code

def test_defaults():
    v = validate_request(_req(), today=TODAY)
    assert (v.start, v.end) == (dt.date(2025, 10, 26), TODAY)
    assert v.benchmarks == catalog.DEFAULT_BENCHMARKS

def test_empty_benchmarks_is_valid():
    assert validate_request(_req(benchmarks=[]), today=TODAY).benchmarks == []

def test_period_inverted():
    assert _code(lambda: validate_request(_req(start=TODAY, end=dt.date(2026, 1, 1)), today=TODAY)) == (422, "invalid_period")

def test_unknown_benchmark_lists_invalid(assets_csv):
    with pytest.raises(ApiError) as ei:
        validate_request(_req(benchmarks=["CDI", "PETR4.SA BRL", "XPTO"]), today=TODAY)
    assert ei.value.code == "unknown_benchmark" and ei.value.details == {"invalid": ["PETR4.SA BRL", "XPTO"]}

def test_ticker_and_brl_variant_accepted(assets_csv):
    assert validate_request(_req(benchmarks=["PETR4.SA", "AAPL BRL"]), today=TODAY).benchmarks == ["PETR4.SA", "AAPL BRL"]

@pytest.mark.parametrize("syn,ok", [
    ({"base": "CDI", "operador": "+", "valor": 50}, True),
    ({"base": "CDI", "operador": "+", "valor": 50.01}, False),
    ({"base": "CDI", "operador": "+", "valor": 0}, False),
    ({"base": "CDI", "operador": "%", "valor": 500}, True),
    ({"base": "CDI", "operador": "%", "valor": 501}, False),
    ({"base": "SELIC", "operador": "+", "valor": 2}, False),
    ({"base": "IPCA", "operador": "x", "valor": 2}, False),
])
def test_synthetic_bounds(syn, ok): ...  # ok -> sem exceção; senão (422, "invalid_synthetic")

def test_too_many_items():
    req = _req(benchmarks=["CDI"] * 14, synthetics=[{"base": "CDI", "operador": "+", "valor": 1}] * 2)
    assert _code(lambda: validate_request(req, today=TODAY)) == (422, "too_many_benchmarks")

def test_no_token_409(user): ...             # run_analysis -> (409, "dlp_token_missing")
def test_dlp_auth_error_422(user_with_token, monkeypatch): ...      # (422, "dlp_token_rejected")
def test_dlp_unavailable_502(user_with_token, monkeypatch): ...     # DLPUnavailableError e DLPError -> (502, "dlp_unavailable")
def test_empty_portfolio_404(user_with_token, monkeypatch): ...     # fetch devolve (None, None) -> (404, "portfolio_empty")
def test_filters_and_specs_forwarded(user_with_token, monkeypatch):
    # captura kwargs: fetch recebe ativo="PETR4", classe="ACAO"; build_dataset recebe extra_specs=[SyntheticSpec("CDI","+",2.0)]
def test_missing_lists_absent_names(user_with_token, monkeypatch):
    # build_dataset devolve df só com Carteira e CDI; pedidos ["CDI","IMA-B"] + CDI+2 -> missing == ["IMA-B", "CDI + 2%"]
```

Fixtures em `apps/api/tests/conftest.py`: `user_with_token` (`user` + `UserToken` com `"tok-teste"`) e `user_with_token_client` (`Client` com `force_login` desse usuário) — reusadas nas Tasks 10 e 11.

- [x] **Step 2: Rodar e ver falhar** — `.venv/bin/pytest apps/api/tests/test_analysis.py apps/market/tests/test_synthetics.py -q` → FAIL.
- [x] **Step 3: Implementar** `display_name` (e usá-la em `engine/dataset.py`), os schemas e `analysis.py`.
- [x] **Step 4: Rodar e ver passar** — `.venv/bin/pytest -q` (inclusive `engine/tests`).
- [x] **Step 5: Commit**

```bash
git add apps/api/ apps/market/synthetics.py apps/market/tests/test_synthetics.py engine/dataset.py
git commit -m "✨ feat(api): validação do AnalysisRequest e orquestração run_analysis"
```

---

## Task 10: Endpoints de gráfico da carteira

**Files:**
- Create: `apps/api/routers/portfolio.py`, `apps/api/tests/test_portfolio_charts.py`
- Modify: `apps/api/schemas.py` (schemas de resposta), `apps/api/tests/conftest.py`, `config/api.py` (`add_router("/portfolio/", ...)`)

**Interfaces:**
- Consumes: `run_analysis`, `serializers.*`, `engine.compute.*`.
- Produces: rotas `POST /portfolio/{twr-evolution, drawdown, rolling-metrics, risk-return, shadow, yearly-summary, irr-evolution}` com `response=` declarado (Swagger mostra o formato); helper `_envelope(ctx: AnalysisContext, data: dict) -> dict`. Payloads de `data` conforme spec §5.5:
  - `twr-evolution` → `{series, final_returns}`; `drawdown`/`shadow` → `{series}`; `irr-evolution` → `{series: [Serie "TIR"]}`;
  - `rolling-metrics` (`RollingRequest`; `window` fora de `21..756` → 422 `invalid_window`; usa `ctx.selic` como `rf_series`) → `{volatility, sharpe}`;
  - `risk-return` → `{rows: [{name, volatilidade, retorno_cagr, sharpe}]}`; `yearly-summary` → `yearly_to_json`.
- Fixture `portfolio_env` (conftest de `apps/api/tests`): `user_with_token` logado + monkeypatch de `apps.api.analysis.fetch_user_portfolio`, devolvendo uma carteira determinística de ~400 dias úteis (`twr` + `portfolio_df` com `date, vlr_mercado, vlr_investido, proventos, fluxo`, começando em `2024-03-01`), e de `engine.dataset.series.get_many/get_series`, servindo `CDI`/`SELIC` sintéticos a partir de `2024-01-02` (padrão de `engine/tests/test_dataset.py`).

- [x] **Step 1: Escrever os testes que falham:**

```python
P = "/api/v1/portfolio"
BODY = {"start": "2024-01-02", "end": "2025-06-30", "benchmarks": ["CDI"]}

def _post(c, path, body=BODY): return c.post(f"{P}/{path}", body, content_type="application/json")

@pytest.mark.parametrize("path", ["twr-evolution", "drawdown", "rolling-metrics", "risk-return",
                                  "shadow", "yearly-summary", "irr-evolution"])
def test_chart_endpoints_return_envelope(portfolio_env, path):
    r = _post(portfolio_env, path)
    assert r.status_code == 200 and set(r.json()) == {"data", "meta"}

@pytest.mark.parametrize("path", ["twr-evolution", "drawdown"])
def test_chart_requires_session(client, db, path):
    assert _post(client, path).status_code == 401

def test_twr_values_match_engine(portfolio_env):
    # monta o mesmo dataset chamando build_dataset/compute_twr_evolution direto e compara
    # o último ponto de "Carteira" e final_returns com o JSON (abs diff < 1e-9)

def test_rolling_metrics_is_strict_json(portfolio_env):
    def no_nan(x): raise AssertionError(f"constante não-JSON {x}")
    body = json.loads(_post(portfolio_env, "rolling-metrics").content, parse_constant=no_nan)
    assert body["data"]["volatility"][0]["data"][0][1] is None

def test_rolling_window_out_of_range(portfolio_env):
    r = _post(portfolio_env, "rolling-metrics", {**BODY, "window": 10})
    assert r.status_code == 422 and r.json()["code"] == "invalid_window"

def test_missing_benchmark_reported_in_meta(portfolio_env):
    r = _post(portfolio_env, "twr-evolution", {**BODY, "benchmarks": ["CDI", "IMA-B"]})  # IMA-B sem dados no fake
    assert r.status_code == 200 and r.json()["meta"]["missing"] == ["IMA-B"]

def test_meta_reflects_effective_period(portfolio_env):
    assert _post(portfolio_env, "twr-evolution").json()["meta"]["start"] == "2024-03-01"  # carteira começa depois do start

def test_no_token_is_409(auth_client):
    r = _post(auth_client, "twr-evolution")
    assert r.status_code == 409 and r.json()["code"] == "dlp_token_missing"

def test_yearly_summary_shape(portfolio_env):
    d = _post(portfolio_env, "yearly-summary").json()["data"]
    assert d["years"] == [2024, 2025] and {r["name"] for r in d["rows"]} >= {"Carteira", "CDI"}
```

- [x] **Step 2: Rodar e ver falhar** — `.venv/bin/pytest apps/api/tests/test_portfolio_charts.py -q` → FAIL (404).
- [x] **Step 3: Implementar o router** (`tags=["portfolio"]`): cada handler = `ctx = run_analysis(request.user, body)` + um `compute_*` + serializer + `_envelope`.
- [x] **Step 4: Rodar e ver passar** — `.venv/bin/pytest -q`.
- [x] **Step 5: Commit**

```bash
git add apps/api/ config/api.py
git commit -m "✨ feat(api): endpoints de gráfico da carteira (TWR, drawdown, móveis, risco, shadow, anual, TIR)"
```

---

## Task 11: `summary`, `filters` e `refresh`

**Files:**
- Modify: `apps/api/routers/portfolio.py`, `apps/api/schemas.py`
- Create: `apps/api/tests/test_portfolio_misc.py`

**Interfaces:**
- Consumes: `run_analysis`, `get_token`, `dlp.buscar_historico`, `dlp.invalidate_user_cache`, `compute_risk_return`, `compute_drawdown`, `compute_irr_evolution`, `compute_twr_evolution`.
- Produces:
  - `POST /portfolio/summary` → `data = {patrimonio_inicial, aportes_periodo, patrimonio, ganho_periodo, twr_total, cagr, volatilidade, sharpe, max_drawdown, xirr, data_inicio, data_fim}`, com as fórmulas exatas do spec §5.5 (sobre `ctx.portfolio_df` e a coluna `Carteira`).
  - `GET /portfolio/filters` → `{ativos: [str], classes: [str]}` (ordenados, distintos), do `buscar_historico(token)` da carteira inteira; colunas candidatas `ASSET_COLUMNS = ("ativo", "ticker", "codigo")`, `CLASS_COLUMNS = ("classe", "tipo")`, usa a primeira presente; nenhuma → lista vazia + `logger.warning`. Erros DLP mapeados como em `run_analysis` (extrair o mapeamento para um helper compartilhado em `analysis.py`: `dlp_errors()` context manager).
  - `POST /portfolio/refresh` → `dlp.invalidate_user_cache(get_token(user))`, `204`.

- [x] **Step 1: Escrever os testes que falham:**

```python
def test_summary_period_values(portfolio_env):
    d = post(portfolio_env, "summary").json()["data"]
    df = PORTFOLIO_DF  # o mesmo DataFrame determinístico da fixture
    assert d["patrimonio_inicial"] == pytest.approx(df["vlr_mercado"].iloc[0])
    assert d["aportes_periodo"] == pytest.approx(df["vlr_investido"].iloc[-1] - df["vlr_investido"].iloc[0])
    assert d["patrimonio"] == pytest.approx(df["vlr_mercado"].iloc[-1])
    assert d["patrimonio_inicial"] + d["aportes_periodo"] + d["ganho_periodo"] == pytest.approx(d["patrimonio"])
    assert d["max_drawdown"] <= 0

def test_filters_lists_distinct_sorted(user_with_token_client, monkeypatch):
    df = pd.DataFrame({"date": ["2024-01-02"] * 3, "ativo": ["PETR4", "HGLG11", "PETR4"], "classe": ["ACAO", "FII", "ACAO"]})
    monkeypatch.setattr("apps.api.routers.portfolio.dlp.buscar_historico", lambda tok: df)
    assert user_with_token_client.get(f"{P}/filters").json() == {"ativos": ["HGLG11", "PETR4"], "classes": ["ACAO", "FII"]}

def test_filters_unknown_columns_returns_empty(...):     # df sem colunas candidatas -> {"ativos": [], "classes": []}
def test_filters_dlp_rejected_422(...):                  # DLPAuthError -> 422 dlp_token_rejected

def test_refresh_invalidates_user_cache(user_with_token_client, monkeypatch):
    seen = []
    monkeypatch.setattr("apps.api.routers.portfolio.dlp.invalidate_user_cache", seen.append)
    assert user_with_token_client.post(f"{P}/refresh").status_code == 204 and seen == ["tok-teste"]

def test_refresh_without_token_409(auth_client): ...
```

- [x] **Step 2: Rodar e ver falhar.**
- [x] **Step 3: Implementar.**
- [x] **Step 4: Rodar e ver passar** — `.venv/bin/pytest -q`.
- [x] **Step 5: Commit**

```bash
git add apps/api/
git commit -m "✨ feat(api): resumo do período, filtros da carteira e refresh do cache DLP"
```

---

## Task 12: `/benchmarks/catalog` e `/assets/search`

**Files:**
- Create: `apps/api/routers/benchmarks.py`, `apps/api/routers/assets.py`, `apps/api/tests/test_benchmarks_assets.py`
- Modify: `apps/api/schemas.py`, `config/api.py`

**Interfaces:**
- Consumes: `catalog.all_names`, `catalog.resolve`, `catalog.DEFAULT_BENCHMARKS`, `assets.search`.
- Produces:
  - `GET /benchmarks/catalog` → `{groups: [{name, items}], defaults, synthetic: {bases: ["IPCA","CDI"], operadores: ["+","%"]}}`. O agrupamento é do implementador (ex.: por `source`/derivação); só precisa respeitar o teste.
  - `GET /assets/search?q=&limit=20` → `[{sigla, nome, classe, mercado, yf_ticker, brl_variant}]`; `len(q.strip()) < 2` → 422 `invalid_query`; `limit` limitado a `1..50` (fora → 422 `invalid_query`).

- [x] **Step 1: Escrever os testes que falham:**

```python
def test_catalog_groups_only_known_names(auth_client):
    d = auth_client.get("/api/v1/benchmarks/catalog").json()
    names = [n for g in d["groups"] for n in g["items"]]
    assert names and set(names) <= set(catalog.all_names())
    assert d["defaults"] == catalog.DEFAULT_BENCHMARKS
    assert d["synthetic"] == {"bases": ["IPCA", "CDI"], "operadores": ["+", "%"]}

def test_search_returns_assets(auth_client, assets_csv):
    r = auth_client.get("/api/v1/assets/search", {"q": "aapl"})
    assert r.json()[0] == {"sigla": "AAPL", "nome": "APPLE INC", "classe": "STOCK",
                           "mercado": "INTL", "yf_ticker": "AAPL", "brl_variant": True}

@pytest.mark.parametrize("params", [{"q": "a"}, {"q": "  "}, {"q": "pe", "limit": 51}, {"q": "pe", "limit": 0}])
def test_search_invalid_query(auth_client, assets_csv, params):
    r = auth_client.get("/api/v1/assets/search", params)
    assert r.status_code == 422 and r.json()["code"] == "invalid_query"

def test_catalog_and_search_require_session(client, db):
    assert client.get("/api/v1/benchmarks/catalog").status_code == 401
    assert client.get("/api/v1/assets/search", {"q": "pe"}).status_code == 401
```

- [x] **Step 2: Rodar e ver falhar.**
- [x] **Step 3: Implementar.**
- [x] **Step 4: Rodar e ver passar** — `.venv/bin/pytest -q`.
- [x] **Step 5: Commit**

```bash
git add apps/api/ config/api.py
git commit -m "✨ feat(api): catálogo de benchmarks e busca de tickers"
```

---

## Task 13: Integração real, roteiro Swagger, ADRs e documentação

**Files:**
- Create: `apps/api/tests/test_integration.py`, `docs/adr/0013-contrato-da-api-v1.md`, `docs/adr/0014-tickers-sob-demanda.md`
- Modify: `docs/adr/README.md`, `docs/consolidacao_arquitetura_e_migracao.md`, `docs/README.md`, `backend/README.md`, `backend/CLAUDE.md`, `apps/api/routers/portfolio.py` (se a DLP real usar outros nomes de coluna) e `apps/market/services/sources/dlp.py` (se o token inválido não vier como 401/403)

- [x] **Step 1: Escrever os testes `integration`** (pulam sem `DLP_TEST_TOKEN` no ambiente):
  - `test_real_dlp_historico_columns` — `buscar_historico(token)` devolve DataFrame; imprime `sorted(df.columns)` (`-s`) para conferir as colunas de ativo/classe.
  - `test_real_dlp_invalid_token_raises_auth_error` — token `"invalido-xyz"` → `DLPAuthError`.
  - `test_real_yahoo_on_demand_aapl` (db) — `series.get_series("AAPL", hoje − 30 dias, hoje)` não vazio.

- [x] **Step 2: Rodar** — `DLP_TEST_TOKEN=<token> .venv/bin/pytest -m integration apps/api/tests/test_integration.py -s -q`. Se as colunas ou o status de token inválido divergirem do que o código assume, ajustar `ASSET_COLUMNS`/`CLASS_COLUMNS` ou o mapeamento de `_get` com teste unitário novo (ver falhar → passar) e registrar o achado no ADR 0013.

- [x] **Step 3: Roteiro manual no Swagger** (spec §3.3, item 3) com `manage.py runserver`: criar usuário no admin → `GET /auth/csrf` → `POST /auth/login` → `POST /dlp/token` (token real) → `GET /dlp/token-status` → os 8 endpoints de carteira com benchmarks default + sintético `{"base":"CDI","operador":"+","valor":2}` + `"AAPL BRL"` → `GET /portfolio/filters` e um endpoint com `classe` → `POST /portfolio/refresh` → `POST /auth/logout` → endpoint protegido = 401. Conferir no stdout do servidor que nenhum token aparece. Registrar o resultado (ok/achados) no fecho da fase, em `docs/consolidacao_arquitetura_e_migracao.md` §2.0.

- [x] **Step 4: Escrever os ADRs.** 0013 — contrato da API (D1–D5 + tabela de erros + achados da DLP real); 0014 — tickers sob demanda (D7–D12). Adicionar uma linha de cada no índice `docs/adr/README.md`.

- [x] **Step 5: Checklist de documentação** de [`docs/README.md`](../../README.md#checklist-de-atualização-ao-fechar-uma-fase): doc mestre §2.0 (linha "API (Fase 3)"), §2.2 (remover "Sem API HTTP"), §4, §5 (registrar a **Fase 3b — Planejador de IR**), §9 (marcar Fase 3, inserir 3b); `backend/README.md` (Estado, Estrutura, contagem de testes, como abrir o Swagger); `backend/CLAUDE.md` (convenções de `apps/api`: routers finos, orquestração em `analysis.py`, `ApiError`, 422 para token rejeitado); `docs/README.md`.

- [x] **Step 6: Verificação final** — `.venv/bin/pytest -q` verde; `.venv/bin/python manage.py check` limpo; `.venv/bin/python manage.py makemigrations --check --dry-run` → "No changes detected".

- [x] **Step 7: Commit**

```bash
git add -A apps/ ../docs/ README.md CLAUDE.md
git commit -m "📝 docs: fecha a Fase 3 — ADRs 0013/0014, doc mestre, READMEs"
```
