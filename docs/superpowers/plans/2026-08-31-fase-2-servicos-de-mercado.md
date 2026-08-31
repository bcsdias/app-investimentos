# Fase 2 — Serviços de Mercado — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Quebrar `src/data/sources/market_data.py` (796 linhas) num app `apps/market/` com camadas de responsabilidade única, gravando dados **crus** em `core.MarketSeries` (Postgres = fonte de verdade) e usando Redis como cache derivado, com dois comandos: `backfill_market_data` (offline, dos CSVs do repo) e `ingest_market_data` (ao vivo, agendável na Fase 5).

**Architecture:** Camadas: `services/sources/*` só falam com a fonte externa e devolvem `list[RawPoint]` cru (levantam `FetchError`, nunca `return None` silencioso). `repository.py` é o único acesso a `MarketSeries` (upsert em lote, `load_series` → `pd.Series`). `transforms.py` são funções puras de derivação (cumprod, splice ANBIMA→ETF, USD→BRL, sintéticos). `catalog.py` são `SeriesDef` em Python. `series.py` é a fachada de leitura: Redis → `repository` + `transforms` conforme o catálogo → Redis. Os comandos orquestram: percorrem o catálogo, chamam o fetcher (ou leem o CSV), gravam via `repository`, aquecem o cache.

**Tech Stack:** Django 5.2, `pandas~=2.2`, `numpy~=2.1`, `yfinance~=0.2`, `python-bcb~=0.3`, `requests~=2.32`, `selenium~=4.25` (inerte no host), `redis`/`django-redis` (já da Fase 1), `pytest-django`.

**Spec:** `docs/superpowers/specs/2026-08-31-fase-2-e-2-5-servicos-e-engine-design.md`

## Global Constraints

Todo task herda implicitamente esta seção.

- **Branch:** `migracao-django`. Não commitar em `main`/`dev`.
- **Python:** sempre `.venv/bin/python` e `.venv/bin/pytest` a partir de `backend/` (não há `python` no PATH da VM). Config vem só de `backend/.env` (`python-dotenv` puro).
- **Banco:** database e role `appinvest`, sempre. O app **nunca** usa o superusuário `postgres`. As tabelas do app **não** podem cair no `banco_lab`.
- **`MarketSeries` guarda valor CRU** por fonte (taxa CDI %, close do ETF, PU do Tesouro, PTAX venda). Toda derivação acontece em `transforms.py` na leitura. Nunca gravar série já acumulada / com splice / convertida.
- **Fonte de verdade = Postgres (`core.MarketSeries`).** Redis é cache derivado com TTL. Cache miss → reconstrói do Postgres.
- **`kind` por-linha** em `MarketSeries`: `"rate"` (precisa de cumprod: `bcb:12`, `bcb:11`, `bcb:433`) | `"level"` (todo o resto). A ingestão sempre grava `kind` a partir do `catalog.py`.
- **Taxonomia:** `source ∈ {BCB, YF, TD, B3, PTAX}`; `series_key` = `bcb:<codigo>` / `yf:<ticker>` / `td:<titulo>:<venc-ISO>` / `b3:<indice>` / `ptax:USDBRL`.
- **Sem silent failures:** `services/sources/*` levantam `FetchError`/`DLPError`, nunca `except: return None`. `repository`/`series` propagam erro de programação. `cache` loga e segue (cache é otimização). `None` só significa "não existe", nunca "deu erro".
- **Chave de cache inclui a janela de datas** (`market:series:<nome>:<start>:<end>`) — corrige o bug do legado.
- **Sintéticos estruturados:** `SyntheticSpec{base, operador, valor}`. A string legada (`"IPCA + 6%"`) só via `parse_legacy_string`.
- **Selenium do B3:** `fetch_b3_selenium` é portado e **testado só com mock** (sem Chrome no host). `ingest` usa `read_b3_static` por padrão; `settings.B3_INGEST_USE_SELENIUM` (default `False`).
- **Logging:** só `logging.getLogger(__name__)` → stdout (ADR 0008). Nada de `print` nos módulos; nos comandos, usar `self.stdout.write` para o resumo e `logger` para o resto.
- **Pandas `pct_change`:** sempre `fill_method=None` explícito (evita o `FutureWarning` do pandas 2.2 e divergência de paridade — spec §5.4).
- **Testes de `repository`/`series`/comandos:** contra `appinvest_test` (o `bulk_create(update_conflicts=True)` exige Postgres). Testes de `transforms`/`synthetics`/`cache` são puros (sem banco).
- **Chamadas ao vivo:** só em testes marcados `@pytest.mark.integration`, desmarcados por padrão.
- **Commits:** Conventional Commits + Gitmoji + imperativo em português. **Nunca** `Co-Authored-By`. Um commit por task, com a mensagem que o task especifica.
- **TDD obrigatório:** escrever o teste, **rodar e ver falhar** com o erro esperado, só então implementar, rodar e ver passar. Os passos "ver falhar" não são opcionais.

---

## Estrutura de arquivos

Tudo em `backend/`. Criar salvo indicação `MODIFICAR`.

```
apps/market/
├── __init__.py
├── apps.py                              # name = "apps.market"; default_auto_field = BigAutoField
├── cache.py                             # get_df / set_df / make_key ; TTL_SERIES, TTL_DLP
├── repository.py                        # upsert_points / load_series / last_date / available_keys
├── synthetics.py                        # SyntheticSpec ; parse_legacy_string
├── transforms.py                        # accumulate / splice_proxy / to_brl / synthetic_from_spec / composite / rebase_100
├── catalog.py                          # SeriesDef + Derivation (ProxySplice/ToBRL/Synthetic/Composite) ; resolve / ingestible / DEFAULT_BENCHMARKS / legacy_csv_name
├── series.py                           # get_series / get_many / _materialize
├── services/
│   ├── __init__.py
│   └── sources/
│       ├── __init__.py
│       ├── base.py                     # RawPoint (dataclass) ; FetchError ; DLPError
│       ├── bcb.py                      # fetch_sgs(codigo, start, end) -> list[RawPoint]
│       ├── ptax.py                     # fetch_ptax(start, end) -> list[RawPoint]
│       ├── yahoo.py                    # fetch_yahoo(ticker, start, end) -> list[RawPoint]
│       ├── tesouro.py                  # fetch_tesouro(titulo, vencimento, start, end) -> list[RawPoint]
│       ├── b3.py                       # read_b3_static(...) ; fetch_b3_selenium(...)
│       └── dlp.py                      # buscar_historico ; buscar_resumo_carteira  (LIVE, por-usuário)
├── management/
│   ├── __init__.py
│   └── commands/
│       ├── __init__.py
│       ├── backfill_market_data.py
│       └── ingest_market_data.py
└── tests/
    ├── __init__.py
    ├── conftest.py                     # fixture paths, mock helpers
    ├── fixtures/                       # CSVs reais copiados de ../data (Task 1)
    │   ├── BCB_12.csv  BCB_433.csv  BCB_12466.csv
    │   ├── YF_GSPC.csv  YF_IMAB11_SA.csv
    │   ├── IFIX_all.csv
    │   └── TD_Tesouro_IPCAmais_15-05-2035.csv
    ├── test_cache.py
    ├── test_repository.py
    ├── test_sources_bcb.py
    ├── test_sources_ptax.py
    ├── test_sources_yahoo.py
    ├── test_sources_tesouro.py
    ├── test_sources_b3.py
    ├── test_sources_dlp.py
    ├── test_synthetics.py
    ├── test_transforms.py
    ├── test_catalog.py
    ├── test_series.py
    └── test_commands.py

apps/core/models.py                     # MODIFICAR — campo `kind` em MarketSeries
apps/core/admin.py                      # MODIFICAR — `kind` no MarketSeriesAdmin
apps/core/migrations/0002_marketseries_kind.py   # via makemigrations
config/settings.py                      # MODIFICAR — INSTALLED_APPS += "apps.market"; B3_INGEST_USE_SELENIUM = False
pytest.ini                              # MODIFICAR — markers + addopts -m "not integration"
requirements.txt                        # MODIFICAR — pandas, numpy, yfinance, python-bcb, requests, selenium, webdriver-manager
```

---

## Ferramentas recomendadas nesta fase

Ver `docs/ferramentas-de-desenvolvimento.md` §"Fase 2 / 2.5":

- **`context7`** (MCP) — API atual de `python-bcb`, `yfinance`, `redis-py` antes de escrever `services/sources/*` e `cache.py`. Travar as versões exatas de patch no `requirements.txt` (Task 1).
- **`pyright-lsp`** — diagnóstico contínuo (muito `pandas`).
- **`pr-review-toolkit` → `silent-failure-hunter`** — **prioritário**. Rodar sobre o diff de `apps/market/` na Task 13. Não pode sobrar `except: ... return None`.
- **`pr-review-toolkit` → `pr-test-analyzer`** — cobertura dos testes de `transforms`/`series` na Task 13.
- **`semgrep`** (MCP) — `get_semgrep_supply_chain_findings` no `requirements.txt` novo (Task 1).
- **`code-review`** + **`code-simplifier`** — ao fechar a Task 13, sobre todo o diff.

---

## Task 1: Fundação — deps, app skeleton, campo `kind`, settings, pytest markers

**Files:**
- Modify: `backend/requirements.txt`
- Modify: `backend/config/settings.py`
- Modify: `backend/pytest.ini`
- Create: `backend/apps/market/__init__.py`
- Create: `backend/apps/market/apps.py`
- Create: `backend/apps/market/tests/__init__.py`
- Create: `backend/apps/market/tests/fixtures/` (copiar CSVs)
- Modify: `backend/apps/core/models.py`
- Modify: `backend/apps/core/admin.py`
- Create: `backend/apps/core/migrations/0002_marketseries_kind.py` (via `makemigrations`)

**Interfaces:**
- Consumes: `core.MarketSeries` (Fase 1).
- Produces:
  - App `apps.market` registrado em `INSTALLED_APPS`.
  - `core.MarketSeries.kind` — `CharField(max_length=8)`, valores `"rate"` | `"level"`.
  - `settings.B3_INGEST_USE_SELENIUM: bool` (default `False`).
  - `pytest.ini` com marker `integration` e `addopts = --reuse-db -m "not integration"`.
  - `backend/apps/market/tests/fixtures/*.csv` — cópias versionadas de `../data/raw` e `../data/static`.

- [ ] **Step 1: Adicionar as dependências ao `requirements.txt`**

Inserir na seção `# --- runtime ---` de `backend/requirements.txt`, logo após `gunicorn~=23.0` (pins aproximados — confirmar patch exato via `context7`):

```
# --- Fase 2: serviços de mercado ---
pandas~=2.2
numpy~=2.1
yfinance~=0.2
python-bcb~=0.3
requests~=2.32
selenium~=4.25
webdriver-manager~=4.0
```

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/pip install -r requirements.txt
.venv/bin/python -c "import pandas, numpy, yfinance, bcb, requests, selenium; print('deps OK')"
```
Expected: imprime `deps OK`.

- [ ] **Step 2: Criar o skeleton do app `apps.market`**

`backend/apps/market/__init__.py`: arquivo vazio.

`backend/apps/market/apps.py`:
```python
from django.apps import AppConfig


class MarketConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.market"
```

`backend/apps/market/tests/__init__.py`: arquivo vazio.

- [ ] **Step 3: Registrar o app e a flag no `settings.py`**

Em `backend/config/settings.py`, no `INSTALLED_APPS`, após `"apps.core",`:
```python
    "apps.core",
    "apps.market",
```

Ao final do arquivo (após o bloco `LOGGING`):
```python
# Fase 2 — a ingestão B3 via Selenium só roda no container com Chrome (Fase 5).
# No host, ingest_market_data usa o fallback de CSV estático.
B3_INGEST_USE_SELENIUM = _env("B3_INGEST_USE_SELENIUM", "0") == "1"
```

- [ ] **Step 4: Adicionar o marker `integration` ao `pytest.ini`**

`backend/pytest.ini` fica:
```ini
[pytest]
DJANGO_SETTINGS_MODULE = config.settings
python_files = test_*.py
pythonpath = .
testpaths = apps
addopts = --reuse-db -m "not integration"
markers =
    integration: chama serviços externos ao vivo (rede) — não roda por padrão
```

- [ ] **Step 5: Adicionar o campo `kind` a `MarketSeries`**

Em `backend/apps/core/models.py`, na classe `MarketSeries`, após o campo `source`:
```python
    source = models.CharField(max_length=16)  # BCB | YF | TD | B3 | PTAX
    kind = models.CharField(max_length=8, default="level")  # "rate" (cumprod) | "level"
```

- [ ] **Step 6: Mostrar `kind` no Admin**

Em `backend/apps/core/admin.py`, no `MarketSeriesAdmin`:
```python
@admin.register(MarketSeries)
class MarketSeriesAdmin(admin.ModelAdmin):
    list_display = ("series_key", "source", "kind", "reference_date", "value")
    list_filter = ("source", "kind")
    date_hierarchy = "reference_date"
    search_fields = ("series_key",)
    ordering = ("series_key", "reference_date")
```

- [ ] **Step 7: Gerar e aplicar a migration**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/python manage.py makemigrations core --name marketseries_kind
.venv/bin/python manage.py migrate
```
Expected: cria `apps/core/migrations/0002_marketseries_kind.py` (AddField `kind`); `migrate` aplica sem erro.

- [ ] **Step 8: Copiar as fixtures de CSV**

Run:
```bash
cd /data/projetos/app-investimentos
mkdir -p backend/apps/market/tests/fixtures
cp data/raw/BCB_12.csv data/raw/BCB_433.csv data/raw/BCB_12466.csv \
   data/raw/YF_GSPC.csv data/raw/YF_IMAB11_SA.csv \
   data/raw/TD_Tesouro_IPCAmais_15-05-2035.csv \
   data/static/IFIX_all.csv \
   backend/apps/market/tests/fixtures/
ls backend/apps/market/tests/fixtures/
```
Expected: 7 arquivos listados. (Se `YF_IMAB11_SA.csv` não existir em `data/raw`, usar `YF_B5P211_SA.csv` no lugar e ajustar o nome nos testes da Task 11.)

- [ ] **Step 9: Rodar a suíte inteira — nada pode ter quebrado**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/python manage.py check
.venv/bin/pytest -q
```
Expected: `check` limpo; os 15 testes das Fases 0+1 seguem verdes (o smoke de cache pode `skip` sem Redis).

- [ ] **Step 10: Commit**

```bash
cd /data/projetos/app-investimentos
git add backend/requirements.txt backend/config/settings.py backend/pytest.ini \
        backend/apps/market/__init__.py backend/apps/market/apps.py \
        backend/apps/market/tests/__init__.py backend/apps/market/tests/fixtures \
        backend/apps/core/models.py backend/apps/core/admin.py backend/apps/core/migrations/0002_marketseries_kind.py
git commit -m "✨ feat(market): fundação da Fase 2 — app apps.market + campo kind

Adiciona apps.market ao projeto, o campo MarketSeries.kind (rate|level) com
migration core.0002, a flag B3_INGEST_USE_SELENIUM, o marker pytest 'integration'
e as fixtures de CSV do repo legado. Deps novas: pandas, numpy, yfinance,
python-bcb, requests, selenium, webdriver-manager."
```

---

## Task 2: `cache.py` — envelope pandas sobre `django.core.cache`

**Files:**
- Create: `backend/apps/market/cache.py`
- Create: `backend/apps/market/tests/test_cache.py`

**Interfaces:**
- Consumes: `django.core.cache.cache` (configurado na Fase 1 → `django-redis`).
- Produces:
  - `apps.market.cache.make_key(*parts) -> str` — `"market:" + ":".join(str(p) for p in parts)`.
  - `apps.market.cache.set_df(key: str, value, ttl: int | None) -> None` — `pd.Series`/`pd.DataFrame` viram `{"_type": "pd.Series"|"pd.DataFrame", "payload": <to_json(orient="table")>}`; erro de cache é logado e engolido.
  - `apps.market.cache.get_df(key: str) -> pd.Series | pd.DataFrame | None` — desembrulha o envelope; `pd.Series` volta como `Series` (não `DataFrame`); erro/miss → `None`.
  - `apps.market.cache.TTL_SERIES = 86400`, `apps.market.cache.TTL_DLP = 600`.

- [ ] **Step 1: Escrever os testes (devem falhar)**

`backend/apps/market/tests/test_cache.py`:
```python
import numpy as np
import pandas as pd
import pytest

from apps.market import cache as mc


@pytest.fixture(autouse=True)
def _locmem_cache(settings):
    settings.CACHES = {
        "default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}
    }
    from django.core.cache import cache

    cache.clear()


def test_make_key_namespaced_and_windowed():
    assert mc.make_key("series", "IMA-B", "2020-01-01", "2024-12-31") == (
        "market:series:IMA-B:2020-01-01:2024-12-31"
    )


def test_series_roundtrip_returns_series_not_frame():
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    s = pd.Series([1.5, 2.5, 3.5], index=idx, name="CDI")
    mc.set_df("market:t:s", s, ttl=60)
    out = mc.get_df("market:t:s")
    assert isinstance(out, pd.Series)
    pd.testing.assert_series_equal(out, s, check_names=False)


def test_dataframe_roundtrip():
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}, index=idx)
    mc.set_df("market:t:df", df, ttl=60)
    out = mc.get_df("market:t:df")
    assert isinstance(out, pd.DataFrame)
    pd.testing.assert_frame_equal(out, df)


def test_plain_value_roundtrip():
    mc.set_df("market:t:plain", {"x": 1, "y": [1, 2, 3]}, ttl=60)
    assert mc.get_df("market:t:plain") == {"x": 1, "y": [1, 2, 3]}


def test_miss_returns_none():
    assert mc.get_df("market:t:absent") is None


def test_set_failure_is_swallowed(monkeypatch, caplog):
    def boom(*a, **k):
        raise RuntimeError("redis down")

    from django.core.cache import cache

    monkeypatch.setattr(cache, "set", boom)
    mc.set_df("market:t:s", pd.Series([1.0]), ttl=60)  # não levanta
    assert "redis down" in caplog.text or "cache" in caplog.text.lower()


def test_multiindex_columns_are_flattened():
    idx = pd.to_datetime(["2024-01-02"])
    df = pd.DataFrame([[1.0]], index=idx)
    df.columns = pd.MultiIndex.from_tuples([("Close", "PETR4")])
    mc.set_df("market:t:mi", df, ttl=60)
    out = mc.get_df("market:t:mi")
    assert list(out.columns) == ["Close_PETR4"]
```

- [ ] **Step 2: Rodar e ver falhar**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/pytest apps/market/tests/test_cache.py -q
```
Expected: FALHA com `ModuleNotFoundError: No module named 'apps.market.cache'`.

- [ ] **Step 3: Implementar `cache.py`**

`backend/apps/market/cache.py`:
```python
"""Cache de séries de mercado — envelope pandas sobre django.core.cache.

Porta do envelope de serialização de src/data/cache.py (mesmo formato JSON), mas
backed pelo django-redis configurado na Fase 1. O cache é uma OTIMIZAÇÃO: qualquer
falha é logada e engolida, nunca propaga.
"""
from __future__ import annotations

import io
import json
import logging

import pandas as pd
from django.core.cache import cache

logger = logging.getLogger(__name__)

TTL_SERIES = 86400  # 24h — séries derivadas
TTL_DLP = 600       # 10min — dados por-usuário da DLP

_SERIES = "pd.Series"
_FRAME = "pd.DataFrame"


def make_key(*parts) -> str:
    return "market:" + ":".join(str(p) for p in parts)


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            "_".join(map(str, col)).strip("_") if isinstance(col, tuple) else str(col)
            for col in df.columns.to_flat_index()
        ]
    return df


def set_df(key: str, value, ttl: int | None) -> None:
    try:
        payload = value
        if isinstance(value, (pd.Series, pd.DataFrame)):
            frame = value.to_frame() if isinstance(value, pd.Series) else value
            frame = _flatten_columns(frame)
            payload = {
                "_type": _SERIES if isinstance(value, pd.Series) else _FRAME,
                "payload": json.loads(frame.to_json(orient="table")),
            }
        cache.set(key, payload, timeout=ttl)
    except Exception as exc:  # cache é otimização — não propaga
        logger.warning("cache set falhou para %s: %s", key, exc)


def get_df(key: str):
    try:
        raw = cache.get(key)
    except Exception as exc:
        logger.warning("cache get falhou para %s: %s", key, exc)
        return None
    if raw is None:
        return None
    if isinstance(raw, dict) and raw.get("_type") in (_SERIES, _FRAME):
        frame = pd.read_json(io.StringIO(json.dumps(raw["payload"])), orient="table")
        if raw["_type"] == _SERIES:
            return frame.iloc[:, 0]
        return frame
    return raw
```

- [ ] **Step 4: Rodar e ver passar**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/pytest apps/market/tests/test_cache.py -q
```
Expected: PASSA (7 testes).

- [ ] **Step 5: Commit**

```bash
cd /data/projetos/app-investimentos
git add backend/apps/market/cache.py backend/apps/market/tests/test_cache.py
git commit -m "✨ feat(market): cache.py — envelope pandas sobre django.core.cache

Porta do envelope {\"_type\", \"payload\": to_json(orient=table)} de src/data/cache.py,
backed pelo django-redis. make_key namespaced ('market:...') e com a janela de
datas na chave. Falha de cache loga e não propaga."
```

---

## Task 3: `repository.py` — único acesso a `core.MarketSeries`

**Files:**
- Create: `backend/apps/market/repository.py`
- Create: `backend/apps/market/tests/test_repository.py`

**Interfaces:**
- Consumes: `apps.core.models.MarketSeries`; `apps.market.services.sources.base.RawPoint` (Task 4 — mas o teste desta task usa um stub local; ver Step 1).
- Produces:
  - `repository.upsert_points(series_key: str, source: str, kind: str, points: Iterable) -> int` — cada `point` tem `.date` (`datetime.date`) e `.value` (`float`); `bulk_create(update_conflicts=True, unique_fields=["series_key","reference_date"], update_fields=["value","kind","updated_at","source"])`; `float` → `Decimal(str(v)).quantize(Decimal("1e-8"))`; descarta `NaN`/`inf` com log; retorna nº de pontos gravados.
  - `repository.load_series(series_key: str, start: date, end: date) -> pd.Series` — `pd.Series(dtype=float, index=DatetimeIndex, name=series_key)`, ordenada; vazia (não `None`) se não houver linhas.
  - `repository.last_date(series_key: str) -> date | None`.
  - `repository.available_keys() -> list[str]` — `distinct series_key` ordenado.

- [ ] **Step 1: Escrever os testes (devem falhar)**

`backend/apps/market/tests/test_repository.py`:
```python
import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from apps.core.models import MarketSeries
from apps.market import repository as repo

pytestmark = pytest.mark.django_db


@dataclass(frozen=True)
class P:
    date: dt.date
    value: float


def test_upsert_inserts_and_counts():
    n = repo.upsert_points(
        "bcb:12", "BCB", "rate",
        [P(dt.date(2024, 1, 2), 0.0437), P(dt.date(2024, 1, 3), 0.0438)],
    )
    assert n == 2
    assert MarketSeries.objects.filter(series_key="bcb:12").count() == 2
    row = MarketSeries.objects.get(series_key="bcb:12", reference_date=dt.date(2024, 1, 2))
    assert row.kind == "rate"
    assert str(row.value) == "0.04370000"


def test_upsert_is_idempotent_and_updates_value():
    repo.upsert_points("bcb:12", "BCB", "rate", [P(dt.date(2024, 1, 2), 0.0437)])
    repo.upsert_points("bcb:12", "BCB", "rate", [P(dt.date(2024, 1, 2), 0.0999)])
    qs = MarketSeries.objects.filter(series_key="bcb:12", reference_date=dt.date(2024, 1, 2))
    assert qs.count() == 1
    assert str(qs.get().value) == "0.09990000"


def test_upsert_drops_nan_and_inf_with_log(caplog):
    n = repo.upsert_points(
        "yf:^GSPC", "YF", "level",
        [P(dt.date(2024, 1, 2), 100.0), P(dt.date(2024, 1, 3), float("nan")),
         P(dt.date(2024, 1, 4), np.inf)],
    )
    assert n == 1
    assert MarketSeries.objects.filter(series_key="yf:^GSPC").count() == 1
    assert "descart" in caplog.text.lower()


def test_load_series_returns_sorted_float_series():
    repo.upsert_points(
        "bcb:12", "BCB", "rate",
        [P(dt.date(2024, 1, 4), 3.0), P(dt.date(2024, 1, 2), 1.0), P(dt.date(2024, 1, 3), 2.0)],
    )
    s = repo.load_series("bcb:12", dt.date(2024, 1, 1), dt.date(2024, 1, 31))
    assert isinstance(s, pd.Series)
    assert s.dtype == float
    assert list(s.values) == [1.0, 2.0, 3.0]
    assert s.index.is_monotonic_increasing
    assert s.name == "bcb:12"


def test_load_series_respects_window():
    repo.upsert_points(
        "bcb:12", "BCB", "rate",
        [P(dt.date(2023, 12, 31), 9.0), P(dt.date(2024, 1, 2), 1.0), P(dt.date(2024, 2, 1), 5.0)],
    )
    s = repo.load_series("bcb:12", dt.date(2024, 1, 1), dt.date(2024, 1, 31))
    assert list(s.values) == [1.0]


def test_load_series_empty_is_series_not_none():
    s = repo.load_series("bcb:999", dt.date(2024, 1, 1), dt.date(2024, 1, 31))
    assert isinstance(s, pd.Series)
    assert s.empty


def test_last_date_and_available_keys():
    repo.upsert_points("bcb:12", "BCB", "rate", [P(dt.date(2024, 1, 2), 1.0), P(dt.date(2024, 1, 9), 2.0)])
    repo.upsert_points("yf:^GSPC", "YF", "level", [P(dt.date(2024, 1, 5), 100.0)])
    assert repo.last_date("bcb:12") == dt.date(2024, 1, 9)
    assert repo.last_date("bcb:absent") is None
    assert repo.available_keys() == ["bcb:12", "yf:^GSPC"]
```

- [ ] **Step 2: Rodar e ver falhar**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/pytest apps/market/tests/test_repository.py -q
```
Expected: FALHA com `ModuleNotFoundError: No module named 'apps.market.repository'`.

- [ ] **Step 3: Implementar `repository.py`**

`backend/apps/market/repository.py`:
```python
"""Único ponto de acesso a core.MarketSeries. Postgres <-> pandas, sem derivação."""
from __future__ import annotations

import datetime as dt
import logging
import math
from decimal import Decimal
from typing import Iterable

import pandas as pd
from django.db.models import Max

from apps.core.models import MarketSeries

logger = logging.getLogger(__name__)

_Q = Decimal("1e-8")


def upsert_points(series_key: str, source: str, kind: str, points: Iterable) -> int:
    rows: list[MarketSeries] = []
    dropped = 0
    for p in points:
        v = float(p.value)
        if math.isnan(v) or math.isinf(v):
            dropped += 1
            continue
        rows.append(
            MarketSeries(
                series_key=series_key,
                source=source,
                kind=kind,
                reference_date=p.date,
                value=Decimal(str(v)).quantize(_Q),
            )
        )
    if dropped:
        logger.warning("%s: %d ponto(s) descartado(s) (NaN/inf)", series_key, dropped)
    if not rows:
        return 0
    MarketSeries.objects.bulk_create(
        rows,
        update_conflicts=True,
        unique_fields=["series_key", "reference_date"],
        update_fields=["value", "kind", "source", "updated_at"],
    )
    return len(rows)


def load_series(series_key: str, start: dt.date, end: dt.date) -> pd.Series:
    qs = (
        MarketSeries.objects.filter(
            series_key=series_key, reference_date__range=(start, end)
        )
        .order_by("reference_date")
        .values_list("reference_date", "value")
    )
    pairs = list(qs)
    if not pairs:
        return pd.Series(dtype=float, name=series_key)
    idx = pd.to_datetime([d for d, _ in pairs])
    vals = [float(v) for _, v in pairs]
    return pd.Series(vals, index=idx, name=series_key, dtype=float)


def last_date(series_key: str) -> dt.date | None:
    return (
        MarketSeries.objects.filter(series_key=series_key)
        .aggregate(m=Max("reference_date"))
        .get("m")
    )


def available_keys() -> list[str]:
    return sorted(
        MarketSeries.objects.values_list("series_key", flat=True).distinct()
    )
```

- [ ] **Step 4: Rodar e ver passar**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/pytest apps/market/tests/test_repository.py -q
```
Expected: PASSA (7 testes).

- [ ] **Step 5: Commit**

```bash
cd /data/projetos/app-investimentos
git add backend/apps/market/repository.py backend/apps/market/tests/test_repository.py
git commit -m "✨ feat(market): repository.py — único acesso a core.MarketSeries

upsert_points (bulk_create update_conflicts, quantização Decimal(1e-8), descarte
de NaN/inf com log), load_series (-> pd.Series float ordenada; vazia != None),
last_date e available_keys. Sem derivação — Postgres <-> pandas."
```

---

## Task 4: `services/sources/` — `base.py` + `bcb.py` + `ptax.py`

**Files:**
- Create: `backend/apps/market/services/__init__.py`
- Create: `backend/apps/market/services/sources/__init__.py`
- Create: `backend/apps/market/services/sources/base.py`
- Create: `backend/apps/market/services/sources/bcb.py`
- Create: `backend/apps/market/services/sources/ptax.py`
- Create: `backend/apps/market/tests/test_sources_bcb.py`
- Create: `backend/apps/market/tests/test_sources_ptax.py`

**Interfaces:**
- Consumes: `bcb.sgs` (`python-bcb`), `requests`.
- Produces:
  - `base.RawPoint` — `@dataclass(frozen=True)` com `date: datetime.date`, `value: float`.
  - `base.FetchError(Exception)`, `base.DLPError(Exception)`.
  - `bcb.fetch_sgs(codigo: int, start: date, end: date) -> list[RawPoint]` — valor CRU da coluna `str(codigo)`; *chunking* de 5 anos; `raise FetchError` em falha; `[]` só se a API responder sem dados.
  - `ptax.fetch_ptax(start: date, end: date) -> list[RawPoint]` — `cotacaoVenda` diária da API Olinda; 1 ponto por dia (último do dia); `raise FetchError` em falha.

- [ ] **Step 1: Escrever `base.py`**

`backend/apps/market/services/__init__.py`: vazio.
`backend/apps/market/services/sources/__init__.py`: vazio.

`backend/apps/market/services/sources/base.py`:
```python
"""Contrato comum das fontes de dados de mercado.

Regra de ouro: uma fonte SEMPRE levanta FetchError/DLPError em falha de
rede/parse. Devolver [] / None significa APENAS 'a fonte não tem esse dado' —
nunca 'deu erro'.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass


@dataclass(frozen=True)
class RawPoint:
    date: dt.date
    value: float


class FetchError(Exception):
    """Falha ao buscar dados de uma fonte de mercado (rede, parse, formato)."""


class DLPError(Exception):
    """Falha na API DLP (por-usuário)."""
```

- [ ] **Step 2: Escrever os testes de `bcb.py` (devem falhar)**

`backend/apps/market/tests/test_sources_bcb.py`:
```python
import datetime as dt

import pandas as pd
import pytest

from apps.market.services.sources import bcb
from apps.market.services.sources.base import FetchError, RawPoint


def test_fetch_sgs_returns_raw_points(monkeypatch):
    def fake_get(codes, start, end):
        idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
        return pd.DataFrame({"12": [0.0437, 0.0438]}, index=idx)

    monkeypatch.setattr(bcb.sgs, "get", fake_get)
    pts = bcb.fetch_sgs(12, dt.date(2024, 1, 1), dt.date(2024, 1, 31))
    assert pts == [
        RawPoint(dt.date(2024, 1, 2), 0.0437),
        RawPoint(dt.date(2024, 1, 3), 0.0438),
    ]


def test_fetch_sgs_chunks_five_year_windows(monkeypatch):
    calls = []

    def fake_get(codes, start, end):
        calls.append((pd.Timestamp(start).date(), pd.Timestamp(end).date()))
        return pd.DataFrame({"12": [1.0]}, index=pd.to_datetime([start]))

    monkeypatch.setattr(bcb.sgs, "get", fake_get)
    bcb.fetch_sgs(12, dt.date(2010, 1, 1), dt.date(2024, 1, 1))
    assert len(calls) >= 3  # 2010-15, 2015-20, 2020-24
    assert calls[0][0] == dt.date(2010, 1, 1)


def test_fetch_sgs_raises_fetcherror_on_api_failure(monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("bcb down")

    monkeypatch.setattr(bcb.sgs, "get", boom)
    with pytest.raises(FetchError):
        bcb.fetch_sgs(12, dt.date(2024, 1, 1), dt.date(2024, 1, 31))


def test_fetch_sgs_empty_response_returns_empty_list(monkeypatch):
    monkeypatch.setattr(bcb.sgs, "get", lambda *a, **k: pd.DataFrame())
    assert bcb.fetch_sgs(12, dt.date(2024, 1, 1), dt.date(2024, 1, 31)) == []
```

- [ ] **Step 3: Escrever os testes de `ptax.py` (devem falhar)**

`backend/apps/market/tests/test_sources_ptax.py`:
```python
import datetime as dt

import pytest

from apps.market.services.sources import ptax
from apps.market.services.sources.base import FetchError, RawPoint


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("http error")


def test_fetch_ptax_returns_daily_last_quote(monkeypatch):
    payload = {
        "value": [
            {"cotacaoVenda": 5.40, "dataHoraCotacao": "2024-01-02 10:00:00.0"},
            {"cotacaoVenda": 5.42, "dataHoraCotacao": "2024-01-02 13:00:00.0"},
            {"cotacaoVenda": 5.50, "dataHoraCotacao": "2024-01-03 13:00:00.0"},
        ]
    }
    monkeypatch.setattr(ptax.requests, "get", lambda *a, **k: _Resp(payload))
    pts = ptax.fetch_ptax(dt.date(2024, 1, 1), dt.date(2024, 1, 31))
    assert pts == [
        RawPoint(dt.date(2024, 1, 2), 5.42),
        RawPoint(dt.date(2024, 1, 3), 5.50),
    ]


def test_fetch_ptax_empty_returns_empty_list(monkeypatch):
    monkeypatch.setattr(ptax.requests, "get", lambda *a, **k: _Resp({"value": []}))
    assert ptax.fetch_ptax(dt.date(2024, 1, 1), dt.date(2024, 1, 31)) == []


def test_fetch_ptax_raises_on_http_failure(monkeypatch):
    monkeypatch.setattr(ptax.requests, "get", lambda *a, **k: _Resp({}, status=500))
    with pytest.raises(FetchError):
        ptax.fetch_ptax(dt.date(2024, 1, 1), dt.date(2024, 1, 31))
```

- [ ] **Step 4: Rodar e ver falhar**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/pytest apps/market/tests/test_sources_bcb.py apps/market/tests/test_sources_ptax.py -q
```
Expected: FALHA com `ModuleNotFoundError` para `apps.market.services.sources.bcb` / `ptax`.

- [ ] **Step 5: Implementar `bcb.py`**

`backend/apps/market/services/sources/bcb.py`:
```python
"""Fonte BCB (SGS). Valor CRU: taxa % para 11/12/433; número-índice para 12466...

Porta de src/data/sources/market_data.py::buscar_dados_bcb — SEM cumprod, SEM
proxy splice, SEM cache. Só busca a janela pedida (com chunking de 5 anos porque
o SGS limita o range).
"""
from __future__ import annotations

import datetime as dt
import logging

import pandas as pd
from bcb import sgs

from .base import FetchError, RawPoint

logger = logging.getLogger(__name__)

_CHUNK_YEARS = 5


def fetch_sgs(codigo: int, start: dt.date, end: dt.date) -> list[RawPoint]:
    frames: list[pd.DataFrame] = []
    cur = pd.Timestamp(start)
    stop = pd.Timestamp(end)
    try:
        while cur <= stop:
            chunk_end = min(cur + pd.DateOffset(years=_CHUNK_YEARS), stop)
            df = sgs.get({str(codigo): codigo}, start=cur.date(), end=chunk_end.date())
            if df is not None and not df.empty:
                df.index = pd.to_datetime(df.index)
                frames.append(df)
            cur = chunk_end + pd.Timedelta(days=1)
    except Exception as exc:  # noqa: BLE001
        raise FetchError(f"BCB SGS {codigo}: {exc}") from exc

    if not frames:
        return []
    full = pd.concat(frames)
    full = full[~full.index.duplicated(keep="last")].sort_index()
    col = str(codigo)
    series = pd.to_numeric(full[col], errors="coerce").dropna()
    return [RawPoint(ts.date(), float(v)) for ts, v in series.items()]
```

- [ ] **Step 6: Implementar `ptax.py`**

`backend/apps/market/services/sources/ptax.py`:
```python
"""Fonte PTAX (USD/BRL venda) via API Olinda do BCB.

Porta de src/data/sources/market_data.py::buscar_dolar_bcb. Valor CRU:
cotacaoVenda diária (último registro do dia).
"""
from __future__ import annotations

import datetime as dt
import logging

import pandas as pd
import requests

from .base import FetchError, RawPoint

logger = logging.getLogger(__name__)

_URL = (
    "https://olinda.bcb.gov.br/olinda/servico/PTAX/versao/v1/odata/"
    "CotacaoDolarPeriodo(dataInicial=@dataInicial,dataFinalCotacao=@dataFinalCotacao)"
    "?@dataInicial='{ini}'&@dataFinalCotacao='{fim}'"
    "&$top=100000&$format=json&$select=cotacaoVenda,dataHoraCotacao"
)


def fetch_ptax(start: dt.date, end: dt.date) -> list[RawPoint]:
    url = _URL.format(
        ini=start.strftime("%m-%d-%Y"), fim=end.strftime("%m-%d-%Y")
    )
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        raise FetchError(f"PTAX Olinda: {exc}") from exc

    values = data.get("value") or []
    if not values:
        return []
    df = pd.DataFrame(values)
    df["dt"] = pd.to_datetime(df["dataHoraCotacao"])
    df["day"] = df["dt"].dt.normalize()
    df = df.sort_values("dt").drop_duplicates(subset=["day"], keep="last")
    return [
        RawPoint(day.date(), float(v))
        for day, v in zip(df["day"], df["cotacaoVenda"])
    ]
```

- [ ] **Step 7: Rodar e ver passar**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/pytest apps/market/tests/test_sources_bcb.py apps/market/tests/test_sources_ptax.py -q
```
Expected: PASSA (7 testes).

- [ ] **Step 8: Commit**

```bash
cd /data/projetos/app-investimentos
git add backend/apps/market/services backend/apps/market/tests/test_sources_bcb.py backend/apps/market/tests/test_sources_ptax.py
git commit -m "✨ feat(market): sources base + bcb + ptax

RawPoint/FetchError/DLPError. fetch_sgs (valor cru, chunking de 5 anos, sem
cumprod/proxy/cache) e fetch_ptax (cotacaoVenda diária via Olinda). Ambos
levantam FetchError em falha; [] só para 'sem dados'."
```

---

## Task 5: `services/sources/` — `yahoo.py` + `tesouro.py`

**Files:**
- Create: `backend/apps/market/services/sources/yahoo.py`
- Create: `backend/apps/market/services/sources/tesouro.py`
- Create: `backend/apps/market/tests/test_sources_yahoo.py`
- Create: `backend/apps/market/tests/test_sources_tesouro.py`

**Interfaces:**
- Consumes: `yfinance` (`yf.download`), `requests`, `pandas`.
- Produces:
  - `yahoo.fetch_yahoo(ticker: str, start: date, end: date) -> list[RawPoint]` — coluna `Close` (`auto_adjust=True`); trata colunas MultiIndex; `raise FetchError` em falha; `[]` se o download vier vazio.
  - `tesouro.fetch_tesouro(titulo: str, vencimento: date, start: date, end: date) -> list[RawPoint]` — `PU Base Manhã` filtrado por `Tipo Titulo == titulo` e `Data Vencimento == vencimento`; `raise FetchError` em falha de download/parse; `[]` se o filtro não casar nada.

- [ ] **Step 1: Escrever os testes de `yahoo.py` (devem falhar)**

`backend/apps/market/tests/test_sources_yahoo.py`:
```python
import datetime as dt

import pandas as pd
import pytest

from apps.market.services.sources import yahoo
from apps.market.services.sources.base import FetchError, RawPoint


def test_fetch_yahoo_simple_columns(monkeypatch):
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    df = pd.DataFrame({"Open": [1, 2], "Close": [459.1, 460.7]}, index=idx)
    monkeypatch.setattr(yahoo.yf, "download", lambda *a, **k: df)
    pts = yahoo.fetch_yahoo("^GSPC", dt.date(2024, 1, 1), dt.date(2024, 1, 31))
    assert pts == [
        RawPoint(dt.date(2024, 1, 2), 459.1),
        RawPoint(dt.date(2024, 1, 3), 460.7),
    ]


def test_fetch_yahoo_multiindex_columns(monkeypatch):
    idx = pd.to_datetime(["2024-01-02"])
    df = pd.DataFrame([[1.0, 459.1]], index=idx)
    df.columns = pd.MultiIndex.from_tuples([("Open", "^GSPC"), ("Close", "^GSPC")])
    monkeypatch.setattr(yahoo.yf, "download", lambda *a, **k: df)
    pts = yahoo.fetch_yahoo("^GSPC", dt.date(2024, 1, 1), dt.date(2024, 1, 31))
    assert pts == [RawPoint(dt.date(2024, 1, 2), 459.1)]


def test_fetch_yahoo_empty_returns_empty_list(monkeypatch):
    monkeypatch.setattr(yahoo.yf, "download", lambda *a, **k: pd.DataFrame())
    assert yahoo.fetch_yahoo("^GSPC", dt.date(2024, 1, 1), dt.date(2024, 1, 31)) == []


def test_fetch_yahoo_raises_on_failure(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("yf down")

    monkeypatch.setattr(yahoo.yf, "download", boom)
    with pytest.raises(FetchError):
        yahoo.fetch_yahoo("^GSPC", dt.date(2024, 1, 1), dt.date(2024, 1, 31))
```

- [ ] **Step 2: Escrever os testes de `tesouro.py` (devem falhar)**

`backend/apps/market/tests/test_sources_tesouro.py`:
```python
import datetime as dt
import io

import pytest

from apps.market.services.sources import tesouro
from apps.market.services.sources.base import FetchError, RawPoint

_CSV = (
    "Tipo Titulo;Data Vencimento;Data Base;PU Base Manha\n"
    "Tesouro IPCA+;15/05/2035;08/03/2010;405,61\n"
    "Tesouro IPCA+;15/05/2035;09/03/2010;403,91\n"
    "Tesouro IPCA+;15/05/2045;08/03/2010;250,00\n"
    "Tesouro Selic;01/03/2027;08/03/2010;9000,00\n"
)


class _Resp:
    status_code = 200

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield _CSV.encode("latin-1")


def test_fetch_tesouro_filters_titulo_and_vencimento(monkeypatch, tmp_path):
    monkeypatch.setattr(tesouro, "_DOWNLOAD_DIR", tmp_path)
    monkeypatch.setattr(tesouro.requests, "get", lambda *a, **k: _Resp())
    pts = tesouro.fetch_tesouro(
        "Tesouro IPCA+", dt.date(2035, 5, 15), dt.date(2010, 1, 1), dt.date(2010, 12, 31)
    )
    assert pts == [
        RawPoint(dt.date(2010, 3, 8), 405.61),
        RawPoint(dt.date(2010, 3, 9), 403.91),
    ]


def test_fetch_tesouro_no_match_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(tesouro, "_DOWNLOAD_DIR", tmp_path)
    monkeypatch.setattr(tesouro.requests, "get", lambda *a, **k: _Resp())
    assert tesouro.fetch_tesouro(
        "Tesouro Prefixado", dt.date(2031, 1, 1), dt.date(2010, 1, 1), dt.date(2010, 12, 31)
    ) == []


def test_fetch_tesouro_raises_on_download_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(tesouro, "_DOWNLOAD_DIR", tmp_path)

    def boom(*a, **k):
        raise ConnectionError("tesouro down")

    monkeypatch.setattr(tesouro.requests, "get", boom)
    with pytest.raises(FetchError):
        tesouro.fetch_tesouro(
            "Tesouro IPCA+", dt.date(2035, 5, 15), dt.date(2010, 1, 1), dt.date(2010, 12, 31)
        )
```

- [ ] **Step 3: Rodar e ver falhar**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/pytest apps/market/tests/test_sources_yahoo.py apps/market/tests/test_sources_tesouro.py -q
```
Expected: FALHA com `ModuleNotFoundError`.

- [ ] **Step 4: Implementar `yahoo.py`**

`backend/apps/market/services/sources/yahoo.py`:
```python
"""Fonte Yahoo Finance. Valor CRU: Close ajustado (auto_adjust=True).

Porta de src/data/sources/market_data.py::buscar_dados_benchmark — sem cache,
sem CSV local. Só a janela pedida.
"""
from __future__ import annotations

import datetime as dt
import logging

import pandas as pd
import yfinance as yf

from .base import FetchError, RawPoint

logger = logging.getLogger(__name__)


def fetch_yahoo(ticker: str, start: dt.date, end: dt.date) -> list[RawPoint]:
    try:
        # end exclusivo no yfinance — soma 1 dia para incluir 'end'
        raw = yf.download(
            ticker,
            start=start.isoformat(),
            end=(end + dt.timedelta(days=1)).isoformat(),
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:  # noqa: BLE001
        raise FetchError(f"Yahoo {ticker}: {exc}") from exc

    if raw is None or raw.empty:
        return []

    close = raw["Close"]
    if isinstance(close, pd.DataFrame):  # colunas MultiIndex -> uma coluna por ticker
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce").dropna()
    close = close[~close.index.duplicated(keep="last")].sort_index()
    return [RawPoint(ts.date(), float(v)) for ts, v in close.items()]
```

- [ ] **Step 5: Implementar `tesouro.py`**

`backend/apps/market/services/sources/tesouro.py`:
```python
"""Fonte Tesouro Direto (Tesouro Transparente). Valor CRU: PU Base Manhã.

Porta de src/data/sources/market_data.py::buscar_dados_tesouro — sem cache. Baixa
o CSV nacional para _DOWNLOAD_DIR e filtra por título + vencimento.
"""
from __future__ import annotations

import datetime as dt
import logging
import tempfile
from pathlib import Path

import pandas as pd
import requests

from .base import FetchError, RawPoint

logger = logging.getLogger(__name__)

_DOWNLOAD_DIR = Path(tempfile.gettempdir())
_URL = (
    "https://www.tesourotransparente.gov.br/ckan/dataset/"
    "df56aa42-484a-4a59-8184-7676580c81e3/resource/"
    "796d2059-14e9-44e3-80c9-2d9e30b405c1/download/precotaxatesourodireto.csv"
)


def fetch_tesouro(
    titulo: str, vencimento: dt.date, start: dt.date, end: dt.date
) -> list[RawPoint]:
    path = _DOWNLOAD_DIR / "PrecoTaxaTesouroDireto.csv"
    try:
        resp = requests.get(_URL, stream=True, timeout=120)
        resp.raise_for_status()
        with open(path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=8192):
                fh.write(chunk)
        df = pd.read_csv(
            path,
            sep=";",
            decimal=",",
            encoding="latin-1",
            usecols=["Tipo Titulo", "Data Vencimento", "Data Base", "PU Base Manha"],
        )
    except Exception as exc:  # noqa: BLE001
        raise FetchError(f"Tesouro {titulo} {vencimento}: {exc}") from exc

    df["Data Vencimento"] = pd.to_datetime(df["Data Vencimento"], dayfirst=True)
    df["Data Base"] = pd.to_datetime(df["Data Base"], dayfirst=True)
    venc = pd.Timestamp(vencimento)
    mask = (df["Tipo Titulo"] == titulo) & (df["Data Vencimento"] == venc)
    sel = (
        df[mask]
        .set_index("Data Base")
        .sort_index()
        .loc[str(start) : str(end), "PU Base Manha"]
    )
    sel = sel[~sel.index.duplicated(keep="last")]
    return [RawPoint(ts.date(), float(v)) for ts, v in sel.items()]
```

- [ ] **Step 6: Rodar e ver passar**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/pytest apps/market/tests/test_sources_yahoo.py apps/market/tests/test_sources_tesouro.py -q
```
Expected: PASSA (7 testes).

- [ ] **Step 7: Commit**

```bash
cd /data/projetos/app-investimentos
git add backend/apps/market/services/sources/yahoo.py backend/apps/market/services/sources/tesouro.py \
        backend/apps/market/tests/test_sources_yahoo.py backend/apps/market/tests/test_sources_tesouro.py
git commit -m "✨ feat(market): sources yahoo + tesouro

fetch_yahoo (Close ajustado, trata MultiIndex, janela pedida) e fetch_tesouro
(PU Base Manhã filtrado por título+vencimento). FetchError em falha; [] para
'sem dados'."
```

---

## Task 6: `services/sources/` — `b3.py` + `dlp.py`

**Files:**
- Create: `backend/apps/market/services/sources/b3.py`
- Create: `backend/apps/market/services/sources/dlp.py`
- Create: `backend/apps/market/tests/test_sources_b3.py`
- Create: `backend/apps/market/tests/test_sources_dlp.py`

**Interfaces:**
- Consumes: `pandas`, `requests`; `django.conf.settings` (`B3_INGEST_USE_SELENIUM` — só lida no comando, não aqui); `apps.market.cache` (para o cache por-usuário da DLP).
- Produces:
  - `b3.read_b3_static(indice: str, start: date, end: date, static_dir: Path | None = None) -> list[RawPoint]` — lê `<static_dir>/<indice>_all.csv` (default: `<repo>/data/static`); `[]` se o arquivo não existir; `raise FetchError` em erro de parse.
  - `b3.fetch_b3_selenium(indice: str, anos: list[int], download_dir: Path | None = None) -> list[RawPoint]` — porta do scraper; **não roda no host** (só testado com mock do `webdriver`); `raise FetchError` em falha.
  - `dlp.buscar_historico(token: str, ativo=None, classe=None, corretora=None) -> pd.DataFrame | None` — LIVE; cache Redis TTL 600s; `raise DLPError` em erro HTTP; `None` só para "sem histórico".
  - `dlp.buscar_resumo_carteira(token: str) -> dict | None` — LIVE; cache Redis TTL 600s; `raise DLPError` em erro HTTP; `None` só para "sem resumo".

- [ ] **Step 1: Escrever os testes de `b3.py` (devem falhar)**

`backend/apps/market/tests/test_sources_b3.py`:
```python
import datetime as dt

import pytest

from apps.market.services.sources import b3
from apps.market.services.sources.base import FetchError, RawPoint

FIXTURES = "apps/market/tests/fixtures"


def test_read_b3_static_reads_and_filters(tmp_path):
    csv = tmp_path / "IFIX_all.csv"
    csv.write_text(
        "Date,Close\n2023-12-29,3000.0\n2024-01-02,3010.5\n2024-06-03,3200.0\n"
    )
    pts = b3.read_b3_static("IFIX", dt.date(2024, 1, 1), dt.date(2024, 3, 31), static_dir=tmp_path)
    assert pts == [RawPoint(dt.date(2024, 1, 2), 3010.5)]


def test_read_b3_static_missing_file_returns_empty(tmp_path):
    assert b3.read_b3_static("NOPE", dt.date(2024, 1, 1), dt.date(2024, 12, 31), static_dir=tmp_path) == []


def test_read_b3_static_uses_repo_fixture():
    # o IFIX_all.csv da fixture existe e tem dados
    pts = b3.read_b3_static("IFIX", dt.date(2010, 1, 1), dt.date(2030, 1, 1),
                            static_dir=__import__("pathlib").Path(FIXTURES))
    assert len(pts) > 0
    assert isinstance(pts[0], RawPoint)


def test_fetch_b3_selenium_raises_when_webdriver_unavailable(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no chrome")

    monkeypatch.setattr(b3.webdriver, "Chrome", boom)
    with pytest.raises(FetchError):
        b3.fetch_b3_selenium("IFIX", [2024])
```

- [ ] **Step 2: Escrever os testes de `dlp.py` (devem falhar)**

`backend/apps/market/tests/test_sources_dlp.py`:
```python
import pandas as pd
import pytest

from apps.market.services.sources import dlp
from apps.market.services.sources.base import DLPError


@pytest.fixture(autouse=True)
def _locmem_cache(settings):
    settings.CACHES = {
        "default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}
    }
    from django.core.cache import cache

    cache.clear()


class _Resp:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status

    def json(self):
        return self._p

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("http error")


def test_buscar_historico_returns_dataframe_and_caches(monkeypatch):
    calls = []

    def fake_get(url, headers=None, params=None, timeout=None):
        calls.append(url)
        return _Resp({"historico": [{"data": "2024-01-02", "vlr_mercado": 100}]})

    monkeypatch.setattr(dlp.requests, "get", fake_get)
    df1 = dlp.buscar_historico("DLP-tok")
    df2 = dlp.buscar_historico("DLP-tok")  # segundo hit vem do cache
    assert isinstance(df1, pd.DataFrame)
    assert "date" in df1.columns  # 'data' normalizado para 'date'
    assert len(calls) == 1


def test_buscar_historico_empty_returns_none(monkeypatch):
    monkeypatch.setattr(dlp.requests, "get", lambda *a, **k: _Resp({"historico": []}))
    assert dlp.buscar_historico("DLP-tok") is None


def test_buscar_historico_http_error_raises_dlperror(monkeypatch):
    monkeypatch.setattr(dlp.requests, "get", lambda *a, **k: _Resp({}, status=500))
    with pytest.raises(DLPError):
        dlp.buscar_historico("DLP-tok")


def test_buscar_resumo_carteira_roundtrip(monkeypatch):
    monkeypatch.setattr(dlp.requests, "get", lambda *a, **k: _Resp({"wallet": [1, 2]}))
    assert dlp.buscar_resumo_carteira("DLP-tok") == {"wallet": [1, 2]}
```

- [ ] **Step 3: Rodar e ver falhar**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/pytest apps/market/tests/test_sources_b3.py apps/market/tests/test_sources_dlp.py -q
```
Expected: FALHA com `ModuleNotFoundError`.

- [ ] **Step 4: Implementar `b3.py`**

`backend/apps/market/services/sources/b3.py`:
```python
"""Fonte B3 — índices de evolução diária.

read_b3_static: lê os CSVs estáticos data/static/<indice>_all.csv (porta de
src/data/sources/b3_source.py). É o caminho usado no host (sem Chrome).

fetch_b3_selenium: porta do scraper de src/data/sources/market_data.py. NÃO roda
no host; só é exercitado com mock. A Fase 5 o liga no container com Chrome.
"""
from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import pandas as pd
from selenium import webdriver  # noqa: F401 — usado por fetch_b3_selenium / mockado nos testes

from .base import FetchError, RawPoint

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[5]  # backend/apps/market/services/sources/b3.py
_DEFAULT_STATIC_DIR = _REPO_ROOT / "data" / "static"


def read_b3_static(
    indice: str, start: dt.date, end: dt.date, static_dir: Path | None = None
) -> list[RawPoint]:
    base = static_dir or _DEFAULT_STATIC_DIR
    path = Path(base) / f"{indice}_all.csv"
    if not path.exists():
        logger.info("B3 estático ausente para %s: %s", indice, path)
        return []
    try:
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date").sort_index()
    except Exception as exc:  # noqa: BLE001
        raise FetchError(f"B3 estático {indice}: {exc}") from exc
    mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
    series = pd.to_numeric(df.loc[mask, "Close"], errors="coerce").dropna()
    return [RawPoint(ts.date(), float(v)) for ts, v in series.items()]


def fetch_b3_selenium(
    indice: str, anos: list[int], download_dir: Path | None = None
) -> list[RawPoint]:
    """Scraper Selenium (headless Chrome). Porta de run_b3_downloader +
    parsing melt/unpivot de buscar_dados_b3. Só roda em ambiente com Chrome."""
    try:
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import Select
        from selenium.webdriver.support.wait import WebDriverWait
        from webdriver_manager.chrome import ChromeDriverManager

        opts = webdriver.ChromeOptions()
        opts.add_argument("--headless=new")
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
        try:
            frames: list[pd.DataFrame] = []
            meses = {
                "Jan": 1, "Fev": 2, "Mar": 3, "Abr": 4, "Mai": 5, "Jun": 6,
                "Jul": 7, "Ago": 8, "Set": 9, "Out": 10, "Nov": 11, "Dez": 12,
            }
            for ano in anos:
                url = (
                    "https://sistemaswebb3-listados.b3.com.br/indexStatisticsPage/"
                    f"daily-evolution/{indice}?language=pt-br"
                )
                driver.get(url)
                sel = Select(WebDriverWait(driver, 30).until(
                    lambda d: d.find_element(By.ID, "selectYear")
                ))
                sel.select_by_value(str(ano))
                # ... (raspagem da tabela; ver buscar_dados_b3 no legado) ...
                table_html = driver.find_element(By.TAG_NAME, "table").get_attribute("outerHTML")
                raw = pd.read_html(table_html)[0]
                raw = raw[pd.to_numeric(raw["Dia"], errors="coerce").notna()]
                melted = raw.melt(id_vars=["Dia"], var_name="Mes", value_name="Close").dropna()
                melted["Close"] = (
                    melted["Close"].astype(str)
                    .str.replace(".", "", regex=False)
                    .str.replace(",", ".", regex=False)
                    .astype(float)
                )
                melted["month"] = melted["Mes"].map(meses)
                melted["year"] = ano
                melted = melted.dropna(subset=["month"])
                melted["Date"] = pd.to_datetime(
                    melted[["year", "month", "Dia"]].rename(columns={"Dia": "day"})
                )
                frames.append(melted[["Date", "Close"]])
        finally:
            driver.quit()
    except Exception as exc:  # noqa: BLE001
        raise FetchError(f"B3 Selenium {indice}: {exc}") from exc

    if not frames:
        return []
    full = pd.concat(frames).sort_values("Date").drop_duplicates("Date", keep="last")
    return [RawPoint(ts.date(), float(v)) for ts, v in zip(full["Date"], full["Close"])]
```

> Nota: a raspagem exata da tabela (Steps de clique/wait) é a de `buscar_dados_b3` no
> legado. Como o path não roda no host, o teste só cobre "levanta `FetchError` quando o
> `webdriver.Chrome` não está disponível". A Fase 5 valida o fluxo real no container.

- [ ] **Step 5: Implementar `dlp.py`**

`backend/apps/market/services/sources/dlp.py`:
```python
"""Cliente da API DLP (carteira pessoal). LIVE, por-usuário.

Porta de src/data/sources/market_data.py::buscar_historico / buscar_resumo_carteira.
NÃO grava em MarketSeries (é dado por-usuário). Cacheia no Redis com TTL 600s.
Levanta DLPError em erro HTTP; devolve None só para 'sem dados'.
"""
from __future__ import annotations

import hashlib
import json
import logging

import pandas as pd
import requests

from apps.market import cache as mc

from .base import DLPError

logger = logging.getLogger(__name__)

_BASE = "https://users.dlombelloplanilhas.com"


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()[:12]


def _norm_date_col(df: pd.DataFrame) -> pd.DataFrame:
    rename = {c: "date" for c in df.columns if c.lower() in ("date", "data")}
    return df.rename(columns=rename) if rename else df


def buscar_historico(token, ativo=None, classe=None, corretora=None):
    params = {k: v for k, v in {"ativo": ativo, "classe": classe, "corretora": corretora}.items() if v}
    pstr = json.dumps(params, sort_keys=True)
    key = mc.make_key("dlp", "hist", _token_hash(token), hashlib.sha256(pstr.encode()).hexdigest()[:12])
    cached = mc.get_df(key)
    if cached is not None:
        return _norm_date_col(cached) if isinstance(cached, pd.DataFrame) else cached

    try:
        resp = requests.get(
            f"{_BASE}/historico",
            headers={"Content-Type": "application/json", "Authorization": token},
            params=params,
            timeout=60,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:  # noqa: BLE001
        raise DLPError(f"DLP /historico: {exc}") from exc

    df = pd.DataFrame(payload.get("historico", []))
    if df.empty:
        return None
    df = _norm_date_col(df)
    mc.set_df(key, df, ttl=mc.TTL_DLP)
    return df


def buscar_resumo_carteira(token):
    key = mc.make_key("dlp", "resumo", _token_hash(token))
    cached = mc.get_df(key)
    if cached is not None:
        return cached

    try:
        resp = requests.get(
            f"{_BASE}/resumo",
            headers={"Content-Type": "application/json", "Authorization": token},
            params={"summary": "true", "wallet": "true", "multiwallet": "false", "notifications": "false"},
            timeout=60,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:  # noqa: BLE001
        raise DLPError(f"DLP /resumo: {exc}") from exc

    if not payload:
        return None
    mc.set_df(key, payload, ttl=mc.TTL_DLP)
    return payload
```

- [ ] **Step 6: Rodar e ver passar**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/pytest apps/market/tests/test_sources_b3.py apps/market/tests/test_sources_dlp.py -q
```
Expected: PASSA (8 testes).

- [ ] **Step 7: Commit**

```bash
cd /data/projetos/app-investimentos
git add backend/apps/market/services/sources/b3.py backend/apps/market/services/sources/dlp.py \
        backend/apps/market/tests/test_sources_b3.py backend/apps/market/tests/test_sources_dlp.py
git commit -m "✨ feat(market): sources b3 + dlp

b3.read_b3_static (CSV estático — caminho do host) + b3.fetch_b3_selenium
(portado, testado só com mock — sem Chrome no host). dlp.buscar_historico /
buscar_resumo_carteira: LIVE por-usuário, cache Redis TTL 600s, DLPError em erro
HTTP, None só para 'sem dados'. Não gravam em MarketSeries."
```

---

## Task 7: `synthetics.py` — contrato estruturado dos benchmarks sintéticos

**Files:**
- Create: `backend/apps/market/synthetics.py`
- Create: `backend/apps/market/tests/test_synthetics.py`

**Interfaces:**
- Consumes: nada.
- Produces:
  - `synthetics.SyntheticSpec` — `@dataclass(frozen=True)` com `base: str` (`"IPCA"`|`"CDI"`), `operador: str` (`"+"`|`"%"`), `valor: float`.
  - `synthetics.parse_legacy_string(s: str) -> SyntheticSpec | None` — `"IPCA + 6%"` → `SyntheticSpec("IPCA","+",6.0)`; `"CDI + 2%"` → `("CDI","+",2.0)`; `"110% do CDI"` → `("CDI","%",110.0)`; lixo → `None`.

- [ ] **Step 1: Escrever os testes (devem falhar)**

`backend/apps/market/tests/test_synthetics.py`:
```python
import pytest

from apps.market.synthetics import SyntheticSpec, parse_legacy_string


@pytest.mark.parametrize(
    "text,expected",
    [
        ("IPCA + 6%", SyntheticSpec("IPCA", "+", 6.0)),
        ("IPCA + 6.5%", SyntheticSpec("IPCA", "+", 6.5)),
        ("CDI + 2%", SyntheticSpec("CDI", "+", 2.0)),
        ("110% do CDI", SyntheticSpec("CDI", "%", 110.0)),
        ("95.5% do CDI", SyntheticSpec("CDI", "%", 95.5)),
    ],
)
def test_parse_legacy_string_known_forms(text, expected):
    assert parse_legacy_string(text) == expected


@pytest.mark.parametrize("text", ["", "IBOV", "IPCA menos 2%", "banana", "6% do SELIC"])
def test_parse_legacy_string_garbage_returns_none(text):
    assert parse_legacy_string(text) is None


def test_spec_is_frozen():
    with pytest.raises(Exception):
        SyntheticSpec("IPCA", "+", 6.0).valor = 7.0
```

- [ ] **Step 2: Rodar e ver falhar**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/pytest apps/market/tests/test_synthetics.py -q
```
Expected: FALHA com `ModuleNotFoundError: No module named 'apps.market.synthetics'`.

- [ ] **Step 3: Implementar `synthetics.py`**

`backend/apps/market/synthetics.py`:
```python
"""Contrato estruturado dos benchmarks sintéticos.

Substitui as 3 regexes espalhadas em financial_report.build_dataset. A API da
Fase 3 aceita SyntheticSpec direto; parse_legacy_string é só para migrar strings
antigas e como açúcar de frontend.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_SPREAD = re.compile(r"^\s*(IPCA|CDI)\s*\+\s*(\d+(?:\.\d+)?)\s*%\s*$", re.IGNORECASE)
_PCT = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*%\s*do\s*(CDI)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class SyntheticSpec:
    base: str        # "IPCA" | "CDI"
    operador: str    # "+" (spread a.a.) | "%" (percentual do índice)
    valor: float


def parse_legacy_string(s: str) -> SyntheticSpec | None:
    if not s:
        return None
    m = _SPREAD.match(s)
    if m:
        return SyntheticSpec(m.group(1).upper(), "+", float(m.group(2)))
    m = _PCT.match(s)
    if m:
        return SyntheticSpec(m.group(2).upper(), "%", float(m.group(1)))
    return None
```

- [ ] **Step 4: Rodar e ver passar**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/pytest apps/market/tests/test_synthetics.py -q
```
Expected: PASSA (11 testes).

- [ ] **Step 5: Commit**

```bash
cd /data/projetos/app-investimentos
git add backend/apps/market/synthetics.py backend/apps/market/tests/test_synthetics.py
git commit -m "✨ feat(market): synthetics.py — SyntheticSpec + parser da string legada

SyntheticSpec{base, operador, valor}. parse_legacy_string cobre 'IPCA + X%',
'CDI + X%' e 'X% do CDI'; lixo -> None. Substitui as 3 regexes de build_dataset."
```

---

## Task 8: `transforms.py` — funções puras de derivação

**Files:**
- Create: `backend/apps/market/transforms.py`
- Create: `backend/apps/market/tests/test_transforms.py`

**Interfaces:**
- Consumes: `apps.market.synthetics.SyntheticSpec` (Task 7); `pandas`, `numpy`.
- Produces (todas puras — `pd.Series`/`DataFrame` entra, sai; sem I/O):
  - `transforms.accumulate(rate: pd.Series, periodicity: str = "daily") -> pd.Series` — `(rate/100 + 1).cumprod()`; 1º NaN → 1. `periodicity` não altera o cumprod cru (fica para `synthetic_from_spec`).
  - `transforms.splice_proxy(bcb_level: pd.Series, etf_price: pd.Series) -> pd.Series` — usa `bcb_level` até sua última data; daí projeta `bcb_last * (etf / etf_base)`; sem duplicar o dia-base; sem NaN na emenda.
  - `transforms.to_brl(usd_series: pd.Series, ptax: pd.Series) -> pd.Series` — `usd_series * ptax.reindex(usd_series.index, method="ffill")`.
  - `transforms.synthetic_from_spec(spec: SyntheticSpec, base: pd.Series) -> pd.Series` — fórmula legada do caminho das regexes de `build_dataset` (`**(1/252)` para `"+"`), com `pct_change(fill_method=None)`.
  - `transforms.composite(members: dict[str, pd.Series], weights: dict[str, float]) -> pd.Series` — soma dos retornos ponderados → `(1 + ret).cumprod()`.
  - `transforms.rebase_100(series: pd.Series, at: pd.Timestamp | None = None) -> pd.Series` — `(series / base_val) * 100`, `base_val` no 1º índice válido `>= at`.

- [ ] **Step 1: Escrever os testes (devem falhar)**

`backend/apps/market/tests/test_transforms.py`:
```python
import numpy as np
import pandas as pd
import pytest

from apps.market import transforms as tf
from apps.market.synthetics import SyntheticSpec


def _s(values, start="2024-01-02"):
    idx = pd.bdate_range(start=start, periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


def test_accumulate_cumprod_from_rate():
    rate = _s([1.0, 1.0, 1.0])  # 1% ao dia
    out = tf.accumulate(rate)
    np.testing.assert_allclose(out.values, [1.01, 1.0201, 1.030301])


def test_accumulate_first_nan_becomes_one():
    rate = pd.Series([np.nan, 1.0], index=pd.bdate_range("2024-01-02", periods=2))
    out = tf.accumulate(rate)
    assert out.iloc[0] == 1.0


def test_splice_proxy_continuous_at_seam():
    bcb = _s([100.0, 110.0], start="2024-01-02")            # termina em 110 em 2024-01-03
    etf = _s([50.0, 55.0, 60.5], start="2024-01-03")         # base 50 no dia da emenda
    out = tf.splice_proxy(bcb, etf)
    # até 2024-01-03: valores do BCB. Depois: 110 * (etf / 50)
    assert out.loc["2024-01-03"] == 110.0
    np.testing.assert_allclose(out.loc["2024-01-04"], 110.0 * 55.0 / 50.0)
    np.testing.assert_allclose(out.loc["2024-01-05"], 110.0 * 60.5 / 50.0)
    assert not out.isna().any()


def test_to_brl_multiplies_by_ffilled_ptax():
    usd = _s([100.0, 100.0, 100.0])
    ptax = pd.Series([5.0], index=[usd.index[0]])
    out = tf.to_brl(usd, ptax)
    np.testing.assert_allclose(out.values, [500.0, 500.0, 500.0])


def test_synthetic_spread_matches_legacy_formula():
    base = _s([100.0, 101.0, 102.0])
    spec = SyntheticSpec("CDI", "+", 2.0)
    out = tf.synthetic_from_spec(spec, base)
    ret = base.pct_change(fill_method=None).fillna(0)
    expected = (1 + ((1 + ret) * (1 + 2.0 / 100) ** (1 / 252) - 1)).cumprod() * 100
    pd.testing.assert_series_equal(out, expected, check_names=False)


def test_synthetic_percent_of_index():
    base = _s([100.0, 110.0, 121.0])
    spec = SyntheticSpec("CDI", "%", 50.0)
    out = tf.synthetic_from_spec(spec, base)
    ret = base.pct_change(fill_method=None).fillna(0)
    expected = (1 + ret * 0.5).cumprod() * 100
    pd.testing.assert_series_equal(out, expected, check_names=False)


def test_composite_weighted_returns():
    a = _s([100.0, 110.0])   # +10%
    b = _s([100.0, 100.0])   # 0%
    out = tf.composite({"a": a, "b": b}, {"a": 0.5, "b": 0.5})
    np.testing.assert_allclose(out.values, [1.0, 1.05])


def test_rebase_100():
    s = _s([200.0, 220.0, 180.0])
    out = tf.rebase_100(s)
    np.testing.assert_allclose(out.values, [100.0, 110.0, 90.0])
```

- [ ] **Step 2: Rodar e ver falhar**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/pytest apps/market/tests/test_transforms.py -q
```
Expected: FALHA com `ModuleNotFoundError: No module named 'apps.market.transforms'`.

- [ ] **Step 3: Implementar `transforms.py`**

`backend/apps/market/transforms.py`:
```python
"""Derivações puras de séries de mercado. pd.Series/DataFrame entra, sai.
Zero I/O, zero Django. É a superfície que a paridade da Fase 6 valida contra o
processar_benchmarks legado.
"""
from __future__ import annotations

import pandas as pd

from apps.market.synthetics import SyntheticSpec


def accumulate(rate: pd.Series, periodicity: str = "daily") -> pd.Series:
    factor = (rate / 100.0) + 1.0
    return factor.cumprod().fillna(1.0)


def splice_proxy(bcb_level: pd.Series, etf_price: pd.Series) -> pd.Series:
    bcb_level = bcb_level.dropna().sort_index()
    etf_price = etf_price.dropna().sort_index()
    if bcb_level.empty:
        return etf_price
    last_date = bcb_level.index[-1]
    tail = etf_price[etf_price.index >= last_date]
    if tail.empty:
        return bcb_level
    base = tail.iloc[0]
    projected = bcb_level.iloc[-1] * (tail.iloc[1:] / base)
    return pd.concat([bcb_level, projected]).sort_index()


def to_brl(usd_series: pd.Series, ptax: pd.Series) -> pd.Series:
    aligned = ptax.reindex(usd_series.index, method="ffill")
    return usd_series * aligned


def synthetic_from_spec(spec: SyntheticSpec, base: pd.Series) -> pd.Series:
    ret = base.pct_change(fill_method=None).fillna(0)
    if spec.operador == "+":
        daily = (1 + ret) * (1 + spec.valor / 100.0) ** (1 / 252) - 1
        return (1 + daily).cumprod() * 100
    if spec.operador == "%":
        return (1 + ret * (spec.valor / 100.0)).cumprod() * 100
    raise ValueError(f"operador sintético desconhecido: {spec.operador!r}")


def composite(members: dict[str, pd.Series], weights: dict[str, float]) -> pd.Series:
    parts = []
    for name, weight in weights.items():
        s = members[name]
        parts.append(s.pct_change(fill_method=None).fillna(0) * weight)
    combined = pd.concat(parts, axis=1).fillna(0).sum(axis=1)
    return (1 + combined).cumprod()


def rebase_100(series: pd.Series, at: pd.Timestamp | None = None) -> pd.Series:
    s = series if at is None else series[series.index >= at]
    first = s.first_valid_index()
    if first is None:
        return series
    base_val = s.loc[first]
    if base_val == 0:
        return series
    return (series / base_val) * 100
```

- [ ] **Step 4: Rodar e ver passar**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/pytest apps/market/tests/test_transforms.py -q
```
Expected: PASSA (8 testes).

- [ ] **Step 5: Commit**

```bash
cd /data/projetos/app-investimentos
git add backend/apps/market/transforms.py backend/apps/market/tests/test_transforms.py
git commit -m "✨ feat(market): transforms.py — derivações puras

accumulate (cumprod de taxa), splice_proxy (emenda ANBIMA->ETF), to_brl
(USD*PTAX ffill), synthetic_from_spec (fórmula legada do caminho das regexes,
pct_change fill_method=None), composite (_calc_portfolio), rebase_100. Puras,
sem I/O — alvo da paridade da Fase 6."
```

---

## Task 9: `catalog.py` — `SeriesDef` em Python puro

**Files:**
- Create: `backend/apps/market/catalog.py`
- Create: `backend/apps/market/tests/test_catalog.py`

**Interfaces:**
- Consumes: `apps.market.synthetics.SyntheticSpec` (Task 7).
- Produces:
  - `catalog.ProxySplice(bcb_key: str, etf_key: str)`, `catalog.ToBRL(base_name: str)`, `catalog.Synthetic(spec: SyntheticSpec)`, `catalog.Composite(weights: dict[str, float])` — dataclasses frozen (a união `Derivation`).
  - `catalog.SeriesDef` — `@dataclass(frozen=True)` com `name: str`, `source: str | None`, `series_key: str | None`, `kind: str`, `periodicity: str` (`"daily"`|`"monthly"`), `derivation: object | None`.
  - `catalog.resolve(name: str) -> SeriesDef` — `raise KeyError` se não existir.
  - `catalog.ingestible() -> list[SeriesDef]` — só as com `source`/`series_key` concretos (o que `ingest_market_data` percorre). Inclui `SELIC`.
  - `catalog.DEFAULT_BENCHMARKS: list[str]`.
  - `catalog.legacy_csv_name(sdef: SeriesDef) -> str | None` — nome do CSV do repo legado que corresponde à série (para o `backfill`): `bcb:12` → `"BCB_12.csv"`; `yf:^GSPC` → `"YF_GSPC.csv"`; `yf:IVVB11.SA` → `"YF_IVVB11_SA.csv"`; `b3:IFIX` → `"IFIX_all.csv"`; `td:...` → `"TD_Tesouro_IPCAmais_15-05-2035.csv"`; `ptax:USDBRL` → `None` (não há CSV legado).

- [ ] **Step 1: Escrever os testes (devem falhar)**

`backend/apps/market/tests/test_catalog.py`:
```python
import pytest

from apps.market import catalog as cat
from apps.market.synthetics import SyntheticSpec


def test_resolve_known_and_unknown():
    cdi = cat.resolve("CDI")
    assert cdi.source == "BCB" and cdi.series_key == "bcb:12" and cdi.kind == "rate"
    with pytest.raises(KeyError):
        cat.resolve("Bananas")


def test_ipca_is_monthly_rate():
    ipca = cat.resolve("IPCA")
    assert ipca.kind == "rate" and ipca.periodicity == "monthly"


def test_ima_b_has_proxy_splice_derivation():
    ima = cat.resolve("IMA-B")
    assert isinstance(ima.derivation, cat.ProxySplice)
    assert ima.derivation.bcb_key == "bcb:12466"
    assert ima.derivation.etf_key.startswith("yf:")


def test_ingestible_excludes_pure_derivations_and_includes_selic():
    keys = {d.series_key for d in cat.ingestible()}
    assert "bcb:11" in keys           # SELIC
    assert "bcb:12" in keys           # CDI
    assert "ptax:USDBRL" in keys
    # séries puramente derivadas (Composite/Synthetic sem series_key) ficam de fora
    assert all(d.series_key for d in cat.ingestible())


def test_every_seriesdef_is_well_formed():
    for name in cat.all_names():
        d = cat.resolve(name)
        assert d.kind in ("rate", "level")
        assert d.periodicity in ("daily", "monthly")
        if d.derivation is None:
            assert d.series_key and d.source


def test_legacy_csv_name_mapping():
    assert cat.legacy_csv_name(cat.resolve("CDI")) == "BCB_12.csv"
    assert cat.legacy_csv_name(cat.resolve("S&P 500")) == "YF_GSPC.csv"
    assert cat.legacy_csv_name(cat.resolve("IFIX")) == "IFIX_all.csv"
    assert cat.legacy_csv_name(cat.resolve("USD/BRL")) is None


def test_default_benchmarks_are_resolvable():
    for name in cat.DEFAULT_BENCHMARKS:
        assert cat.resolve(name)
```

- [ ] **Step 2: Rodar e ver falhar**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/pytest apps/market/tests/test_catalog.py -q
```
Expected: FALHA com `ModuleNotFoundError: No module named 'apps.market.catalog'`.

- [ ] **Step 3: Implementar `catalog.py`**

`backend/apps/market/catalog.py`:
```python
"""Catálogo de séries lógicas. Python puro — porta de src/data/benchmarks_config.py.

Mapeia um nome lógico ('CDI', 'IMA-B', 'S&P 500 BRL') para: onde buscar o dado
cru (source/series_key) e como derivar a série servível (derivation).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from apps.market.synthetics import SyntheticSpec


@dataclass(frozen=True)
class ProxySplice:
    bcb_key: str
    etf_key: str


@dataclass(frozen=True)
class ToBRL:
    base_name: str


@dataclass(frozen=True)
class Synthetic:
    spec: SyntheticSpec


@dataclass(frozen=True)
class Composite:
    weights: dict[str, float]


@dataclass(frozen=True)
class SeriesDef:
    name: str
    source: str | None
    series_key: str | None
    kind: str                  # "rate" | "level"
    periodicity: str = "daily"  # "daily" | "monthly"
    derivation: object | None = None


# --- Séries CRUAS (têm source/series_key; a ingestão grava estas) ---
_RAW: list[SeriesDef] = [
    SeriesDef("CDI", "BCB", "bcb:12", "rate"),
    SeriesDef("SELIC", "BCB", "bcb:11", "rate"),
    SeriesDef("IPCA", "BCB", "bcb:433", "rate", periodicity="monthly"),
    SeriesDef("IMA-B (BCB)", "BCB", "bcb:12466", "level"),
    SeriesDef("IMA-B 5 (BCB)", "BCB", "bcb:12467", "level"),
    SeriesDef("IMA-B 5+ (BCB)", "BCB", "bcb:12468", "level"),
    SeriesDef("IRF-M (BCB)", "BCB", "bcb:12461", "level"),
    SeriesDef("IMA-S (BCB)", "BCB", "bcb:12469", "level"),
    SeriesDef("S&P 500", "YF", "yf:^GSPC", "level"),
    SeriesDef("IVVB11", "YF", "yf:IVVB11.SA", "level"),
    SeriesDef("BOVA11", "YF", "yf:BOVA11.SA", "level"),
    SeriesDef("IMID", "YF", "yf:IMID.L", "level"),
    SeriesDef("Bitcoin", "YF", "yf:BTC-USD", "level"),
    SeriesDef("IMAB11 (proxy)", "YF", "yf:IMAB11.SA", "level"),
    SeriesDef("IRFM11 (proxy)", "YF", "yf:IRFM11.SA", "level"),
    SeriesDef("B5P211 (proxy)", "YF", "yf:B5P211.SA", "level"),
    SeriesDef("IB5M11 (proxy)", "YF", "yf:IB5M11.SA", "level"),
    SeriesDef("LFTS11 (proxy)", "YF", "yf:LFTS11.SA", "level"),
    SeriesDef("IFIX", "B3", "b3:IFIX", "level"),
    SeriesDef("SMLL", "B3", "b3:SMLL", "level"),
    SeriesDef("IDIV", "B3", "b3:IDIV", "level"),
    SeriesDef("USD/BRL", "PTAX", "ptax:USDBRL", "level"),
]

# --- Séries DERIVADAS (calculadas na leitura; a ingestão NÃO grava estas) ---
_DERIVED: list[SeriesDef] = [
    SeriesDef("IMA-B", None, None, "level",
              derivation=ProxySplice("bcb:12466", "yf:IMAB11.SA")),
    SeriesDef("IMA-B 5", None, None, "level",
              derivation=ProxySplice("bcb:12467", "yf:B5P211.SA")),
    SeriesDef("IMA-B 5+", None, None, "level",
              derivation=ProxySplice("bcb:12468", "yf:IB5M11.SA")),
    SeriesDef("IRF-M", None, None, "level",
              derivation=ProxySplice("bcb:12461", "yf:IRFM11.SA")),
    SeriesDef("IMA-S", None, None, "level",
              derivation=ProxySplice("bcb:12469", "yf:LFTS11.SA")),
    SeriesDef("S&P 500 BRL", None, None, "level", derivation=ToBRL("S&P 500")),
    SeriesDef("IMID BRL", None, None, "level", derivation=ToBRL("IMID")),
    SeriesDef("Bitcoin BRL", None, None, "level", derivation=ToBRL("Bitcoin")),
    SeriesDef("IPCA + 6%", None, None, "level",
              derivation=Synthetic(SyntheticSpec("IPCA", "+", 6.0))),
]

_BY_NAME: dict[str, SeriesDef] = {d.name: d for d in (_RAW + _DERIVED)}

DEFAULT_BENCHMARKS: list[str] = ["CDI", "IMA-B", "IMA-B 5", "IMA-B 5+", "IPCA + 6%"]


def all_names() -> list[str]:
    return list(_BY_NAME)


def resolve(name: str) -> SeriesDef:
    try:
        return _BY_NAME[name]
    except KeyError as exc:
        raise KeyError(f"série desconhecida no catálogo: {name!r}") from exc


def ingestible() -> list[SeriesDef]:
    return [d for d in _RAW if d.series_key]


def legacy_csv_name(sdef: SeriesDef) -> str | None:
    key = sdef.series_key
    if not key:
        return None
    prefix, _, ident = key.partition(":")
    if prefix == "bcb":
        return f"BCB_{ident}.csv"
    if prefix == "yf":
        return "YF_" + ident.lstrip("^").replace(".", "_") + ".csv"
    if prefix == "b3":
        return f"{ident}_all.csv"
    if prefix == "td":
        titulo, _, venc = ident.partition(":")
        return f"TD_{titulo.replace(' ', '_').replace('+', 'mais')}_{venc[8:10]}-{venc[5:7]}-{venc[0:4]}.csv"
    return None  # ptax não tem CSV legado
```

- [ ] **Step 4: Rodar e ver passar**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/pytest apps/market/tests/test_catalog.py -q
```
Expected: PASSA (7 testes). Se `test_legacy_csv_name_mapping` falhar em `S&P 500`, conferir que `yf:^GSPC` → `YF_GSPC.csv` (o `lstrip("^")` cuida do `^`).

- [ ] **Step 5: Commit**

```bash
cd /data/projetos/app-investimentos
git add backend/apps/market/catalog.py backend/apps/market/tests/test_catalog.py
git commit -m "✨ feat(market): catalog.py — SeriesDef em Python puro

Porta de benchmarks_config.py: séries cruas (com source/series_key) e derivadas
(ProxySplice/ToBRL/Synthetic/Composite). resolve/ingestible/DEFAULT_BENCHMARKS e
legacy_csv_name (mapeia SeriesDef -> nome do CSV do repo legado, p/ o backfill)."
```

---

## Task 10: `series.py` — a fachada de leitura

**Files:**
- Create: `backend/apps/market/series.py`
- Create: `backend/apps/market/tests/test_series.py`

**Interfaces:**
- Consumes: `apps.market.repository` (Task 3), `apps.market.cache` (Task 2), `apps.market.transforms` (Task 8), `apps.market.catalog` (Task 9).
- Produces:
  - `series.get_series(name: str, start: date, end: date) -> pd.Series | None` — Redis (`cache.make_key("series", name, start, end)`) → `_materialize` → Redis (`ttl=cache.TTL_SERIES`). `None` se a série materializar vazia (com log).
  - `series.get_many(names: list[str], start: date, end: date) -> dict[str, pd.Series]` — itera `get_series`; nomes vazios/ausentes ficam de fora do dict (com log).
  - `series._materialize(sdef, start, end) -> pd.Series | None` — despacho por `sdef.derivation` (tabela no spec §4.9).

- [ ] **Step 1: Escrever os testes (devem falhar)**

`backend/apps/market/tests/test_series.py`:
```python
import datetime as dt

import pandas as pd
import pytest

from apps.market import series as sr

pytestmark = pytest.mark.django_db

S, E = dt.date(2024, 1, 1), dt.date(2024, 12, 31)


@pytest.fixture(autouse=True)
def _locmem_cache(settings):
    settings.CACHES = {
        "default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}
    }
    from django.core.cache import cache

    cache.clear()


def _bdays(n, start="2024-01-02"):
    return pd.bdate_range(start=start, periods=n)


def test_get_series_rate_applies_accumulate(monkeypatch):
    raw = pd.Series([1.0, 1.0, 1.0], index=_bdays(3), name="bcb:12")
    monkeypatch.setattr(sr.repository, "load_series", lambda key, s, e: raw)
    out = sr.get_series("CDI", S, E)
    # accumulate: 1.01, 1.0201, 1.030301
    assert round(out.iloc[-1], 6) == 1.030301


def test_get_series_level_passthrough(monkeypatch):
    raw = pd.Series([100.0, 110.0], index=_bdays(2), name="yf:^GSPC")
    monkeypatch.setattr(sr.repository, "load_series", lambda key, s, e: raw)
    out = sr.get_series("S&P 500", S, E)
    assert list(out.values) == [100.0, 110.0]


def test_get_series_caches_second_call(monkeypatch):
    calls = []
    raw = pd.Series([100.0, 110.0], index=_bdays(2), name="yf:^GSPC")

    def fake_load(key, s, e):
        calls.append(key)
        return raw

    monkeypatch.setattr(sr.repository, "load_series", fake_load)
    sr.get_series("S&P 500", S, E)
    sr.get_series("S&P 500", S, E)
    assert len(calls) == 1  # segundo veio do cache


def test_cache_key_includes_window(monkeypatch):
    raw = pd.Series([100.0, 110.0], index=_bdays(2), name="yf:^GSPC")
    monkeypatch.setattr(sr.repository, "load_series", lambda key, s, e: raw)
    from apps.market import cache as mc

    sr.get_series("S&P 500", S, E)
    assert mc.get_df(mc.make_key("series", "S&P 500", S, E)) is not None
    assert mc.get_df(mc.make_key("series", "S&P 500", dt.date(2023, 1, 1), E)) is None


def test_get_series_proxy_splice(monkeypatch):
    def fake_load(key, s, e):
        if key == "bcb:12466":
            return pd.Series([100.0, 110.0], index=_bdays(2, "2024-01-02"), name=key)
        if key == "yf:IMAB11.SA":
            return pd.Series([50.0, 55.0, 60.5], index=_bdays(3, "2024-01-03"), name=key)
        return pd.Series(dtype=float, name=key)

    monkeypatch.setattr(sr.repository, "load_series", fake_load)
    out = sr.get_series("IMA-B", S, E)
    assert out.loc["2024-01-03"] == 110.0
    assert round(out.loc["2024-01-04"], 4) == round(110.0 * 55.0 / 50.0, 4)


def test_get_series_empty_returns_none(monkeypatch):
    monkeypatch.setattr(sr.repository, "load_series", lambda key, s, e: pd.Series(dtype=float))
    assert sr.get_series("S&P 500", S, E) is None


def test_get_many_skips_missing(monkeypatch):
    def fake_load(key, s, e):
        if key == "bcb:12":
            return pd.Series([1.0, 1.0], index=_bdays(2), name=key)
        return pd.Series(dtype=float, name=key)

    monkeypatch.setattr(sr.repository, "load_series", fake_load)
    out = sr.get_many(["CDI", "S&P 500"], S, E)
    assert set(out) == {"CDI"}
```

- [ ] **Step 2: Rodar e ver falhar**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/pytest apps/market/tests/test_series.py -q
```
Expected: FALHA com `ModuleNotFoundError: No module named 'apps.market.series'`.

- [ ] **Step 3: Implementar `series.py`**

`backend/apps/market/series.py`:
```python
"""Fachada de leitura de séries de mercado.

get_series(name, start, end): Redis -> repository + transforms (conforme catálogo)
-> Redis. Substitui o processar_benchmarks de 80 linhas do legado. Recursão rasa
(Composite -> membros -> talvez ToBRL -> base + PTAX); sem DAG genérico.
"""
from __future__ import annotations

import datetime as dt
import logging

import pandas as pd

from apps.market import cache, catalog, repository, transforms

logger = logging.getLogger(__name__)


def get_series(name: str, start: dt.date, end: dt.date) -> pd.Series | None:
    key = cache.make_key("series", name, start, end)
    hit = cache.get_df(key)
    if hit is not None:
        return hit
    sdef = catalog.resolve(name)
    series = _materialize(sdef, start, end)
    if series is None or series.empty:
        logger.info("série '%s' vazia no período %s..%s", name, start, end)
        return None
    cache.set_df(key, series, ttl=cache.TTL_SERIES)
    return series


def get_many(names: list[str], start: dt.date, end: dt.date) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    for name in names:
        s = get_series(name, start, end)
        if s is not None and not s.empty:
            out[name] = s
        else:
            logger.info("get_many: '%s' ausente/vazia — fora do resultado", name)
    return out


def _materialize(sdef: catalog.SeriesDef, start: dt.date, end: dt.date) -> pd.Series | None:
    d = sdef.derivation

    if d is None:
        raw = repository.load_series(sdef.series_key, start, end)
        if raw.empty:
            return None
        if sdef.kind == "rate":
            return transforms.accumulate(raw, sdef.periodicity)
        return raw

    if isinstance(d, catalog.ProxySplice):
        bcb = repository.load_series(d.bcb_key, start, end)
        etf = repository.load_series(d.etf_key, start, end)
        if bcb.empty and etf.empty:
            return None
        return transforms.splice_proxy(bcb, etf)

    if isinstance(d, catalog.ToBRL):
        base = get_series(d.base_name, start, end)
        ptax = get_series("USD/BRL", start, end)
        if base is None or ptax is None:
            return None
        return transforms.to_brl(base, ptax)

    if isinstance(d, catalog.Synthetic):
        base = get_series(d.spec.base, start, end)
        if base is None:
            return None
        return transforms.synthetic_from_spec(d.spec, base)

    if isinstance(d, catalog.Composite):
        members = {m: get_series(m, start, end) for m in d.weights}
        if any(v is None for v in members.values()):
            return None
        return transforms.composite(members, d.weights)

    raise TypeError(f"derivação não suportada: {type(d).__name__}")
```

- [ ] **Step 4: Rodar e ver passar**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/pytest apps/market/tests/test_series.py -q
```
Expected: PASSA (7 testes).

- [ ] **Step 5: Commit**

```bash
cd /data/projetos/app-investimentos
git add backend/apps/market/series.py backend/apps/market/tests/test_series.py
git commit -m "✨ feat(market): series.py — fachada de leitura

get_series (Redis -> repository + transforms conforme catálogo -> Redis; chave
com a janela), get_many (pula ausentes), _materialize (despacho por derivação:
raw/rate, ProxySplice, ToBRL, Synthetic, Composite). Substitui processar_benchmarks."
```

---

## Task 11: `backfill_market_data` — bootstrap offline dos CSVs do repo

**Files:**
- Create: `backend/apps/market/management/__init__.py`
- Create: `backend/apps/market/management/commands/__init__.py`
- Create: `backend/apps/market/management/commands/backfill_market_data.py`
- Create: `backend/apps/market/tests/test_commands.py` (parte 1 — backfill)

**Interfaces:**
- Consumes: `apps.market.catalog.ingestible` / `legacy_csv_name` (Task 9), `apps.market.repository.upsert_points` (Task 3), `apps.market.services.sources.base.RawPoint` (Task 4).
- Produces:
  - Comando `backfill_market_data` com args `--raw-dir`, `--static-dir`, `--only KEY[,KEY...]`, `--dry-run`.
  - Módulo expõe `_read_csv(path, prefix) -> list[RawPoint]` (parser por família de CSV) — testável isolado.
  - Ao fim: resumo `N séries / M pontos / K puladas` via `self.stdout`; `sys.exit(1)` se alguma série mapeada falhar ao ler.

- [ ] **Step 1: Escrever os testes (devem falhar)**

`backend/apps/market/tests/test_commands.py`:
```python
import datetime as dt
from io import StringIO
from pathlib import Path

import pytest
from django.core.management import call_command

from apps.core.models import MarketSeries

pytestmark = pytest.mark.django_db

FIX = Path("apps/market/tests/fixtures")


def test_backfill_loads_fixture_csvs(tmp_path):
    out = StringIO()
    call_command(
        "backfill_market_data",
        "--raw-dir", str(FIX),
        "--static-dir", str(FIX),
        stdout=out,
    )
    # BCB_12 (rate), BCB_433 (rate), BCB_12466 (level), YF_GSPC, IFIX_all, TD_...
    assert MarketSeries.objects.filter(series_key="bcb:12", kind="rate").exists()
    assert MarketSeries.objects.filter(series_key="bcb:12466", kind="level").exists()
    assert MarketSeries.objects.filter(series_key="yf:^GSPC").exists()
    assert MarketSeries.objects.filter(series_key="b3:IFIX").exists()
    assert "pontos" in out.getvalue().lower()


def test_backfill_is_idempotent():
    call_command("backfill_market_data", "--raw-dir", str(FIX), "--static-dir", str(FIX))
    n1 = MarketSeries.objects.count()
    call_command("backfill_market_data", "--raw-dir", str(FIX), "--static-dir", str(FIX))
    assert MarketSeries.objects.count() == n1


def test_backfill_dry_run_writes_nothing():
    call_command("backfill_market_data", "--raw-dir", str(FIX), "--static-dir", str(FIX), "--dry-run")
    assert MarketSeries.objects.count() == 0


def test_backfill_only_filter():
    call_command(
        "backfill_market_data", "--raw-dir", str(FIX), "--static-dir", str(FIX),
        "--only", "bcb:12",
    )
    assert set(MarketSeries.objects.values_list("series_key", flat=True).distinct()) == {"bcb:12"}


def test_backfill_skips_unmapped_files(tmp_path):
    (tmp_path / "IBLV-2005.csv").write_text("Dia;Jan\n1;100,0\n")
    (tmp_path / "BCB_12.csv").write_text("Date,12\n2024-01-02,0.0437\n")
    out = StringIO()
    call_command("backfill_market_data", "--raw-dir", str(tmp_path), "--static-dir", str(tmp_path), stdout=out)
    assert MarketSeries.objects.filter(series_key="bcb:12").exists()
    assert "IBLV-2005.csv" in out.getvalue() or "pulad" in out.getvalue().lower()
```

- [ ] **Step 2: Rodar e ver falhar**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/pytest apps/market/tests/test_commands.py -q
```
Expected: FALHA com `CommandError: Unknown command: 'backfill_market_data'`.

- [ ] **Step 3: Implementar o comando**

`backend/apps/market/management/__init__.py`: vazio.
`backend/apps/market/management/commands/__init__.py`: vazio.

`backend/apps/market/management/commands/backfill_market_data.py`:
```python
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

_REPO_ROOT = Path(__file__).resolve().parents[6]


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
                    self.stdout.write(f"  puladо (não mapeado): {p.name}")

        verb = "simularia" if dry else "gravou"
        self.stdout.write(
            self.style.SUCCESS(
                f">> backfill {verb} {n_series} série(s), {n_points} pontos; {n_failed} falha(s)."
            )
        )
        if n_failed:
            sys.exit(1)
```

- [ ] **Step 4: Rodar e ver passar**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/pytest apps/market/tests/test_commands.py -q
```
Expected: PASSA (5 testes). Se `YF_GSPC.csv` tiver formato diferente do esperado (`skiprows=3`), ajustar `_read_csv` — o header real é `Price,Close` / `Ticker,^GSPC` / `Date,` (3 linhas antes dos dados).

- [ ] **Step 5: Commit**

```bash
cd /data/projetos/app-investimentos
git add backend/apps/market/management backend/apps/market/tests/test_commands.py
git commit -m "✨ feat(market): comando backfill_market_data

Bootstrap offline de MarketSeries a partir dos CSVs de data/raw + data/static.
Percorre o catálogo, mapeia SeriesDef -> CSV via legacy_csv_name, faz parse por
família (bcb/yf/b3/td) e grava via repository (idempotente). --only, --dry-run;
arquivos não mapeados são pulados com log; exit 1 se alguma leitura falhar."
```

---

## Task 12: `ingest_market_data` — ingestão ao vivo

**Files:**
- Create: `backend/apps/market/management/commands/ingest_market_data.py`
- Modify: `backend/apps/market/tests/test_commands.py` (parte 2 — ingest)

**Interfaces:**
- Consumes: `catalog.ingestible` (Task 9), `repository.upsert_points` / `repository.last_date` (Task 3), `services.sources.*` (Tasks 4-6), `series.get_series` (Task 10), `settings.B3_INGEST_USE_SELENIUM` (Task 1).
- Produces:
  - Comando `ingest_market_data` com args `--only KEY[,KEY...]`, `--full`, `--since YYYY-MM-DD`.
  - Módulo expõe `_fetch(sdef, start, end) -> list[RawPoint]` (despacho por `source`) — testável isolado.
  - Política de falha: uma série falha → `logger.error` + contador, **segue**; resumo `N ok / M falhas` + chaves; `sys.exit(1)` se `M > 0`. Após gravar, aquece o cache dos `DEFAULT_BENCHMARKS` (janela 5 anos e "tudo").

- [ ] **Step 1: Escrever os testes (devem falhar)**

Anexar a `backend/apps/market/tests/test_commands.py`:
```python
# --- ingest_market_data ---

import datetime as dt

from apps.market.services.sources.base import FetchError, RawPoint


def test_ingest_incremental_from_last_date(monkeypatch, db):
    from apps.core.models import MarketSeries
    from apps.market import repository

    repository.upsert_points("bcb:12", "BCB", "rate", [RawPoint(dt.date(2024, 1, 2), 0.04)])

    seen = {}

    def fake_fetch(sdef, start, end):
        seen[sdef.series_key] = (start, end)
        return [RawPoint(dt.date(2024, 1, 3), 0.05)]

    from apps.market.management.commands import ingest_market_data as cmd

    monkeypatch.setattr(cmd, "_fetch", fake_fetch)
    monkeypatch.setattr(cmd, "_warm_cache", lambda *a, **k: None)
    call_command("ingest_market_data", "--only", "bcb:12")
    assert seen["bcb:12"][0] == dt.date(2024, 1, 3)  # last_date + 1
    assert MarketSeries.objects.filter(series_key="bcb:12").count() == 2


def test_ingest_full_starts_from_1995(monkeypatch, db):
    seen = {}
    from apps.market.management.commands import ingest_market_data as cmd

    def fake_fetch(sdef, start, end):
        seen[sdef.series_key] = start
        return []

    monkeypatch.setattr(cmd, "_fetch", fake_fetch)
    monkeypatch.setattr(cmd, "_warm_cache", lambda *a, **k: None)
    call_command("ingest_market_data", "--only", "bcb:12", "--full")
    assert seen["bcb:12"] == dt.date(1995, 1, 1)


def test_ingest_one_source_fails_others_continue(monkeypatch, db):
    from apps.core.models import MarketSeries
    from apps.market.management.commands import ingest_market_data as cmd

    def fake_fetch(sdef, start, end):
        if sdef.series_key == "bcb:12":
            raise FetchError("bcb down")
        return [RawPoint(dt.date(2024, 6, 3), 1.0)]

    monkeypatch.setattr(cmd, "_fetch", fake_fetch)
    monkeypatch.setattr(cmd, "_warm_cache", lambda *a, **k: None)
    with pytest.raises(SystemExit) as ei:
        call_command("ingest_market_data", "--only", "bcb:12,bcb:11")
    assert ei.value.code == 1
    assert MarketSeries.objects.filter(series_key="bcb:11").exists()  # a outra passou
```

- [ ] **Step 2: Rodar e ver falhar**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/pytest apps/market/tests/test_commands.py -q
```
Expected: FALHA com `CommandError: Unknown command: 'ingest_market_data'`.

- [ ] **Step 3: Implementar o comando**

`backend/apps/market/management/commands/ingest_market_data.py`:
```python
"""Ingestão AO VIVO de séries de mercado -> core.MarketSeries (fonte de verdade).

Percorre catalog.ingestible(); para cada série busca da última data gravada até
hoje (incremental), grava o valor CRU via repository e, ao fim, aquece o cache
dos DEFAULT_BENCHMARKS. Uma série falhar NÃO aborta o lote; exit 1 se houve falha.
Agendar este comando é da Fase 5 (ofelia/cron).
"""
from __future__ import annotations

import datetime as dt
import logging
import sys

from django.conf import settings
from django.core.management.base import BaseCommand

from apps.market import catalog, repository, series
from apps.market.services.sources import b3, bcb, ptax, tesouro, yahoo
from apps.market.services.sources.base import FetchError, RawPoint

logger = logging.getLogger(__name__)

_EPOCH = dt.date(1995, 1, 1)


def _fetch(sdef: catalog.SeriesDef, start: dt.date, end: dt.date) -> list[RawPoint]:
    prefix, _, ident = sdef.series_key.partition(":")
    if prefix == "bcb":
        return bcb.fetch_sgs(int(ident), start, end)
    if prefix == "yf":
        return yahoo.fetch_yahoo(ident, start, end)
    if prefix == "ptax":
        return ptax.fetch_ptax(start, end)
    if prefix == "td":
        titulo, _, venc = ident.partition(":")
        return tesouro.fetch_tesouro(titulo, dt.date.fromisoformat(venc), start, end)
    if prefix == "b3":
        if settings.B3_INGEST_USE_SELENIUM:
            anos = list(range(start.year, end.year + 1))
            return b3.fetch_b3_selenium(ident, anos)
        return b3.read_b3_static(ident, start, end)
    raise FetchError(f"prefixo sem fonte: {prefix}")


def _warm_cache(today: dt.date) -> None:
    for name in catalog.DEFAULT_BENCHMARKS:
        for start in (today.replace(year=today.year - 5), _EPOCH):
            try:
                series.get_series(name, start, today)
            except Exception as exc:  # noqa: BLE001
                logger.warning("warm cache falhou para %s: %s", name, exc)


class Command(BaseCommand):
    help = "Ingere séries de mercado ao vivo em core.MarketSeries."

    def add_arguments(self, parser):
        parser.add_argument("--only", default="")
        parser.add_argument("--full", action="store_true")
        parser.add_argument("--since", default="")

    def handle(self, *args, **opts):
        only = {k.strip() for k in opts["only"].split(",") if k.strip()}
        today = dt.date.today()
        ok = failed = 0
        failed_keys: list[str] = []

        for sdef in catalog.ingestible():
            if only and sdef.series_key not in only:
                continue
            if opts["full"]:
                start = _EPOCH
            elif opts["since"]:
                start = dt.date.fromisoformat(opts["since"])
            else:
                last = repository.last_date(sdef.series_key)
                start = (last + dt.timedelta(days=1)) if last else _EPOCH
            if start > today:
                self.stdout.write(f"  {sdef.series_key}: já atualizado")
                continue
            try:
                points = _fetch(sdef, start, today)
            except FetchError as exc:
                failed += 1
                failed_keys.append(sdef.series_key)
                logger.error("ingest %s falhou: %s", sdef.series_key, exc)
                continue
            n = repository.upsert_points(sdef.series_key, sdef.source, sdef.kind, points)
            ok += 1
            if n:
                self.stdout.write(f"  ok {sdef.series_key}: +{n} pontos")
            else:
                self.stdout.write(f"  {sdef.series_key}: sem novos pontos")

        _warm_cache(today)

        self.stdout.write(
            self.style.SUCCESS(f">> ingest: {ok} ok, {failed} falha(s).")
        )
        if failed_keys:
            self.stderr.write("falharam: " + ", ".join(failed_keys))
            sys.exit(1)
```

- [ ] **Step 4: Rodar e ver passar**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/pytest apps/market/tests/test_commands.py -q
```
Expected: PASSA (backfill 5 + ingest 3 = 8 testes).

- [ ] **Step 5: Rodar a suíte de `apps/market` inteira**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/pytest apps/market -q
```
Expected: PASSA (todos os `test_*.py` de `apps/market`; nada marcado `integration`).

- [ ] **Step 6: Commit**

```bash
cd /data/projetos/app-investimentos
git add backend/apps/market/management/commands/ingest_market_data.py backend/apps/market/tests/test_commands.py
git commit -m "✨ feat(market): comando ingest_market_data

Ingestão ao vivo incremental (last_date+1 -> hoje) por série do catálogo, grava
valor cru via repository, aquece o cache dos DEFAULT_BENCHMARKS. --only/--full/
--since. Falha de uma série não aborta o lote; exit 1 se houve falha. B3 usa
read_b3_static salvo B3_INGEST_USE_SELENIUM."
```

---

## Task 13: Verificação ponta a ponta + documentação viva

**Files:**
- Modify: `docs/consolidacao_arquitetura_e_migracao.md` (§2.0 / §2.2)
- Modify: `backend/README.md` (comandos novos)
- Modify: `backend/CLAUDE.md` (convenções de `apps/market/`, se surgirem)
- Create: `docs/adr/0011-marketseries-valor-cru-kind.md` (decisão de guardar valor cru + `kind`)

**Interfaces:**
- Consumes: tudo das Tasks 1-12.
- Produces: DoD da Fase 2 (spec §3.3) verificada; docs atualizadas; ADR da modelagem.

- [ ] **Step 1: Suíte completa + `check` + `makemigrations --check`**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/python manage.py check
.venv/bin/python manage.py makemigrations --check --dry-run
.venv/bin/pytest -q
```
Expected: `check` limpo; "No changes detected"; **toda** a suíte verde (Fases 0+1: 15; Fase 2: ~60).

- [ ] **Step 2: Backfill real + contagem por `source`**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/python manage.py backfill_market_data
.venv/bin/python manage.py shell -c "
from apps.core.models import MarketSeries
from django.db.models import Count
print('total:', MarketSeries.objects.count())
for r in MarketSeries.objects.values('source').annotate(n=Count('id')).order_by('source'):
    print(r)
print('kinds:', list(MarketSeries.objects.values_list('kind', flat=True).distinct()))
"
```
Expected: `total` > 0; linhas por `source` (`BCB`, `YF`, `B3`, talvez `TD`); `kinds` = `['level', 'rate']` — **nenhuma linha sem `kind`**.

- [ ] **Step 3: `ingest --only bcb:12 --since` + idempotência**

Run:
```bash
cd /data/projetos/app-investimentos/backend
SINCE=$(.venv/bin/python -c "import datetime; print((datetime.date.today()-datetime.timedelta(days=30)).isoformat())")
.venv/bin/python manage.py ingest_market_data --only bcb:12 --since "$SINCE"
BEFORE=$(.venv/bin/python -c "from apps.core.models import MarketSeries; print(MarketSeries.objects.filter(series_key='bcb:12').count())")
.venv/bin/python manage.py ingest_market_data --only bcb:12 --since "$SINCE"
AFTER=$(.venv/bin/python -c "from apps.core.models import MarketSeries; print(MarketSeries.objects.filter(series_key='bcb:12').count())")
echo "antes=$BEFORE depois=$AFTER"
```
Expected: `antes == depois` (idempotente — sem linha duplicada). O comando pode logar "sem novos pontos" se o BCB não tiver dado novo, o que é aceitável.

- [ ] **Step 4: `series.get_series` — CDI, IMA-B (splice), e cache hit**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/python manage.py shell -c "
import datetime as dt, logging
logging.basicConfig(level=logging.INFO)
from apps.market import series
s, e = dt.date(2020,1,1), dt.date(2024,12,31)
cdi = series.get_series('CDI', s, e)
ima = series.get_series('IMA-B', s, e)
print('CDI pts:', 0 if cdi is None else len(cdi), '| IMA-B pts:', 0 if ima is None else len(ima))
ima2 = series.get_series('IMA-B', s, e)   # deve logar cache hit (nenhuma query de repository)
print('2a chamada IMA-B ok:', ima2 is not None)
"
```
Expected: `CDI pts` > 0 e `IMA-B pts` > 0 (o splice usou `bcb:12466` + `yf:IMAB11.SA` do backfill; se `YF_IMAB11_SA.csv` não estava no repo, a série IMA-B pode vir só do trecho BCB — aceitável, registrar). A 2ª chamada não deve emitir log de `repository`.

- [ ] **Step 5: Confirmar que nada foi para `banco_lab`**

Run:
```bash
docker exec -i lab-postgres psql -U postgres -d banco_lab -c "\dt"
docker exec -i lab-postgres psql -U postgres -d appinvest -c "SELECT count(*) FROM core_marketseries;"
```
Expected: `banco_lab` sem relations; `appinvest.core_marketseries` com contagem > 0.

- [ ] **Step 6: `silent-failure-hunter` sobre o diff de `apps/market/`**

Rodar o agente `pr-review-toolkit` → `silent-failure-hunter` com o diff de `backend/apps/market/` (todas as Tasks). Corrigir qualquer `except: ... return None` / exceção engolida que ele apontar, com commit próprio (`🐛 fix(market): ...`). Confirmar: fontes levantam `FetchError`; `series`/`repository` propagam; só `cache` engole (com log).

- [ ] **Step 7: Atualizar `docs/consolidacao_arquitetura_e_migracao.md`**

- Em **§2.0** (tabela "Já implementado"), adicionar linha **Fase 2**: `apps/market/` (sources, repository, cache, catalog, synthetics, transforms, series), migration `core.0002` (`kind`), comandos `ingest_market_data` / `backfill_market_data`, ~60 testes.
- Em **§2.2** ("O que não existe ainda"), remover o bullet "Sem serviços de mercado (`apps/market/`)"; ajustar o de "Dados ainda no legado" (o cache agora é `lab-redis`; os tokens seguem no Supabase até a fase de migração de dados).
- Marcar item **3** do checklist §9 como concluído.

- [ ] **Step 8: Atualizar `backend/README.md`**

Na seção "Rodar", adicionar:
```bash
.venv/bin/python manage.py backfill_market_data      # bootstrap: semeia MarketSeries dos CSVs do repo (uma vez)
.venv/bin/python manage.py ingest_market_data        # ingestão ao vivo (incremental); agendamento é Fase 5
```
Na "Estrutura", acrescentar `apps/market/` com uma linha por submódulo.

- [ ] **Step 9: Criar `docs/adr/0011-marketseries-valor-cru-kind.md`**

Registrar: `MarketSeries` guarda **valor cru** por fonte + campo `kind` (`rate`/`level`); toda derivação em `transforms.py` na leitura; Postgres = fonte de verdade, Redis = cache com a janela na chave; contrato sintético estruturado (`SyntheticSpec`). Contexto: corrige o bug do cache windowed do legado e mantém o banco auditável. Consequências: `+` auditável, algoritmo de splice evolui sem re-ingestão; `−` `cumprod` a cada cache-miss (mitigado por Redis). Referenciar o spec.
Atualizar `docs/adr/README.md` com a linha do 0011.

- [ ] **Step 10: `code-review` + `code-simplifier` no diff completo de `apps/market/`**

Rodar `/code-review` sobre `backend/apps/market/`. Aplicar correções de correção num commit `🐛 fix(market): ...`; limpezas do `code-simplifier` num `♻️ refactor(market): ...`. Rodar `.venv/bin/pytest apps/market -q` de novo — verde.

- [ ] **Step 11: Commit da documentação**

```bash
cd /data/projetos/app-investimentos
git add docs/consolidacao_arquitetura_e_migracao.md docs/adr/0011-marketseries-valor-cru-kind.md \
        docs/adr/README.md backend/README.md backend/CLAUDE.md
git commit -m "📝 docs: fecha a Fase 2 — §2 do doc mestre, ADR 0011, README do backend

Documentação viva: estado atual com apps/market/ + os dois comandos, ADR da
modelagem MarketSeries (valor cru + kind), comandos novos no README."
```

---

## Self-Review

**1. Cobertura do spec**

| Item do spec | Task |
|---|---|
| §4.1 estrutura `apps/market/` | Tasks 1-12 (um módulo por task) |
| §4.2 `MarketSeries.kind` + migration `core.0002` | Task 1 Steps 5-7 |
| §4.2 taxonomia `series_key`/`source`/`kind` | Task 9 (`catalog._RAW`) |
| §4.3 `services/sources/*` (busca pura, `FetchError`, `RawPoint`) | Tasks 4-6 |
| §4.4 `repository.py` (upsert/load/last_date/available_keys) | Task 3 |
| §4.5 `cache.py` (envelope, `make_key` com janela, TTLs, falha não-fatal) | Task 2 |
| §4.6 `catalog.py` (`SeriesDef`, `Derivation`, `resolve`, `ingestible`, `DEFAULT_BENCHMARKS`) | Task 9 |
| §4.7 `synthetics.py` (`SyntheticSpec`, `parse_legacy_string`) | Task 7 |
| §4.8 `transforms.py` (6 funções puras, `pct_change(fill_method=None)`) | Task 8 |
| §4.9 `series.py` (`get_series`/`get_many`/`_materialize`, despacho por derivação) | Task 10 |
| §4.10 `ingest_market_data` (incremental, política de falha, warm cache, flag Selenium) | Task 12 |
| §4.11 `backfill_market_data` (mapeia nome→série, pula órfãos, `--dry-run`) | Task 11 |
| §4.12 tratamento de erros sem silent failures | Tasks 4-6 (design) + Task 13 Step 6 (`silent-failure-hunter`) |
| §4.13 fixtures de CSV | Task 1 Step 8 |
| §5 (Fase 2.5) | **fora deste plano** — Plano 2 |
| §6.1 testes de `apps/market` | cada task tem seus `test_*.py`; `@pytest.mark.integration` no `pytest.ini` (Task 1) |
| §7 deps novas no `requirements.txt` | Task 1 Step 1 |
| §3.3 Definition of Done (Fase 2), itens 1-9 | Task 13 Steps 1-6 |
| Documentação viva (§2 doc mestre, ADR, README) | Task 13 Steps 7-11 |

Sem lacunas para o escopo da Fase 2. A Fase 2.5 é o Plano 2.

**2. Placeholders**

Nenhum "TBD/TODO". Os pontos deliberadamente deixados para o implementador estão marcados como tal e não bloqueiam nenhum passo:
- Raspagem exata da tabela B3 em `fetch_b3_selenium` (Task 6 Step 4) — o path não roda no host; o teste cobre só o `raise FetchError`. A Fase 5 valida.
- Pins exatos de patch no `requirements.txt` (Task 1) — confirmados via `context7` na execução; os `~=` já limitam a major/minor.
- Nome do CSV proxy YF (`YF_IMAB11_SA.csv` vs `YF_B5P211_SA.csv`) — Task 1 Step 8 e Task 13 Step 4 já preveem o fallback e o registro.

**3. Consistência de tipos**

- `RawPoint(date, value)` — definido em `base.py` (Task 4), consumido por `repository.upsert_points` (Task 3 usa um stub `P` idêntico no teste; a assinatura bate: `.date`, `.value`), pelos comandos (Tasks 11-12) e por todos os `fetch_*`.
- `repository.load_series(series_key, start, end) -> pd.Series` — assinatura idêntica em Task 3 (def), Task 10 (`series._materialize` chama), Task 10 testes (monkeypatch com `lambda key, s, e`).
- `cache.make_key(*parts)`, `cache.get_df`, `cache.set_df`, `cache.TTL_SERIES`, `cache.TTL_DLP` — definidos em Task 2, usados em Tasks 6 (`dlp`), 10 (`series`).
- `catalog.SeriesDef` campos (`name`, `source`, `series_key`, `kind`, `periodicity`, `derivation`) — definidos em Task 9, lidos em Task 10 (`_materialize`), Tasks 11-12 (comandos).
- `catalog.ProxySplice.bcb_key`/`etf_key`, `ToBRL.base_name`, `Synthetic.spec`, `Composite.weights` — definidos em Task 9, despachados em Task 10.
- `catalog.ingestible()` / `catalog.legacy_csv_name(sdef)` / `catalog.DEFAULT_BENCHMARKS` — Task 9 (def), Tasks 11-12 (uso).
- `synthetics.SyntheticSpec(base, operador, valor)` — Task 7 (def), Task 8 (`synthetic_from_spec`), Task 9 (`Synthetic`).
- `services.sources.<mod>.fetch_*` — assinaturas fixadas nos "Interfaces" das Tasks 4-6 e reusadas em `ingest._fetch` (Task 12).
- `FetchError` / `DLPError` — `base.py` (Task 4), levantadas nas fontes (Tasks 4-6), capturadas em `ingest` (Task 12).

Sem inconsistências.

---

## Execution Handoff

Plano completo e salvo em `docs/superpowers/plans/2026-08-31-fase-2-servicos-de-mercado.md`.

O **Plano 2 (Fase 2.5)** está em `docs/superpowers/plans/2026-08-31-fase-2-5-refatorar-engine.md` e **só começa depois da Fase 2 fechada e revisada**.

Duas opções de execução para a Fase 2:

1. **Subagent-Driven (recomendado)** — um subagente novo por task, revisão em dois estágios entre tasks, iteração rápida.
2. **Inline Execution** — executar as tasks nesta sessão com `executing-plans`, em lote com checkpoints de revisão.

Qual abordagem?
