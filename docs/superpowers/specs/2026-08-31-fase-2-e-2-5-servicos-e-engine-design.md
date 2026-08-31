# Design — Fase 2 (Serviços de Mercado) + Fase 2.5 (Refatorar o Engine)

> **Data:** 2026-08-31
> **Branch:** `migracao-django`
> **Documento mestre:** [`../../consolidacao_arquitetura_e_migracao.md`](../../consolidacao_arquitetura_e_migracao.md)
> **Fases anteriores:** 0 + 1 concluídas em 2026-08-28 — ver spec [`2026-08-27-fundacao-backend-fases-0-1-design.md`](2026-08-27-fundacao-backend-fases-0-1-design.md) e [`../../adr/`](../../adr/README.md).
> **Escopo deste design:** segundo incremento da migração. Cobre a Fase 2 (serviços de
> mercado, ingestão, `MarketSeries` como fonte de verdade) e a Fase 2.5 (extrair cálculo da
> plotagem no engine). Sai **um spec** e **dois planos de implementação** independentes.

---

## 1. Contexto

Depois das Fases 0 + 1, o backend Django tem infraestrutura (Postgres/Redis no LAB), modelo de
usuário, wrapper Fernet e os modelos `UserToken` / `MarketSeries` — mas `MarketSeries` está
**vazio** e não há nada que produza dados de mercado. Toda a lógica de dados ainda vive no
Streamlit legado (`src/`), num único arquivo de 796 linhas (`src/data/sources/market_data.py`)
que mistura cliente DLP, Yahoo, BCB, Tesouro, PTAX e scraping B3 via Selenium, com cache
apenas em Redis (Upstash) e `except: return None` por toda parte.

O engine de cálculo (`src/engine/{twr,irr,metrics}.py`) é NumPy/pandas puro e testado, mas o
orquestrador (`src/engine/financial_report.py`, 417 linhas) importa `matplotlib` no topo e tem
7 métodos `plot_*` que **misturam cálculo e plotagem**, retornando `(fig, DataFrame)`. Há ainda
um bug conhecido: `self.logger` é referenciado (linha 191) sem existir, quebrando o caminho de
dados vazios.

Este design descreve como:

- **Fase 2** — quebrar `market_data.py` em `apps/market/`, com camadas de responsabilidade
  única; gravar dados **crus** em `MarketSeries` (Postgres = fonte de verdade), com Redis como
  cache derivado; entregar dois comandos (`ingest_market_data` ao vivo, `backfill_market_data`
  offline a partir dos CSVs do repo).
- **Fase 2.5** — copiar o engine puro sem alteração; extrair as funções `compute_*` dos
  `plot_*`; adaptar `build_dataset` / `fetch_user_portfolio` para consumir os serviços novos;
  preservar os `plot_*` Matplotlib num `report_legacy.py` só-CLI; corrigir o bug do `self.logger`.

Resultado esperado: `apps/market/` funcional (dados no Postgres, cache no Redis, séries
derivadas sob demanda) e `engine/` com uma superfície de cálculo pura e testável, pronta para
os endpoints da Fase 3.

---

## 2. Decisões travadas (brainstorming 2026-08-31)

| # | Tema | Decisão |
|---|---|---|
| 1 | Estrutura do incremento | **Um spec** cobrindo Fase 2 + 2.5; **dois planos** de implementação separados (fecha/revisa a Fase 2 antes de mexer no engine). |
| 2 | Fonte de verdade | `core.MarketSeries` (Postgres) é a fonte de verdade. Redis é **cache derivado** com TTL. Cache miss → reconstrói do Postgres. |
| 3 | Benchmarks sintéticos | Contrato **estruturado** no backend: `SyntheticSpec{base, operador, valor}`. A string legada (`"IPCA + 6%"`) vira açúcar de frontend / migração, via um parser. |
| 4 | Estratégia de teste | **Mock no HTTP** (`requests`, `yfinance`, `bcb.sgs`) + **CSVs reais do repo como fixtures** de replay. Chamadas ao vivo atrás de `@pytest.mark.integration` (desmarcado por padrão). |
| 5 | O que uma linha de `MarketSeries` guarda | **Valor cru canônico por fonte** (taxa CDI %, close do ETF, PU do Tesouro, PTAX venda). Toda derivação (cumprod, splice ANBIMA→ETF, USD→BRL, sintéticos, rebase) acontece numa **camada de transformação determinística na leitura**, cacheada no Redis. |
| 6 | Schema de `MarketSeries` | **Adiciona `kind`** (`rate` \| `level`) → migration `core.0002`. `kind` por-linha (denormalizado): torna o `MarketSeries` auto-descritivo. `price` colapsa em `level`. |
| 7 | Splice ANBIMA→ETF + USD→BRL | Vivem **só na camada de leitura**, compondo as linhas cruas independentes (`bcb:12466` + `yf:IMAB11.SA` + `ptax:USDBRL`). Mudar o algoritmo não exige re-ingestão. |
| 8 | Bootstrap dos dados | `backfill_market_data` (command offline, lê os ~166 CSVs de `data/raw`+`data/static`) **+** `ingest_market_data` (ao vivo, agendável na Fase 5). Complementares: backfill = passado profundo; ingest = ponta recente. |
| 9 | Engine `twr`/`irr`/`metrics` | **Cópia literal** na Fase 2.5 (preserva a validade da paridade da Fase 6). Débito de refatoração registrado em ADR, revisão **pós-Fase 6**. Exceção: alinhar o comportamento de `pandas.pct_change` entre os dois lados já na 2.5 (afeta a própria paridade). |
| 10 | `dataset.py` vs classe `FinancialReport` | Desvio consciente da árvore do doc mestre: `engine/dataset.py` (funções) no lugar da classe god-object. |
| 11 | `ativos.csv` / catálogo de ativos | **Fora da Fase 2.** O catálogo de benchmarks (`catalog.py`) são `SeriesDef` em Python puro. O `ativos.csv` (lista-universo de ~7.500 tickers) é consumido só na Fase 3 (endpoint de catálogo). |

---

## 3. Escopo e Definition of Done

### 3.1 Dentro deste design

**Fase 2 — `apps/market/`**
- Camadas: `services/sources/*` (busca pura), `repository.py` (único acesso a `MarketSeries`),
  `cache.py` (`django.core.cache` + envelope pandas), `catalog.py` (`SeriesDef` em Python),
  `synthetics.py` (`SyntheticSpec` + parser), `transforms.py` (funções puras), `series.py`
  (fachada de leitura).
- Migration `core.0002` — campo `kind` em `MarketSeries`.
- Comandos `ingest_market_data` e `backfill_market_data`.
- Suíte `pytest` (unit com mock; smoke `integration` opt-in).

**Fase 2.5 — `backend/engine/`**
- Cópia literal de `twr.py`, `irr.py`, `metrics.py` (+ testes portados).
- `compute.py` — 7 funções puras extraídas dos `plot_*`.
- `dataset.py` — `fetch_user_portfolio` + `build_dataset`, consumindo `apps.market.series`;
  correção do bug `self.logger`.
- `report_legacy.py` — os `plot_*` Matplotlib, só-CLI.
- ADR do débito técnico do engine + atualização do doc mestre §2.3.

### 3.2 Fora deste design (fases seguintes)

- `ativos.csv`, endpoint `GET /api/v1/benchmarks/catalog` → Fase 3.
- Endpoints HTTP, `serializers.py` (pandas → formato ECharts), auth de sessão do Ninja → Fase 3.
- Planejador de IR (`apps/migration_ir/`) → Fase 3.
- Scheduler (`ofelia`/cron), `Dockerfile.ingestion` com Chrome, ligar
  `B3_INGEST_USE_SELENIUM` → Fase 5.
- `docker-compose.yml` do app, nginx unificado → Fase 5.
- Import das linhas `user_tokens` do Supabase → fase de migração de dados.
- Remoção de `src/`, `streamlit`, `altair`, `supabase`, `upstash-redis` → Fase 6.
- Harness de paridade numérica completo → Fase 6 (a 2.5 só deixa `transforms`/`compute_*`
  testáveis e determinísticos).

### 3.3 Definition of Done

**Fase 2** — a partir de um checkout limpo da `migracao-django`, com Fases 0 + 1 aplicadas:

1. `.venv/bin/pytest apps/market -q` verde (sem `-m integration`).
2. `manage.py backfill_market_data` popula `MarketSeries` a partir de `data/raw` +
   `data/static`; `MarketSeries.objects.count()` > 0; contagem por `source` coerente com os
   arquivos existentes.
3. `manage.py ingest_market_data --only bcb:12 --since <hoje-30d>` grava pontos novos (ou loga
   "sem novidade" sem erro); rodar 2× é idempotente (nenhuma linha duplicada — a
   `UniqueConstraint` segura).
4. `apps.market.series.get_series("CDI", start, end)` e `get_series("IMA-B", start, end)`
   (que exercita o `splice_proxy`) retornam `pd.Series` não-vazia; a 2ª chamada idêntica é
   **cache hit** no Redis (verificável por log / `cache.get_df`).
5. `migration core.0002` aplicada; `kind` preenchido em **todas** as linhas de `MarketSeries`.
6. Admin: `MarketSeries` navegável, filtro por `source`, `date_hierarchy` por
   `reference_date`.
7. `manage.py check` limpo; `manage.py makemigrations --check --dry-run` diz "No changes
   detected".
8. `docker exec lab-postgres psql -U postgres -d banco_lab -c "\dt"` — segue sem relations
   do app.
9. `pr-review-toolkit` → `silent-failure-hunter` rodado sobre o diff de `apps/market/` — sem
   `except: ... return None` remanescente; todo erro levanta ou loga+conta.

**Fase 2.5**:

1. `.venv/bin/pytest engine -q` verde; `test_twr.py` / `test_irr.py` / `test_metrics.py`
   portados passam **sem nenhuma alteração** no código de cálculo (prova da cópia literal).
2. Cada `compute_*` tem ≥ 1 teste reproduzindo um número conhecido do fluxo legado.
3. `build_dataset` consome `apps.market.series.get_many` — `processar_benchmarks` não é mais
   importado. O caminho de dados vazios não quebra (bug `self.logger` corrigido, com teste de
   regressão).
4. `report_legacy` gera os PNGs por CLI; `grep -rn matplotlib backend/apps backend/config`
   não retorna nada (a API não importa Matplotlib).
5. ADR `docs/adr/0011-engine-copia-literal-fase-2-5.md` criado; `consolidacao_arquitetura_e_migracao.md`
   §2 e §2.3 atualizadas.

---

## 4. Fase 2 — `apps/market/`

### 4.1 Estrutura de arquivos

```
backend/apps/market/
├── __init__.py
├── apps.py                      # name = "apps.market"; default_auto_field = BigAutoField
├── catalog.py                   # SeriesDef (Python puro) — porta de benchmarks_config.py
├── synthetics.py                # SyntheticSpec{base, operador, valor} + parse_legacy_string
├── repository.py                # ÚNICO acesso a core.MarketSeries
├── cache.py                     # django.core.cache + envelope pandas
├── transforms.py                # funções PURAS de derivação
├── series.py                    # FACHADA DE LEITURA: get_series / get_many
├── services/
│   ├── __init__.py
│   └── sources/
│       ├── __init__.py
│       ├── base.py              # RawPoint (dataclass), FetchError, DLPError
│       ├── bcb.py               # fetch_sgs(codigo, start, end) -> list[RawPoint]
│       ├── yahoo.py             # fetch_yahoo(ticker, start, end)
│       ├── tesouro.py           # fetch_tesouro(titulo, vencimento, start, end)
│       ├── ptax.py              # fetch_ptax(start, end)  (Olinda BCB)
│       ├── b3.py                # read_b3_static(...)  +  fetch_b3_selenium(...)
│       └── dlp.py               # buscar_historico / buscar_resumo_carteira (LIVE, por-usuário)
├── management/
│   ├── __init__.py
│   └── commands/
│       ├── __init__.py
│       ├── ingest_market_data.py
│       └── backfill_market_data.py
└── tests/
    ├── __init__.py
    ├── fixtures/                # ~6 CSVs representativos copiados do repo legado
    ├── test_sources_bcb.py
    ├── test_sources_yahoo.py
    ├── test_sources_tesouro.py
    ├── test_sources_ptax.py
    ├── test_sources_b3.py
    ├── test_sources_dlp.py
    ├── test_repository.py
    ├── test_cache.py
    ├── test_transforms.py
    ├── test_synthetics.py
    ├── test_catalog.py
    ├── test_series.py
    └── test_commands.py
```

`apps.market` entra em `INSTALLED_APPS` (para os management commands e o test discovery). Não
tem modelos próprios na Fase 2 (`MarketSeries` fica em `apps.core`, conforme ADR 0007).

### 4.2 Modelo de dados — `core.MarketSeries` + migration `core.0002`

`core.MarketSeries` hoje (Fase 1):

```python
series_key      = CharField(max_length=120, db_index=True)
source          = CharField(max_length=16)
reference_date  = DateField()
value           = DecimalField(max_digits=20, decimal_places=8)
updated_at      = DateTimeField(auto_now=True)
# UniqueConstraint(["series_key", "reference_date"], name="uniq_series_point")
# Index(["series_key", "reference_date"]); ordering por essas colunas
```

**Migration `core.0002`** adiciona:

```python
kind = CharField(max_length=8)   # "rate" | "level"
```

- `default="level"` na migration (tabela vazia — nenhuma linha a migrar; o `default` some do
  modelo depois se preferirmos, ou fica como default benigno).
- `kind` é **por-linha e denormalizado**: todos os pontos de `bcb:12` têm `kind="rate"`. A
  ingestão sempre grava `kind` a partir do `catalog.py`, então não diverge. Ganho: dá para ler
  o banco e saber interpretar o valor sem o catálogo de código.
- Semântica:
  - `rate` — taxa em % que **precisa de `cumprod`** para virar índice: `bcb:12` (CDI),
    `bcb:11` (SELIC), `bcb:433` (IPCA).
  - `level` — já é magnitude (número-índice, preço, PU, cotação): `bcb:12466…` (IMA-B/IRF-M/
    IMA-S — o BCB devolve número-índice), `yf:*`, `b3:*`, `td:*`, `ptax:USDBRL`.

**Taxonomia de `series_key` / `source`:**

| `source` | `series_key` | Exemplos | `kind` |
|---|---|---|---|
| `BCB` | `bcb:<codigo_sgs>` | `bcb:12`, `bcb:11`, `bcb:433` | `rate` |
| `BCB` | `bcb:<codigo_sgs>` | `bcb:12466`, `bcb:12461`, `bcb:12469` | `level` |
| `YF` | `yf:<ticker>` | `yf:^GSPC`, `yf:IVVB11.SA`, `yf:BTC-USD`, `yf:IMAB11.SA` | `level` |
| `TD` | `td:<titulo>:<venc-ISO>` | `td:Tesouro IPCA+:2035-05-15` (dormente hoje) | `level` |
| `B3` | `b3:<indice>` | `b3:IFIX`, `b3:SMLL`, `b3:IDIV` | `level` |
| `PTAX` | `ptax:USDBRL` | `ptax:USDBRL` | `level` |

Metadados que **não** entram no banco (ficam no `catalog.py`): periodicidade (`daily` /
`monthly` — IPCA é mensal), rótulo humano, e a receita de derivação.

### 4.3 `services/sources/` — busca pura

`base.py`:

```python
@dataclass(frozen=True)
class RawPoint:
    date: datetime.date
    value: float            # convertido p/ Decimal(20,8) só no repository

class FetchError(Exception): ...     # fontes LEVANTAM isto — nunca "return None" silencioso
class DLPError(Exception): ...       # falha na API DLP (por-usuário)
```

Cada módulo **só** fala com a fonte e devolve `list[RawPoint]` com o **valor cru**. Sem banco,
sem cache, sem cumprod, sem splice. Levanta `FetchError(...) from e` em qualquer falha de
rede/parse — não devolve `None` silencioso.

| Módulo | Função | Devolve (cru) | Notas |
|---|---|---|---|
| `bcb.py` | `fetch_sgs(codigo, start, end)` | taxa % (`12/11/433`); número-índice (`12466…`) | mantém o *chunking* de 5 anos (limite de range do SGS); porta de `buscar_dados_bcb` **sem** a parte de cumprod/proxy/cache |
| `yahoo.py` | `fetch_yahoo(ticker, start, end)` | `Close` ajustado (`auto_adjust=True`) | trata colunas MultiIndex do yfinance novo |
| `tesouro.py` | `fetch_tesouro(titulo, vencimento, start, end)` | `PU Base Manhã` filtrado por tipo+vencimento | path dormente (`CATALOGO_TD` vazio) mas implementado + testado com fixture sintética |
| `ptax.py` | `fetch_ptax(start, end)` | `cotacaoVenda` diária | API Olinda `CotacaoDolarPeriodo`; porta de `buscar_dolar_bcb` |
| `b3.py` | `read_b3_static(indice, start, end)` | `Close` do índice, dos CSVs `data/static/<T>_all.csv` | **path usado agora** (host, sem Chrome); porta de `b3_source.get_b3_index` |
| `b3.py` | `fetch_b3_selenium(indice, anos)` | idem, via scraping | portado e testado **com mock**; não roda no host; Fase 5 pluga no container com Chrome |
| `dlp.py` | `buscar_historico(token, ...)`, `buscar_resumo_carteira(token)` | `DataFrame` / `dict` da API DLP | **ao vivo, por-usuário. NÃO grava em `MarketSeries`.** Cache Redis TTL 600s. Levanta `DLPError` em erro HTTP; `return None` só para "sem dados" genuíno. |

**Comportamento incremental** ("busca só o que falta") **não** vive aqui — os fetchers são
stateless. Quem decide o `start` é o comando `ingest_market_data`, que conhece o
`repository.last_date(key)`.

### 4.4 `repository.py` — único acesso a `core.MarketSeries`

```python
def upsert_points(series_key: str, source: str, kind: str,
                  points: Iterable[RawPoint]) -> int:
    """bulk_create(update_conflicts=True,
                   unique_fields=["series_key", "reference_date"],
                   update_fields=["value", "kind", "updated_at"]).
    float -> Decimal(str(v)).quantize(Decimal('1e-8')). Descarta NaN/inf: loga a
    contagem descartada (WARNING), não silencia. Retorna nº de pontos gravados."""

def load_series(series_key: str, start: date, end: date) -> pd.Series:
    """filter(series_key=..., reference_date__range=(start, end))
       .order_by("reference_date").values_list("reference_date", "value")
    -> pd.Series(dtype=float, index=DatetimeIndex, name=series_key).
    Sem linhas -> pd.Series vazia (NÃO None). Decimal -> float aqui."""

def last_date(series_key: str) -> date | None:
    """Maior reference_date da série, ou None. Para o fetch incremental do comando."""

def available_keys() -> list[str]:
    """distinct series_key — Admin / debug."""
```

Sem transforms, sem cache. Postgres ↔ pandas, só isso.

### 4.5 `cache.py` — `django.core.cache` + envelope pandas

Porta do envelope de serialização de `src/data/cache.py`, por cima do `django-redis` que a
Fase 1 já ligou no `lab-redis`. **Sai** o cliente Upstash REST; **sai** o dict `_LOCAL_CACHE`
em processo (o `django-redis` já tem cache de conexão; e o L1 do legado causava *staleness*).

```python
def get_df(key: str) -> pd.Series | pd.DataFrame | None:
    """cache.get(key); se dict com _type in {"pd.Series", "pd.DataFrame"}:
    pd.read_json(StringIO(payload), orient="table"); Series -> .iloc[:, 0].
    Senão devolve como está. Erro de cache -> log WARNING + None."""

def set_df(key: str, value, ttl: int | None) -> None:
    """pd.Series/DataFrame: achata colunas MultiIndex (guarda do legado),
    to_json(orient="table"), embrulha em {"_type": ..., "payload": ...}, cache.set(key, env, ttl).
    Senão cache.set(key, value, ttl). Erro de cache -> log WARNING + segue (cache é otimização)."""

def make_key(*parts) -> str:
    """'market:' + ':'.join(str(p) for p in parts).
    Ex.: make_key('series', 'IMA-B', '2020-01-01', '2024-12-31')
      -> 'market:series:IMA-B:2020-01-01:2024-12-31'."""
```

- **A chave inclui a janela de datas** → corrige o bug do legado (chave `bcb:12` ignorava a
  janela usada no cálculo do acumulado).
- Namespace `market:*` em tudo que este app escreve — greppável e flushável.
- Constantes de TTL: `TTL_SERIES = 86400` (24h, séries derivadas); `TTL_DLP = 600` (10min,
  dados por-usuário da DLP).
- O envelope permanece JSON-compatível com o formato legado de propósito (rede de segurança
  durante a coexistência).

### 4.6 `catalog.py` — `SeriesDef` em Python puro

Porta de `src/data/benchmarks_config.py` (`CATALOGO_YF`, `CATALOGO_B3`, `CATALOGO_BCB`,
`CATALOGO_TD`, `CATALOGO_CRYPTO`, `BENCHMARKS_ATIVOS`). **Sem CSV** — são estruturas Python.

```python
@dataclass(frozen=True)
class SeriesDef:
    name: str                     # "CDI", "IMA-B", "S&P 500 BRL"
    source: str | None            # "BCB"/"YF"/"TD"/"B3"/"PTAX"; None se puramente derivada
    series_key: str | None        # "bcb:12", "yf:^GSPC"; None se puramente derivada
    kind: str                     # "rate" | "level"
    periodicity: str              # "daily" | "monthly"
    derivation: Derivation | None # None = usa cru (com accumulate implícito se kind=="rate")

# Derivation — união marcada, fechada (NÃO um DAG genérico):
class ProxySplice:  bcb_key: str;  etf_key: str
class ToBRL:        base_name: str
class Synthetic:    spec: SyntheticSpec
class Composite:    weights: dict[str, float]
```

- `catalog.resolve(name) -> SeriesDef` — levanta `KeyError` explícito se o nome não existe.
- `catalog.ingestible() -> list[SeriesDef]` — os que têm `source`/`series_key` concretos (o
  que `ingest_market_data` percorre). Puramente derivadas são calculadas na leitura, nunca
  gravadas.
- `DEFAULT_BENCHMARKS: list[str]` — a lista de exibição padrão (porta de `BENCHMARKS_ATIVOS`),
  como constante Python.
- `SELIC` sempre entra em `ingestible()` (o engine usa a série da SELIC como taxa livre de
  risco no Sharpe).

### 4.7 `synthetics.py` — contrato estruturado

```python
@dataclass(frozen=True)
class SyntheticSpec:
    base: str        # "IPCA" | "CDI"
    operador: str    # "+"  (spread a.a.)   |  "%"  (percentual do índice)
    valor: float     # 6.0 => +6% a.a.       |  110.0 => 110% do CDI

def parse_legacy_string(s: str) -> SyntheticSpec | None:
    # "IPCA + 6%"   -> SyntheticSpec("IPCA", "+", 6.0)
    # "CDI + 2%"    -> SyntheticSpec("CDI",  "+", 2.0)
    # "110% do CDI" -> SyntheticSpec("CDI",  "%", 110.0)
    # lixo          -> None
```

As 3 regexes de `financial_report.build_dataset` (linhas 133‑146) viram **um** parser. A API
da Fase 3 aceita o `SyntheticSpec` direto; `parse_legacy_string` serve para migrar configs
antigas e como açúcar de frontend. As fórmulas ficam em `transforms.synthetic_from_spec`,
idênticas ao legado.

### 4.8 `transforms.py` — funções puras (alvo da paridade da Fase 6)

Toda função: `pd.Series/DataFrame` entra → sai. Zero I/O, zero Django, determinística.

```python
def accumulate(rate: pd.Series, periodicity: str) -> pd.Series:
    """kind='rate': (rate/100 + 1).cumprod(); 1º valor NaN -> 1.
    periodicity só afeta a anualização em synthetic_from_spec, não o cumprod cru."""

def splice_proxy(bcb_level: pd.Series, etf_price: pd.Series) -> pd.Series:
    """Porta de BCB_PROXIES em buscar_dados_bcb. Usa bcb_level até a última data válida;
    daí em diante projeta a variação do ETF: bcb_last * (etf / etf_base).
    Sem NaN na emenda; sem duplicar o dia-base."""

def to_brl(usd_series: pd.Series, ptax: pd.Series) -> pd.Series:
    """usd_series * ptax.reindex(usd_series.index, method='ffill')."""

def synthetic_from_spec(spec: SyntheticSpec, base: pd.Series) -> pd.Series:
    """spec.operador '+': (1 + ((1 + base.pct_change(fill_method=None).fillna(0))
                                 * (1 + spec.valor/100)**(1/252) - 1)).cumprod() * 100
       spec.operador '%': (1 + (base.pct_change(fill_method=None).fillna(0)
                                 * spec.valor/100)).cumprod() * 100
    Idêntico ao caminho das regexes de build_dataset (com fill_method explícito — §5.4).
    ATENÇÃO: o legado tem duas fórmulas conflitantes para "IPCA + X%" (§8) — esta segue a
    de build_dataset (**(1/252)), não a hardcoded de processar_benchmarks (**(1/12))."""

def composite(members: dict[str, pd.Series], weights: dict[str, float]) -> pd.Series:
    """Porta de _calc_portfolio: soma dos retornos ponderados -> (1 + ret).cumprod()."""

def rebase_100(series: pd.Series, at: pd.Timestamp | None = None) -> pd.Series:
    """(series / base_val) * 100 — o rebase hoje espalhado no build_dataset."""
```

### 4.9 `series.py` — a fachada de leitura

O que a Fase 3 e o `engine/dataset.py` chamam. Substitui inteiramente o `processar_benchmarks`
de 80 linhas do legado.

```python
def get_series(name: str, start: date, end: date) -> pd.Series | None:
    key = cache.make_key("series", name, start, end)
    if (hit := cache.get_df(key)) is not None:
        return hit
    sdef = catalog.resolve(name)
    s = _materialize(sdef, start, end)
    if s is None or s.empty:
        logger.info("série '%s' vazia no período %s..%s", name, start, end)
        return None
    cache.set_df(key, s, ttl=cache.TTL_SERIES)
    return s

def get_many(names: list[str], start: date, end: date) -> dict[str, pd.Series]:
    """Itera get_series resolvendo dependências entre nomes (Composite puxa membros,
    ToBRL puxa base + 'USD/BRL'). Nomes ausentes/vazios ficam de fora do dict (com log)."""
```

`_materialize(sdef, start, end)` despacha por `sdef.derivation`:

| Caso | Ação |
|---|---|
| `None` + `kind == "rate"` | `transforms.accumulate(repository.load_series(sdef.series_key, start, end), sdef.periodicity)` |
| `None` + `kind == "level"` | `repository.load_series(sdef.series_key, start, end)` direto |
| `ProxySplice(bcb_key, etf_key)` | `transforms.splice_proxy(repository.load_series(bcb_key, ...), repository.load_series(etf_key, ...))` |
| `ToBRL(base_name)` | `transforms.to_brl(get_series(base_name, ...), get_series("USD/BRL", ...))` |
| `Synthetic(spec)` | `transforms.synthetic_from_spec(spec, get_series(spec.base, ...))` |
| `Composite(weights)` | `transforms.composite({m: get_series(m, ...) for m in weights}, weights)` |

Recursão rasa — profundidade real ≤ 3 (`Composite` → membros → talvez `ToBRL` → base + PTAX).
Não há resolvedor de DAG genérico.

### 4.10 `ingest_market_data` — ao vivo (agendável na Fase 5)

```
manage.py ingest_market_data [--only KEY[,KEY...]] [--full] [--since YYYY-MM-DD]
```

- Percorre `catalog.ingestible()`.
- Para cada série: `start = repository.last_date(key) + 1 dia` (incremental). `--full` volta a
  `1995-01-01`; `--since` força a data.
- `points = <dispatch por source>.fetch_*(...)`; `repository.upsert_points(key, source, kind, points)`.
- Ao fim de todas as gravações, **aquece o cache**: chama `series.get_series` para
  `catalog.DEFAULT_BENCHMARKS` em janelas comuns (1a, 5a, tudo) → o 1º request da Fase 3 já é
  hit.
- **B3**: usa `read_b3_static` por padrão. `settings.B3_INGEST_USE_SELENIUM` (default `False`)
  liga `fetch_b3_selenium` — a Fase 5 ativa no container `lab-ingestion`. Fallback automático
  Selenium→estático quando o scraping falha (§7.5 do doc mestre) já embutido.
- **Política de falha**: uma série falhar → `logger.error` + incrementa contador, **segue** as
  outras (job em lote não aborta por um ticker que o Yahoo limitou). Fecha com resumo
  `N ok / M falhas` + as chaves que falharam; **exit code ≠ 0** se houve qualquer falha. Porta
  o dict `stats` + retry/backoff de `scripts/sync_to_redis.py`.
- **Séries descontinuadas (ANBIMA, meados de 2023)**: `repository.last_date("bcb:12466")` fica
  parada em 2023; a busca incremental volta vazia toda execução → `logger.info("bcb:12466:
  sem novos pontos (possivelmente descontinuada)")`, **não conta como falha**. O
  `splice_proxy` na leitura preenche o gap com `yf:IMAB11.SA` (que segue atualizando via
  ingestão normal).
- Logging estruturado para stdout (ADR 0008).

### 4.11 `backfill_market_data` — bootstrap offline (descartável)

```
manage.py backfill_market_data [--raw-dir PATH] [--static-dir PATH] [--only KEY...] [--dry-run]
```

- Dirs default: `<repo>/data/raw` e `<repo>/data/static` (relativos a `BASE_DIR`).
  Configuráveis.
- Mapeia nome de arquivo → `series_key` (porta o parsing de `sync_to_redis.py`):
  - `BCB_<codigo>.csv` → `bcb:<codigo>` (`source=BCB`).
  - `YF_<t>.csv` → `yf:<ticker>` (`_`→`.`; `BVSP`→`^BVSP`).
  - `TD_<titulo>_<venc>.csv` → `td:<titulo>:<venc-ISO>`.
  - `<TICKER>_all.csv` (em `static/`) → `b3:<TICKER>`.
- Lê o CSV → `list[RawPoint]` → `repository.upsert_points`. `kind` vem do `catalog`. Idempotente.
- **Só toca em arquivo que mapeia com confiança para uma série do catálogo.** Arquivos órfãos
  (`IBLV-2005.csv` e demais CSVs anuais de B3, já superados pelos `_all.csv`; tickers fora do
  catálogo) são **pulados com log**, nunca adivinhados.
- `--dry-run` relata o que gravaria, sem escrever.
- Mesmo resumo + disciplina de exit-code do `ingest`.
- Documentado no `backend/README.md` como ferramenta de **bootstrap único**, não de rotina.

### 4.12 Tratamento de erros — sem silent failures

`src/data/sources/market_data.py` é `except Exception: return None` por toda parte. O
`silent-failure-hunter` é **prioritário** nesta fase (guia de ferramentas §Fase 2/2.5).
Postura nova, camada a camada:

| Camada | Falha externa / transitória | Bug / dado corrompido |
|---|---|---|
| `services/sources/*` | `raise FetchError(...) from e` — nunca `return None` | propaga |
| `repository` | `raise` (erro de banco é grave) | `raise`; ponto NaN/inf → descarta **o ponto**, loga contagem, segue |
| `cache` | `logger.warning` + segue (cache é otimização) | idem |
| `series.get_series` | dependência vazia → `None` **com log dizendo qual série faltou** | propaga |
| comandos | série falha → `error` + contador, segue; exit ≠ 0 no fim | propaga (bug quebra o build) |
| `dlp.py` | `raise DLPError` em erro HTTP; `return None` **só** para "sem dados" real | — |

Regra única: **"sem dados" ≠ "deu erro"**. `None` só significa "não existe". Erro sempre
levanta ou loga+conta.

### 4.13 Fixtures de teste

Copiar para `backend/apps/market/tests/fixtures/` (versionado, independente de `src/`/`data/`
que somem na Fase 6):

- `BCB_12.csv` — série `rate` (CDI).
- `BCB_12466.csv` — série `level` descontinuada (exercita `splice_proxy`).
- `YF_GSPC.csv` — série `price`/`level`.
- `IFIX_all.csv` — índice B3 estático.
- `PTAX_sample.csv` — recorte de cotação PTAX.
- `TD_sample.csv` — amostra sintética do Tesouro (não existe hoje; montada à mão).

---

## 5. Fase 2.5 — `backend/engine/`

### 5.1 Estrutura de arquivos

```
backend/engine/
├── __init__.py
├── twr.py            # cópia literal de src/engine/twr.py
├── irr.py            # cópia literal de src/engine/irr.py
├── metrics.py        # cópia literal de src/engine/metrics.py  (+ ajuste pct_change — §5.4)
├── compute.py        # NOVO — compute_* puros extraídos dos plot_*
├── dataset.py        # NOVO — fetch_user_portfolio + build_dataset, consumindo apps.market.series
├── report_legacy.py  # os plot_* Matplotlib, só-CLI; NÃO importado pela API
└── tests/
    ├── __init__.py
    ├── test_twr.py       # portado de tests/test_twr.py
    ├── test_irr.py       # portado
    ├── test_metrics.py   # portado
    ├── test_compute.py   # NOVO
    └── test_dataset.py   # NOVO
```

`engine/` é pacote Python puro, **irmão de `apps/`** (não é Django app — não tem modelos).
`config/settings.py` não muda; `pytest.ini` passa a ter `testpaths = apps engine`.

### 5.2 `compute.py` — extrair cálculo dos `plot_*`

Cada `plot_*` de `financial_report.py` hoje mistura cálculo + Matplotlib e retorna
`(fig, DataFrame)`. A parte de cálculo sai para funções **puras** (pandas dentro, pandas/dict
fora; zero matplotlib, zero I/O):

| Legado (`financial_report.py`) | Função pura em `compute.py` | Devolve |
|---|---|---|
| `plot_twr_evolution` | `compute_twr_evolution(df_combined)` | `{"series": DataFrame base-100, "final_returns": dict[str, float]}` |
| `plot_drawdown` | `compute_drawdown(df_combined)` | `DataFrame` (wrap de `metrics.calculate_drawdown`) |
| `plot_risk_return_scatter` | `compute_risk_return(df_combined)` | `DataFrame[Volatilidade, "Retorno (CAGR)", Sharpe]` (hoje inline, linhas 249‑255) |
| `plot_rolling_volatility` + `plot_rolling_sharpe` | `compute_rolling_metrics(df_combined, rf_series, window=252)` | `{"volatility": DataFrame, "sharpe": DataFrame}` |
| `plot_irr_evolution` | `compute_irr_evolution(portfolio_df)` | `Series` (caminhada XIRR mensal, linhas 313‑349) |
| `simulate_shadow_portfolios` | `compute_shadow(portfolio_df, df_combined)` | `DataFrame` (shadow wealth, linhas 361‑384) |
| `generate_summary_table` | `compute_yearly_summary(df_combined)` | `DataFrame` (retorno ano a ano + total acumulado) |

A serialização para o formato que o ECharts espera (`[[timestamp, valor], ...]`) é da Fase 3
(`apps/api/serializers.py`) — `compute.py` para em pandas.

### 5.3 `dataset.py` — montagem do dataset, sem framework

Porta de `FinancialReport.fetch_user_portfolio` + `build_dataset`, com três mudanças:

1. **Consome `apps.market.series`**: `build_dataset` chama `series.get_many(names, start, end)`
   no lugar de `processar_benchmarks(...)`. As 3 regexes de sintético somem — `series.get_series`
   já resolve `Synthetic` via catálogo. Sintéticos ad-hoc (o usuário digita "IPCA + 7%" na UI)
   entram por um parâmetro opcional `extra_specs: list[SyntheticSpec]`, resolvido via
   `transforms.synthetic_from_spec`.
2. **`fetch_user_portfolio` chama `apps.market.services.dlp.buscar_historico`** (não o import
   legado). Usa `engine.twr.calculate_twr`.
3. **Sem `self.`**: em vez de mutar `self.df_combined` / `self.portfolio_df` /
   `self.selic_series`, devolve um `DatasetResult` (dataclass: `df_combined` base-100,
   `selic_series`, `portfolio_df`). **Corrige o bug do `self.logger`** (linha 191:
   `self.logger.warning(...)` sem `self.logger` existir — quebra no caminho de dados vazios) →
   `logger = logging.getLogger(__name__)` no módulo.

### 5.4 Exceção do `pandas.pct_change` (não adiar)

`src/engine/metrics.py` e `financial_report.build_dataset` chamam `.pct_change()` sem
`fill_method`. No pandas 2.2 (que o backend vai pinar) o default muda de comportamento e emite
`FutureWarning`. O legado roda pandas sem pin. **Se os dois lados rodarem pandas diferente, a
paridade da Fase 6 diverge exatamente por isso.**

Decisão: na Fase 2.5, aplicar `fill_method=None` explícito em **todos** os `pct_change` do
engine portado e do `dataset.py`, **e** registrar no ADR do débito que o mesmo ajuste deve ser
feito no `src/` legado antes da suíte de paridade da Fase 6 (ou o pin de pandas dos dois lados
alinhado). Este é o único ponto onde a "cópia literal" recebe um ajuste consciente.

### 5.5 `report_legacy.py` — os `plot_*` Matplotlib preservados

Funções `plot_*(df_combined, portfolio_df, ...)` que chamam as `compute_*` e desenham com
Matplotlib, para **export CLI** (a pasta `reports/`). **Nunca importado pela API.**
`import matplotlib` vive só aqui. Serve para gerar os PNGs antigos durante a coexistência e
como conferência visual na paridade da Fase 6. Sem testes além de "importa e produz um objeto
`Figure`".

### 5.6 Direção das dependências

```
engine.twr / irr / metrics  → (puro, nada de app)
engine.compute              → (puro, nada de app)
engine.dataset             → apps.market.series , apps.market.services.dlp , engine.twr
engine.report_legacy       → engine.compute , engine.dataset , matplotlib
```

Unidirecional: `engine` depende de `apps.market`, nunca o contrário. A API da Fase 3 depende
dos dois.

### 5.7 Débito técnico do engine — revisão pós-Fase 6

A cópia literal de `twr.py` / `irr.py` / `metrics.py` preserva a validade da paridade da Fase
6 (compara o backend novo contra o Streamlit rodando **este mesmo código**). Refatorar no port
tiraria o valor do teste. A hora certa de revisar é **pós-Fase 6**, quando existe (a) o harness
de paridade como rede de regressão, (b) o `src/` sendo deletado — acaba o "duas cópias em
sincronia", (c) o contrato da API estável.

Registrar em `docs/adr/0011-engine-copia-literal-fase-2-5.md` os pontos concretos:

- **`irr.py`** — Newton-Raphson puro sem fallback (bisseção/Brent); pode não convergir em
  fluxos irregulares e devolver `None` ou `r` ruim com o cutoff mágico `abs(npv) < 0.1`.
  Chute inicial `0.1` fixo; `365.0` fixo (ignora bissexto); clamp `r <= -1.0 → -0.99` que pode
  oscilar; não guarda contra `dates` fora de ordem. Candidato a `scipy.optimize.brentq` /
  `numpy_financial.xirr`.
- **`twr.py`** — `np.where(denominador != 0, ...)` com igualdade float sem epsilon
  (denominador quase-zero → HPR gigante); normalização de nome de coluna duplicada com
  `fetch_user_portfolio` (o `dataset.py` pode centralizar); `fluxo = vlr_investido.diff().fillna(iloc[0])`
  assume 1ª linha como capital inicial limpo.
- **`metrics.py`** — `252` e `rf_constant=0.10` mágicos espalhados; `pct_change` já ajustado na
  2.5 (§5.4), mas a limpeza dos constantes fica para depois.

Item correspondente adicionado a `consolidacao_arquitetura_e_migracao.md` §2.3.

---

## 6. Testes

### 6.1 Fase 2 (`apps/market/tests/`)

| Arquivo | Cobre | Como |
|---|---|---|
| `test_sources_*.py` | `fetch_*` devolve `RawPoint` correto; levanta `FetchError`/`DLPError` em erro HTTP; distingue "sem dados" de "erro" | mock em `requests` / `yfinance.download` / `bcb.sgs.get`; payloads recortados |
| `test_repository.py` | upsert idempotente; quantização Decimal; descarte NaN/inf com log; `load_series` vazio ≠ None; `last_date` | banco de teste `appinvest_test` |
| `test_cache.py` | round-trip `set_df`/`get_df` (Series e DataFrame); achatar MultiIndex; erro de cache não propaga; `make_key` inclui a janela | mock/real `django.core.cache` (locmem nos testes) |
| `test_transforms.py` | `accumulate` = cumprod manual; `splice_proxy` contínuo na emenda; `to_brl` alinhamento ffill; `synthetic_from_spec` = fórmula legada exata; `composite` pesos; `rebase_100` | puro, fixtures pandas à mão — **semente da paridade da Fase 6** |
| `test_synthetics.py` | `parse_legacy_string` (3 formas) + lixo → None | — |
| `test_catalog.py` | todo `SeriesDef` resolve; sem `series_key` órfão; `kind`/`periodicity` presentes; `ingestible()` não inclui puramente derivadas | — |
| `test_series.py` | cache hit (mock `cache`); cache miss (mock `repository`, `transforms` real); resolução de dependência (`Composite` puxa membros); chave inclui a janela (regressão do bug legado) | — |
| `test_commands.py` | `ingest`: start incremental do `last_date`; falha de 1 fonte → segue + exit ≠ 0 + resumo; aquecimento de cache. `backfill`: mapeia nomes, pula órfãos, idempotente, `--dry-run` | mock em `sources.fetch_*`; CSVs de `tests/fixtures/` |
| `@pytest.mark.integration` | 1 smoke ao vivo por fonte: BCB CDI (30d), Yahoo `^GSPC`, PTAX | **desmarcado por padrão** (`addopts = -m "not integration"`); job de CI dedicado |

### 6.2 Fase 2.5 (`engine/tests/`)

- `test_twr.py` / `test_irr.py` / `test_metrics.py` — portados de `tests/`, só troca de import
  (`src.engine` → `engine`). **Verdes sem alteração no código de cálculo** (o ajuste de
  `pct_change` do §5.4 pode exigir revisão pontual dos asserts de `test_metrics.py` — se
  exigir, documentar por quê).
- `test_compute.py` — cada `compute_*` contra fixture montada à mão; ≥ 1 caso por função
  reproduzindo um número conhecido do fluxo legado.
- `test_dataset.py` — `build_dataset` com `series.get_many` mockado; **caminho de dados
  vazios** (regressão do crash `self.logger`); `fetch_user_portfolio` com `dlp.buscar_historico`
  mockado.

### 6.3 `pytest.ini` (atualização)

```ini
[pytest]
DJANGO_SETTINGS_MODULE = config.settings
python_files = test_*.py
pythonpath = .
testpaths = apps engine
addopts = --reuse-db -m "not integration"
markers =
    integration: chama serviços externos ao vivo (rede) — não roda por padrão
```

---

## 7. Dependências novas — `backend/requirements.txt`

Runtime (Fase 2):

| Pacote | Pin (aprox.; exato via context7 na implementação) | Para quê |
|---|---|---|
| `pandas` | `~=2.2` | toda a manipulação de série |
| `numpy` | `~=2.1` | idem |
| `yfinance` | `~=0.2` | `services/sources/yahoo.py` |
| `python-bcb` | `~=0.3` | `services/sources/bcb.py` (`sgs`) |
| `requests` | `~=2.32` | Tesouro, PTAX Olinda, DLP |
| `selenium` | `~=4.25` | `fetch_b3_selenium` (inerte no host; Fase 5) |
| `webdriver-manager` | `~=4.0` | idem |

Fase 2.5:

| Pacote | Pin | Para quê |
|---|---|---|
| `matplotlib` | `~=3.9` | **só** `engine/report_legacy.py` (CLI) |

`fpdf2` continua fora (é da Fase 3, planejador de IR).

---

## 8. Riscos e pendências

- **`pandas.pct_change` (§5.4):** tratado na 2.5 nos dois lados ou pin alinhado — se ignorado,
  a paridade da Fase 6 divergirá por causa disso.
- **Duas fórmulas legadas conflitantes para `"IPCA + X%"`:** `financial_report.build_dataset`
  (linha 136, caminho das regexes) usa `(1 + spec.valor/100)**(1/252)` (composição diária);
  `market_data.processar_benchmarks` (linha 766, caminho hardcoded "IPCA + 6%") usa
  `(1.06 ** (1/12)) - 1` (composição mensal). São resultados diferentes. `synthetic_from_spec`
  precisa **escolher uma deliberadamente** — recomendação: seguir o caminho das regexes
  (`**(1/252)`), que é o genérico e o que a `build_dataset` já usava para qualquer `X`, e
  documentar a escolha no teste. Conferir na paridade da Fase 6 contra qual dos dois o número
  do Streamlit efetivamente bate.
- **Selenium no host:** `fetch_b3_selenium` não é exercitável sem Chrome; a Fase 2 confia no
  `read_b3_static` + `backfill`. Risco de o path Selenium ter regredido só ser detectado na
  Fase 5. Mitigação: testes com mock cobrem a lógica de parsing; a Fase 5 valida no container.
- **Séries ANBIMA descontinuadas:** dependem do `splice_proxy` e de o ETF proxy
  (`yf:IMAB11.SA` etc.) continuar existindo e sendo ingerido. Se o Yahoo mudar o ticker, a
  série IMA-B trava na data de 2023. Monitorar via o resumo do `ingest`.
- **`bulk_create(update_conflicts=True)`** exige Postgres (ok — é o alvo) e Django ≥ 4.1 (ok —
  5.2). Não funciona no SQLite, então nenhum teste de `repository` pode usar SQLite; todos
  contra `appinvest_test`.
- **Volume de `MarketSeries`:** ~30 séries × ~30 anos × ~250 pregões ≈ 225 mil linhas após o
  backfill completo. Trivial para o Postgres; `load_series` com index em `(series_key,
  reference_date)` é rápido. Sem particionamento nesta fase.
- **`data/raw` tem 166 arquivos, nem todos mapeáveis:** o `backfill` pula os órfãos com log —
  conferir no resumo que os pulados são de fato lixo (CSVs anuais de B3 pré-`_all`), não séries
  que deveriam entrar.
- **Formato `series_key` do Tesouro:** `td:<titulo>:<venc-ISO>` tem `:` no meio e o `titulo`
  tem espaço/`+` ("Tesouro IPCA+"). `CATALOGO_TD` está vazio hoje, então o risco é teórico,
  mas o parser do `backfill` e o `make_key` precisam lidar com isso — testar com a fixture
  sintética.

---

## 9. Divisão dos planos

**Plano 1 — `docs/superpowers/plans/2026-08-31-fase-2-servicos-de-mercado.md`**

| Task | Conteúdo |
|---|---|
| T1 | Deps no `requirements.txt` + `pytest.ini` (marker `integration`, `testpaths`) + migration `core.0002` (`kind`) |
| T2 | `cache.py` + `repository.py` (+ testes) |
| T3 | `services/sources/*` — um módulo por vez, TDD com mock (+ `base.py`) |
| T4 | `synthetics.py` + `transforms.py` (+ testes puros) |
| T5 | `catalog.py` + `series.py` (+ testes) |
| T6 | `backfill_market_data` (+ fixtures + testes) |
| T7 | `ingest_market_data` (+ testes) |
| T8 | Verificação ponta a ponta (DoD §3.3) + `silent-failure-hunter` + docs (§2 do doc mestre; ADR se surgir decisão nova) |

**Plano 2 — `docs/superpowers/plans/2026-08-31-fase-2-5-refatorar-engine.md`**

| Task | Conteúdo |
|---|---|
| T1 | Cópia literal `twr`/`irr`/`metrics` + portar os 3 testes + `pytest.ini testpaths` + ajuste `pct_change` (§5.4) |
| T2 | `engine/compute.py` + `test_compute.py` |
| T3 | `engine/dataset.py` (consome `series`, corrige `self.logger`) + `test_dataset.py` |
| T4 | `engine/report_legacy.py` |
| T5 | ADR `0011-engine-copia-literal-fase-2-5.md` + `consolidacao_arquitetura_e_migracao.md` §2/§2.3 + verificação (DoD §3.3) |

---

## 10. Próximo passo

Após aprovação e revisão deste spec: gerar os **dois planos de implementação** (skill
`writing-plans`), começando pelo Plano 1 (Fase 2). O Plano 2 (Fase 2.5) só começa depois da
Fase 2 fechada e revisada.
