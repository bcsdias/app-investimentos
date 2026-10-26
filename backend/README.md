# Backend — `app-investimentos`

Backend Django da migração Streamlit → Django + React (ver [`../docs/consolidacao_arquitetura_e_migracao.md`](../docs/consolidacao_arquitetura_e_migracao.md)).
Nesta fase roda **no host** (Python 3.12); conteinerização é a Fase 5.

**Estado:** Fases 0 + 1 + 2 + 2.5 + 3 + 3c concluídas — infra LAB + esqueleto Django (`User` customizado,
wrapper Fernet, modelos `UserToken` / `MarketSeries`, Admin), app `apps.market` (fontes de
mercado, `repository`, `cache`, `catalog`, `assets` + `ativos.csv`, `synthetics`, `transforms`, `series` com fetch-on-read, comandos
`ingest`/`backfill`), pacote puro `engine/` (`twr`/`irr`/`metrics` + `compute` + `dataset` +
`report_legacy`) e API REST JSON em `/api/v1/` (`apps.api` + `apps.accounts.api`, Swagger em `/api/v1/docs`). Suíte `pytest` verde (249 testes). O `src/` (Streamlit legado) continua funcionando em
paralelo até a Fase 6.

---

## Pré-requisitos

- **Python 3.12** + pacote `python3.12-venv` (`sudo apt install python3.12-venv`).
- **Docker** + Docker Compose v2.
- Acesso ao daemon Docker do host (a infra LAB roda em containers).
- `../.streamlit/secrets.toml` com `[security].fernet_key` (a chave é reusada — ver
  [ADR 0003](../docs/adr/0003-reuso-da-chave-fernet-do-legado.md)).

---

## Setup

### 1. Ambiente virtual

```bash
cd backend
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

Sempre use `.venv/bin/python` / `.venv/bin/pytest` — não há `python` no PATH da VM.

### 2. Infra LAB (Postgres + Redis compartilhados)

Os compose files ficam **fora do repo** (`/data/projetos/lab-postgres/`, `/data/projetos/lab-redis/`).
Ver [ADR 0001](../docs/adr/0001-infra-lab-compartilhada.md).

```bash
docker network create lab-net 2>/dev/null || true
docker compose -f /data/projetos/lab-postgres/docker-compose.yml up -d
docker compose -f /data/projetos/lab-redis/docker-compose.yml up -d
docker inspect --format '{{.Name}} {{.State.Health.Status}}' lab-postgres lab-redis
```

### 3. Database `appinvest`

```bash
APPINVEST_DB_PASSWORD='<senha-forte>' bash scripts/bootstrap_db.sh
```

Cria o role + database `appinvest` (idempotente). O app **nunca** usa o superusuário `postgres`,
e as tabelas do app **não** caem no `banco_lab` compartilhado
([ADR 0006](../docs/adr/0006-prefixo-lab-containers-appinvest-codigo.md)).

### 4. `.env`

```bash
cp .env.example .env
# preencher DJANGO_SECRET_KEY (get_random_secret_key), APPINVEST_DB_PASSWORD (a mesma do passo 3),
# FERNET_KEY (copiar de ../.streamlit/secrets.toml -> [security].fernet_key)
```

`.env` é git-ignored. **Nunca** commitar; **nunca** imprimir `FERNET_KEY` em log ou saída de teste.

### 5. Migrations + superusuário

```bash
.venv/bin/python manage.py migrate
.venv/bin/python manage.py createsuperuser
```

---

## Rodar

```bash
.venv/bin/python manage.py runserver 0.0.0.0:8000   # Admin em /admin/, Swagger em /api/v1/docs
.venv/bin/python manage.py check
.venv/bin/pytest -q                                 # 249 testes (+ 4 integration com -m integration)
.venv/bin/python manage.py backfill_market_data      # bootstrap: semeia MarketSeries dos CSVs do repo (uma vez)
.venv/bin/python manage.py ingest_market_data        # ingestão ao vivo (incremental); agendamento é Fase 5
.venv/bin/python manage.py shell -c "from engine import dataset, report_legacy; ..."  # export CLI dos gráficos
```

---

## Estrutura

```
backend/
├── config/                 projeto Django
│   ├── api.py               NinjaAPI (/api/v1/, Swagger em /api/v1/docs) + handler ApiError
│   ├── settings.py          DB appinvest, cache Redis, Fernet, logging stdout, TZ America/Sao_Paulo
│   ├── urls.py              /admin/, /api/v1/
│   └── wsgi.py / asgi.py
├── apps/
│   ├── accounts/           User(AbstractUser) — AUTH_USER_MODEL (ADR 0002) + api.py (/auth/*)
│   ├── api/                API da Carteira em Django Ninja (Fase 3, ADR 0013)
│   │   ├── errors.py        ApiError(status, code, message, details)
│   │   ├── csrf.py          ApiSessionAuth (sessão + CSRF no envelope ApiError)
│   │   ├── schemas.py       schemas Ninja (request/response)
│   │   ├── serializers.py   pandas -> JSON estrito no formato do ECharts
│   │   ├── analysis.py      validate_request, get_token, dlp_errors, run_analysis
│   │   └── routers/         dlp.py, portfolio.py, benchmarks.py, assets.py
│   ├── core/
│   │   ├── security.py      wrapper Fernet: encrypt() / decrypt()  (ADR 0003)
│   │   ├── models.py        UserToken (OneToOne, cifrado em repouso) + MarketSeries (ADR 0007, ADR 0011)
│   │   └── admin.py         UserToken sem "add", encrypted_token read-only
│   └── market/             serviços de mercado (Fases 2 + 3, ADR 0014)
│       ├── services/sources/ clientes de mercado (BCB, PTAX, Yahoo, Tesouro, B3, DLP)
│       ├── data/ativos.csv  universo empacotado de tickers (~7.500 linhas)
│       ├── assets.py        resolução de tickers Yahoo + variante BRL + search()
│       ├── repository.py    acesso exclusivo a core.MarketSeries (Postgres)
│       ├── cache.py         envelope pandas sobre Redis (django-redis) + single_flight
│       ├── synthetics.py    contrato estruturado SyntheticSpec + display_name + parser legado
│       ├── transforms.py    derivações puras (accumulate, splice, to_brl, composite)
│       ├── catalog.py       definição e resolução de séries (SeriesDef)
│       ├── series.py        fachada de leitura (Redis -> Repo + Transforms + fetch-on-read -> Redis)
│       └── management/commands/ backfill_market_data, ingest_market_data
├── engine/                 núcleo de cálculo puro (Fase 2.5, ADR 0012)
│   ├── twr.py               cópia literal: cálculo TWR base 100
│   ├── irr.py               cópia literal: cálculo TIR / XIRR
│   ├── metrics.py           cópia literal + pct_change(fill_method=None): drawdown, vol, sharpe
│   ├── compute.py           7 funções puras extraídas dos plot_* de financial_report
│   ├── dataset.py           fetch_user_portfolio + build_dataset (consome apps.market.series)
│   └── report_legacy.py     export CLI dos gráficos históricos (Matplotlib)
├── scripts/
│   ├── bootstrap_db.sh      cria role/database appinvest (idempotente)  — versionado
│   └── rotate_db_password.sh rotaciona a senha: gera + aplica no PG + espelha no .env
├── requirements.txt        pins com ~= (runtime + dev/test)
└── pytest.ini              DJANGO_SETTINGS_MODULE=config.settings, testpaths=apps engine, --reuse-db
```


Convenções para trabalhar aqui: [`CLAUDE.md`](CLAUDE.md).


---

## Variáveis de ambiente (`.env`)

| Variável | Exemplo | Obrigatória | Papel |
|---|---|---|---|
| `DJANGO_SECRET_KEY` | *(gerada)* | sim | assinatura de sessões/CSRF |
| `DJANGO_DEBUG` | `1` | não (default `0`) | modo debug |
| `DJANGO_ALLOWED_HOSTS` | `localhost,127.0.0.1` | não | hosts aceitos |
| `LOG_LEVEL` | `INFO` | não | nível do root logger (stdout) |
| `POSTGRES_HOST` / `POSTGRES_PORT` | `localhost` / `5432` | não | conexão ao `lab-postgres` |
| `APPINVEST_DB_PASSWORD` | *(senha do role)* | sim | senha do role `appinvest` |
| `DLP_BASE_URL` | `https://users.dlpinvest.com.br` | não | URL base da API DLP Invest |
| `REDIS_URL` | `redis://localhost:6379/1` | não | cache (`django-redis`) |
| `FERNET_KEY` | *(base64, 44 chars)* | sim | cripto dos tokens DLP — **mesma chave do app legado** |
