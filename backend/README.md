# Backend — `app-investimentos`

Backend Django da migração Streamlit → Django + React (ver [`../docs/consolidacao_arquitetura_e_migracao.md`](../docs/consolidacao_arquitetura_e_migracao.md)).
O desenvolvimento roda **no host** (Python 3.12); o app empacotado roda nos containers do `../docker-compose.yml` ([ADR 0016](../docs/adr/0016-empacotamento-no-lab.md)).

**Estado:** Fases 0 + 1 + 2 + 2.5 + 3 + 3b + 3c + 4 + 5 + 6 + 7a concluídas — infra LAB + esqueleto Django (`User` customizado,
wrapper Fernet, modelos `UserToken` / `MarketSeries`, Admin), app `apps.market` (fontes de
mercado, `repository`, `cache`, `catalog`, `assets` + `ativos.csv`, `synthetics`, `transforms`, `series` com fetch-on-read, comandos
`ingest`/`backfill`), pacote puro `engine/` com metodologia DLP alinhada (`flows`/`twr`/`irr`/`metrics` + `compute` + `dataset` +
`report_legacy`, [ADR 0018](../docs/adr/0018-metodologia-de-retorno-dlp.md)), API REST JSON em `/api/v1/` (`apps.api` + `apps.accounts.api`, Swagger em `/api/v1/docs`), planejador de IR (`apps.migration_ir`, `/api/v1/migration/*`), SPA React em `../frontend/` e empacotamento Docker (`lab-appinvest-{web,backend,ingestion}` em `127.0.0.1:8080`, Chromium ARM64 + `supercronic`). Suíte `pytest` verde (385 testes unitários/contrato + 8 integration). Paridade com o Streamlit legado validada na Fase 6 ([ADR 0017](../docs/adr/0017-paridade-legado.md)) e nova linha de base da Fase 7 em `reports/parity/20261028-163150-fase7/baseline` (39 testes em `../scripts/parity/`).

---

## Pré-requisitos

- **Python 3.12** + pacote `python3.12-venv` (`sudo apt install python3.12-venv`).
- **Docker** + Docker Compose v2.
- Acesso ao daemon Docker do host (a infra LAB roda em containers).
- A `FERNET_KEY` em uso (a chave do app legado foi reusada — ver
  [ADR 0003](../docs/adr/0003-reuso-da-chave-fernet-do-legado.md)); sem ela os tokens DLP já
  cadastrados não decifram.

---

## Setup

### 1. Ambiente virtual

```bash
cd backend
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements-dev.txt
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
# FERNET_KEY (a mesma do ambiente atual — ver ADR 0003; instalação nova: Fernet.generate_key())
```

`.env` é git-ignored. **Nunca** commitar; **nunca** imprimir `FERNET_KEY` em log ou saída de teste.

### 5. Migrations + superusuário

```bash
.venv/bin/python manage.py migrate
.venv/bin/python manage.py createsuperuser
```

---

## Rodar (desenvolvimento no host)

```bash
.venv/bin/python manage.py runserver 0.0.0.0:8000   # Admin em /admin/, Swagger em /api/v1/docs
.venv/bin/python manage.py check
.venv/bin/pytest -q                                 # 385 testes (+ 8 integration com -m integration)
.venv/bin/python manage.py backfill_market_data      # bootstrap: semeia MarketSeries dos CSVs do repo (uma vez)
.venv/bin/python manage.py ingest_market_data        # ingestão ao vivo (incremental; no host B3 lê CSV estático)
.venv/bin/python manage.py shell -c "from engine import dataset, report_legacy; ..."  # export CLI dos gráficos
```

---

## Containers (app rodando no LAB)

O `docker-compose.yml` na **raiz do repositório** sobe os três containers do app (`lab-appinvest-web`, `lab-appinvest-backend`, `lab-appinvest-ingestion`) na rede externa `lab-net`, compartilhando `lab-postgres` (database `appinvest`), `lab-redis` (DB `1`) e `backend/.env` com o ambiente de desenvolvimento ([ADR 0016](../docs/adr/0016-empacotamento-no-lab.md)).

### Subir / reconstruir e acessar

```bash
# A partir da raiz do repositório (/data/projetos/app-investimentos):
docker compose up -d --build
docker compose ps
```

- **Acesso (VS Code Port Forwarding):** encaminhe a porta **`8080`** na aba *Ports* do VS Code e abra `http://localhost:8080` (SPA em `/`, Django Admin em `/admin/`, API em `/api/v1/`). Nenhuma porta do app é exposta em `0.0.0.0`.
- **Criar superusuário (ou usar o já existente no banco `appinvest`):**
  ```bash
  docker compose exec backend python manage.py createsuperuser
  ```

### Carga inicial de mercado e ingestão manual

No container `ingestion`, `B3_INGEST_USE_SELENIUM=1` está ativo com Chromium + `chromium-driver` nativos do Debian ARM64 e agendamento via `supercronic` (`0 6,19 * * 1-5`, horário de São Paulo):

```bash
# 1. Carga inicial dos CSVs versionados (idempotente):
docker compose exec ingestion python manage.py backfill_market_data

# 2. Ingestão incremental ao vivo (BCB, PTAX, Yahoo e B3 via Chromium):
docker compose exec ingestion python manage.py ingest_market_data

# Ingestão filtrada (ex.: apenas índices B3):
docker compose exec ingestion python manage.py ingest_market_data --only b3:IFIX,b3:SMLL,b3:IDIV
```

### Logs

```bash
docker compose logs -f
docker compose logs -f backend ingestion
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
│   ├── migration_ir/       planejador de migração com isenção de IR (Fase 3b, ADR 0015)
│   │   ├── wallet.py        /resumo -> Holding (puro; price_fx/pm_fx, agrega corretoras)
│   │   ├── tax.py           tabela de classes, limite 20k, IR estimado, DISCLAIMER (puro)
│   │   ├── planner.py       validate_plan_input + build_plan (guloso no teto + rebalanceamento) (puro)
│   │   ├── prices.py        cotação dos alvos: price_fx da carteira -> apps.market (assets.lookup)
│   │   ├── schemas.py       schemas Ninja (PlanRequest, WalletOut, PlanOut)
│   │   ├── pdf.py           render_plan_pdf (fpdf2)
│   │   └── api.py           /migration/wallet, /migration/plan, /migration/plan.pdf
│   └── market/             serviços de mercado (Fases 2 + 3, ADR 0014)
│       ├── services/sources/ clientes de mercado (BCB, PTAX, Yahoo, Tesouro, B3, DLP)
│       ├── data/ativos.csv  universo empacotado de tickers (~7.500 linhas)
│       ├── assets.py        resolução de tickers Yahoo + variante BRL + search() + lookup(sigla)
│       ├── repository.py    acesso exclusivo a core.MarketSeries (Postgres)
│       ├── cache.py         envelope pandas sobre Redis (django-redis) + single_flight
│       ├── synthetics.py    contrato estruturado SyntheticSpec + display_name + parser legado
│       ├── transforms.py    derivações puras (accumulate, splice, to_brl, composite)
│       ├── catalog.py       definição e resolução de séries (SeriesDef)
│       ├── series.py        fachada de leitura (Redis -> Repo + Transforms + fetch-on-read -> Redis)
│       └── management/commands/ backfill_market_data, ingest_market_data
├── engine/                 núcleo de cálculo puro (Fases 2.5 + 7a, ADR 0012, ADR 0018)
│   ├── flows.py             fluxo de caixa líquido com lucro de trades e proventos
│   ├── twr.py               cálculo TWR base 100 (HPR aporte-início / resgate-fim)
│   ├── irr.py               cálculo TIR / XIRR Actual/365 invariante com fallback bisseção
│   ├── metrics.py           drawdown, vol móvel, Sharpe contra CDI do período
│   ├── compute.py           7 funções puras extraídas dos plot_* de financial_report
│   ├── dataset.py           fetch_user_portfolio + build_dataset (corte end_date, gap)
│   └── report_legacy.py     export CLI dos gráficos históricos (Matplotlib)
├── scripts/
│   ├── bootstrap_db.sh      cria role/database appinvest (idempotente)  — versionado
│   └── rotate_db_password.sh rotaciona a senha: gera + aplica no PG + espelha no .env
├── deploy/                 entrypoint-web.sh (migrate + collectstatic + gunicorn) e crontab (supercronic)
├── Dockerfile              multi-target: backend (Gunicorn) e ingestion (Chromium ARM64 + supercronic)
├── requirements.txt        pins com ~= (runtime — usado nos containers)
├── requirements-dev.txt    -r requirements.txt + pytest*, model-bakery, matplotlib (host)
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
