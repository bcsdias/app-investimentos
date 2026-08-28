# Backend — `app-investimentos`

Backend Django da migração Streamlit → Django + React (ver [`../docs/consolidacao_arquitetura_e_migracao.md`](../docs/consolidacao_arquitetura_e_migracao.md)).
Nesta fase roda **no host** (Python 3.12); conteinerização é a Fase 5.

**Estado:** Fases 0 + 1 concluídas — infra LAB + esqueleto Django com `User` customizado,
wrapper Fernet, modelos `UserToken` / `MarketSeries`, Django Admin, suíte `pytest` verde.
O `src/` (Streamlit legado) continua funcionando em paralelo até a Fase 6.

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
.venv/bin/python manage.py runserver 0.0.0.0:8000   # Admin em /admin/
.venv/bin/python manage.py check
.venv/bin/pytest -q                                 # 15 testes (o smoke de cache pode 'skip' sem Redis)
```

---

## Estrutura

```
backend/
├── config/                 projeto Django
│   ├── settings.py          DB appinvest, cache Redis, Fernet, logging stdout, TZ America/Sao_Paulo
│   ├── urls.py              /admin/  (+ stub /api/ para a Fase 3)
│   └── wsgi.py / asgi.py
├── apps/
│   ├── accounts/           User(AbstractUser) — AUTH_USER_MODEL (ADR 0002)
│   └── core/
│       ├── security.py      wrapper Fernet: encrypt() / decrypt()  (ADR 0003)
│       ├── models.py        UserToken (OneToOne, cifrado em repouso) + MarketSeries (ADR 0007)
│       └── admin.py         UserToken sem "add", encrypted_token read-only
├── scripts/
│   ├── bootstrap_db.sh      cria role/database appinvest (idempotente)  — versionado
│   └── rotate_db_password.sh rotaciona a senha: gera + aplica no PG + espelha no .env
├── requirements.txt        pins com ~= (runtime + dev/test)
└── pytest.ini              DJANGO_SETTINGS_MODULE=config.settings, testpaths=apps, --reuse-db
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
| `REDIS_URL` | `redis://localhost:6379/1` | não | cache (`django-redis`) |
| `FERNET_KEY` | *(base64, 44 chars)* | sim | cripto dos tokens DLP — **mesma chave do app legado** |
