# Fundação Backend (Fases 0 + 1) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Levantar a infraestrutura local (Postgres/Redis compartilhados no Docker LAB) e um esqueleto Django funcional em `backend/`, com modelo de usuário customizado, criptografia de token DLP validada por teste, modelos `UserToken`/`MarketSeries`, Django Admin e suíte `pytest-django` verde.

**Architecture:** Serviços LAB compartilhados — um `lab-postgres` e um `lab-redis` únicos servem o LAB inteiro, ligados por uma rede Docker externa `lab-net`; o `app-investimentos` recebe um database dedicado `appinvest`. O backend Django vive em `backend/` na raiz do repo, ao lado do `src/` (Streamlit legado, que permanece até a Fase 6). Nesta fase o Django roda **no host** (Python 3.12); conteinerização é Fase 5. `settings.py` já é escrito para os dois modos, trocando apenas variáveis de ambiente.

**Tech Stack:** Django 5.2 LTS, django-ninja 1.4 (stub apenas), psycopg 3, django-redis, python-dotenv, cryptography (Fernet), pytest-django, PostgreSQL 16, Redis 7, Docker Compose.

**Spec:** `docs/superpowers/specs/2026-08-27-fundacao-backend-fases-0-1-design.md`

## Global Constraints

Todo task herda implicitamente esta seção.

- **Branch:** trabalhar na `lab-dev` (branch dedicada da migração). Não commitar em `main`/`dev`.
- **Python:** 3.12 (host). Ambiente virtual em `backend/.venv/`.
- **Pins de dependência (runtime):** `django~=5.2`, `django-ninja~=1.4`, `psycopg[binary]~=3.2`, `django-redis~=5.4`, `redis~=5.2`, `cryptography~=43.0`, `python-dotenv~=1.0`, `gunicorn~=23.0`.
- **Pins de dependência (dev/test):** `pytest~=8.3`, `pytest-django~=4.9`, `pytest-cov~=5.0`, `model-bakery~=1.19`.
- **Banco:** database `appinvest`, role `appinvest` (com `CREATEDB`). O app **nunca** usa o superusuário `postgres`. As tabelas do app **não** podem cair no `banco_lab`.
- **`AUTH_USER_MODEL = "accounts.User"`** — definido antes do primeiro `migrate`.
- **Chave Fernet:** reusar o valor de `[security].fernet_key` em `/data/projetos/app-investimentos/.streamlit/secrets.toml`. **Nunca** commitar esse valor, **nunca** imprimi-lo em log ou saída de teste. Ele vai só para `backend/.env` (git-ignored).
- **Nomes:** containers mantêm prefixo `lab-*` (infra do LAB); database e código usam `appinvest`.
- **Sem `django-cors-headers`** (SPA + API na mesma origem via nginx, Fase 5).
- **Logging:** somente `StreamHandler` (stdout). Nada de `RotatingFileHandler`.
- **Timezone:** `TIME_ZONE = "America/Sao_Paulo"`, `USE_TZ = True`.
- **Commits:** Conventional Commits + Gitmoji + descrição no imperativo em português (padrão de `.agent/skills/git-commit-formatter.md`). **Nunca** adicionar trailer `Co-Authored-By`. Identidade git local do repo já configurada como `Bruno Dias <bcsdias@gmail.com>`.
- **Compose files do LAB** (`/data/projetos/lab-postgres/`, `/data/projetos/lab-redis/`) ficam **fora** do repo do app (aqueles diretórios não são repositórios git). Só `backend/scripts/bootstrap_db.sh` é versionado.

---

## Estrutura de arquivos

**Fase 0 (infra — fora do repo git, exceto o script):**
- `/data/projetos/lab-redis/docker-compose.yml` — CRIAR. Serviço `redis:7-alpine` como `lab-redis`.
- `/data/projetos/lab-postgres/docker-compose.yml` — MODIFICAR. Adicionar healthcheck + `lab-net`.
- `/data/projetos/app-investimentos/backend/scripts/bootstrap_db.sh` — CRIAR (versionado). Cria role+database `appinvest` idempotentemente.
- Rede Docker `lab-net` — CRIAR via CLI.
- `/data/projetos/lab-nginx/` — REMOVER.

**Fase 1 (Django — tudo em `backend/`, versionado):**
- `backend/requirements.txt` — dependências pinadas.
- `backend/.env.example` — chaves sem valores (versionado).
- `backend/.env` — valores reais (git-ignored, criado na verificação).
- `backend/pytest.ini` — config do pytest-django.
- `backend/manage.py` — entrypoint padrão do Django.
- `backend/config/{__init__,settings,urls,wsgi,asgi}.py` — projeto Django. `settings.py` concentra DB/cache/Fernet/logging; `urls.py` só `/admin/` + stub `/api/`.
- `backend/apps/__init__.py` — pacote container dos apps.
- `backend/apps/accounts/{__init__,apps,models,admin}.py` + `migrations/__init__.py` — app do usuário. `models.py` = `User(AbstractUser)`.
- `backend/apps/accounts/tests/{__init__,test_user}.py` — testes do modelo de usuário.
- `backend/apps/core/{__init__,apps,security,models,admin}.py` + `migrations/__init__.py` — app de domínio. `security.py` = wrapper Fernet; `models.py` = `UserToken` + `MarketSeries`.
- `backend/apps/core/tests/{__init__,test_security,test_models,test_smoke}.py` — testes de criptografia, modelos e smoke de infra.
- `.gitignore` (raiz) — MODIFICAR: acrescentar `.pytest_cache/`, `backend/staticfiles/`, `.coverage`.

---

## Task 1: Fase 0 — Infraestrutura local

**Files:**
- Create: `/data/projetos/lab-redis/docker-compose.yml`
- Modify: `/data/projetos/lab-postgres/docker-compose.yml`
- Create: `/data/projetos/app-investimentos/backend/scripts/bootstrap_db.sh`
- Delete: `/data/projetos/lab-nginx/` (diretório inteiro)

**Interfaces:**
- Consumes: nada.
- Produces:
  - Rede Docker externa `lab-net`.
  - Container `lab-postgres` (porta host `5432`), *healthy*, na `lab-net`, com database `appinvest` e role `appinvest` (senha = `APPINVEST_DB_PASSWORD`, atributo `CREATEDB`).
  - Container `lab-redis` (porta host `6379`), *healthy*, na `lab-net`.
  - Script versionado `backend/scripts/bootstrap_db.sh` (idempotente).

- [ ] **Step 1: Criar a rede compartilhada `lab-net`**

Run:
```bash
docker network create lab-net 2>/dev/null || echo "lab-net já existe"
docker network inspect lab-net --format '{{.Name}} {{.Driver}}'
```
Expected: imprime `lab-net bridge`.

- [ ] **Step 2: Escrever `/data/projetos/lab-redis/docker-compose.yml`**

```yaml
services:
  redis:
    image: redis:7-alpine
    container_name: lab-redis
    restart: unless-stopped
    command: ["redis-server", "--appendonly", "yes"]
    ports:
      - "6379:6379"
    volumes:
      - ./dados_redis:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5
    deploy:
      resources:
        limits:
          memory: 384M
          cpus: "0.5"
    networks: [lab-net]

networks:
  lab-net:
    external: true
```

- [ ] **Step 3: Substituir `/data/projetos/lab-postgres/docker-compose.yml`**

Conteúdo novo (preserva imagem, `banco_lab`, usuário `postgres`, senha, volume e limites; adiciona healthcheck + `lab-net`):
```yaml
services:
  postgres:
    image: postgres:16-alpine
    container_name: lab-postgres
    restart: unless-stopped
    ports:
      - "5432:5432"
    environment:
      POSTGRES_DB: banco_lab
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: MinhaSenhaForte123!
      PGDATA: /var/lib/postgresql/data/pgdata
    volumes:
      - ./dados_banco:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 3s
      retries: 5
    deploy:
      resources:
        limits:
          memory: 1024M
          cpus: "1.0"
    networks: [lab-net]

networks:
  lab-net:
    external: true
```

- [ ] **Step 4: Subir/recriar os dois serviços e aguardar healthy**

Run:
```bash
docker compose -f /data/projetos/lab-postgres/docker-compose.yml up -d
docker compose -f /data/projetos/lab-redis/docker-compose.yml up -d
sleep 8
docker ps --filter name=lab-postgres --filter name=lab-redis --format 'table {{.Names}}\t{{.Status}}'
```
Expected: ambos `Up ... (healthy)`.

- [ ] **Step 5: Escrever `backend/scripts/bootstrap_db.sh`**

```bash
#!/usr/bin/env bash
# Cria o role e o database 'appinvest' no container lab-postgres (idempotente).
# Requer acesso ao daemon Docker do host.
#
# Uso:
#   APPINVEST_DB_PASSWORD='...' bash backend/scripts/bootstrap_db.sh
set -euo pipefail

CONTAINER="${LAB_POSTGRES_CONTAINER:-lab-postgres}"
SUPERUSER="${POSTGRES_SUPERUSER:-postgres}"
DB_NAME="appinvest"
DB_ROLE="appinvest"
DB_PASS="${APPINVEST_DB_PASSWORD:?defina APPINVEST_DB_PASSWORD}"

run_psql() { docker exec -i "$CONTAINER" psql -v ON_ERROR_STOP=1 -U "$SUPERUSER" "$@"; }

echo ">> Garantindo role '$DB_ROLE'..."
if [ "$(run_psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='$DB_ROLE'")" = "1" ]; then
  run_psql -c "ALTER ROLE $DB_ROLE LOGIN CREATEDB PASSWORD '$DB_PASS';"
  echo "   role já existia; atributos/senha atualizados."
else
  run_psql -c "CREATE ROLE $DB_ROLE LOGIN CREATEDB PASSWORD '$DB_PASS';"
  echo "   role criado."
fi

echo ">> Garantindo database '$DB_NAME'..."
if [ "$(run_psql -tAc "SELECT 1 FROM pg_database WHERE datname='$DB_NAME'")" = "1" ]; then
  echo "   database já existia."
else
  run_psql -c "CREATE DATABASE $DB_NAME OWNER $DB_ROLE;"
  echo "   database criado."
fi

run_psql -c "GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_ROLE;"
echo ">> OK: $DB_ROLE@$DB_NAME pronto."
```

Run:
```bash
chmod +x /data/projetos/app-investimentos/backend/scripts/bootstrap_db.sh
```

- [ ] **Step 6: Rodar o bootstrap e validar o database**

Run (escolha uma senha e guarde — vai para `backend/.env` na Task 5):
```bash
APPINVEST_DB_PASSWORD='troque-esta-senha' bash /data/projetos/app-investimentos/backend/scripts/bootstrap_db.sh
docker exec -i lab-postgres psql -U postgres -tAc "SELECT datname FROM pg_database WHERE datname='appinvest';"
docker exec -i lab-postgres psql -U postgres -tAc "SELECT rolname, rolcreatedb FROM pg_roles WHERE rolname='appinvest';"
```
Expected: primeira query imprime `appinvest`; segunda imprime `appinvest|t`.

- [ ] **Step 7: Rodar o bootstrap DE NOVO (prova de idempotência)**

Run:
```bash
APPINVEST_DB_PASSWORD='troque-esta-senha' bash /data/projetos/app-investimentos/backend/scripts/bootstrap_db.sh
```
Expected: termina com `>> OK: appinvest@appinvest pronto.` e as linhas "já existia" — sem erro.

- [ ] **Step 8: Remover o `lab-nginx` de teste**

Run:
```bash
docker compose -f /data/projetos/lab-nginx/docker-compose.yml down
rm -rf /data/projetos/lab-nginx
docker ps -a --filter name=lab-nginx-teste --format '{{.Names}}'
```
Expected: última linha vazia (container removido); diretório não existe mais.

- [ ] **Step 9: Commit do script versionado**

```bash
cd /data/projetos/app-investimentos
git add backend/scripts/bootstrap_db.sh
git commit -m "🐳 chore(lab): adiciona bootstrap idempotente do database appinvest

Cria role/database 'appinvest' no lab-postgres via docker exec. A infra
compartilhada (rede lab-net, lab-redis novo, healthcheck no lab-postgres,
remoção do lab-nginx de teste) foi aplicada fora do repo, em /data/projetos."
```

---

## Task 2: Esqueleto Django + modelo de usuário customizado

**Files:**
- Create: `backend/requirements.txt`
- Create: `backend/.env.example`
- Create: `backend/pytest.ini`
- Create: `backend/manage.py`
- Create: `backend/config/__init__.py`
- Create: `backend/config/settings.py`
- Create: `backend/config/urls.py`
- Create: `backend/config/wsgi.py`
- Create: `backend/config/asgi.py`
- Create: `backend/apps/__init__.py`
- Create: `backend/apps/accounts/__init__.py`
- Create: `backend/apps/accounts/apps.py`
- Create: `backend/apps/accounts/models.py`
- Create: `backend/apps/accounts/admin.py`
- Create: `backend/apps/accounts/migrations/__init__.py`
- Create: `backend/apps/accounts/tests/__init__.py`
- Create: `backend/apps/accounts/tests/test_user.py`
- Create: `backend/apps/core/tests/__init__.py` *(pacote de testes já criado aqui; usado na Task 3+)*
- Create: `backend/apps/core/tests/test_smoke.py`
- Create: `backend/apps/core/__init__.py`
- Create: `backend/apps/core/apps.py`
- Modify: `.gitignore` (raiz)

**Interfaces:**
- Consumes (da Task 1): database `appinvest` / role `appinvest` em `localhost:5432`; `lab-redis` em `localhost:6379`.
- Produces:
  - Módulo `config.settings` com `DATABASES["default"]["NAME"] == "appinvest"`, `CACHES["default"]` no Redis, `AUTH_USER_MODEL == "accounts.User"`, `settings.FERNET_KEY` (str).
  - `apps.accounts.models.User` (subclasse de `AbstractUser`, sem campos extras).
  - `manage.py` operante (`check`, `migrate`, `createsuperuser`, `runserver`, `shell`).
  - `pytest` executável a partir de `backend/` com `pytest-django`.
  - Pacote `apps.core` (vazio de modelos ainda) e `apps.core.tests`.

- [ ] **Step 1: Criar o virtualenv e o `requirements.txt`**

`backend/requirements.txt`:
```
# --- runtime ---
django~=5.2
django-ninja~=1.4
psycopg[binary]~=3.2
django-redis~=5.4
redis~=5.2
cryptography~=43.0
python-dotenv~=1.0
gunicorn~=23.0

# --- dev / test ---
pytest~=8.3
pytest-django~=4.9
pytest-cov~=5.0
model-bakery~=1.19
```

Run:
```bash
cd /data/projetos/app-investimentos/backend
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/python -c "import django; print('Django', django.get_version())"
```
Expected: imprime `Django 5.2.x`.

- [ ] **Step 2: Criar os arquivos do projeto Django**

`backend/manage.py`:
```python
#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys


def main():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and available "
            "on your PYTHONPATH? Did you forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
```

`backend/config/__init__.py`: arquivo vazio.

`backend/config/settings.py`:
```python
"""Django settings — app-investimentos backend (Fases 0+1)."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _env(key: str, default=None, *, required: bool = False) -> str:
    val = os.getenv(key, default)
    if required and not val:
        raise RuntimeError(f"Variável de ambiente obrigatória ausente: {key}")
    return val


SECRET_KEY = _env("DJANGO_SECRET_KEY", required=True)
DEBUG = _env("DJANGO_DEBUG", "0") == "1"
ALLOWED_HOSTS = [
    h.strip()
    for h in _env("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
    if h.strip()
]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "apps.accounts",
    "apps.core",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": "appinvest",
        "USER": "appinvest",
        "PASSWORD": _env("APPINVEST_DB_PASSWORD", required=True),
        "HOST": _env("POSTGRES_HOST", "localhost"),
        "PORT": _env("POSTGRES_PORT", "5432"),
    }
}

CACHES = {
    "default": {
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": _env("REDIS_URL", "redis://localhost:6379/1"),
        "OPTIONS": {"CLIENT_CLASS": "django_redis.client.DefaultClient"},
    }
}

AUTH_USER_MODEL = "accounts.User"

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "pt-br"
TIME_ZONE = "America/Sao_Paulo"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

FERNET_KEY = _env("FERNET_KEY", required=True)

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {"format": "{asctime} {levelname} {name} {message}", "style": "{"},
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler", "formatter": "verbose"},
    },
    "root": {"handlers": ["console"], "level": _env("LOG_LEVEL", "INFO")},
}
```

`backend/config/urls.py`:
```python
from django.contrib import admin
from django.urls import path

urlpatterns = [
    path("admin/", admin.site.urls),
    # path("api/v1/", api.urls)  # Django Ninja — adicionado na Fase 3
]
```

`backend/config/wsgi.py`:
```python
import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
application = get_wsgi_application()
```

`backend/config/asgi.py`:
```python
import os

from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
application = get_asgi_application()
```

- [ ] **Step 3: Criar o app `accounts` com o `User` customizado**

`backend/apps/__init__.py`: vazio.
`backend/apps/accounts/__init__.py`: vazio.
`backend/apps/accounts/migrations/__init__.py`: vazio.
`backend/apps/accounts/tests/__init__.py`: vazio.
`backend/apps/core/__init__.py`: vazio.
`backend/apps/core/tests/__init__.py`: vazio.

`backend/apps/accounts/apps.py`:
```python
from django.apps import AppConfig


class AccountsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.accounts"
```

`backend/apps/accounts/models.py`:
```python
from django.contrib.auth.models import AbstractUser


class User(AbstractUser):
    """Modelo de usuário próprio — sem campos extras ainda, mas trocável desde o dia 1."""

    pass
```

`backend/apps/accounts/admin.py`:
```python
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin

from .models import User

admin.site.register(User, UserAdmin)
```

`backend/apps/core/apps.py`:
```python
from django.apps import AppConfig


class CoreConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.core"
```

- [ ] **Step 4: Criar `pytest.ini` e o `.env.example`**

`backend/pytest.ini`:
```ini
[pytest]
DJANGO_SETTINGS_MODULE = config.settings
python_files = test_*.py
pythonpath = .
testpaths = apps
addopts = --reuse-db
```

`backend/.env.example`:
```
# Django
DJANGO_SECRET_KEY=
DJANGO_DEBUG=1
DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1
LOG_LEVEL=INFO

# PostgreSQL — database dedicado do projeto no lab-postgres
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
APPINVEST_DB_PASSWORD=

# Redis — lab-redis
REDIS_URL=redis://localhost:6379/1

# Fernet — copiar de .streamlit/secrets.toml -> [security].fernet_key
FERNET_KEY=
```

- [ ] **Step 5: Criar o `backend/.env` real (não versionado) para rodar os próximos passos**

Run:
```bash
cd /data/projetos/app-investimentos/backend
cp .env.example .env
SECRET=$(.venv/bin/python -c "from django.core.management.utils import get_random_secret_key as g; print(g())")
FKEY=$(.venv/bin/python -c "import tomllib,pathlib; print(tomllib.loads(pathlib.Path('../.streamlit/secrets.toml').read_text())['security']['fernet_key'])")
python3 - "$SECRET" "$FKEY" <<'PY'
import sys, pathlib
secret, fkey = sys.argv[1], sys.argv[2]
p = pathlib.Path(".env")
txt = p.read_text()
txt = txt.replace("DJANGO_SECRET_KEY=", f"DJANGO_SECRET_KEY={secret}")
txt = txt.replace("APPINVEST_DB_PASSWORD=", "APPINVEST_DB_PASSWORD=troque-esta-senha")
txt = txt.replace("FERNET_KEY=", f"FERNET_KEY={fkey}")
p.write_text(txt)
PY
grep -c '=$' .env
```
Expected: `grep -c '=$'` imprime `0` (nenhuma chave ficou sem valor). Use a MESMA senha do `APPINVEST_DB_PASSWORD` da Task 1 Step 6.

> Nota de segurança: `.env` está coberto pelo `.gitignore` da raiz (`.env`). Não exiba o conteúdo de `FERNET_KEY` em nenhuma saída.

- [ ] **Step 6: Escrever o teste de usuário (deve falhar)**

`backend/apps/accounts/tests/test_user.py`:
```python
import pytest
from django.contrib.auth import get_user_model

from apps.accounts.models import User

pytestmark = pytest.mark.django_db


def test_get_user_model_is_custom():
    assert get_user_model() is User


def test_superuser_flags():
    su = User.objects.create_superuser(
        username="root", email="r@example.com", password="pw-abc-12345"
    )
    assert su.is_staff and su.is_superuser
```

- [ ] **Step 7: Escrever o smoke de infra (deve falhar por falta de migração)**

`backend/apps/core/tests/test_smoke.py`:
```python
import pytest
from django.core.cache import cache
from django.db import connection

pytestmark = pytest.mark.django_db


def test_database_is_appinvest():
    with connection.cursor() as cur:
        cur.execute("SELECT current_database()")
        assert "appinvest" in cur.fetchone()[0]


def test_cache_set_get():
    try:
        cache.set("smoke-key", 123, timeout=10)
    except Exception as exc:  # Redis fora do ar
        pytest.skip(f"Redis indisponível: {exc}")
    assert cache.get("smoke-key") == 123
```

- [ ] **Step 8: Rodar os testes e ver falhar**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/pytest -q
```
Expected: FALHA. Erros esperados: `django.db.utils.ProgrammingError` / "relation ... does not exist" (sem migrations) e/ou coleta falhando antes de `migrate`.

- [ ] **Step 9: Gerar e aplicar as migrations**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/python manage.py makemigrations accounts core
.venv/bin/python manage.py migrate
```
Expected: cria `apps/accounts/migrations/0001_initial.py`; `migrate` aplica `contenttypes`, `auth`, `admin`, `sessions`, `accounts`, `core` sem erro. (O app `core` ainda não tem modelos — nenhuma migration de `core` é gerada; tudo bem.)

- [ ] **Step 10: Rodar os testes e ver passar**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/pytest -q
```
Expected: PASSA (`test_user.py` 2 testes, `test_smoke.py` 2 testes — o de cache pode aparecer como `skipped` se o Redis não responder, o que é aceitável).

- [ ] **Step 11: `manage.py check` e confirmação de que nada foi para o `banco_lab`**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/python manage.py check
docker exec -i lab-postgres psql -U postgres -d appinvest  -c "\dt" | grep -E 'accounts_user|django_migrations' && echo "OK: tabelas em appinvest"
docker exec -i lab-postgres psql -U postgres -d banco_lab -c "\dt"
```
Expected: `check` sem erros; `appinvest` contém `accounts_user` e `django_migrations`; `banco_lab` responde `Did not find any relations.` (segue vazio).

- [ ] **Step 12: Ajustar o `.gitignore` da raiz**

Acrescentar ao final de `/data/projetos/app-investimentos/.gitignore`:
```
# Backend Django
.pytest_cache/
backend/staticfiles/
.coverage
```

- [ ] **Step 13: Commit**

```bash
cd /data/projetos/app-investimentos
git add backend/requirements.txt backend/.env.example backend/pytest.ini backend/manage.py \
        backend/config backend/apps .gitignore
git status --porcelain   # conferir que backend/.env NÃO aparece
git commit -m "✨ feat(backend): esqueleto Django + modelo de usuário customizado

Projeto config/ (settings com Postgres appinvest, cache Redis, Fernet, logging
stdout), apps accounts (User(AbstractUser)) e core (vazio). pytest-django
configurado; testes de usuário e smoke de conectividade (DB=appinvest, cache
Redis) passando. Migrations iniciais aplicadas."
```

---

## Task 3: `core/security.py` — wrapper de criptografia Fernet

**Files:**
- Create: `backend/apps/core/security.py`
- Create: `backend/apps/core/tests/test_security.py`

**Interfaces:**
- Consumes: `settings.FERNET_KEY` (str, definido na Task 2).
- Produces:
  - `apps.core.security.encrypt(plaintext: str) -> str`
  - `apps.core.security.decrypt(ciphertext: str) -> str`
  - (`decrypt` levanta `cryptography.fernet.InvalidToken` para entrada inválida ou cifrada com outra chave.)

- [ ] **Step 1: Escrever o teste (deve falhar)**

`backend/apps/core/tests/test_security.py`:
```python
import base64

import pytest
from cryptography.fernet import Fernet, InvalidToken

from apps.core import security


def test_encrypt_decrypt_roundtrip():
    assert security.decrypt(security.encrypt("DLP-abc123")) == "DLP-abc123"


def test_ciphertext_differs_from_plaintext():
    plain = "DLP-abc123"
    assert security.encrypt(plain) != plain


def test_decrypt_rejects_garbage():
    with pytest.raises(InvalidToken):
        security.decrypt("isto-nao-e-um-token-fernet")


def test_decrypt_rejects_token_from_other_key():
    foreign = Fernet(Fernet.generate_key()).encrypt(b"DLP-abc123").decode()
    with pytest.raises(InvalidToken):
        security.decrypt(foreign)


def test_configured_key_is_32_bytes_base64():
    raw = security._cipher()  # noqa: SLF001 — sanity check do wrapper
    assert isinstance(raw, Fernet)
    from django.conf import settings

    key = settings.FERNET_KEY
    key = key.encode() if isinstance(key, str) else key
    assert len(base64.urlsafe_b64decode(key)) == 32
```

- [ ] **Step 2: Rodar o teste e ver falhar**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/pytest apps/core/tests/test_security.py -q
```
Expected: FALHA com `ModuleNotFoundError: No module named 'apps.core.security'`.

- [ ] **Step 3: Implementar `security.py`**

`backend/apps/core/security.py`:
```python
"""Criptografia simétrica (Fernet) para os tokens DLP em repouso.

Porta de src/data/user_store.py::_get_cipher — MESMA chave, para que os tokens
cifrados pelo app Streamlit legado continuem decifráveis após a migração.
"""
from cryptography.fernet import Fernet
from django.conf import settings


def _cipher() -> Fernet:
    key = settings.FERNET_KEY
    return Fernet(key.encode() if isinstance(key, str) else key)


def encrypt(plaintext: str) -> str:
    return _cipher().encrypt(plaintext.encode()).decode()


def decrypt(ciphertext: str) -> str:
    return _cipher().decrypt(ciphertext.encode()).decode()
```

- [ ] **Step 4: Rodar o teste e ver passar**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/pytest apps/core/tests/test_security.py -q
```
Expected: PASSA (5 testes).

- [ ] **Step 5: Commit**

```bash
cd /data/projetos/app-investimentos
git add backend/apps/core/security.py backend/apps/core/tests/test_security.py
git commit -m "✨ feat(backend): wrapper Fernet para tokens DLP (core.security)

encrypt/decrypt sobre settings.FERNET_KEY (mesma chave do app legado, para
manter os tokens do Supabase decifráveis). Testes: round-trip, cifrado != plano,
rejeição de lixo e de token de outra chave, sanidade da chave (32 bytes base64)."
```

---

## Task 4: `core/models.py` — `UserToken` e `MarketSeries`

**Files:**
- Create: `backend/apps/core/models.py`
- Create: `backend/apps/core/tests/test_models.py`
- Create: `backend/apps/core/migrations/0001_initial.py` (via `makemigrations`)
- Create: `backend/apps/core/migrations/__init__.py`

**Interfaces:**
- Consumes: `apps.core.security.encrypt` / `decrypt` (Task 3); `settings.AUTH_USER_MODEL` (Task 2).
- Produces:
  - `apps.core.models.UserToken` — `OneToOneField` `user` (`related_name="dlp_token"`, `on_delete=CASCADE`); campo `encrypted_token: TextField`; `created_at`/`updated_at`. Métodos: `set_token(raw: str) -> None`, `get_token() -> str`.
  - `apps.core.models.MarketSeries` — campos `series_key: CharField(120)`, `source: CharField(16)`, `reference_date: DateField`, `value: DecimalField(20, 8)`, `updated_at`. `UniqueConstraint(["series_key", "reference_date"], name="uniq_series_point")`.

- [ ] **Step 1: Criar `backend/apps/core/migrations/__init__.py`**

Arquivo vazio.

- [ ] **Step 2: Escrever o teste (deve falhar)**

`backend/apps/core/tests/test_models.py`:
```python
import datetime as dt

import pytest
from django.contrib.auth import get_user_model
from django.db import IntegrityError

from apps.core.models import MarketSeries, UserToken

pytestmark = pytest.mark.django_db


def _user(username="alice"):
    return get_user_model().objects.create_user(username=username, password="pw-abc-12345")


def test_usertoken_roundtrip_persisted():
    tok = UserToken(user=_user())
    tok.set_token("DLP-secret-xyz")
    tok.save()
    assert UserToken.objects.get(pk=tok.pk).get_token() == "DLP-secret-xyz"


def test_usertoken_column_is_not_plaintext():
    tok = UserToken(user=_user())
    tok.set_token("DLP-secret-xyz")
    tok.save()
    assert "DLP-secret-xyz" not in UserToken.objects.get(pk=tok.pk).encrypted_token


def test_usertoken_is_one_per_user():
    user = _user()
    UserToken.objects.create(user=user, encrypted_token="x")
    with pytest.raises(IntegrityError):
        UserToken.objects.create(user=user, encrypted_token="y")


def test_usertoken_related_name():
    tok = UserToken(user=_user())
    tok.set_token("DLP-1")
    tok.save()
    assert tok.user.dlp_token.get_token() == "DLP-1"


def test_marketseries_unique_point():
    MarketSeries.objects.create(
        series_key="bcb:12", source="BCB",
        reference_date=dt.date(2024, 1, 2), value="0.00043739",
    )
    with pytest.raises(IntegrityError):
        MarketSeries.objects.create(
            series_key="bcb:12", source="BCB",
            reference_date=dt.date(2024, 1, 2), value="0.00099999",
        )


def test_marketseries_allows_same_key_other_date():
    MarketSeries.objects.create(
        series_key="bcb:12", source="BCB",
        reference_date=dt.date(2024, 1, 2), value="0.0004",
    )
    MarketSeries.objects.create(
        series_key="bcb:12", source="BCB",
        reference_date=dt.date(2024, 1, 3), value="0.0004",
    )
    assert MarketSeries.objects.filter(series_key="bcb:12").count() == 2
```

- [ ] **Step 3: Rodar o teste e ver falhar**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/pytest apps/core/tests/test_models.py -q
```
Expected: FALHA com `ImportError: cannot import name 'MarketSeries' from 'apps.core.models'` (ou módulo inexistente).

- [ ] **Step 4: Implementar `models.py`**

`backend/apps/core/models.py`:
```python
from django.conf import settings
from django.db import models

from apps.core import security


class UserToken(models.Model):
    """Token da API DLP de um usuário, cifrado em repouso com Fernet."""

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="dlp_token",
    )
    encrypted_token = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def set_token(self, raw: str) -> None:
        self.encrypted_token = security.encrypt(raw)

    def get_token(self) -> str:
        return security.decrypt(self.encrypted_token)

    def __str__(self) -> str:
        return f"UserToken<{self.user}>"


class MarketSeries(models.Model):
    """Um ponto diário de uma série de mercado (benchmark, índice, taxa)."""

    series_key = models.CharField(max_length=120, db_index=True)  # ex.: "bcb:12", "yf:^BVSP"
    source = models.CharField(max_length=16)  # BCB | YF | TD | B3 | PTAX
    reference_date = models.DateField()
    value = models.DecimalField(max_digits=20, decimal_places=8)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["series_key", "reference_date"], name="uniq_series_point"
            )
        ]
        indexes = [models.Index(fields=["series_key", "reference_date"])]
        ordering = ["series_key", "reference_date"]

    def __str__(self) -> str:
        return f"{self.series_key}@{self.reference_date}={self.value}"
```

- [ ] **Step 5: Gerar e aplicar a migration**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/python manage.py makemigrations core
.venv/bin/python manage.py migrate
```
Expected: cria `apps/core/migrations/0001_initial.py` com `UserToken` + `MarketSeries` + constraint `uniq_series_point`; `migrate` aplica sem erro.

- [ ] **Step 6: Rodar o teste e ver passar**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/pytest apps/core/tests/test_models.py -q
```
Expected: PASSA (6 testes).

- [ ] **Step 7: Rodar a suíte inteira**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/pytest -q
```
Expected: PASSA (accounts 2 + core security 5 + core models 6 + smoke 2 = 15; cache smoke pode ficar `skipped`).

- [ ] **Step 8: Commit**

```bash
cd /data/projetos/app-investimentos
git add backend/apps/core/models.py backend/apps/core/tests/test_models.py \
        backend/apps/core/migrations
git commit -m "✨ feat(backend): modelos UserToken e MarketSeries

UserToken: OneToOne com o usuário, token DLP cifrado via core.security
(set_token/get_token). MarketSeries: ponto diário de série de mercado com
UniqueConstraint (series_key, reference_date). Migration 0001 aplicada;
testes de round-trip, não-vazamento do plano, unicidade e related_name."
```

---

## Task 5: Django Admin + verificação ponta a ponta

**Files:**
- Create: `backend/apps/core/admin.py`

**Interfaces:**
- Consumes: `apps.core.models.UserToken`, `apps.core.models.MarketSeries` (Task 4); `apps.accounts.models.User` admin (Task 2).
- Produces: painel `/admin` com os três modelos navegáveis; superusuário criado.

- [ ] **Step 1: Implementar `core/admin.py`**

`backend/apps/core/admin.py`:
```python
from django.contrib import admin

from .models import MarketSeries, UserToken


@admin.register(UserToken)
class UserTokenAdmin(admin.ModelAdmin):
    list_display = ("user", "updated_at")
    readonly_fields = ("encrypted_token", "created_at", "updated_at")
    search_fields = ("user__username", "user__email")

    def has_add_permission(self, request):
        # tokens entram pela API (Fase 3), nunca digitados no Admin (evita
        # gravar valor não-cifrado no campo encrypted_token).
        return False


@admin.register(MarketSeries)
class MarketSeriesAdmin(admin.ModelAdmin):
    list_display = ("series_key", "source", "reference_date", "value")
    list_filter = ("source",)
    date_hierarchy = "reference_date"
    search_fields = ("series_key",)
    ordering = ("series_key", "reference_date")
```

- [ ] **Step 2: `manage.py check` + `makemigrations --check`**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/python manage.py check
.venv/bin/python manage.py makemigrations --check --dry-run
```
Expected: `check` sem erros; `makemigrations --check` diz "No changes detected" (exit 0).

- [ ] **Step 3: Criar o superusuário**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/python manage.py createsuperuser --username admin --email admin@example.com
```
Expected: cria o usuário (definir a senha no prompt).

- [ ] **Step 4: Subir o servidor e checar o Admin manualmente**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/python manage.py runserver 0.0.0.0:8000
```
Verificação manual (outro terminal ou navegador):
- `http://localhost:8000/admin/` → login com o superusuário.
- Ver as seções **ACCOUNTS › Users**, **CORE › User tokens**, **CORE › Market series**.
- Em *User tokens*, confirmar que **não há botão "Add"** e que `encrypted_token` é somente leitura.
- Encerrar o servidor (`Ctrl+C`).

- [ ] **Step 5: Round-trip de token via shell**

Run:
```bash
cd /data/projetos/app-investimentos/backend
.venv/bin/python manage.py shell -c "
from django.contrib.auth import get_user_model
from apps.core.models import UserToken
U = get_user_model()
u, _ = U.objects.get_or_create(username='rt-check')
t, _ = UserToken.objects.get_or_create(user=u)
t.set_token('DLP-roundtrip-999'); t.save()
assert UserToken.objects.get(user__username='rt-check').get_token() == 'DLP-roundtrip-999'
print('round-trip OK')
u.delete()
"
```
Expected: imprime `round-trip OK`.

- [ ] **Step 6: Verificação completa da Definition of Done (spec §2)**

Run:
```bash
cd /data/projetos/app-investimentos/backend
docker network inspect lab-net --format '{{.Name}}'                       # lab-net
docker inspect --format '{{.State.Health.Status}}' lab-postgres lab-redis # healthy / healthy
.venv/bin/python manage.py check                                          # sem erros
.venv/bin/pytest -q                                                       # verde
docker exec -i lab-postgres psql -U postgres -d appinvest -c "\dt" | grep -E 'core_usertoken|core_marketseries'
docker exec -i lab-postgres psql -U postgres -d banco_lab -c "\dt"        # segue vazio
```
Expected: `lab-net` existe; ambos containers `healthy`; `check` limpo; `pytest` verde; `appinvest` tem `core_usertoken` e `core_marketseries`; `banco_lab` sem relations.

- [ ] **Step 7: Commit**

```bash
cd /data/projetos/app-investimentos
git add backend/apps/core/admin.py
git commit -m "✨ feat(backend): registra UserToken e MarketSeries no Django Admin

UserTokenAdmin sem 'add' e com encrypted_token somente-leitura (token nunca
é digitado no Admin nem exibido decifrado). MarketSeriesAdmin com filtro por
source e navegação por data. Fecha as Fases 0+1: infra LAB + esqueleto Django
verificados ponta a ponta."
```

---

## Self-Review

**1. Cobertura do spec**

| Item do spec | Task |
|---|---|
| §3.1 rede `lab-net` | Task 1 Step 1 |
| §3.2 `lab-postgres` healthcheck + `lab-net` | Task 1 Step 3 |
| §3.3 `lab-redis` novo | Task 1 Step 2 |
| §3.4 remover `lab-nginx` | Task 1 Step 8 |
| §3.5 database/role `appinvest` + `bootstrap_db.sh` idempotente + `CREATEDB` | Task 1 Steps 5–7 |
| §3.6 validação Fase 0 | Task 1 Step 4/6 + Task 5 Step 6 |
| §4.2 estrutura de `backend/` | Task 2 Steps 2–4 |
| §4.3 `requirements.txt` pinado | Task 2 Step 1 |
| §4.4 `settings.py` (DB appinvest, cache Redis, `AUTH_USER_MODEL`, `FERNET_KEY`, logging stdout, TZ, sem CORS) | Task 2 Step 2 |
| §4.5 `urls.py` só `/admin/` + stub `/api/` | Task 2 Step 2 |
| §4.6 `User(AbstractUser)` | Task 2 Step 3 |
| §4.7 `core/security.py` | Task 3 |
| §4.8 `UserToken` + `MarketSeries` | Task 4 |
| §4.9 Admin (sem exibir token decifrado) | Task 5 Step 1 |
| §4.10 migrations + superusuário | Task 2 Step 9, Task 4 Step 5, Task 5 Step 3 |
| §5 `pytest.ini` + casos (security, models, user, smoke) | Task 2 Steps 4/6/7, Task 3 Step 1, Task 4 Step 2 |
| §6 decisões travadas | refletidas em Global Constraints |
| §8 verificação ponta a ponta | Task 5 Step 6 |
| Reuso da chave Fernet do `secrets.toml` | Task 2 Step 5 |
| Ajuste `.gitignore` | Task 2 Step 12 |

Sem lacunas.

**2. Placeholders**

Nenhum "TBD/TODO/depois". Todo passo de código traz o conteúdo completo. O único ponto deliberadamente aberto — formato exato de `series_key`/`source` — está no spec §4.8 como refinamento da Fase 2 e não afeta nenhum passo aqui (os testes usam valores concretos `"bcb:12"`/`"BCB"`).

**3. Consistência de tipos**

- `security.encrypt(str) -> str` / `security.decrypt(str) -> str` — definidos na Task 3, usados na Task 4 (`set_token`/`get_token`) com a mesma assinatura.
- `UserToken.user` `related_name="dlp_token"` — definido na Task 4 Step 4, exercitado em `test_usertoken_related_name` (Task 4 Step 2) e na Task 5 Step 5 (`user__username` lookup) — consistente.
- `MarketSeries` constraint `name="uniq_series_point"` — nome idêntico no modelo (Task 4 Step 4) e citado na tabela de cobertura.
- `settings.FERNET_KEY` — produzido na Task 2 Step 2, consumido na Task 3 Step 3.
- Nome do database `appinvest` — idêntico em `bootstrap_db.sh` (Task 1), `settings.py` `DATABASES` (Task 2), e todas as queries de verificação.
- `APPINVEST_DB_PASSWORD` — mesma variável na Task 1 Step 6 e no `.env` da Task 2 Step 5 (instrução explícita de usar a mesma senha).

Sem inconsistências.

---

## Execution Handoff

Plano completo e salvo em `docs/superpowers/plans/2026-08-27-fundacao-backend-fases-0-1.md`. Duas opções de execução:

1. **Subagent-Driven (recomendado)** — um subagente novo por task, revisão entre tasks, iteração rápida.
2. **Inline Execution** — executar as tasks nesta sessão com `executing-plans`, em lote com checkpoints de revisão.

Qual abordagem?
