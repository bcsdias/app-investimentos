# Design — Fundação Backend: Fases 0 + 1

> **Data:** 2026-08-27
> **Branch:** `lab-dev`
> **Documento mestre:** [`../../consolidacao_arquitetura_e_migracao.md`](../../consolidacao_arquitetura_e_migracao.md)
> **Escopo deste design:** primeiro incremento da migração — infraestrutura local (Fase 0) e
> esqueleto do backend Django (Fase 1). As demais fases (serviços de mercado, refatoração do
> engine, API, React, dockerização, paridade) terão cada uma seu próprio ciclo design → plano →
> implementação.

---

## 1. Contexto

O `app-investimentos` hoje é um app Streamlit monолítico que depende de serviços cloud pagos
(Supabase para tokens, Upstash Redis para cache) e OIDC do Google. O documento mestre define
a migração para **PostgreSQL local + Redis local + Django 5 + Django Ninja + React SPA**, rodando
no "Docker LAB" (`/data/projetos/`).

Nada dessa arquitetura-alvo existe ainda: não há Django, Docker Compose, `backend/` nem
`frontend/` no repositório. Antes de portar qualquer lógica de negócio é preciso existir uma
base: banco e cache locais de pé, e um projeto Django que conecta neles, com modelo de dados
para tokens e séries de mercado, painel Admin e testes automatizados.

Este design cobre exatamente essa base. Resultado esperado: um backend Django vazio de regras de
negócio, porém plenamente funcional — migrations aplicadas, Admin no ar, criptografia de token
validada por teste — pronto para receber os serviços de mercado na Fase 2.

### Decisão de topologia

Adotada a **Abordagem 1 — serviços LAB compartilhados**, seguindo o padrão já existente em
`/data/projetos/` (uma pasta por serviço). Um único PostgreSQL e um único Redis servem o LAB
inteiro; o `app-investimentos` recebe um *database* dedicado (`appinvest`). Alternativa
descartada: stack auto-contida com Postgres/Redis dentro do compose do próprio app (foge do
padrão do LAB e multiplica consumo de RAM se surgir um segundo projeto).

---

## 2. Escopo e Definition of Done

### Dentro deste design

- **Fase 0 — infraestrutura**
  - Rede Docker externa compartilhada `lab-net`.
  - `lab-redis` novo (`/data/projetos/lab-redis/`).
  - `lab-postgres` padronizado (healthcheck + `lab-net`), sem recriar o volume.
  - Database e role `appinvest` no `lab-postgres`, via script idempotente versionado.
  - Remoção do `lab-nginx` de teste.
- **Fase 1 — esqueleto Django** em `backend/`
  - Projeto `config/` + apps `accounts` (User customizado) e `core`.
  - `settings.py` conectando em `appinvest` e no Redis, lendo segredos de `backend/.env`.
  - Modelos `UserToken` (token DLP criptografado com Fernet) e `MarketSeries`.
  - `core/security.py` — wrapper de cifragem (porta de `src/data/user_store.py`).
  - Django Admin registrando os modelos.
  - Migrations aplicadas; superusuário criado.
  - Scaffold `pytest-django` + testes de criptografia e dos modelos.

### Fora deste design (fases seguintes)

Serviços de mercado / ingestão (Fase 2), refatoração de `financial_report.py` (2.5), endpoints
da API e auth de sessão (3), React SPA (4), conteinerização do backend + `docker-compose.yml` do
app + nginx unificado (5), import das linhas de `user_tokens` do Supabase (fase de migração de
dados), validação de paridade (6).

### Definition of Done

A partir de um checkout limpo da `lab-dev`, seguindo o plano de implementação:

1. `docker network inspect lab-net` existe; `lab-postgres` e `lab-redis` de pé, *healthy*, na
   `lab-net`, acessíveis do host (`pg_isready`, `redis-cli ping`).
2. Database `appinvest` e role `appinvest` (com `CREATEDB`) existem; `psql -h localhost -U appinvest
   -d appinvest` conecta.
3. `cd backend && python manage.py check` sem erros; `migrate` aplicado — confirmado que as
   tabelas estão em `appinvest` e **não** em `banco_lab`.
4. `/admin` acessível em `http://localhost:8000/admin` com login do superusuário; `UserToken` e
   `MarketSeries` aparecem no painel.
5. No shell do Django: criar um `User` + `UserToken`, chamar `set_token()/get_token()` e obter o
   token DLP original de volta, usando a chave Fernet recuperada.
6. `pytest` verde (criptografia, modelos, User customizado, smoke de infra).

---

## 3. Fase 0 — Infraestrutura

### 3.1. Rede compartilhada `lab-net`

```bash
docker network create lab-net        # bridge; idempotência tratada no script (|| true)
```

Todos os serviços do LAB passam a declarar `lab-net` como rede externa. As redes
`*_default` geradas por compose deixam de ser o ponto de contato entre serviços.

### 3.2. `/data/projetos/lab-postgres/docker-compose.yml` — ajuste (não recriar)

Mudanças mínimas, preservando imagem, `POSTGRES_DB=banco_lab`, usuário `postgres`, senha atual,
volume `./dados_banco` e limites de memória:

- adicionar `networks: [lab-net]` no serviço e bloco `networks: { lab-net: { external: true } }`;
- adicionar `healthcheck` (`pg_isready -U postgres`, `interval: 10s`, `retries: 5`);
- manter `ports: ["5432:5432"]` publicado — o `manage.py` roda no host na Fase 1.

O `banco_lab` permanece intocado; `appinvest` é criado ao lado.

### 3.3. `/data/projetos/lab-redis/docker-compose.yml` — novo

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

### 3.4. Remover `lab-nginx` de teste

```bash
docker compose -f /data/projetos/lab-nginx/docker-compose.yml down
rm -rf /data/projetos/lab-nginx
```

Conteúdo atual: um `index.html` de 353 bytes. O `app-investimentos` traz seu próprio nginx na
Fase 5. A rede `lab-nginx_default` some junto.

### 3.5. Database e role `appinvest`

O `pgdata` do `lab-postgres` já está inicializado, então o hook
`/docker-entrypoint-initdb.d/` **não** dispara. Bootstrap via script idempotente **versionado no
repo do app**: `backend/scripts/bootstrap_db.sh`.

Comportamento do script (executa `psql` dentro do container via `docker exec`):

1. cria o role `appinvest` com `LOGIN`, `CREATEDB` (necessário para o banco de teste do
   pytest-django) e senha vinda de `APPINVEST_DB_PASSWORD` — só se o role não existir;
2. cria o database `appinvest` com `OWNER appinvest` — só se não existir
   (`SELECT 1 FROM pg_database WHERE datname='appinvest'` como guarda);
3. `GRANT ALL PRIVILEGES ON DATABASE appinvest TO appinvest`.

O app **nunca** usa o superusuário `postgres`.

### 3.6. Validação da Fase 0

```bash
docker network inspect lab-net
docker exec lab-postgres pg_isready -U postgres
docker exec lab-redis redis-cli ping                 # PONG
psql -h localhost -U appinvest -d appinvest -c '\conninfo'
redis-cli -h localhost ping                          # PONG
```

---

## 4. Fase 1 — Esqueleto Django

### 4.1. Local e runtime

- `backend/` na raiz do repositório, ao lado de `src/` (o Streamlit permanece até a Fase 6).
- Roda no **host**, Python 3.12: `python -m venv backend/.venv && backend/.venv/bin/pip install -r
  backend/requirements.txt`.
- Nenhum container do backend nesta fase (isso é Fase 5). O `settings.py`, porém, já é escrito
  para funcionar nos dois modos, trocando apenas variáveis de ambiente.

### 4.2. Estrutura de arquivos

```
backend/
├── .venv/                     # git-ignored
├── .env                       # git-ignored — segredos reais
├── .env.example               # versionado — chaves sem valores
├── requirements.txt
├── pytest.ini
├── manage.py
├── scripts/
│   └── bootstrap_db.sh
├── config/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
└── apps/
    ├── __init__.py
    ├── accounts/
    │   ├── __init__.py
    │   ├── apps.py
    │   ├── models.py          # User(AbstractUser)
    │   ├── admin.py
    │   ├── migrations/
    │   └── tests/
    │       └── test_user.py
    └── core/
        ├── __init__.py
        ├── apps.py
        ├── models.py          # UserToken, MarketSeries
        ├── security.py        # wrapper Fernet
        ├── admin.py
        ├── migrations/
        └── tests/
            ├── test_security.py
            └── test_models.py
```

### 4.3. `requirements.txt` (pinado — subconjunto da Fase 1)

Runtime:

| Pacote | Pin | Nota |
|---|---|---|
| `django` | `~=5.2` | 5.2 LTS (suporte estendido até 2028) |
| `django-ninja` | `~=1.4` | incluído já; endpoints só na Fase 3, `urls.py` nasce com stub |
| `psycopg[binary]` | `~=3.2` | psycopg 3, suportado nativamente pelo Django 5 |
| `django-redis` | `~=5.4` | backend de cache |
| `redis` | `~=5.2` | cliente TCP (substitui `upstash-redis`) |
| `cryptography` | `~=43.0` | Fernet |
| `python-dotenv` | `~=1.0` | leitura de `backend/.env` |
| `gunicorn` | `~=23.0` | só para já constar; uso na Fase 5 |

Dev:

| Pacote | Pin |
|---|---|
| `pytest` | `~=8.3` |
| `pytest-django` | `~=4.9` |
| `pytest-cov` | `~=5.0` |
| `model-bakery` | `~=1.19` |

`pandas`, `numpy`, `yfinance`, `python-bcb`, `fpdf2` **não** entram agora — chegam nas Fases 2/2.5/3.
Versões exatas de patch são travadas no momento da implementação (o plano confirma via context7).

### 4.4. `config/settings.py` — pontos-chave

- `from dotenv import load_dotenv; load_dotenv(BASE_DIR / ".env")` no topo.
- `SECRET_KEY = os.environ["DJANGO_SECRET_KEY"]` — chave nova, exclusiva do Django, **sem relação
  com a Fernet**.
- `DEBUG = os.getenv("DJANGO_DEBUG", "0") == "1"`.
- `ALLOWED_HOSTS` a partir de env (`localhost,127.0.0.1` em dev).
- `DATABASES["default"]`:
  ```python
  {
    "ENGINE": "django.db.backends.postgresql",
    "NAME": "appinvest",
    "USER": "appinvest",
    "PASSWORD": os.environ["APPINVEST_DB_PASSWORD"],
    "HOST": os.getenv("POSTGRES_HOST", "localhost"),
    "PORT": os.getenv("POSTGRES_PORT", "5432"),
  }
  ```
  No host resolve `localhost`; conteinerizado (Fase 5) vira `lab-postgres` só trocando o env.
- `CACHES["default"]`: `django_redis.cache.RedisCache`, `LOCATION = os.getenv("REDIS_URL",
  "redis://localhost:6379/1")`.
- `AUTH_USER_MODEL = "accounts.User"`.
- `INSTALLED_APPS`: apps padrão do Django + `apps.accounts` + `apps.core`.
- `FERNET_KEY = os.environ["FERNET_KEY"]` — valor copiado de `[security].fernet_key` do
  `.streamlit/secrets.toml` para `backend/.env` (a mesma chave que cifrou os tokens hoje no
  Supabase; preservá-la é o que permite o import futuro sem recifrar).
- `LOGGING`: um handler `console` (`logging.StreamHandler`), nível via `LOG_LEVEL` (default
  `INFO`). Sem `RotatingFileHandler` (contêiner coleta stdout).
- Sem `django-cors-headers` (SPA e API na mesma origem via nginx, Fase 5).
- `TIME_ZONE = "America/Sao_Paulo"`, `USE_TZ = True`.

### 4.5. `config/urls.py`

Apenas `path("admin/", admin.site.urls)`. Bloco `/api/v1/` presente como comentário/stub, a ser
preenchido na Fase 3.

### 4.6. `apps/accounts/models.py`

```python
from django.contrib.auth.models import AbstractUser

class User(AbstractUser):
    pass
```

Modelo de usuário customizado desde o primeiro `migrate` — trocar depois é caro. Sem campos
extras por enquanto. O `accounts/admin.py` registra `User` com o `UserAdmin` padrão.

### 4.7. `apps/core/security.py`

Porta de `src/data/user_store.py::_get_cipher`:

```python
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

### 4.8. `apps/core/models.py`

```python
class UserToken(models.Model):
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="dlp_token"
    )
    encrypted_token = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def set_token(self, raw: str) -> None:
        self.encrypted_token = security.encrypt(raw)

    def get_token(self) -> str:
        return security.decrypt(self.encrypted_token)


class MarketSeries(models.Model):
    series_key = models.CharField(max_length=120, db_index=True)   # ex.: "bcb:12", "yf:^BVSP"
    source = models.CharField(max_length=16)                        # BCB | YF | TD | B3 | PTAX
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
```

O formato exato de `series_key` / `source` será refinado na Fase 2, quando os serviços de mercado
forem portados. Aqui basta para fixar o schema e exercitar os testes.

### 4.9. `apps/core/admin.py`

- `UserTokenAdmin`: `list_display = ("user", "updated_at")`; `encrypted_token`, `created_at`,
  `updated_at` em `readonly_fields`; **nunca** exibe o token decifrado. Opcional: método
  `token_preview` mostrando `****` + últimos 4 caracteres do valor decifrado.
- `MarketSeriesAdmin`: `list_display = ("series_key", "source", "reference_date", "value")`;
  `list_filter = ("source",)`; `date_hierarchy = "reference_date"`; `search_fields = ("series_key",)`.

### 4.10. Migrations

```bash
python manage.py makemigrations accounts core
python manage.py migrate
python manage.py createsuperuser
```

Greenfield + User customizado nascem juntos, sem o problema clássico de trocar `AUTH_USER_MODEL`
com migrations já aplicadas.

---

## 5. Testes (`pytest-django`)

`backend/pytest.ini`:

```ini
[pytest]
DJANGO_SETTINGS_MODULE = config.settings
python_files = test_*.py
addopts = --reuse-db
```

Banco de teste: `appinvest_test`, criado e derrubado automaticamente pelo pytest-django — por
isso o role `appinvest` recebe `CREATEDB` no `bootstrap_db.sh`.

### Casos da Fase 1

**`apps/core/tests/test_security.py`**
- `encrypt` seguido de `decrypt` devolve o texto original;
- texto cifrado ≠ texto plano;
- `decrypt` de um valor corrompido levanta `cryptography.fernet.InvalidToken`;
- sanidade: `settings.FERNET_KEY` tem 44 caracteres base64.

**`apps/core/tests/test_models.py`**
- `UserToken.set_token()` / `get_token()` faz round-trip persistindo no banco;
- a coluna `encrypted_token` gravada não contém o valor plano;
- a `UniqueConstraint` de `MarketSeries` barra `(series_key, reference_date)` duplicado
  (`IntegrityError`);
- o `OneToOneField` impede dois `UserToken` para o mesmo `user`.

**`apps/accounts/tests/test_user.py`**
- `get_user_model()` é `apps.accounts.models.User`;
- superusuário criado tem `is_staff` e `is_superuser`.

**Smoke de infra**
- `manage.py check` sem erros;
- `django.core.cache.cache.set()/get()` contra o Redis funciona (marca `@pytest.mark.django_db`
  não necessária; pode ser `skip` se `REDIS_URL` não responder).

Sem meta numérica de cobertura na Fase 1; a exigência é `security.py` e os dois modelos 100%
exercitados.

---

## 6. Decisões travadas

| Tema | Decisão |
|---|---|
| Topologia de infra | Abordagem 1 — `lab-postgres` / `lab-redis` compartilhados via `lab-net`; database `appinvest` dedicado |
| `lab-nginx` de teste | Remover |
| Mover / reestruturar o repo | Não mover; `backend/` e `frontend/` são aditivos; `src/` fica até a Fase 6 |
| Prefixo `lab-` | Containers mantêm `lab-*` (infra do LAB); database e código usam nome de projeto (`appinvest`). Merge de `lab-dev` → `main` é decisão à parte, não bloqueia |
| Modelo de usuário | `accounts.User(AbstractUser)` customizado desde o dia 1 |
| Driver Postgres | `psycopg[binary]` 3.x |
| `django-ninja` | Já no `requirements.txt` da Fase 1 (stub em `urls.py`) |
| Gerência de env | `python-dotenv` puro |
| Chave Fernet | Reusar `[security].fernet_key` do `secrets.toml` (viabiliza import futuro do Supabase sem recifrar) |
| Multi-usuário | Confirmado — `UserToken` por usuário; `MarketSeries` global |

---

## 7. Riscos e pendências para fases futuras

- **Segredos em disco:** `.env` e `.streamlit/secrets.toml` (com `service_role` do Supabase e
  `client_secret` do Google) estão no host, git-ignored. Mover para um gerenciador de segredos do
  LAB é trabalho de fase posterior.
- **Ordem de subida:** `depends_on` não cruza arquivos compose. Mitigação simples: um
  `lab-up.sh` que sobe `lab-postgres` + `lab-redis` e aguarda healthy antes de qualquer coisa do
  app. Entra na Fase 5, junto do compose do app.
- **Contagem de linhas no Supabase `user_tokens`:** desconhecida. Verificar (com a `service_role`
  key + URL já disponíveis) antes de desenhar a fase de migração de dados.
- **`series_key` / `source`:** formato provisório; a Fase 2 pode exigir migration de ajuste no
  `MarketSeries` quando os serviços reais forem portados.
- **`bootstrap_db.sh` roda via `docker exec`:** assume que o operador tem acesso ao daemon Docker
  do host. Documentar no README do backend.

---

## 8. Verificação ponta a ponta

Sequência completa a rodar ao final da implementação (resumo — o plano detalha cada passo):

```bash
# --- Fase 0 ---
docker network create lab-net || true
docker compose -f /data/projetos/lab-postgres/docker-compose.yml up -d
docker compose -f /data/projetos/lab-redis/docker-compose.yml up -d
bash /data/projetos/app-investimentos/backend/scripts/bootstrap_db.sh
docker compose -f /data/projetos/lab-nginx/docker-compose.yml down && rm -rf /data/projetos/lab-nginx

docker exec lab-postgres pg_isready -U postgres
docker exec lab-redis redis-cli ping
psql -h localhost -U appinvest -d appinvest -c '\conninfo'

# --- Fase 1 ---
cd /data/projetos/app-investimentos/backend
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env    # preencher FERNET_KEY, DJANGO_SECRET_KEY, APPINVEST_DB_PASSWORD
.venv/bin/python manage.py check
.venv/bin/python manage.py migrate
.venv/bin/python manage.py createsuperuser
.venv/bin/pytest -q

# confirmar que as tabelas estão em appinvest, não em banco_lab
psql -h localhost -U appinvest -d appinvest -c '\dt'
psql -h localhost -U postgres  -d banco_lab -c '\dt'   # deve seguir vazio

# Admin e round-trip de token
.venv/bin/python manage.py runserver   # abrir http://localhost:8000/admin
.venv/bin/python manage.py shell -c "
from django.contrib.auth import get_user_model
from apps.core.models import UserToken
u = get_user_model().objects.create_user('teste', password='x')
t = UserToken(user=u); t.set_token('DLP-abc123'); t.save()
assert UserToken.objects.get(pk=t.pk).get_token() == 'DLP-abc123'
print('round-trip OK')
"
```

---

## 9. Próximo passo

Após aprovação deste design: gerar o **plano de implementação** (skill `writing-plans`) das
Fases 0 + 1, com tarefas passo a passo e checkpoints de teste. As fases seguintes entram em
ciclos próprios.
