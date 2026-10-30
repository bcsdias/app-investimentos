# Fase 5 — Dockerização no LAB — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Empacotar o app em três containers (`lab-appinvest-web`, `lab-appinvest-backend`, `lab-appinvest-ingestion`) que são "o app rodando" no LAB, acessado por port forwarding do VS Code, com ingestão de mercado agendada incluindo B3 via Chromium em ARM64.

**Architecture:** O compose fica na raiz do repo e entra na rede externa `lab-net`, usando `lab-postgres`/`lab-redis` existentes. Um `backend/Dockerfile` com dois targets (`backend`: Gunicorn; `ingestion`: + Chromium + supercronic) e um `frontend/Dockerfile` multi-stage (build Vite → nginx, que faz proxy de `/api/` e `/admin/`). O desenvolvimento continua no host; duas mudanças de código pequenas e testadas em `apps/market` habilitam o Chromium do sistema e a semântica de falha do fallback B3.

**Tech Stack:** Docker Compose v5, `python:3.12-slim-bookworm` (arm64), Debian `chromium`/`chromium-driver`, supercronic, Gunicorn `gthread`, `node:22-alpine`, `nginx:alpine`, Django 5.2, pytest.

**Spec:** [`docs/superpowers/specs/2026-10-27-fase-5-dockerizacao-design.md`](../specs/2026-10-27-fase-5-dockerizacao-design.md)

## Global Constraints

- Host é **aarch64** (2 vCPU, 11 GiB): toda imagem base precisa ter `linux/arm64`; nada de Google Chrome nem `ChromeDriverManager` no container.
- Única porta publicada: `127.0.0.1:8080:80` (nginx). Nenhuma porta do app em `0.0.0.0`; 8000 e 5173 são do dev no host.
- Nomes de container: `lab-appinvest-web`, `lab-appinvest-backend`, `lab-appinvest-ingestion`; compose `name: appinvest`; rede externa `lab-net`.
- Segredos só via `env_file: backend/.env` (o mesmo do host) + overrides em `environment:`; `.env` nunca entra em imagem nem em commit.
- Overrides do container: `DJANGO_DEBUG=0`, `DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1`, `POSTGRES_HOST=lab-postgres`, `REDIS_URL=redis://lab-redis:6379/1`.
- Timeouts: Gunicorn `--timeout 180`; nginx `proxy_read_timeout 180s` em `/api/`.
- Gunicorn: 2 workers `gthread`, 4 threads. Limites de memória: backend ~1 GiB, ingestão ~1,5 GiB, web ~128 MiB.
- Agenda da ingestão: 06:00 e 19:00, seg–sex, `America/Sao_Paulo`, via supercronic (não ofelia).
- `restart: unless-stopped` em todos os serviços.
- Não tocar em `backend/engine/` (ADR 0012) nem nos composes de `/data/projetos/lab-*`.
- Fluxo do host intacto: `runserver` + Vite + `.venv/bin/pytest` + `npm test`.
- Commits: gitmoji + português, como o histórico; **sem** linha `Co-Authored-By`.

## Review Focus

1. **VS Code encaminha 8080 para outra porta local** (ex.: 8081) → login ainda funciona, porque `Origin` e `Host` acompanham a porta local e o nginx repassa `$http_host`. Coberto no smoke da Task 6 com `Host`/`Origin` em `localhost:8081`.
2. **Selenium quebrado (site da B3 mudou)** → a série ainda recebe o histórico do CSV, mas o comando sai com exit 1 nomeando `b3:*`. Pinado em `test_ingest_b3_selenium_failure_falls_back_and_marks_failed` (Task 3).
3. **Rota profunda da SPA recarregada** (`/migracao-ir` com F5) → o nginx devolve `index.html`, não 404. Smoke na Task 5.
4. **`lab-postgres` fora do ar na subida** → o backend reinicia em loop com erro claro no log, sem ficar "healthy" falso. Healthcheck da Task 4 bate na API (que toca sessão/DB), verificado na Task 6.
5. **Imagem vazando segredo** → nenhum `.env` dentro das imagens. Verificação explícita na Task 4.

---

## File Structure

| Arquivo | Responsabilidade |
|---|---|
| `backend/requirements.txt` | Só runtime (sai pytest*, model-bakery, matplotlib). |
| `backend/requirements-dev.txt` (novo) | `-r requirements.txt` + pytest, pytest-django, pytest-cov, model-bakery, matplotlib. |
| `backend/config/settings.py` | + `CHROME_BINARY`, `CHROMEDRIVER_PATH` (env, default `None`). |
| `backend/apps/market/services/sources/b3.py` | `fetch_b3_selenium` usa o Chromium do sistema quando configurado. |
| `backend/apps/market/management/commands/ingest_market_data.py` | Fallback Selenium → CSV com falha registrada. |
| `.dockerignore` (raiz, novo) | Contexto de build das imagens Python. |
| `backend/Dockerfile` (novo) | Targets `backend` e `ingestion`. (Um arquivo com dois targets em vez de `Dockerfile` + `Dockerfile.ingestion`, porque a spec §9 deixa a organização ao builder e isso evita duplicar a base.) |
| `backend/deploy/entrypoint-web.sh` (novo) | `migrate` → `collectstatic` → `exec gunicorn`. |
| `backend/deploy/crontab` (novo) | Agenda da ingestão. |
| `frontend/Dockerfile`, `frontend/.dockerignore`, `frontend/nginx.conf` (novos) | Build Vite + nginx. |
| `docker-compose.yml` (raiz, novo) | Os três serviços. |
| `docs/adr/0016-empacotamento-no-lab.md` (novo), `docs/adr/README.md`, `docs/consolidacao_arquitetura_e_migracao.md`, `backend/README.md`, `backend/CLAUDE.md`, `backend/.env.example` | Docs (spec §7). |

---

### Task 1: Separar requirements de runtime e de dev

**Files:**
- Modify: `backend/requirements.txt`
- Create: `backend/requirements-dev.txt`
- Modify: `backend/README.md` (seção "Setup → 1. Ambiente virtual"), `backend/CLAUDE.md` (onde menciona instalar deps)

**Interfaces:**
- Produces: `backend/requirements.txt` instalável sem ferramentas de teste (usado pela Task 4); `requirements-dev.txt` para o host.

- [ ] **Step 1:** Mova `pytest`, `pytest-django`, `pytest-cov`, `model-bakery` e `matplotlib~=3.9` (com o comentário "Fase 2.5: export CLI…") para `requirements-dev.txt`, que começa com `-r requirements.txt`. `selenium` e `webdriver-manager` **ficam** no runtime, porque `b3.py` importa `selenium` no topo.
- [ ] **Step 2:** Troque `pip install -r requirements.txt` por `pip install -r requirements-dev.txt` no README/CLAUDE.md do backend.
- [ ] **Step 3: Verificar**
  Run: `cd backend && .venv/bin/pip install -q -r requirements-dev.txt && .venv/bin/pytest -q`
  Expected: todos verdes (350 passed, integration deselected).
- [ ] **Step 4: Commit** — `🔧 chore(backend): separa requirements de runtime e de desenvolvimento`

---

### Task 2: `fetch_b3_selenium` com Chromium do sistema

**Files:**
- Modify: `backend/config/settings.py` (perto de `B3_INGEST_USE_SELENIUM`)
- Modify: `backend/apps/market/services/sources/b3.py:46-56`
- Modify: `backend/.env.example`
- Test: `backend/apps/market/tests/test_sources_b3.py`

**Interfaces:**
- Produces: settings `CHROME_BINARY: str | None` e `CHROMEDRIVER_PATH: str | None` (env de mesmo nome; vazio → `None`). A assinatura de `fetch_b3_selenium(indice, anos, download_dir=None)` não muda.

- [ ] **Step 1: Escrever os testes que falham**

```python
def test_fetch_b3_selenium_uses_system_chromium_when_configured(monkeypatch, settings):
    settings.CHROME_BINARY = "/usr/bin/chromium"
    settings.CHROMEDRIVER_PATH = "/usr/bin/chromedriver"
    seen = {}

    def fake_chrome(service=None, options=None):
        seen["service"], seen["options"] = service, options
        raise RuntimeError("stop after construction")

    monkeypatch.setattr(b3.webdriver, "Chrome", fake_chrome)
    import webdriver_manager.chrome as wdm
    monkeypatch.setattr(wdm.ChromeDriverManager, "install",
                        lambda self: pytest.fail("não deve baixar driver"))
    with pytest.raises(FetchError):
        b3.fetch_b3_selenium("IFIX", [2024])
    assert seen["service"].path == "/usr/bin/chromedriver"
    assert seen["options"].binary_location == "/usr/bin/chromium"
    args = seen["options"].arguments
    assert "--headless=new" in args and "--no-sandbox" in args and "--disable-dev-shm-usage" in args


def test_fetch_b3_selenium_without_settings_keeps_webdriver_manager(monkeypatch, settings):
    settings.CHROME_BINARY = None
    settings.CHROMEDRIVER_PATH = None
    import webdriver_manager.chrome as wdm
    monkeypatch.setattr(wdm.ChromeDriverManager, "install", lambda self: "/tmp/fake-driver")
    seen = {}

    def fake_chrome(service=None, options=None):
        seen["service"] = service
        raise RuntimeError("stop")

    monkeypatch.setattr(b3.webdriver, "Chrome", fake_chrome)
    with pytest.raises(FetchError):
        b3.fetch_b3_selenium("IFIX", [2024])
    assert seen["service"].path == "/tmp/fake-driver"
```

O segundo teste também corrige a pendência do §2.3 do mestre (o teste antigo passava porque `ChromeDriverManager().install()` falhava antes do mock).

- [ ] **Step 2:** Run `.venv/bin/pytest apps/market/tests/test_sources_b3.py -v`. Expected: FAIL (`AttributeError: Settings has no CHROME_BINARY` / `--no-sandbox` ausente).
- [ ] **Step 3: Implementar.** Em `settings.py`: `CHROME_BINARY = _env("CHROME_BINARY") or None`, idem `CHROMEDRIVER_PATH`, com um comentário de uma linha dizendo que só o container de ingestão (ARM64, Chromium Debian) os define. Em `b3.py`: sempre adicione `--no-sandbox` e `--disable-dev-shm-usage`; se `settings.CHROME_BINARY`, defina `opts.binary_location`; se `settings.CHROMEDRIVER_PATH`, use `Service(executable_path=...)`; senão, mantenha `Service(ChromeDriverManager().install())`, com o import do `webdriver_manager` só nesse ramo. Ajuste a docstring do módulo ("A Fase 5 o liga no container com Chromium do sistema"). Em `.env.example`, inclua `# CHROME_BINARY=` e `# CHROMEDRIVER_PATH=` comentados, com uma linha "(só no container de ingestão)".
- [ ] **Step 4:** Run `.venv/bin/pytest apps/market -q`. Expected: PASS.
- [ ] **Step 5: Commit** — `✨ feat(market): scraper B3 usa Chromium do sistema quando configurado (ARM64)`

---

### Task 3: Fallback Selenium → CSV que não esconde a falha

**Files:**
- Modify: `backend/apps/market/management/commands/ingest_market_data.py`
- Test: `backend/apps/market/tests/test_commands.py`

**Interfaces:**
- Consumes: `b3.fetch_b3_selenium`, `b3.read_b3_static` (assinaturas atuais).
- Produces: `class _PartialFetch(FetchError)` no módulo do comando, com atributo `points: list[RawPoint]`. É levantada por `_fetch` quando o Selenium falha e o CSV foi lido.

Comportamento: com `B3_INGEST_USE_SELENIUM=True`, `_fetch` tenta `fetch_b3_selenium`. Se ele levantar `FetchError`, faz `logger.error`, lê `read_b3_static(ident, start, end)` e levanta `_PartialFetch(msg, points)`. No `handle`, um `except _PartialFetch` **antes** do `except (FetchError, DatabaseError)` grava `exc.points` via `upsert_points` (quando houver) e marca a série em `failed`/`failed_keys`. O exit 1 continua vindo do caminho existente. Atualize a docstring do módulo: a linha "Agendar este comando é da Fase 5 (ofelia/cron)" passa a "agendado via supercronic no container de ingestão (Fase 5)".

- [ ] **Step 1: Escrever os testes que falham** (seguindo o padrão `monkeypatch` + `call_command` do arquivo)

```python
def test_ingest_b3_selenium_failure_falls_back_and_marks_failed(monkeypatch, db, settings):
    settings.B3_INGEST_USE_SELENIUM = True
    monkeypatch.setattr(cmd.b3, "fetch_b3_selenium",
                        lambda *a, **k: (_ for _ in ()).throw(FetchError("site mudou")))
    monkeypatch.setattr(cmd.b3, "read_b3_static",
                        lambda ident, s, e: [RawPoint(dt.date(2024, 6, 3), 3200.0)])
    monkeypatch.setattr(cmd, "_warm_cache", lambda *a, **k: None)
    err = io.StringIO()
    with pytest.raises(SystemExit) as exc:
        call_command("ingest_market_data", "--only", "b3:IFIX", stderr=err)
    assert exc.value.code == 1
    assert "b3:IFIX" in err.getvalue()
    assert repository.last_date("b3:IFIX") == dt.date(2024, 6, 3)  # histórico do CSV gravado


def test_ingest_b3_selenium_success_does_not_read_static(monkeypatch, db, settings):
    settings.B3_INGEST_USE_SELENIUM = True
    monkeypatch.setattr(cmd.b3, "fetch_b3_selenium",
                        lambda *a, **k: [RawPoint(dt.date(2026, 10, 26), 3500.0)])
    monkeypatch.setattr(cmd.b3, "read_b3_static",
                        lambda *a: pytest.fail("não deve ler CSV"))
    monkeypatch.setattr(cmd, "_warm_cache", lambda *a, **k: None)
    call_command("ingest_market_data", "--only", "b3:IFIX")  # sem SystemExit
    assert repository.last_date("b3:IFIX") == dt.date(2026, 10, 26)
```

(Ajuste os imports e o alias `cmd` ao que o arquivo já usa.)

- [ ] **Step 2:** Run `.venv/bin/pytest apps/market/tests/test_commands.py -k b3 -v`. Expected: o primeiro FAIL (não lê o CSV / não grava).
- [ ] **Step 3:** Implemente conforme o bloco de comportamento acima.
- [ ] **Step 4:** Run `.venv/bin/pytest -q`. Expected: tudo verde.
- [ ] **Step 5: Commit** — `🐛 fix(market): falha do Selenium cai no CSV da B3 mas continua marcando a série como falha`

---

### Task 4: Imagens Python (`backend` e `ingestion`)

**Files:**
- Create: `.dockerignore`, `backend/Dockerfile`, `backend/deploy/entrypoint-web.sh`, `backend/deploy/crontab`

**Interfaces:**
- Produces: targets `backend` e `ingestion` (build context = raiz do repo; `-f backend/Dockerfile`). Layout: código em `/app/backend`, `data/static` + `data/raw` em `/app/data`, workdir `/app/backend`, usuário não-root `app` (uid 1000), diretório `/app/backend/staticfiles` criado e com `chown` para `app` (vira volume na Task 6).

Pontos que o Dockerfile precisa fixar:
- Stage `base`: `python:3.12-slim-bookworm`, `PYTHONDONTWRITEBYTECODE=1`, `PYTHONUNBUFFERED=1`, `pip install --no-cache-dir -r backend/requirements.txt`. Copie `backend/` e `data/static`, `data/raw`.
- Target `backend`: `EXPOSE 8000`; `ENTRYPOINT ["/app/backend/deploy/entrypoint-web.sh"]`. O entrypoint (`set -e`) roda `python manage.py migrate --noinput`, `python manage.py collectstatic --noinput` e `exec gunicorn config.wsgi --bind 0.0.0.0:8000 --workers 2 --worker-class gthread --threads 4 --timeout 180 --access-logfile - --error-logfile -`.
- Target `ingestion`: `apt-get install --no-install-recommends chromium chromium-driver tzdata`; supercronic em versão pinada (release `linux-arm64`, binário escolhido por `TARGETARCH`, com checagem de SHA conforme o README do supercronic); `ENV CHROME_BINARY=/usr/bin/chromium CHROMEDRIVER_PATH=/usr/bin/chromedriver B3_INGEST_USE_SELENIUM=1 TZ=America/Sao_Paulo`; `CMD ["supercronic", "/app/backend/deploy/crontab"]`.
- `crontab`: `0 6,19 * * 1-5 cd /app/backend && python manage.py ingest_market_data`.
- `.dockerignore` (raiz): `.git`, `**/.env`, `**/.env.*` (mantendo `!**/.env.example`), `**/.venv`, `**/__pycache__`, `**/.pytest_cache`, `backend/staticfiles`, `frontend`, `src`, `tests`, `.streamlit`, `.gemini`, `.agent`, `.superpowers`, `.impeccable`, `.vscode`.

- [ ] **Step 1:** Escreva os quatro arquivos; `chmod +x backend/deploy/entrypoint-web.sh`.
- [ ] **Step 2: Build**
  Run: `docker build -f backend/Dockerfile --target backend -t appinvest-backend:dev . && docker build -f backend/Dockerfile --target ingestion -t appinvest-ingestion:dev .`
  Expected: as duas imagens são construídas; `docker image inspect appinvest-ingestion:dev --format '{{.Architecture}}'` → `arm64`.
- [ ] **Step 3: Sem segredo e layout correto**
  Run: `docker run --rm --entrypoint sh appinvest-backend:dev -c 'test ! -e /app/backend/.env && ls /app/data/static/IFIX_all.csv && python -c "import matplotlib" 2>&1 | tail -1'`
  Expected: imprime o caminho do CSV e `ModuleNotFoundError: No module named 'matplotlib'`.
- [ ] **Step 4: Chromium funciona no ARM**
  Run: `docker run --rm --entrypoint sh appinvest-ingestion:dev -c 'chromium --version && chromedriver --version && supercronic -test /app/backend/deploy/crontab'`
  Expected: versões impressas; o supercronic valida o crontab sem erro.
- [ ] **Step 5: Commit** — `🐳 build(backend): imagens backend (Gunicorn) e ingestion (Chromium + supercronic) para ARM64`

---

### Task 5: Imagem web (SPA + nginx)

**Files:**
- Create: `frontend/Dockerfile`, `frontend/.dockerignore`, `frontend/nginx.conf`

**Interfaces:**
- Consumes: o backend em `http://lab-appinvest-backend:8000` (nome do container na Task 6).
- Produces: imagem que serve em `:80`; `/static/` vem de `/usr/share/nginx/static` (volume montado na Task 6).

`nginx.conf` (substitui `/etc/nginx/conf.d/default.conf`):
- `location /api/` e `location /admin/` → `proxy_pass http://lab-appinvest-backend:8000;` com `proxy_set_header Host $http_host;`, `X-Forwarded-For $proxy_add_x_forwarded_for`, `X-Forwarded-Proto $scheme`. Só em `/api/`: `proxy_read_timeout 180s; proxy_send_timeout 180s;`.
- `location /static/ { alias /usr/share/nginx/static/; }`
- `location / { try_files $uri $uri/ /index.html; }`
- `client_max_body_size 1m;`
- Use `resolver 127.0.0.11 valid=30s;` com o upstream numa variável, para o nginx subir mesmo que o backend ainda esteja reiniciando (com `proxy_pass` via variável, use `$request_uri` como no padrão do nginx).

Dockerfile: stage `node:22-alpine` → `npm ci` → `npm run build`; stage `nginx:alpine` copia `dist/` para `/usr/share/nginx/html` e o `nginx.conf`. `.dockerignore`: `node_modules`, `dist`, `coverage`.

- [ ] **Step 1:** Escreva os três arquivos.
- [ ] **Step 2: Build e rota da SPA**
  Run: `docker build -t appinvest-web:dev frontend && docker run -d --rm --name appinvest-web-smoke -p 127.0.0.1:18080:80 appinvest-web:dev && sleep 1 && curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:18080/migracao-ir && curl -s http://127.0.0.1:18080/migracao-ir | grep -c '<div id="root">'; docker stop appinvest-web-smoke`
  Expected: `200` e `1` (o deep link `/migracao-ir` devolve `index.html`).
- [ ] **Step 3: Commit** — `🐳 build(frontend): imagem nginx com o build da SPA e proxy para o backend`

---

### Task 6: `docker-compose.yml` e smoke de ponta a ponta

**Files:**
- Create: `docker-compose.yml` (raiz)

**Interfaces:**
- Consumes: targets da Task 4, imagem da Task 5, `backend/.env`.

Conteúdo a fixar:
- `name: appinvest`; `networks: lab-net: external: true`; volume `staticfiles`.
- `backend`: `build: {context: ., dockerfile: backend/Dockerfile, target: backend}`, `container_name: lab-appinvest-backend`, `env_file: backend/.env`, `environment:` com os overrides das Global Constraints, volume `staticfiles:/app/backend/staticfiles`, healthcheck `python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/api/v1/auth/csrf').status==204 else 1)"` (interval 15s, start_period 30s), `mem_limit: 1g`.
- `ingestion`: `target: ingestion`, `container_name: lab-appinvest-ingestion`, mesmo `env_file`/overrides, `shm_size: 1gb`, `mem_limit: 1536m`.
- `web`: `build: frontend`, `container_name: lab-appinvest-web`, `ports: ["127.0.0.1:8080:80"]`, volume `staticfiles:/usr/share/nginx/static:ro`, `depends_on: backend: condition: service_healthy`, healthcheck `wget -qO- http://localhost/ >/dev/null`, `mem_limit: 128m`.
- `restart: unless-stopped` nos três.

- [ ] **Step 1:** Escreva o `docker-compose.yml`; `docker compose config -q` sem erro.
- [ ] **Step 2: Subir**
  Run: `docker compose up -d --build && sleep 45 && docker compose ps --format '{{.Name}} {{.Status}} {{.Ports}}'`
  Expected: os três `Up … (healthy)` (a ingestão não tem healthcheck: `Up`); o web com `127.0.0.1:8080->80/tcp` e nenhum `0.0.0.0`.
- [ ] **Step 3: Login por CSRF na porta 8080 e numa porta local diferente** (Review Focus 1). Crie um usuário temporário e faça o login com Host/Origin `localhost:8080` e depois `localhost:8081`:

```bash
docker compose exec backend python manage.py shell -c "from django.contrib.auth import get_user_model as g; g().objects.create_user('smoke-f5','Smoke-f5-senha!')"
for P in 8080 8081; do
  J=$(mktemp); curl -s -c $J -H "Host: localhost:$P" http://127.0.0.1:8080/api/v1/auth/csrf
  T=$(awk '/csrftoken/{print $7}' $J)
  curl -s -b $J -c $J -H "Host: localhost:$P" -H "Origin: http://localhost:$P" -H "X-CSRFToken: $T" \
    -H 'Content-Type: application/json' -d '{"username":"smoke-f5","password":"Smoke-f5-senha!"}' \
    -o /dev/null -w "$P login %{http_code}\n" http://127.0.0.1:8080/api/v1/auth/login
  curl -s -b $J -H "Host: localhost:$P" -o /dev/null -w "$P me %{http_code}\n" http://127.0.0.1:8080/api/v1/auth/me
done
docker compose exec backend python manage.py shell -c "from django.contrib.auth import get_user_model as g; g().objects.filter(username='smoke-f5').delete()"
```
  Expected: `login 200` e `me 200` nas duas portas.
- [ ] **Step 4: Admin com CSS** — `curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8080/static/admin/css/base.css` → `200`.
- [ ] **Step 5: Postgres indisponível (Review Focus 4)** — `docker stop lab-postgres && docker compose restart backend && sleep 40 && docker compose ps backend` → não `healthy`, e `docker compose logs --tail 20 backend` mostra erro de conexão com o Postgres; depois `docker start lab-postgres` e, em até ~1 min, `healthy` de novo. (Avise o usuário antes: derruba o Postgres do LAB por menos de um minuto.)
- [ ] **Step 6: Restart** — `docker compose restart && sleep 45 && docker compose ps` → tudo `Up`/`healthy`.
- [ ] **Step 7: Dev do host convive** — com os containers de pé, `cd backend && .venv/bin/pytest -q` e `cd frontend && npm test` verdes; `ss -ltn | grep -E ':(8000|5173|8080)'` mostra 8080 só em `127.0.0.1`.
- [ ] **Step 8: `banco_lab` vazio (ADR 0006)** — `docker exec lab-postgres psql -U postgres -d banco_lab -c '\dt'` → `Did not find any relations.`
- [ ] **Step 9: Commit** — `🐳 build: docker-compose do app no LAB (web, backend, ingestion) na lab-net`

---

### Task 7: Carga inicial de mercado e ingestão B3 ao vivo

**Files:** nenhum, a não ser que o scraper falhe (aí só `docs/consolidacao_arquitetura_e_migracao.md`, na Task 8).

- [ ] **Step 1: Backfill idempotente** — `docker compose exec ingestion python manage.py backfill_market_data` → termina sem falha; a contagem total de `MarketSeries` não diminui (anote antes/depois com `docker compose exec backend python manage.py shell -c "from apps.core.models import MarketSeries as M; print(M.objects.count())"`).
- [ ] **Step 2: Ingestão ao vivo** — `docker compose exec ingestion python manage.py ingest_market_data; echo exit=$?`.
- [ ] **Step 3: Conferir** — último ponto por série:
  `docker compose exec backend python manage.py shell -c "from apps.core.models import MarketSeries as M; from django.db.models import Max; [print(r) for r in M.objects.values('series_key').annotate(b=Max('reference_date')).order_by('series_key')]"`
  Expected: `b3:IFIX/SMLL/IDIV` no último dia útil **ou**, se o `exit=1` nomeou `b3:*`, registre a saída de erro para a Task 8 (spec §6: não bloqueia). As séries ANBIMA `bcb:1246x` podem continuar em 10/04/2026 (descontinuadas; emenda com ETF).
- [ ] **Step 4: Agenda ativa** — `docker compose logs ingestion | head` mostra o supercronic carregando o job `0 6,19 * * 1-5`.
- [ ] **Step 5: App de ponta a ponta (manual, com o usuário)** — pedir ao usuário que encaminhe a porta 8080 no VS Code, faça login com a conta real, confira a carteira (8 análises, benchmark IMA-B contínuo depois de abril) e baixe um PDF no planejador. Esse passo não gera commit.

---

### Task 8: Documentação

**Files:**
- Create: `docs/adr/0016-empacotamento-no-lab.md`
- Modify: `docs/adr/README.md`, `docs/consolidacao_arquitetura_e_migracao.md`, `backend/README.md`, `backend/CLAUDE.md`

- [ ] **Step 1: ADR 0016** (formato das ADRs existentes: Status/Data/Fase, Contexto, Decisão, Consequências): dev no host × containers como app rodando; banco/Redis compartilhados com o dev; nomes `lab-appinvest-*`; `127.0.0.1:8080`; Chromium Debian por ser ARM64; supercronic em vez de ofelia; timeouts de 180 s; fallback B3 que marca falha; Supabase descartado sem migração. Acrescente a entrada no índice `docs/adr/README.md`.
- [ ] **Step 2: Doc mestre** — os itens listados na spec §7: §2.0 (linha da Fase 5 + contagem de testes atualizada), §2.2, §2.3 (riscar a pendência do teste Selenium "pelo motivo errado"; tirar "timeout DLP × gunicorn" do backlog; se a Task 7 falhou no scraper, adicionar a pendência com a mensagem de erro), §2.4 (Supabase: descartado), §3.5 (Chromium ARM), §5 Fase 5 item 4, §6.2 (riscar, com o motivo), §7.4 (decidido: supercronic), §9 (marcar Fase 5; trocar o item 10 por "carga inicial de mercado" marcada).
- [ ] **Step 3: `backend/README.md`** — nova seção "Containers (app rodando no LAB)": subir/reconstruir (`docker compose up -d --build`), port forwarding da 8080, `createsuperuser` no container, ingestão manual, carga inicial (backfill → ingest), logs (`docker compose logs -f`). `backend/CLAUDE.md`: uma linha dizendo que o dev continua no host e o compose é só o app rodando.
- [ ] **Step 4: Commit** — `📝 docs: fecha a Fase 5 (ADR 0016, doc mestre, README de containers)`
