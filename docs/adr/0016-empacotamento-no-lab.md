# ADR 0016 — Empacotamento do app no LAB (`docker-compose.yml` + Chromium ARM64 + `supercronic`)

- **Status:** Aceita
- **Data:** 2026-10-27
- **Fase:** 5 (Dockerização e unificação no LAB)
- **Spec:** [`../superpowers/specs/2026-10-27-fase-5-dockerizacao-design.md`](../superpowers/specs/2026-10-27-fase-5-dockerizacao-design.md)

## Contexto

Até a Fase 4 o aplicativo rodava apenas como dois processos de desenvolvimento no host (`runserver` em `:8000` e Vite em `:5173`), sem agendamento da ingestão de mercado e sem navegador para o scraper Selenium da B3. A VM do LAB é `aarch64` (ARM64, 2 vCPU, 11 GiB RAM), onde não há pacote oficial do Google Chrome e o `webdriver-manager` baixa `chromedriver` incompatível. Além disso, o LAB é compartilhado com outros projetos na rede `lab-net` ([ADR 0001](0001-infra-lab-compartilhada.md)) e o serviço legado do Supabase usado por `user_store.py` encontra-se instável e sem dados relevantes para migrar.

## Decisão

1. **Dev no host × containers como "o app rodando":** o fluxo de desenvolvimento (`runserver`, Vite, `.venv/bin/pytest`, `npm test`) permanece no host, sem bind mount nem hot reload em container. Os containers representam o app em execução contínua (`restart: unless-stopped`), reconstruídos sob demanda com `docker compose up -d --build`. Host e containers compartilham o mesmo database `appinvest` no `lab-postgres` e o mesmo Redis DB `1` no `lab-redis`, reutilizando `backend/.env` via `env_file` com overrides de `POSTGRES_HOST=lab-postgres`, `REDIS_URL=redis://lab-redis:6379/1`, `DJANGO_DEBUG=0` e `DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1`.
2. **Topologia em 3 serviços (`name: appinvest`):**
   - `lab-appinvest-web` (`frontend/Dockerfile`, multi-stage `node:22-alpine` → `nginx:alpine`): incorpora o build estático da SPA (`dist/`), serve `/static/` do volume compartilhado `staticfiles` e faz proxy reverso de `/api/` e `/admin/` para `http://lab-appinvest-backend:8000` repassando `Host $http_host`. Única porta publicada: **`127.0.0.1:8080:80`** (acesso via port forwarding do VS Code, nunca em `0.0.0.0`).
   - `lab-appinvest-backend` (`backend/Dockerfile`, target `backend`, `python:3.12-slim-bookworm` sem navegador): executa `migrate --noinput`, `collectstatic --noinput` e Gunicorn (`2 workers gthread`, `4 threads`, `--timeout 180`).
   - `lab-appinvest-ingestion` (`backend/Dockerfile`, target `ingestion`): adiciona `chromium` + `chromium-driver` do Debian Bookworm (`arm64`) e `supercronic` (`v0.2.33` pinado por SHA-1).
3. **Requirements separados:** `backend/requirements.txt` contém apenas dependências de runtime; `backend/requirements-dev.txt` inclui `-r requirements.txt` + `pytest*`, `model-bakery` e `matplotlib` (usado só em `report_legacy.py` no host).
4. **Chromium do sistema em ARM64 e 2 semestres na B3:** `settings.CHROME_BINARY` e `settings.CHROMEDRIVER_PATH` (definidos no target `ingestion`) instruem `fetch_b3_selenium` a usar o Chromium/driver do sistema com `--headless=new`, `--no-sandbox` e `--disable-dev-shm-usage`, sem chamar `ChromeDriverManager().install()`. O scraper itera pelos dois semestres (`1º semestre` e `2º semestre` em `select#semester`) para cobrir o ano inteiro.
5. **Fallback B3 que não mascara falha:** quando `B3_INGEST_USE_SELENIUM=1` e o Selenium falha, `ingest_market_data` lê `read_b3_static` (`data/static/*_all.csv`) e grava os pontos para preservar o histórico inicial, mas **continua marcando a série em `failed_keys` e saindo com código `1`**, evitando que um scraper quebrado passe despercebido com dados congelados no último CSV estático.
6. **Agendamento com `supercronic`:** `CMD ["/usr/local/bin/supercronic", "/app/backend/deploy/crontab"]` roda `python manage.py ingest_market_data` às **06:00 e 19:00 de segunda a sexta** (`TZ=America/Sao_Paulo`), logando em stdout ([ADR 0008](0008-logging-stdout-apenas.md)), sem precisar montar `/var/run/docker.sock` (como exigiria o `ofelia`).
7. **Timeout DLP × Gunicorn/nginx:** fixado em **180 s** (`--timeout 180` no Gunicorn e `proxy_read_timeout 180s` / `proxy_send_timeout 180s` em `/api/` no nginx), cobrindo o pior caso do `single_flight` (até 90 s de espera + 60 s da chamada DLP).
8. **Supabase descartado sem migração:** o import de `user_tokens` do Supabase é cancelado. Usuários são criados via `createsuperuser` ou `/admin` e o token DLP é cadastrado pela própria SPA.

## Consequências

### Positivas
- App disponível permanentemente em `127.0.0.1:8080` (sobrevive a reboot da VM) sem expor portas em `0.0.0.0` e sem interferir nas portas de dev (`8000` e `5173`).
- Ingestão de índices B3 (`IFIX`, `SMLL`, `IDIV`) funcional em ARM64 com Chromium nativo do Debian e atualizada até o último dia útil.
- Imagem `backend` enxuta (sem navegador, sem pacotes de teste/matplotlib) e sem segredos embutidos.

### Negativas / débitos
- Como host e containers compartilham o mesmo banco `appinvest` e Redis DB `1`, migrations aplicadas no dev afetam imediatamente os containers (e vice-versa) — aceito para ambiente LAB de um único usuário.
