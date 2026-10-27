# Design — Fase 5 (Dockerização e unificação no LAB)

> **Data:** 2026-10-27
> **Branch:** `migracao-django`
> **Documento mestre:** [`../../consolidacao_arquitetura_e_migracao.md`](../../consolidacao_arquitetura_e_migracao.md) (§3.3, §3.5, §4, §5 Fase 5, §7.4, §7.5)
> **Fases anteriores:** 0 + 1, 2 + 2.5, 3 + 3b + 3c, 4 — specs em `docs/superpowers/specs/`
> **ADRs relevantes:** [0001](../../adr/0001-infra-lab-compartilhada.md) (infra compartilhada), [0005](../../adr/0005-sem-django-cors-headers.md) (sem CORS), [0006](../../adr/0006-prefixo-lab-containers-appinvest-codigo.md) (prefixo `lab-`), [0008](../../adr/0008-logging-stdout-apenas.md) (logs em stdout)

---

## 1. Intenção e porquê

Hoje o app só existe como dois processos de desenvolvimento no host (`runserver` em
`127.0.0.1:8000` e Vite em `127.0.0.1:5173`), e a ingestão de mercado só roda quando alguém
lembra de rodá-la. A parte de B3 nunca rodou ao vivo, porque o Selenium precisa de um
navegador que o host não tem.

A Fase 5 empacota o app num conjunto de containers que **é "o app rodando" no LAB**:
sobe com um comando, volta sozinho depois de reboot, atualiza os dados de mercado sozinho
duas vezes por dia útil e é acessado pelo usuário via **port forwarding do VS Code**.

Sucesso: o usuário encaminha uma porta no VS Code, abre `http://localhost:<porta>`, faz
login, vê a carteira com benchmarks atualizados (incluindo índices B3 até a data corrente)
e usa o planejador de IR — sem nenhum processo de dev rodando.

## 2. Decisões tomadas com o usuário

| # | Tema | Decisão |
|---|---|---|
| 1 | **Ambiente** | LAB, um usuário (o próprio dono), acesso **somente** via port forwarding do VS Code para a VM. Sem domínio, sem HTTPS, sem exposição pública. |
| 2 | **Dev × containers** | **Desenvolvimento continua no host** (`runserver` + Vite + `.venv` + Vitest, exatamente como hoje). **Os containers são "o app rodando"**, reconstruídos sob demanda (`docker compose up -d --build`). Não há bind mount de código nem hot reload em container. |
| 3 | **Supabase** | **Fora do escopo.** O import de `user_tokens` (§6.2 do mestre, item 10 do checklist §9) é **cancelado**: o Supabase está com problemas e não há dado de usuário relevante lá. Usuários são criados via `createsuperuser`/`/admin`; o token DLP é cadastrado pela SPA. |
| 4 | **Dados de mercado** | Usar o que já está no Postgres local + os CSVs do repo (`data/raw`, `data/static`) e **completar os faltantes** pela ingestão ao vivo do container (ver §6). |

## 3. Não-objetivos e anti-objetivos

**Não-objetivos (ficam para outras fases):**
- Paridade com o Streamlit e remoção de `src/`, `redis.py`, `ativos.csv` duplicado, scratch do Gemini, `requirements.txt` da raiz (Fase 6 — o legado ainda precisa rodar).
- Qualquer mudança em `backend/engine/` ou nos números (Fase 7, [ADR 0012](../../adr/0012-engine-copia-literal-fase-2-5.md)).
- HTTPS, domínio, SSO, exposição pública, CI/CD, registry de imagens.
- Remover o prefixo `lab-` / merge na `main` (§7.3 do mestre).
- Itens do backlog da revisão de 2026-10-27, **exceto** o timeout DLP × Gunicorn, que é decidido aqui (§5.2).
- Containerizar `lab-postgres`/`lab-redis` de novo — eles já existem e continuam fora do repo (ADR 0001).

**Anti-objetivos (se acontecer, a fase falhou mesmo "funcionando"):**
- Qualquer porta do app publicada em `0.0.0.0` (a VM tem IP público; Postgres e Redis já estão em `0.0.0.0` por decisão anterior e **não** são tocados aqui, mas o app não piora isso).
- Segredos (`.env`, `FERNET_KEY`, senha do banco) dentro de uma imagem ou commitados.
- Selenium/Chrome no caminho de request (imagem do backend **sem** navegador — §3.5 do mestre).
- O fluxo de desenvolvimento do host quebrar ou exigir passos novos.
- Tabela do app caindo no `banco_lab` (ADR 0006).

## 4. Restrições do ambiente (levantadas em 2026-10-27)

- **VM Oracle `aarch64` (ARM64)**, 2 vCPU, 11 GiB RAM, 45 GiB livres em `/data`. Docker Compose v5.
  - **Consequência central:** não existe Google Chrome para Linux ARM64, e o
    `webdriver-manager` (`ChromeDriverManager().install()`, usado hoje em
    `backend/apps/market/services/sources/b3.py`) baixa chromedriver x86 — **não funciona**.
    A imagem de ingestão usa **`chromium` + `chromium-driver` do Debian** (há builds arm64) e o
    scraper passa a usar o driver do sistema (§5.3).
  - Todas as imagens base precisam ter variante `linux/arm64` (as oficiais `python`, `node`,
    `nginx` têm).
- `lab-postgres` (`:5432`) e `lab-redis` (`:6379`) rodam na rede externa `lab-net`; o app
  usa o database `appinvest` e o Redis DB `1`.
- Portas livres no host: 80/443/8080 livres; 8000 e 5173 são do dev e **não** podem ser usadas pelos containers.
- `settings.py` já é "dual host/container": tudo vem de env (`POSTGRES_HOST`, `REDIS_URL`,
  `DJANGO_*`, `FERNET_KEY`, `APPINVEST_DB_PASSWORD`, `B3_INGEST_USE_SELENIUM`, `LOG_LEVEL`).
  `load_dotenv` **não sobrescreve** variáveis já presentes no ambiente.
- `b3.py` e `backfill_market_data.py` resolvem `data/static`/`data/raw` como
  `Path(__file__).parents[5]` = **raiz do repo**. No container o layout precisa reproduzir isso
  (ver §5.1).

## 5. Desenho

### 5.1. Topologia

Um `docker-compose.yml` na **raiz do repo**, `name: appinvest`, três serviços, todos na rede
externa `lab-net`:

| Serviço / container | Imagem | Papel | Porta |
|---|---|---|---|
| `lab-appinvest-web` | `frontend/Dockerfile` (multi-stage: `node` build Vite → `nginx:alpine` com o `dist/`) | Serve a SPA e faz proxy de `/api/` e `/admin/` para o backend; serve `/static/` (CSS do admin) | **`127.0.0.1:8080:80`** |
| `lab-appinvest-backend` | `backend/Dockerfile` (`python:3.12-slim`, **sem** navegador) | Gunicorn servindo Django | só rede interna (`8000`) |
| `lab-appinvest-ingestion` | `backend/Dockerfile.ingestion` (base Debian + `chromium` + `chromium-driver` + `supercronic`) | Roda `ingest_market_data` agendado | nenhuma |

Chamada minha — nomes `lab-appinvest-*` em vez de `lab-backend`/`lab-nginx` do desenho
original: o LAB é compartilhado (ADR 0001) e nomes genéricos colidiriam com outro projeto.
Continua respeitando o prefixo `lab-` da ADR 0006.

Chamada minha — **sem container `lab-frontend` separado**: o build do Vite vai direto para a
imagem do nginx (o desenho original tinha "assets" + nginx). Um container a menos, mesmo resultado.

Layout dentro das imagens Python: código em `/app/backend`, e `data/static` + `data/raw` do
repo copiados para `/app/data/` — assim `parents[5]` continua apontando para `/app` e o
código não muda. (`data/raw` e `data/static` são versionados — 166 arquivos, ~17 MB — então
o build context da raiz já os inclui.)

`restart: unless-stopped` em todos. Healthcheck no backend (ex.: `GET /api/v1/auth/csrf`) e no
nginx; o nginx depende do backend saudável. `lab-postgres`/`lab-redis` **não** entram no
compose (são externos); o backend deve falhar com log claro se não alcançá-los.

Recursos (chamada minha, cabe em 2 vCPU / 11 GiB ao lado do resto do LAB): backend com
**2 workers Gunicorn** (`gthread`, 4 threads cada), limite ~1 GiB; ingestão ~1,5 GiB
(Chromium); nginx ~128 MiB.

### 5.2. Backend (`backend/Dockerfile`)

- `python:3.12-slim` arm64, usuário não-root, `requirements` de **runtime** só.
  Chamada minha: dividir `backend/requirements.txt` em `requirements.txt` (runtime) e
  `requirements-dev.txt` (`-r requirements.txt` + pytest, pytest-django, pytest-cov,
  model-bakery). `matplotlib` (só `report_legacy`, nunca importado pela API) vai para o dev.
  `selenium`/`webdriver-manager` ficam no runtime porque `b3.py` importa `selenium` no topo e é
  importado pelo backend. O README/CLAUDE.md do backend passa a mandar instalar o `-dev` no host.
- `.dockerignore` excluindo `.env`, `.venv`, `__pycache__`, `.pytest_cache`, `staticfiles`, `node_modules`.
- Entrypoint: `manage.py migrate --noinput` → `collectstatic --noinput` → `gunicorn config.wsgi`.
  Chamada minha: `migrate` na subida é aceitável aqui porque é um único backend num LAB de um
  usuário; a ingestão **não** roda `migrate`.
- **Timeout DLP × Gunicorn (backlog "decidir na Fase 5"):** a DLP tem `timeout=60` por
  connect/read (não total), e o `single_flight` (`apps/market/cache.py`) espera até 90 s pelo lock
  antes de chamar a DLP ele mesmo — com as 8 análises disparando em paralelo, o pior caso passa de
  90 s. Chamada minha: cobrir o pior caso, sem mexer em `cache.py`: Gunicorn `--timeout 180` e nginx
  `proxy_read_timeout 180s` para `/api/`. Num LAB de um usuário, um worker preso 3 min é aceitável;
  um 504 na primeira análise não é.
- `STATIC_ROOT` é servido pelo nginx: volume nomeado compartilhado (`staticfiles`) entre backend
  (escreve no `collectstatic`) e nginx (lê em `/static/`). Sem `whitenoise`.
- Env do container (via `env_file` + `environment` no compose):
  - `DJANGO_DEBUG=0`, `DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1`
  - `POSTGRES_HOST=lab-postgres`, `REDIS_URL=redis://lab-redis:6379/1`
  - `DJANGO_SECRET_KEY`, `APPINVEST_DB_PASSWORD`, `FERNET_KEY` — **os mesmos** do `backend/.env`
    do host (mesmo banco; a mesma `FERNET_KEY` é obrigatória para o token já salvo continuar
    legível, ADR 0003). Chamada minha: o compose lê `backend/.env` via `env_file` e sobrescreve só
    host/URLs/debug em `environment:`; nenhum arquivo de segredo novo.
- **CSRF atrás do proxy:** o nginx repassa `Host $http_host` (com porta). O acesso é pelo port
  forwarding do VS Code Remote (navegador em `http://localhost:<porta>`) — o mesmo caminho já
  validado com o proxy do Vite em `localhost:5173` (commit `9bf18fb`); Dev Tunnels/`*.devtunnels.ms`
  com HTTPS **não** é o cenário e não é suportado nesta fase. O navegador manda `Origin: http://localhost:<porta-local>` e `Host: localhost:<porta-local>`,
  que batem — mesmo princípio do fix `9bf18fb` no proxy do Vite. Se o VS Code mapear para uma
  porta local diferente de 8080 isso continua funcionando, porque Origin e Host acompanham a porta
  local. `CSRF_TRUSTED_ORIGINS` não é necessário; se o teste manual mostrar o contrário, adicionar
  via env. Cookies **sem** `Secure` (sem HTTPS); `SESSION_COOKIE_SAMESITE=Lax` (default).

### 5.3. Ingestão (`backend/Dockerfile.ingestion`)

- Mesmo código e requirements de runtime do backend + `chromium`, `chromium-driver`
  (Debian bookworm, arm64) + `supercronic` (binário estático arm64).
- `B3_INGEST_USE_SELENIUM=1`, `TZ=America/Sao_Paulo`.
- **Mudança de código necessária em `fetch_b3_selenium`:** usar o Chromium/chromedriver do
  sistema em vez de `ChromeDriverManager().install()`. Proposta: settings `CHROME_BINARY` e
  `CHROMEDRIVER_PATH` (env; defaults `None`), e se presentes usar
  `opts.binary_location = CHROME_BINARY` + `Service(executable_path=CHROMEDRIVER_PATH)`; senão
  manter o comportamento atual. Também `--no-sandbox` e `--disable-dev-shm-usage` (padrão em
  container) e `shm_size` no compose. Mudança fora de `engine/` — não afeta paridade.
- **Fallback (§7.5 do mestre):** se o Selenium falhar para um índice, cair para
  `read_b3_static` naquele índice em vez de só registrar falha. Hoje `_fetch` escolhe um ou
  outro pela flag; o builder implementa o "tenta Selenium, cai no CSV".
  Chamada minha: **falha do Selenium conta como falha da série** (log `ERROR`, entra em
  `failed_keys`, exit 1), mesmo que o CSV tenha sido lido. Motivo: os CSVs estáticos param em
  10/04/2026 e não recebem pontos novos, então um fallback "bem-sucedido" esconderia um scraper
  quebrado com o `b3:*` congelado em abril. O fallback serve para o banco recriado do zero ter o
  histórico, não para mascarar a falha.
- **Agenda:** `supercronic` com crontab versionado (`backend/deploy/crontab` ou similar):
  `ingest_market_data` às **06:00 e 19:00, segunda a sexta** (§3.3 do mestre), horário de São
  Paulo. Chamada minha: `supercronic` em vez de `ofelia` (recomendação original §7.4) porque
  não exige montar o socket do Docker no container. Log do job em stdout (ADR 0008).
- Execução manual a qualquer momento:
  `docker compose run --rm lab-appinvest-ingestion python manage.py ingest_market_data [...]`
  (ou `exec` no container em execução).

### 5.4. Web (`frontend/Dockerfile` + `nginx.conf`)

- Stage 1: `node:22` (mesma major do host) → `npm ci` → `npm run build`.
- Stage 2: `nginx:alpine` com `dist/` em `/usr/share/nginx/html` e um `nginx.conf` versionado:
  - `/api/` e `/admin/` → `proxy_pass http://lab-appinvest-backend:8000` com `Host $http_host`,
    `X-Forwarded-For`, `X-Forwarded-Proto`, `proxy_read_timeout 90s`.
  - `/static/` → volume `staticfiles`.
  - `/` → SPA com `try_files $uri /index.html` (rotas do React Router).
  - `client_max_body_size` pequeno (a API só recebe JSON).
- Só o nginx publica porta: `127.0.0.1:8080:80`.

### 5.5. Dev no host — o que **não** muda

`runserver` + Vite + `.venv/bin/pytest` + `npm test` seguem idênticos, lendo `backend/.env`.
Host e containers compartilham o **mesmo** database `appinvest` e o mesmo Redis DB 1 — de
propósito: token DLP e séries ingeridas pelo container aparecem no dev imediatamente.
Consequência aceita: uma migration aplicada no dev já vale para o container (e vice-versa).

## 6. Dados — carga inicial e faltantes

Estado em 2026-10-27 (`MarketSeries`, 105 mil pontos):

| Série | Até | Situação |
|---|---|---|
| `bcb:11`, `bcb:12`, `ptax:USDBRL`, `bcb:433` | 26/10 (IPCA 01/09) | em dia |
| `yf:*` (13 tickers, incl. `AAPL` sob demanda) | 27/10 | em dia |
| `b3:IFIX`, `b3:SMLL`, `b3:IDIV` | **10/04/2026** | **faltante** — nunca houve ingestão Selenium; vieram do `backfill` dos CSVs estáticos |
| `bcb:12461/12466–12469` (ANBIMA) | **10/04/2026** | parada no SGS; a emenda com ETF (`IMAB11.SA` etc., §1.2 do mestre) cobre o período depois disso — **não** é bloqueio desta fase, só verificar que os gráficos que usam IMA-B estão contínuos |

Procedimento único depois do primeiro `up` (documentado no README, não automatizado):
1. `backfill_market_data` no container de ingestão — idempotente; garante que tudo dos CSVs de
   `data/raw` + `data/static` está no banco (cobre um banco recriado do zero).
2. `ingest_market_data` no container de ingestão — incremental a partir da última data; para B3
   o Selenium busca 2026 inteiro e preenche abril → hoje.
3. Conferir com a mesma query de contagem por `series_key` (último ponto de `b3:*` ≈ último dia útil).

Usuário: `docker compose exec lab-appinvest-backend python manage.py createsuperuser` (ou reusar
o usuário já existente no banco compartilhado). Token DLP: pela SPA.

**Risco conhecido:** o scraper da B3 nunca rodou contra o site atual nesta branch; o site pode
ter mudado desde o legado. Se o Selenium não funcionar, a fase **não** fica bloqueada — o
fallback para CSV mantém o app de pé e o conserto do scraper vira item próprio (registrar no
§2.3 do mestre). Ver §8 para o que conta como pronto.

## 7. Documentação a atualizar

- `docs/consolidacao_arquitetura_e_migracao.md`: §2.0 (linha da Fase 5), §2.2, §2.4 (Supabase:
  "descartado, sem migração"), §3.5 (Chromium ARM, não Chrome), §5 Fase 5 item 4, §6.2 (riscar),
  §7.4 (decidido: `supercronic`), §9 (marcar Fase 5; riscar "Migração de dados — import Supabase" e
  substituir por "carga inicial de mercado"), backlog (timeout DLP decidido).
- ADR nova: **0016 — Empacotamento no LAB** (dev no host × containers como app rodando, nomes
  `lab-appinvest-*`, porta `127.0.0.1:8080`, Chromium ARM, `supercronic`, Supabase descartado).
- `backend/README.md` / `backend/CLAUDE.md`: `requirements-dev.txt`, como subir/reconstruir
  os containers, como rodar ingestão manual, carga inicial.

## 8. Pronto quando

1. `docker compose up -d --build` a partir da raiz sobe os três serviços *healthy* em ARM64;
   `docker compose ps` mostra o nginx em `127.0.0.1:8080` e nenhuma porta do app em `0.0.0.0`.
2. Com o port forwarding do VS Code: login pela SPA, carteira (8 análises) e planejador de IR
   (incl. download do PDF) funcionando; `/admin` com CSS; primeira análise com DLP lenta não dá 502/504.
3. `ingest_market_data` no container roda o Selenium com Chromium e `b3:*` chega ao último dia
   útil — **ou**, se o scraper da B3 estiver quebrado, o comando sai com exit 1 nomeando as
   séries `b3:*`, o resto da ingestão fica em dia, e o conserto do scraper está registrado no
   doc mestre como pendência.
4. O crontab está ativo (log do `supercronic` mostra a próxima execução).
5. `docker compose restart` / reboot da VM: tudo volta sem intervenção.
6. Fluxo do host intacto: `.venv/bin/pytest -q` (com `requirements-dev.txt`) e `npm test` verdes;
   `runserver` + Vite funcionam lado a lado com os containers.
7. Nenhum segredo nas imagens (`docker history`/inspeção do filesystem da imagem sem `.env`);
   `\dt` em `banco_lab` vazio (ADR 0006).
8. Docs da §7 atualizadas.

## 9. Deixado para o builder

- Base exata da imagem de ingestão (`python:3.12-slim-bookworm` + apt `chromium` é o
  caminho esperado) e versão pinada do `supercronic`.
- Forma do healthcheck, nomes de volumes, organização dos arquivos de deploy
  (`backend/deploy/`, `frontend/nginx.conf` etc.).
- Como testar a troca para Chromium do sistema em `b3.py` (unitário com mock dos settings;
  a execução real é o critério 3 da §8).
