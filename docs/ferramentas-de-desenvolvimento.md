# Ferramentas de Desenvolvimento (plugins do Claude Code)

Guia de quais plugins/skills usar em cada fase da migração, para melhorar geração de código,
revisão e segurança. Plugins instalados (marketplace `claude-plugins-official`):

`frontend-design`, `superpowers`, `code-review`, `context7`, `skill-creator`, `code-simplifier`,
`playwright`, `github`, `claude-md-management`, `security-guidance`, `typescript-lsp`,
`claude-code-setup`, `pr-review-toolkit`, `pyright-lsp`, `chrome-devtools-mcp`,
`explanatory-output-style`, `semgrep`.

---

## Transversal (todas as fases)

| Ferramenta | Uso | Quando |
|---|---|---|
| **`superpowers`** | Espinha dorsal do fluxo: `brainstorming` → `writing-plans` → `executing-plans` / `subagent-driven-development`. Também `test-driven-development` (os planos já são TDD), `systematic-debugging` (todo teste que falhar), `verification-before-completion` (antes de dizer "pronto"), `using-git-worktrees` (isolar a execução de cada fase). | Início e execução de cada fase |
| **`context7`** (MCP) | Consultar API e travar versões exatas: Django 5.2 / django-ninja 1.4 / psycopg 3 / django-redis (Fases 1–3); React / Vite / Tailwind / Apache ECharts (Fase 4). Evita código com API desatualizada. | Antes de escrever qualquer código que use lib externa |
| **`pyright-lsp`** | Diagnóstico de tipos/imports em tempo real no `backend/` (Python). Alto valor no engine e nos serviços de mercado, que manipulam muito `pandas`. | Fases 1–3, contínuo |
| **`claude-md-management`** | Criar e manter `backend/CLAUDE.md` (e `frontend/CLAUDE.md`) com convenções do projeto, para sessões futuras não re-derivarem contexto. | Ao final da Fase 1; revisar a cada fase |

---

## Fase 1 — Backend Django (modelos, segurança)

| Ferramenta | Uso |
|---|---|
| **`pyright-lsp`** | Validar `settings.py`, modelos e `security.py` enquanto escreve. |
| **`semgrep`** (MCP) — `get_semgrep_secrets_findings` | Varrer o repo garantindo que a chave Fernet, `service_role` do Supabase e `client_secret` do Google **não** vazaram para arquivos versionados. Rodar após a Task 2 e antes de cada commit de config. |
| **`semgrep`** (MCP) — `get_semgrep_supply_chain_findings` | Checar vulnerabilidades nas dependências pinadas do `requirements.txt`. |
| **`pr-review-toolkit`** → agente `type-design-analyzer` | Revisar o design de `UserToken` / `MarketSeries` (encapsulamento, invariantes). |
| **`code-review`** (`/code-review`) + **`code-simplifier`** | Ao fechar a fase: revisão de correção + limpeza do código recém-escrito. |

---

## Fase 2 / 2.5 — Serviços de mercado e refatoração do engine

| Ferramenta | Uso |
|---|---|
| **`pr-review-toolkit`** → agente `silent-failure-hunter` | **Prioritário.** O `src/data/sources/market_data.py` atual está cheio de `except: return None`. Ao quebrá-lo em `apps/market/services/*`, rodar este agente para não carregar os silent failures adiante. |
| **`pr-review-toolkit`** → agente `pr-test-analyzer` | Conferir cobertura dos testes das funções `compute_*` extraídas de `financial_report.py`. |
| **`context7`** | API atual de `python-bcb`, `yfinance`, `redis-py`. |
| **`superpowers:systematic-debugging`** | Divergências numéricas ao portar TWR/TIR/Sharpe. |

---

## Fase 3 — API (Django Ninja)

| Ferramenta | Uso |
|---|---|
| **`context7`** | Padrões atuais de `django-ninja` 1.4: `Router`, `Schema`, autenticação por sessão (`django_auth`), respostas de arquivo (PDF do planejador de IR). |
| **`semgrep`** (MCP) — `get_semgrep_sast_findings` | SAST nos endpoints: injeção, exposição de dados, auth ausente em rota. |
| **`security-guidance`** | Decisões de auth de sessão, CSRF, isolamento de carteira por usuário. |
| **`pr-review-toolkit`** → `code-reviewer` + `type-design-analyzer` | Revisar rotas e schemas Pydantic in/out. |
| **`github`** (MCP) | Abrir PR da fase, se adotar fluxo de PR (hoje o padrão é commit direto na `lab-dev`). ⚠️ ver "Pendências" abaixo. |

---

## Fase 4 — Frontend React SPA

| Ferramenta | Uso |
|---|---|
| **`frontend-design`** | Direção visual do dashboard — tipografia, layout, tema escuro, evitar cara de template. |
| **`dataviz`** (skill) | **Prioritário.** Os 5 gráficos ECharts (TWR, Drawdown, RollingSharpe, Shadow, RiskReturn) + KPIs: paleta, eixos, tooltip e legenda consistentes, light/dark. Ler antes de escrever o primeiro gráfico. |
| **`context7`** | API atual de React 18, Vite, Tailwind, `echarts` / `echarts-for-react`, Axios. |
| **`typescript-lsp`** | Diagnóstico no `frontend/` (recomendado migrar o plano de JSX para TSX). |
| **`chrome-devtools-mcp`** | Debugar a SPA no navegador, performance (LCP), acessibilidade (a11y-debugging). ⚠️ precisa de Node/npx. |
| **`playwright`** (MCP) | Testes E2E do dashboard (login, filtros, zoom, export PDF). ⚠️ precisa de Node/npx. |

---

## Fase 5 / 6 — Dockerização, merge e paridade

| Ferramenta | Uso |
|---|---|
| **`claude-code-setup`** (automation-recommender) | Depois que a estrutura estabilizar: sugerir hooks (rodar `pytest` + `pyright` a cada edição no `backend/`, lint no pre-commit) e um `run` skill para subir o LAB. |
| **`/security-review`** (built-in) | Varredura de segurança do diff completo antes de fechar a `lab-dev`. |
| **`code-review`** `ultra` | Revisão multi-agente na nuvem do branch inteiro antes do merge. |
| **`skill-creator`** | Opcional: criar skills do repo — "subir o LAB", "rodar a suíte de paridade". |

---

## Pré-requisitos do host

| Item | Versão / método | Para quê |
|---|---|---|
| **Node.js** | 22 LTS via NodeSource (`deb.nodesource.com/setup_22.x`), arm64, system-wide | `npx` no PATH → destrava os MCP `playwright` e `chrome-devtools-mcp`; build do frontend (Fase 4, dev). Em produção o `frontend/Dockerfile` fixa a própria versão. |
| **Chromium** | `apt install chromium-browser` ou `npx playwright install --with-deps chromium` | `chrome-devtools-mcp` (Fase 4) e Selenium da ingestão B3 (Fase 2, container). |
| **Docker + acesso ao daemon** | já presente | `lab-postgres` / `lab-redis` / `bootstrap_db.sh`. |
| **Python** | 3.12 (host) | `backend/` roda no host nas Fases 1–3. |

## Pendências de conexão (MCP que falharam no boot)

| Servidor | Erro | Como resolver |
|---|---|---|
| `chrome-devtools-mcp` | `ENOENT: npx` | Instalar Node (ver "Pré-requisitos do host") e **reiniciar o Claude Code** — MCP reconecta só no startup. |
| `playwright` | `ENOENT: npx` | Idem. |
| `github` | `Authorization header is badly formatted` | Configurar token válido (`gh auth login` ou a variável de ambiente esperada pelo servidor). Só necessário se adotar fluxo de PR. |
| `semgrep` (hook PostToolUse) | `Not logged into Semgrep Guardian` | Logar no guardian MCP, ou remover/condicionar o hook no `settings.json`. As ferramentas `mcp__plugin_semgrep_guardian__*` (SAST/secrets) seguem utilizáveis. |

---

## Não relevantes para código neste projeto

- **`explanatory-output-style`** — estilo de resposta do assistente; preferência pessoal, não afeta o código gerado.
