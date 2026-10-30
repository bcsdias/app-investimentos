# Convenções — `backend/`

Instruções para agentes trabalhando neste diretório. Complementa
[`README.md`](README.md) (setup/rodar) e os planos em `../docs/superpowers/plans/`.

## Contexto

- Parte da migração Streamlit → Django + React. O `src/` (Streamlit legado) foi removido na
  Fase 6 após a paridade aprovada ([ADR 0017](../docs/adr/0017-paridade-legado.md)).
- `../scripts/parity/` é a **linha de base da Fase 7**: com o encerramento da Fase 7c ([ADR 0020](../docs/adr/0020-backlog-spa-api-fase-7c.md)),
  a nova linha de base congelada é `reports/parity/20261029-211840-fase7c/baseline/` (local, nunca versionado).
  Regressões são executadas com `python -m parity.regress --baseline reports/parity/20261029-211840-fase7c/baseline`.
- Cada fase segue o ciclo **design (`specs/`) → plano (`plans/`) → implementação TDD**.
- Branch de trabalho: `migracao-django`. Não commitar em `main` / `dev`.

## Ambiente

- Virtualenv em `backend/.venv/` (instalado com `.venv/bin/pip install -r requirements-dev.txt`). **Sempre** `.venv/bin/python` e `.venv/bin/pytest` —
  não existe `python` no PATH da VM, e `python3` é o do sistema (sem deps).
- Config vem só de `backend/.env` (via `python-dotenv`, sem expansão custom) — ver
  [ADR 0010](../docs/adr/0010-python-dotenv-puro.md).
- O desenvolvimento continua no host (`runserver` + Vite + `.venv/bin/pytest` + `npm test`); o `docker-compose.yml` na raiz é apenas "o app rodando" no LAB (`127.0.0.1:8080`, reconstruído sob demanda com `docker compose up -d --build`) — ver [ADR 0016](../docs/adr/0016-empacotamento-no-lab.md).

## Banco de dados

- Database e role: **`appinvest`**, sempre. O app **nunca** usa o superusuário `postgres`.
- As tabelas do app **não** podem cair no `banco_lab` (database compartilhado do LAB).
  Ver [ADR 0006](../docs/adr/0006-prefixo-lab-containers-appinvest-codigo.md).
- `AUTH_USER_MODEL = "accounts.User"` — definido antes do primeiro `migrate`
  ([ADR 0002](../docs/adr/0002-modelo-user-customizado-dia-1.md)).
- Migrations: `makemigrations <app>` + `migrate`. Commitar a migration junto do modelo.

## Criptografia

- Todo segredo de usuário em repouso passa por `apps.core.security` (`encrypt` / `decrypt`,
  Fernet sobre `settings.FERNET_KEY`). Ver [ADR 0003](../docs/adr/0003-reuso-da-chave-fernet-do-legado.md).
- **Nunca** gravar valor plano numa coluna `*_token` / `encrypted_*`.
- **Nunca** imprimir `FERNET_KEY` (nem trechos) em log, exceção, ou saída de teste.
- **Nunca** commitar `backend/.env`.

## Layout de apps e pacotes

- Um app por domínio em `apps/<nome>/`, com `apps.py` declarando `name = "apps.<nome>"`
  e `default_auto_field = "django.db.models.BigAutoField"`.
- `engine/` é um pacote Python puro (irmão de `apps/`), sem `models.py` e sem acoplamento a Django.
- `tests/` como pacote dentro do app (`apps/<nome>/tests/test_*.py`) ou do pacote (`engine/tests/test_*.py`).
- **Convenções de `apps/api/` (Fase 3, [ADR 0013](../docs/adr/0013-contrato-da-api-v1.md); Fase 7d, [ADR 0022](../docs/adr/0022-otimizacoes-fase-7d.md)):**
  - Única instância `NinjaAPI` em `config/api.py` (`/api/v1/`, Swagger em `/api/v1/docs`), autenticada por `ApiSessionAuth` (`apps/api/csrf.py`).
  - **Routers finos** em `apps/api/routers/*` e `apps/accounts/api.py`: nenhuma lógica de cálculo nos routers; toda orquestração (validação → token DLP → carteira → `build_dataset`) passa por `apps/api/analysis.py` e toda conversão `pandas -> JSON` por `apps/api/serializers.py`.
  - Memo de `AnalysisContext` entre endpoints da carteira (`apps/api/analysis.py`): gravado com `cache.set` direto (pickle), nunca `set_df`, para preservar tipos pandas e a precisão total dos floats. TTL de 300 s; chave `market:analysis:<user.pk>:<versão do cache DLP>:<geração das séries>:<dia>:<digest do pedido>`. Tudo que muda o contexto precisa entrar na chave (ou no digest); parâmetros aplicados depois do contexto (ex.: `window`) não entram.
  - Erros de negócio sempre levantam `ApiError(status, code, message, details)` (`apps/api/errors.py`), renderizado como `{"code": str, "message": str, "details": object|null}` em português.
  - **Nunca** retornar `401` quando a API externa DLP rejeitar o token do usuário (`DLPAuthError`) — usar sempre **`422 dlp_token_rejected`** (`401` é reservado para sessão ausente ou login inválido).
  - Isolamento por construção: o usuário é sempre `request.user`; nenhuma rota recebe ID ou e-mail de usuário.
- **Apps de domínio com API própria (Fase 3b, [ADR 0015](../docs/adr/0015-planejador-de-ir.md)):** o router pode morar no próprio app (`apps/<app>/api.py`, registrado em `config/api.py`), seguindo as mesmas regras de router fino/`ApiError`. Cálculo **novo** (sem paridade a preservar) fica em módulos puros sem Django dentro do app, nunca em `engine/`; esses módulos levantam erro próprio (ex.: `PlanError`) que só o `api.py` converte em `ApiError`.

## TDD

Conforme os planos: escrever o teste, **rodar e ver falhar** com o erro esperado, só então
implementar, rodar e ver passar. Os passos "ver falhar" não são opcionais.

`.venv/bin/pytest -q` a partir de `backend/`. `--reuse-db` já está no `pytest.ini`.

## Commits

- Conventional Commits + Gitmoji + descrição no imperativo em **português**.
- **Nunca** adicionar trailer `Co-Authored-By`.
- Um commit por task do plano, com a mensagem que o plano especifica.

## Proibições

| Não fazer | Por quê |
|---|---|
| Token digitado/editável no Django Admin | grava valor não-cifrado em `encrypted_token`; entra só pela API (Fase 3) |
| `set_df` para memo de `AnalysisContext` | `set_df` grava JSON `orient="table"` com `double_precision=10` e arredonda os floats; o memo usa `cache.set` direto (pickle) |
| `django-cors-headers` | SPA + API na mesma origem via nginx na Fase 5 ([ADR 0005](../docs/adr/0005-sem-django-cors-headers.md)) |
| `RotatingFileHandler` / log em arquivo | logging só `StreamHandler`/stdout ([ADR 0008](../docs/adr/0008-logging-stdout-apenas.md)) |
| Commitar compose files do LAB | ficam fora do repo, em `/data/projetos/lab-*/` |
| `TIME_ZONE` / `USE_TZ` diferentes | `America/Sao_Paulo`, `USE_TZ = True` |

## Ao fechar uma fase

1. Rodar o checklist de verificação do plano (Definition of Done).
2. Rodar o **checklist de atualização de documentação** em
   [`../docs/README.md`](../docs/README.md#checklist-de-atualização-ao-fechar-uma-fase) —
   cobre `consolidacao_arquitetura_e_migracao.md` (§2.0/§2.2/§2.3/§4/§9), `docs/README.md`,
   `backend/README.md` (Estado + Estrutura + contagem de testes), ADRs e este arquivo.
   A última task de todo plano deve incluir esse checklist explicitamente.
3. Se a fase travou alguma decisão nova: novo ADR em `../docs/adr/` + linha no índice.
4. Atualizar este arquivo se surgiram convenções novas de código/estrutura.
