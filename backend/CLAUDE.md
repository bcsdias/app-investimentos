# Convenções — `backend/`

Instruções para agentes trabalhando neste diretório. Complementa
[`README.md`](README.md) (setup/rodar) e os planos em `../docs/superpowers/plans/`.

## Contexto

- Parte da migração Streamlit → Django + React. O `src/` (Streamlit legado) **coexiste
  e permanece funcional** até a Fase 6 — não mexer nele a partir daqui.
- Cada fase segue o ciclo **design (`specs/`) → plano (`plans/`) → implementação TDD**.
- Branch de trabalho: `migracao-django`. Não commitar em `main` / `dev`.

## Ambiente

- Virtualenv em `backend/.venv/`. **Sempre** `.venv/bin/python` e `.venv/bin/pytest` —
  não existe `python` no PATH da VM, e `python3` é o do sistema (sem deps).
- Config vem só de `backend/.env` (via `python-dotenv`, sem expansão custom) — ver
  [ADR 0010](../docs/adr/0010-python-dotenv-puro.md).

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

## Layout de apps

- Um app por domínio em `apps/<nome>/`, com `apps.py` declarando `name = "apps.<nome>"`
  e `default_auto_field = "django.db.models.BigAutoField"`.
- `tests/` como pacote dentro do app (`apps/<nome>/tests/test_*.py`).

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
| `django-cors-headers` | SPA + API na mesma origem via nginx na Fase 5 ([ADR 0005](../docs/adr/0005-sem-django-cors-headers.md)) |
| `RotatingFileHandler` / log em arquivo | logging só `StreamHandler`/stdout ([ADR 0008](../docs/adr/0008-logging-stdout-apenas.md)) |
| Commitar compose files do LAB | ficam fora do repo, em `/data/projetos/lab-*/` |
| `TIME_ZONE` / `USE_TZ` diferentes | `America/Sao_Paulo`, `USE_TZ = True` |

## Ao fechar uma fase

1. Rodar o checklist de verificação do plano (Definition of Done).
2. Atualizar `../docs/consolidacao_arquitetura_e_migracao.md` §2 (estado atual).
3. Se a fase travou alguma decisão nova: novo ADR em `../docs/adr/`.
4. Atualizar este arquivo se surgiram convenções novas.
