# 0002 — `User` customizado desde o dia 1

**Status:** Aceita · **Data:** 2026-08-28 · **Fase:** 1

## Contexto

Trocar o modelo de usuário do Django **depois** que migrations já rodaram é uma das
operações mais dolorosas do framework (envolve recriar tabelas, refazer FKs, migrar
dados). O projeto vai ter autenticação própria (o SSO Google do Streamlit é perdido na
migração) e provavelmente campos extras de perfil no futuro.

## Decisão

Criar `apps.accounts.User(AbstractUser)` **sem campos extras** já na Fase 1, e definir
`AUTH_USER_MODEL = "accounts.User"` **antes do primeiro `migrate`**.

## Consequências

- **+** Qualquer campo/método futuro no usuário é uma migration trivial, não uma cirurgia.
- **+** `get_user_model()` / `settings.AUTH_USER_MODEL` em todo o código desde o início.
- **−** Um app e uma migration a mais sem ganho funcional imediato.
- `apps/accounts/admin.py` registra o `User` com o `UserAdmin` padrão.
