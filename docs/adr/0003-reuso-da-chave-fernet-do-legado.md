# 0003 — Reusar a chave Fernet do app legado

**Status:** Aceita · **Data:** 2026-08-28 · **Fase:** 1

## Contexto

O app Streamlit cifra os tokens da API DLP com **Fernet** (`src/data/user_store.py::_get_cipher`)
usando `st.secrets["security"]["fernet_key"]`, e guarda o resultado no Supabase
(`user_tokens.encrypted_token`). A migração vai importar essas linhas para o modelo
`UserToken` no Postgres local.

Se o backend novo usasse uma chave diferente, todo token existente precisaria ser
**decifrado com a chave antiga e recifrado com a nova** durante a migração — passo extra,
com janela de risco e necessidade de acesso simultâneo às duas chaves.

## Decisão

O backend Django usa **a mesma chave**: `FERNET_KEY` em `backend/.env` recebe o valor de
`[security].fernet_key` do `../.streamlit/secrets.toml`. `apps.core.security._cipher()`
constrói `Fernet(key)` exatamente como o legado.

Testes garantem a compatibilidade nos dois sentidos (token cifrado pelo legado decifra no
backend e vice-versa).

## Consequências

- **+** Import do Supabase vira `INSERT` direto: zero recifragem, zero janela de risco.
- **+** Durante a coexistência (até a Fase 6), os dois apps leem o mesmo token.
- **−** A chave vive em **dois arquivos** no host (`secrets.toml` e `backend/.env`), ambos
  git-ignored. Consolidar num gerenciador de segredos é trabalho de fase posterior.
- **−** Rotacionar a chave Fernet passa a exigir coordenar os dois apps — adiar até o
  Streamlit sair de cena.
- **Invariável:** `FERNET_KEY` nunca vai a log, exceção ou saída de teste; nunca é commitada.
