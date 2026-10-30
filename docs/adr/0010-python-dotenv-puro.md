# 0010 — Gerência de env com `python-dotenv` puro

**Status:** Aceita · **Data:** 2026-08-28 · **Fase:** 1

## Contexto

Django não lê `.env` sozinho. Opções comuns: `django-environ` (parsing rico: `db_url()`,
casts, listas), `pydantic-settings`, ou `python-dotenv` (só carrega o arquivo em
`os.environ`, o resto é manual).

## Decisão

Usar **`python-dotenv~=1.0`**. `settings.py` chama `load_dotenv(BASE_DIR / ".env")` e um
helper local `_env(key, default, *, required)` faz a leitura, com erro explícito quando
uma variável obrigatória falta.

## Consequências

- **+** Zero mágica: o que está no `settings.py` é o que acontece. `DATABASES` é um dict
  Python normal, não uma URL parseada.
- **+** Menos superfície de dependência.
- **−** Sem helpers de parsing — listas (`DJANGO_ALLOWED_HOSTS`) e casts (`DEBUG`) são
  split/comparação manual no `settings.py`. Aceitável no tamanho atual da config.
- Se a config crescer muito, reavaliar `django-environ`.
