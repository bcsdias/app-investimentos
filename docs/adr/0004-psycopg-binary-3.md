# 0004 — Driver Postgres: `psycopg[binary]` 3.x

**Status:** Aceita · **Data:** 2026-08-28 · **Fase:** 1

## Contexto

Django 4.2+ suporta tanto `psycopg2` quanto `psycopg` 3. Precisa escolher o driver e a
forma de distribuição (wheel binária vs. compilação a partir do source, que exige
`libpq-dev` + toolchain de C no host/imagem).

## Decisão

Usar **`psycopg[binary]~=3.2`** — psycopg 3, extra `binary` (wheel autocontida, sem
`libpq` no sistema). `ENGINE = "django.db.backends.postgresql"`.

## Consequências

- **+** `pip install` sem dependência de sistema; funciona no host e simplifica a imagem da Fase 5.
- **+** psycopg 3: melhor suporte a async, pipeline mode, tipagem — alinhado com o futuro do Django.
- **−** A wheel `binary` não é recomendada pelos mantenedores para produção de alta carga
  (preferem `psycopg[c]` compilada). Reavaliar na Fase 5 se a carga justificar.
