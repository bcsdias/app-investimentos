# 0008 — Logging só `StreamHandler` (stdout)

**Status:** Aceita · **Data:** 2026-08-28 · **Fase:** 1

## Contexto

O app legado usa `RotatingFileHandler` e escreve em `log/`. Em container (Fase 5), log em
arquivo é antipadrão: exige volume, rotação própria, e não aparece em `docker logs`.

## Decisão

`LOGGING` com um único handler `console` (`logging.StreamHandler`, stdout), formatter
`verbose` (`{asctime} {levelname} {name} {message}`), `root` no nível de `LOG_LEVEL`
(default `INFO`). **Sem** handler de arquivo, **sem** `RotatingFileHandler`.

## Consequências

- **+** `docker logs` / `journald` / coletor do LAB capturam tudo sem config extra.
- **+** Mesma configuração no host e em container.
- **−** Sem persistência local de log por conta do app — fica a cargo de quem roda o processo
  (redirecionar stdout, ou o runtime de container).
