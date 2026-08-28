# 0006 — Prefixo `lab-` nos containers; `appinvest` no database e no código

**Status:** Aceita · **Data:** 2026-08-28 · **Fase:** 0

## Contexto

A infra compartilhada ([0001](0001-infra-lab-compartilhada.md)) usa nomes com prefixo
`lab-` (`lab-postgres`, `lab-redis`, `lab-net`) porque é infraestrutura do LAB, não do
projeto. Mas o database, o role e o código do app são específicos do `app-investimentos`.
Misturar os dois vocabulários (ex.: `banco_lab` no código) gera confusão e acopla o app à
nomenclatura de infra.

## Decisão

- **Containers e rede:** mantêm o prefixo `lab-*` (pertencem ao LAB).
- **Database, role e tudo em `backend/`:** usam o nome do projeto, **`appinvest`**.
- O database default do `lab-postgres` (`banco_lab`) **segue vazio** do ponto de vista do
  app — nenhuma tabela do `app-investimentos` pode cair nele.

Verificação na Definition of Done de cada fase: `\dt` em `appinvest` mostra as tabelas do
app; `\dt` em `banco_lab` responde "Did not find any relations".

## Consequências

- **+** Fronteira clara: infra vs. aplicação. O código não sabe que existe um "LAB".
- **+** Merge futuro de `migracao-django` → `main` pode remover o prefixo `lab-` dos
  containers sem tocar em nada do `backend/`.
- **−** Exige checagem explícita a cada fase (o isolamento é lógico, não físico).
