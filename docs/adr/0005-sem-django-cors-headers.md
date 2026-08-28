# 0005 — Sem `django-cors-headers`

**Status:** Aceita · **Data:** 2026-08-28 · **Fase:** 1

## Contexto

Arquiteturas SPA + API costumam adicionar `django-cors-headers` porque o front (ex.:
`localhost:5173`) e a API (`localhost:8000`) ficam em origens diferentes no dev.

## Decisão

**Não** incluir `django-cors-headers`. Na Fase 5 o **nginx único** serve a SPA (estáticos)
e faz proxy de `/api/` para o backend — **mesma origem**, CORS não se aplica. No dev, o
Vite faz proxy de `/api/` para o Django, também eliminando cross-origin.

## Consequências

- **+** Uma dependência e uma classe de middleware a menos; sem risco de `CORS_ALLOW_ALL`
  frouxo vazando para produção.
- **−** Quem rodar o front em porta separada **sem** o proxy do Vite vai bater em CORS.
  A convenção é: sempre via proxy (dev) ou nginx (prod).
- Se um dia existir um cliente de origem realmente distinta (app mobile, terceiro),
  esta ADR é revisada.
