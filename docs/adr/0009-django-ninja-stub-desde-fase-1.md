# 0009 — `django-ninja` pinado desde a Fase 1, como stub

**Status:** Aceita · **Data:** 2026-08-28 · **Fase:** 1

## Contexto

A API REST só é implementada na Fase 3. Mas a escolha de framework de API (Django Ninja,
DRF, etc.) é uma decisão de arquitetura do documento mestre (§3.1), e o `requirements.txt`
da Fase 1 quer refletir a stack alvo inteira.

## Decisão

Incluir **`django-ninja~=1.4`** no `requirements.txt` já na Fase 1. Em `config/urls.py`,
só um comentário-stub (`# path("api/v1/", api.urls)  # Fase 3`). Nada em `INSTALLED_APPS`,
nenhum router, nenhum schema ainda.

Motivo do Ninja sobre DRF: schemas Pydantic (mesmo modelo mental do resto do ecossistema
Python moderno), menos boilerplate, tipagem de request/response nativa.

## Consequências

- **+** `requirements.txt` já é a foto da stack final; `pip install` na Fase 3 não muda deps.
- **+** Trava a decisão de framework antes de escrever endpoint, evitando retrabalho.
- **−** Uma dependência instalada sem uso por 2 fases. Custo desprezível.
- O pin `~=1.4` admite 1.x >= 1.4; a Fase 3 confirma a versão exata.
