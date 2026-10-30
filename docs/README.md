# Documentação — `app-investimentos`

Documentação da reformulação arquitetural (migração Streamlit → Django + React no Docker LAB).

## Índice

| Documento | Conteúdo |
|---|---|
| [`consolidacao_arquitetura_e_migracao.md`](consolidacao_arquitetura_e_migracao.md) | **Documento mestre.** Estado atual do repo (§2), decisões de arquitetura, roteiro de fases (0 → 7). |
| [`adr/`](adr/README.md) | **Architecture Decision Records** — uma decisão por arquivo, imutável. O *porquê* consultável a longo prazo. |
| [`../backend/README.md`](../backend/README.md) | Setup, rodar e testar o backend Django. |
| [`../backend/CLAUDE.md`](../backend/CLAUDE.md) | Convenções para agentes trabalhando em `backend/`. |
| [`superpowers/specs/`](superpowers/specs/) | Specs de design, um por incremento (`AAAA-MM-DD-<tema>-design.md`). |
| [`superpowers/plans/`](superpowers/plans/) | Planos de implementação passo a passo, um por incremento. |
| [`ferramentas-de-desenvolvimento.md`](ferramentas-de-desenvolvimento.md) | Quais plugins/skills do Claude Code usar em cada fase (geração de código, revisão, segurança). |
| [`referencia-api-dlp-invest.md`](referencia-api-dlp-invest.md) | **Referência da API externa DLP Invest:** domínios, autenticação (`DLP-...`), guia para agentes de IA, todas as 23 rotas e schemas OpenAPI (`HistoricModel`, `AtivoCarteiraModel`, `SummaryInitModel`, `DarfModel`, etc.). |
| [`seguranca-dados-pessoais.md`](seguranca-dados-pessoais.md) | **Repo público:** o que nunca versionar (carteira, segredos, logs), checklist antes de commitar, como limpar o histórico se algo vazar, registro da limpeza de 2026-10-20. |
| [`historico/`](historico/README.md) | Drafts e revisões anteriores, substituídos pelo documento mestre. |

## Documentação viva

Alguns documentos refletem o **estado atual** do repositório e são atualizados **ao fechar
cada fase**, como último passo explícito da Definition of Done do plano. Ver o
[checklist de pós-fase](#checklist-de-atualização-ao-fechar-uma-fase) abaixo — todo plano
deve incluí-lo na última task para não deixar doc para trás (aconteceu nas Fases 2/2.5:
`docs/README.md` e o §4 do documento mestre ficaram desatualizados e foram corrigidos depois).

### Checklist de atualização ao fechar uma fase

Rodar **todos** os itens abaixo antes de considerar a fase concluída. Marcar o que não se
aplica com "n/a" no commit de docs.

| # | Arquivo / seção | O que atualizar |
|---|---|---|
| 1 | `consolidacao_arquitetura_e_migracao.md` §2.0 ("Já implementado") | Linha nova (ou revisão) da camada entregue: local, situação, contagem de testes **total atual**, ADRs novos. Atualizar o título da seção com as fases cobertas. |
| 2 | `consolidacao_arquitetura_e_migracao.md` §2.2 ("O que **não** existe ainda") | Remover o que passou a existir. |
| 3 | `consolidacao_arquitetura_e_migracao.md` §2.3 ("Dívida técnica") | Marcar itens resolvidos (mantendo o registro histórico); **acrescentar** dívida nova assumida de propósito (pendências de revisão adiadas, decisões "por enquanto"). |
| 4 | `consolidacao_arquitetura_e_migracao.md` §4 ("Estrutura de Diretórios Alvo") | Reconciliar o desenho com o que **de fato** foi construído quando o layout divergir do plano original. |
| 5 | `consolidacao_arquitetura_e_migracao.md` §9 (checklist de ordem) | Marcar `[x]` a(s) fase(s) fechada(s). |
| 6 | `docs/README.md` (este arquivo) — "Fluxo de trabalho" | Atualizar a linha de fases concluídas e o "próximo incremento". |
| 7 | `docs/adr/` + `docs/adr/README.md` | ADR novo por decisão de arquitetura travada na fase; linha correspondente no índice. |
| 8 | `backend/README.md` | Bloco **"Estado"** (topo), seção **"Estrutura"** (novos módulos/pacotes) e a contagem de testes no bloco **"Rodar"** (`# N testes`). |
| 9 | `backend/CLAUDE.md` | Só se surgiram convenções novas de código/estrutura. |
| 10 | `ferramentas-de-desenvolvimento.md` | Só se a fase mudou quais skills/plugins usar dali em diante. |
| 11 | Specs/planos consumidos | Se a implementação divergiu do plano de forma relevante, anotar no commit e (se afeta fases seguintes) no spec correspondente. |

## Fluxo de trabalho

Cada fase do roteiro do documento mestre passa por: **design** (`superpowers/specs/`) →
**plano** (`superpowers/plans/`) → **implementação** (TDD, commits por task na branch `migracao-django`).

Fases **0 + 1** (infra LAB + esqueleto do backend Django) concluídas em 2026-08-28.
Fases **2 + 2.5** (serviços de mercado `apps/market/` + engine puro `backend/engine/`)
concluídas em 2026-08-31, com correções de revisão aplicadas na sequência.
**Fase 3** (API da Carteira em Django Ninja — auth, token DLP, 8 análises + filters/refresh, catálogo de benchmarks, busca e fetch-on-read de tickers) concluída em 2026-10-26.
**Fase 3c** (otimizações DLP: cache único de `/historico` com filtro em memória, `DLP_BASE_URL`) concluída em 2026-10-26.
**Fase 3b** (Planejador de IR `/migration/{wallet,plan,plan.pdf}`, [ADR 0015](adr/0015-planejador-de-ir.md)) concluída em 2026-10-26.
**Fase 4** (SPA React em `frontend/`) concluída em 2026-10-27, com as correções da revisão pós-Fases 3/4 ([plano](superpowers/plans/2026-10-27-correcoes-revisao-fases-3-4.md)).
**Fase 5** (Dockerização no LAB — `lab-appinvest-{web,backend,ingestion}` em `127.0.0.1:8080`, Chromium ARM64 + `supercronic`, carga inicial + ingestão B3 ao vivo, [ADR 0016](adr/0016-empacotamento-no-lab.md)) concluída em 2026-10-27.
**Fase 6** (paridade em duas camadas com o legado, linha de base da Fase 7 em `scripts/parity/` + `reports/parity/`, remoção de `src/` e libs legadas, [ADR 0017](adr/0017-paridade-legado.md)) concluída em 2026-10-27.
**Fase 7a** (correção matemática e de confiabilidade do engine alinhada à planilha DLP, nova linha de base em `reports/parity/20261028-180012-fase7-abertura/baseline/`, revisada após o 1º teste do usuário: abertura do período no fechamento anterior a `start_date`, [ADR 0018](adr/0018-metodologia-de-retorno-dlp.md)) concluída em 2026-10-28.
**Fase 7b** (integridade dos dados de mercado: invalidação de cache por geração global no Redis, módulo puro de detecção de lacunas por cadência `gaps.py`, comando `market_gaps`, re-busca na ingestão, `ffill` com aviso aditivo `meta.gaps` no `PeriodNotice` da SPA, benchmarks com lookback de 40 dias e abertura no último valor conhecido, nova linha de base em `reports/parity/20261029-122417-fase7b/baseline/`, [ADR 0019](adr/0019-integridade-dos-dados-de-mercado.md)) concluída em 2026-10-29.
**Fase 7c** (achados do teste da 7b e resolução completa do backlog SPA/API: rate limit de login 429, 422 em ApiError, sincronização de URL via `?period=`, TIR do período ANBIMA, lock com dono e geração DLP estável no Redis, acessibilidade, campos de data e PDF sem cortes, nova linha de base em `reports/parity/20261029-211840-fase7c/baseline/`, [ADR 0020](adr/0020-backlog-spa-api-fase-7c.md)) concluída em 2026-10-30.
**Fase 7c.1** (índices IMA pela ANBIMA: histórico completo no S3 + fallback por data, eliminação definitiva de projeções por ETF e ProxySplice, remoção das séries `bcb:1246x` e catálogo limpo com um nome por índice, [ADR 0021](adr/0021-indices-ima-pela-anbima.md)) concluída em 2026-10-30.
Próximo e último incremento: **Fase 7d** (otimizações do engine sem alterar números).

