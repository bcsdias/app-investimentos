# app-investimentos

Análise de carteira de investimentos: consome o histórico da carteira na API DLP e cruza com
dados de mercado (Banco Central, B3, Yahoo Finance, PTAX) para calcular rentabilidade, risco
e comparar com benchmarks. Inclui um planejador de migração de carteira com isenção de IR.

## Estado

Migração do antigo app Streamlit para **Django + React no Docker LAB** concluída até a Fase 6:
a paridade numérica com o legado foi validada e o Streamlit (`src/`) foi removido
([ADR 0017](docs/adr/0017-paridade-legado.md)). Próximo passo: Fase 7 (correções do engine
pós-paridade). Desenvolvimento na branch `migracao-django`.

Arquitetura, decisões e roteiro: [`docs/`](docs/README.md) — o
[documento mestre](docs/consolidacao_arquitetura_e_migracao.md) e os
[ADRs](docs/adr/README.md).

## Funcionalidades

- **Rentabilidade (TWR)** base 100, isolando aportes e retiradas, e **TIR (XIRR)** ao longo do tempo.
- **Comparativo com benchmarks**: CDI, SELIC, IPCA + X%, IMA-B (com emenda por ETF), S&P 500
  e outros em reais, índices B3 (IFIX, SMLL, IDIV) e qualquer ticker do Yahoo sob demanda.
- **Risco**: drawdown, volatilidade e Sharpe móveis, risco × retorno (CAGR, vol, Sharpe).
- **Shadow portfolio**: os mesmos aportes aplicados em cada benchmark; tabela de retorno anual.
- **Planejador de IR**: migração gradual de carteira respeitando a isenção de R$ 20 mil/mês,
  com export em PDF.

## Arquitetura

| Parte | Onde | O quê |
|---|---|---|
| Backend | [`backend/`](backend/README.md) | Django 5 + Django Ninja (`/api/v1/`), engine de cálculo puro (`backend/engine/`), serviços de mercado e ingestão (`apps/market/`) |
| Frontend | `frontend/` | SPA React 18 + Vite + Tailwind + ECharts |
| Dados | `lab-postgres` / `lab-redis` (infra compartilhada do LAB) | Séries de mercado no Postgres (database `appinvest`), cache no Redis; token DLP cifrado com Fernet |
| Containers | [`docker-compose.yml`](docker-compose.yml) | `lab-appinvest-web` (`127.0.0.1:8080`), `lab-appinvest-backend` (Gunicorn), `lab-appinvest-ingestion` (cron 06:00/19:00 seg–sex) — [ADR 0016](docs/adr/0016-empacotamento-no-lab.md) |
| Paridade | `scripts/parity/` | Linha de base da Fase 7 (comparador e engine sobre mercado congelado) — [ADR 0017](docs/adr/0017-paridade-legado.md) |

## Como rodar

**O app no LAB** (requer a rede `lab-net` com `lab-postgres` e `lab-redis`):

```bash
docker compose up -d --build     # http://127.0.0.1:8080
```

**Desenvolvimento no host:** backend com `backend/.venv` e `runserver`, frontend com Vite
(proxy de `/api` para o backend). Passo a passo, variáveis de ambiente e testes em
[`backend/README.md`](backend/README.md).

```bash
cd backend && .venv/bin/pytest -q                      # suíte do backend
cd frontend && npm test                                # Vitest
cd scripts/parity && ../../backend/.venv/bin/pytest -q # harness de paridade
```

## Estrutura

- `backend/` — projeto Django (`config/`, `apps/{accounts,api,core,market,migration_ir}`, `engine/`).
- `frontend/` — SPA React.
- `scripts/parity/` — linha de base da Fase 7.
- `data/` — CSVs históricos usados pelo `backfill_market_data` (`data/raw/` e `data/static/`).
- `docs/` — documento mestre, ADRs, specs e planos por fase.
- `reports/` e `log/` — saídas locais (ignoradas pelo git).

O histórico do app Streamlit (v3.x) está no git, até o commit `3e8f155`.

## Créditos

Desenvolvido por **Bruno Dias**.
