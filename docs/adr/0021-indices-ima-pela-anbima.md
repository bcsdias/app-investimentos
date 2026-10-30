# ADR 0021 — Índices IMA pela ANBIMA (Histórico Oficial e Eliminação das Projeções por ETF)

- **Status:** Aceita
- **Data:** 2026-10-30
- **Fase:** 7c.1 (Índices IMA pela ANBIMA)

## Contexto

Na validação da Fase 7c e nos diagnósticos de dados de mercado ([ADR 0019 §4](0019-integridade-dos-dados-de-mercado.md)), constatou-se que a família de índices IMA da ANBIMA apresentava sérias distorções no legado e no banco importado:
1. **SGS do BCB descontinuado:** As séries do Banco Central (`bcb:12461`, `12466`, `12467`, `12468`, `12469`) foram paralisadas em 22/05/2023, pois a ANBIMA interrompeu o repasse desses dados ao SGS.
2. **Projeções espúrias por ETF do legado:** O app legado, via fallback em `src/data/sources/market_data.py`, anexava aos CSVs do BCB uma projeção baseada na cotação de ETFs (`LFTS11.SA`, `IMAB11.SA`, `IRFM11.SA`, `B5P211.SA`, `IB5M11.SA`), gravada com `source=BCB`. Esse artifício gerou distorções graves:
   - No `IMA-S` (`LFTS11.SA`), como o ETF só começou a ser negociado em 21/02/2024, houve um buraco de 9 meses onde o índice ficou artificialmente estagnado (~0% entre 05/2023 e 02/2024, contra +9,49% real apurado pela ANBIMA).
   - Em 10/04/2026 (fim do snapshot do legado), o descasamento de duration e taxas acumulou retornos errôneos (IMA-Geral −4,54% e IMA-B −0,48% em relação aos índices reais).
3. **Rótulo invertido em `bcb:12469`:** A série 12469 no SGS é, na verdade, o **IMA-Geral**, mas vinha catalogada e rotulada no sistema como **IMA-S** (o código real do IMA-S no SGS era 12462, que não constava no catálogo).

Era imperativo restabelecer a integridade matemática e histórica dos comparadores de renda fixa sem recorrer a emendas ou proxies.

## Decisão

1. **Fonte oficial ANBIMA em duas camadas (`apps/market/services/sources/anbima.py`):**
   - **Camada 1 (histórico completo):** Leitura direta dos arquivos XLS públicos disponibilizados pela ANBIMA em bucket S3 (`https://s3-data-prd-use1-precos.s3.us-east-1.amazonaws.com/arquivos/indices-historico/{code}-HISTORICO.xls`). Lido via `pandas` com engine `openpyxl`. Suporta as variações de rótulo adotadas pela ANBIMA ao longo do histórico (ex.: `IMA-S` e `IMA - S`, `IMA-GERAL` e `IMA - Geral`).
   - **Camada 2 (fallback diário):** Para janelas curtas (`FALLBACK_MAX_DAYS = 30` dias corridos), em caso de indisponibilidade transitória do S3, consulta diária ao endpoint ASP legado da ANBIMA (`ima-sh-down.asp`), realizando parse do CSV cp1252 com formatação numérica brasileira (`9.086,067150`).
   - Janelas maiores que 30 dias com o arquivo indisponível levantam `FetchError`, sem disparar rajada de requisições web. Se ambas falharem, vigora o mecanismo padrão do ADR 0019 (`ffill` + aviso transparente em `meta.gaps`).
2. **Eliminação completa de projeções por ETF (`ProxySplice`):**
   - Remoção da classe `ProxySplice`, da derivação em `catalog.py`, de `splice_proxy` em `transforms.py` e de seu desvio em `series._materialize`. **Nunca** mais projetar ou emendar índices oficiais por cotas de ETF.
3. **Unificação dos nomes no catálogo:**
   - O catálogo expõe exatamente os 6 índices oficiais: `IMA-S`, `IMA-B`, `IMA-B 5`, `IMA-B 5+`, `IRF-M` e `IMA-Geral`, todos com `source="ANBIMA"`, agrupados sob `"ANBIMA / Tesouro"`.
   - Remoção dos nomes auxiliares com sufixo `(BCB)` e `(proxy)` sem apelidos retroativos. Registra-se esta remoção como **exceção explícita ao ADR 0013** (contrato puramente aditivo), justificada pelo fato de não existirem consumidores externos da API além da própria SPA do projeto.
4. **Exclusão das séries antigas do banco de dados:**
   - Migration de dados `core.0003_remove_bcb_ima_series.py` remove em definitivo os registros `bcb:12461`, `bcb:12466`, `bcb:12467`, `bcb:12468` e `bcb:12469` de `MarketSeries`, chamando `bump_series_generation()` para invalidar qualquer cache residual.
5. **Dependência runtime:**
   - Inclusão formal de `openpyxl~=3.1` em `backend/requirements.txt`, garantindo compatibilidade nos containers Docker de backend e ingestão.

## Consequências

### Positivas
- **Integridade analítica:** Os índices IMA agora refletem o número-índice oficial da ANBIMA com precisão exata de 6 casas decimais, cobrindo o histórico ininterrupto desde 2000/2003 até D-1.
- **Transparência no catálogo:** O usuário vê um nome limpo para cada índice, sem confusão entre "(BCB)" defasado e "(proxy)" aproximado.
- **Correção da troca de série:** `IMA-Geral` e `IMA-S` agora possuem séries distintas e corretas (`anbima:IMAGERAL` e `anbima:IMAS`).
- **Ingestão automática contínua:** O cron de ingestão existente (06h e 19h seg-sex) mantém as séries da ANBIMA atualizadas sem intervenção manual.
- **Zero lacunas:** Diagnóstico `market_gaps` e chamadas de API com a carteira real mostram 0 lacunas e zero advertências espúrias no período 2024+.

### Neutras / Atenções
- Os valores de rentabilidade dos benchmarks IMA sofrem alteração na SPA em relação ao legado (agora com os dados verdadeiros).
- A matriz de paridade congelada (`scripts/parity/cases.py`) permanece 100% verde e inalterada (`=`), pois os 4 benchmarks de regressão (`CDI`, `IPCA + 6%`, `S&P 500 BRL`, `IFIX`) não dependem dos índices IMA.
- O uso dos dados da ANBIMA segue a política pública de consulta e download gratuito no portal da instituição, adequado para uso pessoal no LAB.
