# Plano de Implementação: Arquitetura Django Ninja + React (Docker LAB)

## Goal Description
O objetivo deste plano é migrar e desacoplar completamente o projeto `app-investimentos` na branch `lab-dev`, substituindo a estrutura anterior em Streamlit e dependências em nuvem (Supabase, Upstash) por um ecossistema moderno, autônomo e de alta performance no **Docker LAB**:
1. **Banco de Dados Local:** PostgreSQL 16 (`lab-postgres`) para tokens criptografados e séries temporais.
2. **Cache Local:** Redis 7 (`lab-redis`) para cotações e séries de mercado.
3. **Backend API:** Django 5 com **Django Ninja** (REST API rápida, tipagem Pydantic, Swagger automático em `/api/docs` e reaproveitamento de 100% dos cálculos em Python).
4. **Frontend SPA:** **React** (com Vite, Tailwind CSS, Lucide Icons e **Apache ECharts**) consumindo a API em JSON.
5. **Rotina de Ingestão:** Comando Django (`manage.py ingest_market_data`) para atualização diária automatizada de BCB, Tesouro Direto e Yahoo Finance.

```mermaid
flowchart TD
    subgraph Browser_Client [Navegador do Usuário]
        ReactApp["React SPA (Vite + Tailwind + ECharts)<br/>:80 / :5173"]
    end

    subgraph Reverse_Proxy [Roteador Local]
        Nginx["Nginx Reverse Proxy (:80 / :8080)"]
    end

    subgraph Backend_Django [Backend Django + Django Ninja (:8000)]
        API["Django Ninja Router (/api/v1/...)"]
        Swagger["Swagger UI Docs (/api/docs)"]
        Admin["Django Admin (/admin)"]
        Engine["Engine Financeiro (TWR, TIR, Sharpe)"]
        Ingestion["Management Command (ingest_market_data)"]
    end

    subgraph Persistence [Infraestrutura Docker LAB]
        Postgres[("lab-postgres (PostgreSQL 16)")]
        Redis[("lab-redis (Redis 7 Alpine)")]
    end

    subgraph External_APIs [Provedores Externos de Mercado]
        DLP["DLP API (Carteira do Usuário)"]
        BCB["BCB SGS (IPCA, CDI, SELIC)"]
        TD["Tesouro Transparente"]
        YF["Yahoo Finance (Ações / FIIs / ETFs)"]
    end

    ReactApp -->|"Requisições HTTP (JSON)"| Nginx
    Nginx -->|"/api/* e /admin/*"| Backend_Django
    Nginx -->|"/*"| ReactApp

    API --> Engine
    Engine --> Postgres
    Engine --> Redis
    Engine --> DLP

    Ingestion --> BCB
    Ingestion --> TD
    Ingestion --> YF
    Ingestion --> Postgres
    Ingestion --> Redis
```

---

## User Review Required

> [!IMPORTANT]
> **Branch de Trabalho:**
> Todas as alterações serão implementadas e testadas exclusivamente na branch **`lab-dev`**, preservando intacta a branch `origin/dev` e a `origin/main`.

> [!TIP]
> **Acesso aos Ambientes durante o Desenvolvimento:**
> - **Frontend React:** `http://localhost:5173` (ou porta mapeada no LAB)
> - **Documentação Interativa da API (Swagger):** `http://localhost:8000/api/docs`
> - **Painel Administrativo do Django:** `http://localhost:8000/admin`

---

## Proposed Changes

A organização de arquivos será dividida em **Backend (Django)**, **Frontend (React)** e **Orquestração Docker**:

```
/data/projetos/app-investimentos/
├── docker-compose.yml                    [NEW/MODIFY] Orquestração do Backend, Redis, Postgres e Nginx
├── backend/
│   ├── Dockerfile                        [NEW] Container Python para o Django Backend
│   ├── requirements.txt                  [NEW] Dependências Django, Ninja, psycopg2, redis, pandas, etc.
│   ├── manage.py                         [NEW] Entrypoint do Django
│   ├── config/
│   │   ├── __init__.py                   [NEW]
│   │   ├── settings.py                   [NEW] Configurações de banco, cache, CORS, secret keys
│   │   ├── urls.py                       [NEW] Roteamento principal (/api/, /admin/)
│   │   └── wsgi.py                       [NEW]
│   ├── apps/
│   │   ├── core/
│   │   │   ├── models.py                 [NEW] Modelos UserToken e MarketSeries
│   │   │   ├── admin.py                  [NEW] Painel Admin do Django
│   │   │   └── security.py               [NEW] Criptografia Fernet (AES-256)
│   │   ├── api/
│   │   │   ├── router.py                 [NEW] Endpoints REST Django Ninja (/api/v1/...)
│   │   │   └── schemas.py                [NEW] Schemas Pydantic para validação de entrada/saída
│   │   └── market/
│   │       ├── services.py               [NEW] Adaptadores para BCB, Tesouro, Yahoo Finance e DLP
│   │       └── management/commands/
│   │           └── ingest_market_data.py [NEW] Comando agendado de atualização de mercado
│   └── engine/                           [NEW] Módulos financeiros migrados e adaptados
│       ├── financial_report.py
│       ├── twr.py
│       ├── irr.py
│       └── metrics.py
│
└── frontend/
    ├── Dockerfile                        [NEW] Multi-stage build Nginx para produção
    ├── package.json                      [NEW] Dependências React 18, Vite, ECharts, Tailwind, Lucide
    ├── vite.config.js                    [NEW] Configuração do Vite com proxy para o Backend
    ├── tailwind.config.js                [NEW]
    ├── src/
    │   ├── main.jsx                      [NEW] Ponto de entrada do React
    │   ├── App.jsx                       [NEW] Layout principal do Dashboard
    │   ├── api/
    │   │   └── client.js                 [NEW] Cliente HTTP para consumir o Django Ninja
    │   └── components/
    │       ├── Header.jsx                [NEW] Cabeçalho com seletor de usuário e token DLP
    │       ├── SummaryCards.jsx          [NEW] Cards de rentabilidade, CAGR, volatilidade e Sharpe
    │       ├── FiltersBar.jsx            [NEW] Seletor de período (1A, 2A, 5A, Tudo) e benchmarks
    │       ├── charts/
    │       │   ├── TwrChart.jsx          [NEW] Gráfico ECharts de evolução da rentabilidade (Base 100)
    │       │   ├── DrawdownChart.jsx     [NEW] Gráfico ECharts de queda máxima
    │       │   ├── RollingSharpeChart.jsx[NEW] Gráfico ECharts de Sharpe móvel
    │       │   ├── ShadowPortfolioChart.jsx [NEW] Gráfico ECharts de simulação de aportes
    │       │   └── RiskReturnChart.jsx   [NEW] Gráfico ECharts de dispersão Risco x Retorno
    │       └── YearlyTable.jsx           [NEW] Tabela de rentabilidade anual consolidada
```

---

### Componente 1: Backend Django + Django Ninja

#### [NEW] `backend/requirements.txt`
```text
django>=5.0,<5.2
django-ninja>=1.1.0
django-cors-headers>=4.3.0
psycopg2-binary>=2.9.9
redis>=5.0.0
django-redis>=5.4.0
cryptography>=42.0.0
pandas>=2.2.0
numpy>=1.26.0
yfinance>=0.2.36
python-bcb>=0.2.0
requests>=2.31.0
python-dotenv>=1.0.0
gunicorn>=21.2.0
```

#### [NEW] `backend/apps/core/models.py`
```python
from django.db import models
from django.conf import settings
from cryptography.fernet import Fernet

class UserToken(models.Model):
    user_email = models.EmailField(primary_key=True)
    encrypted_token = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def set_token(self, raw_token: str):
        cipher = Fernet(settings.FERNET_KEY.encode())
        self.encrypted_token = cipher.encrypt(raw_token.encode()).decode()

    def get_token(self) -> str:
        cipher = Fernet(settings.FERNET_KEY.encode())
        return cipher.decrypt(self.encrypted_token.encode()).decode()

    def __str__(self):
        return self.user_email

class MarketSeries(models.Model):
    ticker = models.CharField(max_length=100, db_index=True)
    source = models.CharField(max_length=50)  # 'BCB', 'YF', 'TD', 'B3'
    reference_date = models.DateField(db_index=True)
    close_price = models.DecimalField(max_digits=18, decimal_places=6)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('ticker', 'reference_date')
        ordering = ['reference_date']
```

#### [NEW] `backend/apps/api/router.py` (Endpoints Django Ninja)
```python
from ninja import NinjaAPI, Schema
from typing import List, Dict, Optional, Any
from django.shortcuts import get_object_or_404
from apps.core.models import UserToken
from backend.engine.financial_report import FinancialReport
from src.utils.logger import logger

api = NinjaAPI(
    title="App Investimentos API",
    version="1.0.0",
    description="API Financeira de Performance, Benchmarks e Carteiras"
)

class TokenInputSchema(Schema):
    user_email: str
    token: str

@api.post("/auth/token")
def save_token(request, payload: TokenInputSchema):
    user, _ = UserToken.objects.get_or_create(user_email=payload.user_email)
    user.set_token(payload.token)
    user.save()
    return {"success": True, "message": "Token salvo e criptografado com sucesso."}

@api.get("/analytics/portfolio-summary")
def get_portfolio_summary(request, user_email: str, start_date: Optional[str] = None, end_date: Optional[str] = None):
    user = get_object_or_404(UserToken, user_email=user_email)
    raw_token = user.get_token()
    
    report = FinancialReport(logger=logger)
    user_series = report.fetch_user_portfolio(token=raw_token, start_date=start_date, end_date=end_date)
    report.build_dataset(user_series=user_series, start_date=start_date, end_date=end_date)
    
    # Retorna métricas consolidadas em JSON estruturado
    return report.get_summary_metrics_json()

@api.get("/analytics/twr-evolution")
def get_twr_evolution(request, user_email: str, benchmarks: Optional[str] = "CDI,IBOV,S&P 500"):
    user = get_object_or_404(UserToken, user_email=user_email)
    raw_token = user.get_token()
    
    active_benches = [b.strip() for b in benchmarks.split(",")]
    report = FinancialReport(logger=logger)
    user_series = report.fetch_user_portfolio(token=raw_token)
    report.build_dataset(user_series=user_series, active_benchmarks=active_benches)
    
    # Formata séries temporais no padrão ideal para o Apache ECharts
    return report.get_twr_echarts_payload()
```

#### [NEW] `backend/apps/market/management/commands/ingest_market_data.py`
```python
from django.core.management.base import BaseCommand
from django.core.cache import cache
from apps.market.services import sync_bcb_data, sync_tesouro_data, sync_yfinance_data

class Command(BaseCommand):
    help = "Rotina diária de ingestão e atualização dos índices de mercado (BCB, Tesouro, Yahoo)"

    def add_arguments(self, parser):
        parser.add_arguments('--force', action='store_true', help='Força download completo')

    def handle(self, *args, **options):
        self.stdout.write("Iniciando rotina de ingestão de dados de mercado...")
        sync_bcb_data()
        sync_tesouro_data()
        sync_yfinance_data()
        cache.clear()
        self.stdout.write(self.style.SUCCESS("Ingestão concluída e cache atualizado!"))
```

---

### Componente 2: Frontend React SPA (Vite + Tailwind + Apache ECharts)

#### [NEW] `frontend/package.json`
```json
{
  "name": "app-investimentos-frontend",
  "private": true,
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "echarts": "^5.5.0",
    "echarts-for-react": "^3.0.2",
    "lucide-react": "^0.344.0",
    "axios": "^1.6.7",
    "clsx": "^2.1.0",
    "tailwind-merge": "^2.2.1"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.2.1",
    "autoprefixer": "^10.4.18",
    "postcss": "^8.4.35",
    "tailwindcss": "^3.4.1",
    "vite": "^5.1.4"
  }
}
```

#### [NEW] `frontend/src/components/charts/TwrChart.jsx` (Exemplo do componente ECharts interativo)
```jsx
import React from 'react';
import ReactECharts from 'echarts-for-react';

export default function TwrChart({ dates, seriesData, title = "Evolução da Rentabilidade (Base 100)" }) {
  const option = {
    title: {
      text: title,
      textStyle: { color: '#e2e8f0', fontSize: 16, fontWeight: '600' }
    },
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(15, 23, 42, 0.9)',
      borderColor: '#334155',
      textStyle: { color: '#f8fafc' },
      valueFormatter: (value) => `${(value - 100).toFixed(2)}%`
    },
    legend: {
      type: 'scroll',
      top: 30,
      textStyle: { color: '#94a3b8' }
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '15%',
      top: '18%',
      containLabel: true
    },
    dataZoom: [
      { type: 'inside', start: 0, end: 100 },
      { type: 'slider', bottom: 10, borderColor: '#334155', textStyle: { color: '#94a3b8' } }
    ],
    xAxis: {
      type: 'category',
      data: dates,
      axisLine: { lineStyle: { color: '#475569' } }
    },
    yAxis: {
      type: 'value',
      axisLine: { lineStyle: { color: '#475569' } },
      splitLine: { lineStyle: { color: '#334155', type: 'dashed' } },
      axisLabel: { formatter: '{value}' }
    },
    series: seriesData.map(item => ({
      name: item.name,
      type: 'line',
      data: item.data,
      smooth: true,
      showSymbol: false,
      lineStyle: { width: item.name === 'Carteira' ? 3 : 1.5 },
      itemStyle: item.name === 'Carteira' ? { color: '#3b82f6' } : undefined
    }))
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg">
      <ReactECharts option={option} style={{ height: '450px', width: '100%' }} />
    </div>
  );
}
```

---

## Verification Plan

### Testes Automatizados e Comandos de Validação

1. **Validação do Backend Django:**
   ```bash
   cd /data/projetos/app-investimentos/backend
   python manage.py check
   python manage.py makemigrations core
   python manage.py migrate
   ```

2. **Validação da Conexão com PostgreSQL e Redis:**
   ```bash
   python manage.py shell -c "from apps.core.models import UserToken; print('Postgres OK'); from django.core.cache import cache; cache.set('test', 1, 10); print('Redis OK:', cache.get('test'))"
   ```

3. **Validação da Ingestão de Dados:**
   ```bash
   python manage.py ingest_market_data
   ```

4. **Validação dos Endpoints da API via Swagger:**
   - Acessar `http://localhost:8000/api/docs` e executar requisições em `/analytics/portfolio-summary` e `/analytics/twr-evolution`.

5. **Validação do Frontend React:**
   ```bash
   cd /data/projetos/app-investimentos/frontend
   npm install
   npm run build
   ```

### Manual Verification
- Abrir o dashboard no navegador, testar os filtros de períodos (1 ano, 2 anos, início).
- Testar zoom no gráfico ECharts com a roda do mouse e slider inferior.
- Clicar nas legendas dos benchmarks para ligar e desligar índices em tempo real.
- Validar simulação de aportes (Shadow Portfolio) e dispersão Risco x Retorno.
