import React, { useState } from 'react';
import { usePortfolioQueries } from '../hooks/usePortfolioQueries';
import { FiltersBar } from '../components/portfolio/FiltersBar';
import { PeriodNotice } from '../components/portfolio/PeriodNotice';
import { SummaryCards } from '../components/portfolio/SummaryCards';
import { Tabs } from '../components/common/Tabs';
import { EmptyState } from '../components/common/EmptyState';
import { Skeleton } from '../components/common/Skeleton';
import { DlpTokenRequiredState } from '../components/dlp/DlpTokenRequiredState';
import { TwrChart } from '../components/portfolio/charts/TwrChart';
import { YearlyTable } from '../components/portfolio/tables/YearlyTable';
import { ShadowChart } from '../components/portfolio/charts/ShadowChart';
import { IrrEvolutionChart } from '../components/portfolio/charts/IrrEvolutionChart';
import { DrawdownChart } from '../components/portfolio/charts/DrawdownChart';
import { RollingSharpeChart } from '../components/portfolio/charts/RollingSharpeChart';
import { RiskReturnChart } from '../components/portfolio/charts/RiskReturnChart';

const DASHBOARD_TABS = [
  { id: 'performance', label: 'Performance' },
  { id: 'risco', label: 'Risco' },
];

export function PortfolioPage() {
  const [activeTab, setActiveTab] = useState('performance');
  const { tokenConfigured, summary, twr, drawdown, rolling, riskReturn, shadow, yearly, irr } =
    usePortfolioQueries();

  const isPortfolioEmpty =
    summary.error?.response?.status === 404 &&
    summary.error?.response?.data?.code === 'portfolio_empty';

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-zinc-100">
            Dashboard Analítico da Carteira
          </h1>
          <p className="text-sm text-zinc-400">
            Acompanhamento de performance ponderada pelo tempo, risco e simulação contra indexadores
          </p>
        </div>
        <Tabs tabs={DASHBOARD_TABS} activeTab={activeTab} onChange={setActiveTab} />
      </div>

      <FiltersBar />
      <PeriodNotice meta={summary.data?.meta} />

      {tokenConfigured === false ? (
        <DlpTokenRequiredState />
      ) : tokenConfigured === undefined ? (
        <div className="space-y-4" data-testid="token-status-loading">
          <Skeleton className="h-24 w-full" />
          <Skeleton className="h-80 w-full" />
        </div>
      ) : isPortfolioEmpty ? (
        <EmptyState
          title="Carteira sem movimentações registradas"
          description="Nenhuma movimentação foi encontrada na DLP Invest para os filtros selecionados. Verifique seu token DLP ou limpe os filtros ativos."
        />
      ) : (
        <>
          <SummaryCards
            data={summary.data}
            isLoading={summary.isLoading}
            isError={summary.isError}
            error={summary.error}
            onRetry={summary.refetch}
          />

          {activeTab === 'performance' ? (
            <div className="space-y-6">
              <TwrChart data={twr.data} isLoading={twr.isLoading} error={twr.error} onRetry={twr.refetch} />
              <YearlyTable data={yearly.data} isLoading={yearly.isLoading} error={yearly.error} onRetry={yearly.refetch} />
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <ShadowChart data={shadow.data} isLoading={shadow.isLoading} error={shadow.error} onRetry={shadow.refetch} />
                <IrrEvolutionChart data={irr.data} isLoading={irr.isLoading} error={irr.error} onRetry={irr.refetch} />
              </div>
            </div>
          ) : (
            <div className="space-y-6">
              <DrawdownChart data={drawdown.data} isLoading={drawdown.isLoading} error={drawdown.error} onRetry={drawdown.refetch} />
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <RollingSharpeChart data={rolling.data} isLoading={rolling.isLoading} error={rolling.error} onRetry={rolling.refetch} />
                <RiskReturnChart data={riskReturn.data} isLoading={riskReturn.isLoading} error={riskReturn.error} onRetry={riskReturn.refetch} />
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default PortfolioPage;

