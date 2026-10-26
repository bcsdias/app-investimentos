import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

vi.mock('echarts-for-react', () => ({
  default: ({ option }) => (
    <div data-testid="echarts-mock" data-option={JSON.stringify(option)} />
  ),
}));

import { PortfolioFilterProvider } from '../context/PortfolioFilterContext';
import { portfolioApi } from '../api/portfolio';
import { usePortfolioQueries } from '../hooks/usePortfolioQueries';
import { TwrChart } from '../components/portfolio/charts/TwrChart';
import { YearlyTable } from '../components/portfolio/tables/YearlyTable';
import { ShadowChart } from '../components/portfolio/charts/ShadowChart';
import { IrrEvolutionChart } from '../components/portfolio/charts/IrrEvolutionChart';

function QueriesInspector() {
  const queries = usePortfolioQueries();
  return (
    <div data-testid="queries-inspector">
      {queries.summary.isSuccess ? 'summary-ok' : 'summary-loading'}
    </div>
  );
}

describe('usePortfolioQueries e Componentes da Aba Performance', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('usePortfolioQueries dispara as 8 queries analíticas e propaga o AbortController.signal', async () => {
    let capturedSignal = null;
    vi.spyOn(portfolioApi, 'fetchSummary').mockImplementation(async (_payload, signal) => {
      capturedSignal = signal;
      return { data: { patrimonio: 100000 } };
    });
    vi.spyOn(portfolioApi, 'fetchTwrEvolution').mockResolvedValue({ data: { series: [] } });
    vi.spyOn(portfolioApi, 'fetchDrawdown').mockResolvedValue({ data: { series: [] } });
    vi.spyOn(portfolioApi, 'fetchRollingMetrics').mockResolvedValue({
      data: { volatility: [], sharpe: [] },
    });
    vi.spyOn(portfolioApi, 'fetchRiskReturn').mockResolvedValue({ data: { rows: [] } });
    vi.spyOn(portfolioApi, 'fetchShadow').mockResolvedValue({ data: { series: [] } });
    vi.spyOn(portfolioApi, 'fetchYearlySummary').mockResolvedValue({
      data: { years: [], rows: [] },
    });
    vi.spyOn(portfolioApi, 'fetchIrrEvolution').mockResolvedValue({ data: { series: [] } });

    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={qc}>
        <PortfolioFilterProvider>
          <QueriesInspector />
        </PortfolioFilterProvider>
      </QueryClientProvider>
    );

    await waitFor(() => {
      expect(screen.getByTestId('queries-inspector')).toHaveTextContent('summary-ok');
    });

    expect(capturedSignal).toBeDefined();
    expect(typeof capturedSignal.aborted).toBe('boolean');
    expect(portfolioApi.fetchTwrEvolution).toHaveBeenCalledTimes(1);
    expect(portfolioApi.fetchDrawdown).toHaveBeenCalledTimes(1);
    expect(portfolioApi.fetchRollingMetrics).toHaveBeenCalledTimes(1);
    expect(portfolioApi.fetchRiskReturn).toHaveBeenCalledTimes(1);
    expect(portfolioApi.fetchShadow).toHaveBeenCalledTimes(1);
    expect(portfolioApi.fetchYearlySummary).toHaveBeenCalledTimes(1);
    expect(portfolioApi.fetchIrrEvolution).toHaveBeenCalledTimes(1);
  });

  it('TwrChart monta configuração ECharts com xAxis time, dataZoom e lida com valores nulos', () => {
    const twrData = {
      data: {
        series: [
          {
            name: 'Carteira',
            data: [
              ['2025-01-01', null],
              ['2025-01-02', 100.0],
              ['2025-06-01', 142.5],
            ],
          },
          {
            name: 'CDI',
            data: [
              ['2025-01-01', 100.0],
              ['2025-06-01', 105.2],
            ],
          },
        ],
        final_returns: { Carteira: 42.5, CDI: 5.2 },
      },
    };

    render(<TwrChart data={twrData} isLoading={false} />);
    const chartEl = screen.getByTestId('echarts-mock');
    const option = JSON.parse(chartEl.getAttribute('data-option'));

    expect(option.xAxis.type).toBe('time');
    expect(Array.isArray(option.dataZoom)).toBe(true);
    expect(option.series).toHaveLength(2);
    expect(option.series[0].name).toBe('Carteira');
    expect(option.series[0].data[0]).toEqual(['2025-01-01', null]);
  });

  it('YearlyTable renderiza cabeçalho com anos e células com percentuais coloridos', () => {
    const yearlyData = {
      data: {
        years: [2024, 2025],
        rows: [
          {
            name: 'Carteira',
            values: { 2024: 0.154, 2025: -0.032 },
            total: 0.117,
          },
          {
            name: 'CDI',
            values: { 2024: 0.108, 2025: 0.112 },
            total: 0.232,
          },
        ],
      },
    };

    render(<YearlyTable data={yearlyData} isLoading={false} />);

    expect(screen.getByText('2024')).toBeInTheDocument();
    expect(screen.getByText('2025')).toBeInTheDocument();
    expect(screen.getByText('Carteira')).toBeInTheDocument();

    const posCell = screen.getByText('+15,40%');
    expect(posCell.className).toMatch(/emerald/);

    const negCell = screen.getByText('-3,20%');
    expect(negCell.className).toMatch(/red/);
  });

  it('ShadowChart e IrrEvolutionChart renderizam séries temporais sem erros com pontos nulos', () => {
    const shadowData = {
      data: {
        series: [
          {
            name: 'Carteira Real',
            data: [
              ['2025-01-01', 100000],
              ['2025-02-01', 110000],
            ],
          },
        ],
      },
    };
    const irrData = {
      data: {
        series: [
          {
            name: 'TIR',
            data: [
              ['2025-01-31', null],
              ['2025-02-28', 14.5],
            ],
          },
        ],
      },
    };

    render(
      <div>
        <ShadowChart data={shadowData} isLoading={false} />
        <IrrEvolutionChart data={irrData} isLoading={false} />
      </div>
    );

    const charts = screen.getAllByTestId('echarts-mock');
    expect(charts).toHaveLength(2);
  });
});

