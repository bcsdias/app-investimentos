import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

let lastRenderedOptions = [];

vi.mock('echarts-for-react', () => ({
  default: ({ option }) => {
    lastRenderedOptions.push(option);
    return <div data-testid="echarts-mock" data-option={JSON.stringify(option)} />;
  },
}));

import { PortfolioFilterProvider } from '../context/PortfolioFilterContext';
import { portfolioApi } from '../api/portfolio';
import { benchmarksApi } from '../api/benchmarks';
import { DrawdownChart } from '../components/portfolio/charts/DrawdownChart';
import { RollingSharpeChart } from '../components/portfolio/charts/RollingSharpeChart';
import { RiskReturnChart } from '../components/portfolio/charts/RiskReturnChart';
import { PortfolioPage } from '../pages/PortfolioPage';

describe('Gráficos da Aba Risco e PortfolioPage', () => {
  beforeEach(() => {
    lastRenderedOptions = [];
    vi.restoreAllMocks();
  });

  it('DrawdownChart configura max: 0 e formata o eixo Y em percentual negativo (-15.2%)', () => {
    const ddData = {
      data: {
        series: [
          {
            name: 'Carteira',
            data: [
              ['2025-01-01', 0],
              ['2025-03-01', -0.152],
            ],
          },
        ],
      },
    };

    render(<DrawdownChart data={ddData} isLoading={false} />);
    const opt = lastRenderedOptions[lastRenderedOptions.length - 1];

    expect(opt.yAxis.max).toBe(0);
    expect(opt.yAxis.axisLabel.formatter(-15.2)).toBe('-15.2%');
    expect(opt.series[0].data[1][1]).toBeCloseTo(-15.2, 2);
  });

  it('RollingSharpeChart suporta pontos iniciais null em Sharpe e Volatilidade sem quebrar', () => {
    const rollingData = {
      data: {
        sharpe: [
          {
            name: 'Carteira',
            data: [
              ['2025-01-01', null],
              ['2025-01-02', null],
              ['2025-12-31', 1.35],
            ],
          },
        ],
        volatility: [
          {
            name: 'Carteira',
            data: [
              ['2025-01-01', null],
              ['2025-01-02', null],
              ['2025-12-31', 0.145],
            ],
          },
        ],
      },
    };

    render(<RollingSharpeChart data={rollingData} isLoading={false} />);
    const opt = lastRenderedOptions[lastRenderedOptions.length - 1];

    expect(Array.isArray(opt.yAxis)).toBe(true);
    expect(opt.yAxis).toHaveLength(2);
    expect(opt.series.length).toBeGreaterThanOrEqual(2);
    expect(opt.series[0].data[0]).toEqual(['2025-01-01', null]);
    expect(opt.series[0].data[2]).toEqual(['2025-12-31', 1.35]);
  });

  it('RiskReturnChart monta série do tipo scatter associando nome do ativo/benchmark ao ponto', () => {
    const rrData = {
      data: {
        rows: [
          { name: 'Carteira', volatilidade: 0.14, retorno_cagr: 0.18, sharpe: 0.57 },
          { name: 'IBOV', volatilidade: 0.22, retorno_cagr: 0.11, sharpe: 0.05 },
        ],
      },
    };

    render(<RiskReturnChart data={rrData} isLoading={false} />);
    const opt = lastRenderedOptions[lastRenderedOptions.length - 1];

    expect(opt.series).toHaveLength(2);
    expect(opt.series[0].type).toBe('scatter');
    expect(opt.series[0].name).toBe('Carteira');
    expect(opt.series[0].data[0][0]).toBeCloseTo(14.0, 2);
    expect(opt.series[0].data[0][1]).toBeCloseTo(18.0, 2);
  });

  it('PortfolioPage renderiza aba Performance por padrão e alterna para Risco sem recarregar rede', async () => {
    vi.spyOn(portfolioApi, 'getFilters').mockResolvedValue({ ativos: [], classes: [] });
    vi.spyOn(benchmarksApi, 'getCatalog').mockResolvedValue({
      groups: [],
      defaults: ['CDI', 'IBOV'],
      synthetic: { bases: ['CDI', 'IPCA'], operadores: ['+', '%'] },
    });

    const summarySpy = vi.spyOn(portfolioApi, 'fetchSummary').mockResolvedValue({
      data: {
        patrimonio: 180000,
        aportes_periodo: 150000,
        ganho_periodo: 30000,
        twr_total: 20.0,
        xirr: 15.0,
      },
    });
    const twrSpy = vi
      .spyOn(portfolioApi, 'fetchTwrEvolution')
      .mockResolvedValue({ data: { series: [], final_returns: {} } });
    const ddSpy = vi
      .spyOn(portfolioApi, 'fetchDrawdown')
      .mockResolvedValue({ data: { series: [] } });
    const rollingSpy = vi
      .spyOn(portfolioApi, 'fetchRollingMetrics')
      .mockResolvedValue({ data: { volatility: [], sharpe: [] } });
    const rrSpy = vi
      .spyOn(portfolioApi, 'fetchRiskReturn')
      .mockResolvedValue({ data: { rows: [] } });
    const shadowSpy = vi
      .spyOn(portfolioApi, 'fetchShadow')
      .mockResolvedValue({ data: { series: [] } });
    const yearlySpy = vi
      .spyOn(portfolioApi, 'fetchYearlySummary')
      .mockResolvedValue({ data: { years: [], rows: [] } });
    const irrSpy = vi
      .spyOn(portfolioApi, 'fetchIrrEvolution')
      .mockResolvedValue({ data: { series: [] } });

    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });

    render(
      <QueryClientProvider client={qc}>
        <PortfolioFilterProvider>
          <PortfolioPage />
        </PortfolioFilterProvider>
      </QueryClientProvider>
    );

    await waitFor(() => {
      expect(screen.getByText(/Evolução da Cota \(TWR — Base 100\)/i)).toBeInTheDocument();
      expect(screen.getByText(/Rentabilidade Ano a Ano/i)).toBeInTheDocument();
    });

    expect(summarySpy).toHaveBeenCalledTimes(1);
    expect(ddSpy).toHaveBeenCalledTimes(1);

    const riskTab = screen.getByRole('tab', { name: /Risco/i });
    fireEvent.click(riskTab);

    await waitFor(() => {
      expect(screen.getByText(/Drawdown Subaquático/i)).toBeInTheDocument();
      expect(screen.getByText(/Sharpe e Volatilidade Móvel/i)).toBeInTheDocument();
      expect(screen.getByText(/Dispersão Risco × Retorno/i)).toBeInTheDocument();
    });

    // Nenhuma chamada extra na rede ao trocar de aba (dados já em cache)
    expect(summarySpy).toHaveBeenCalledTimes(1);
    expect(twrSpy).toHaveBeenCalledTimes(1);
    expect(ddSpy).toHaveBeenCalledTimes(1);
    expect(rollingSpy).toHaveBeenCalledTimes(1);
    expect(rrSpy).toHaveBeenCalledTimes(1);
    expect(shadowSpy).toHaveBeenCalledTimes(1);
    expect(yearlySpy).toHaveBeenCalledTimes(1);
    expect(irrSpy).toHaveBeenCalledTimes(1);
  });
});
