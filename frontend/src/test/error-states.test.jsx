import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

vi.mock('echarts-for-react/lib/core', () => ({ default: () => <div data-testid="echarts-mock" /> }));

import { PortfolioFilterProvider } from '../context/PortfolioFilterContext';
import { portfolioApi } from '../api/portfolio';
import { benchmarksApi } from '../api/benchmarks';
import { migrationApi } from '../api/migration';
import { dlpApi } from '../api/dlp';
import { PortfolioPage } from '../pages/PortfolioPage';
import { MigrationPage } from '../pages/MigrationPage';

function apiError(status, code, message) {
  const err = new Error(message);
  err.response = { status, data: { code, message } };
  return err;
}

function renderWithClient(ui) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={qc}>{ui}</QueryClientProvider>);
}

describe('Estados de erro com nova tentativa', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    window.sessionStorage.clear();
    vi.spyOn(dlpApi, 'getTokenStatus').mockResolvedValue({ configured: true });
    vi.spyOn(portfolioApi, 'getFilters').mockResolvedValue({ ativos: [], classes: [] });
    vi.spyOn(benchmarksApi, 'getCatalog').mockResolvedValue({
      groups: [],
      defaults: [],
      synthetic: { bases: [], operadores: [] },
    });
  });

  it('gráfico de drawdown mostra a mensagem do 502 e tenta de novo', async () => {
    [
      'fetchSummary',
      'fetchTwrEvolution',
      'fetchRollingMetrics',
      'fetchRiskReturn',
      'fetchShadow',
      'fetchYearlySummary',
      'fetchIrrEvolution',
    ].forEach((name) => vi.spyOn(portfolioApi, name).mockResolvedValue({ data: {} }));
    const ddSpy = vi
      .spyOn(portfolioApi, 'fetchDrawdown')
      .mockRejectedValue(apiError(502, 'dlp_unavailable', 'DLP indisponível'));

    renderWithClient(
      <PortfolioFilterProvider>
        <PortfolioPage />
      </PortfolioFilterProvider>
    );

    fireEvent.click(await screen.findByRole('tab', { name: /Risco/i }));
    expect(await screen.findByText('DLP indisponível')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Tentar novamente' }));
    await waitFor(() => expect(ddSpy).toHaveBeenCalledTimes(2));
  });

  it('MigrationPage mostra erro da carteira (502) com nova tentativa e sem HoldingsTable', async () => {
    const walletSpy = vi
      .spyOn(migrationApi, 'getWallet')
      .mockRejectedValue(apiError(502, 'dlp_unavailable', 'DLP indisponível'));

    renderWithClient(<MigrationPage />);

    expect(await screen.findByText('DLP indisponível')).toBeInTheDocument();
    expect(screen.queryByText('1. Custódia Atual e Tratamento por Papel')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Tentar novamente' }));
    await waitFor(() => expect(walletSpy).toHaveBeenCalledTimes(2));
  });
});
