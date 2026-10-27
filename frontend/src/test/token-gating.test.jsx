import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

vi.mock('echarts-for-react', () => ({ default: () => <div data-testid="echarts-mock" /> }));

import { PortfolioFilterProvider } from '../context/PortfolioFilterContext';
import { portfolioApi } from '../api/portfolio';
import { benchmarksApi } from '../api/benchmarks';
import { migrationApi } from '../api/migration';
import { dlpApi } from '../api/dlp';
import { PortfolioPage } from '../pages/PortfolioPage';
import { MigrationPage } from '../pages/MigrationPage';

const FETCHERS = [
  'fetchSummary',
  'fetchTwrEvolution',
  'fetchDrawdown',
  'fetchRollingMetrics',
  'fetchRiskReturn',
  'fetchShadow',
  'fetchYearlySummary',
  'fetchIrrEvolution',
];

function renderWithClient(ui) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={qc}>{ui}</QueryClientProvider>);
}

describe('Gating pelo status do token DLP', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    window.sessionStorage.clear();
    vi.spyOn(dlpApi, 'getTokenStatus').mockResolvedValue({ configured: false });
    vi.spyOn(portfolioApi, 'getFilters').mockResolvedValue({ ativos: [], classes: [] });
    vi.spyOn(benchmarksApi, 'getCatalog').mockResolvedValue({
      groups: [],
      defaults: [],
      synthetic: { bases: [], operadores: [] },
    });
  });

  it('PortfolioPage sem token não chama a API da carteira e mostra CTA que abre o modal', async () => {
    const spies = FETCHERS.map((name) => vi.spyOn(portfolioApi, name).mockResolvedValue({ data: {} }));
    const listener = vi.fn();
    window.addEventListener('open-dlp-modal', listener);

    renderWithClient(
      <PortfolioFilterProvider>
        <PortfolioPage />
      </PortfolioFilterProvider>
    );

    expect(await screen.findByText('Configure seu token DLP')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Configurar token' }));
    window.removeEventListener('open-dlp-modal', listener);

    expect(listener).toHaveBeenCalledTimes(1);
    expect(listener.mock.calls[0][0].detail.reason).toBe('missing');
    spies.forEach((spy) => expect(spy).not.toHaveBeenCalled());
    expect(portfolioApi.getFilters).not.toHaveBeenCalled();
  });

  it('MigrationPage sem token não busca a carteira e mostra o mesmo CTA', async () => {
    const walletSpy = vi.spyOn(migrationApi, 'getWallet').mockResolvedValue({ data: [] });

    renderWithClient(<MigrationPage />);

    expect(await screen.findByText('Configure seu token DLP')).toBeInTheDocument();
    expect(walletSpy).not.toHaveBeenCalled();
  });

  it('PortfolioPage com token configurado consulta a carteira', async () => {
    dlpApi.getTokenStatus.mockResolvedValue({ configured: true });
    const summarySpy = vi.spyOn(portfolioApi, 'fetchSummary').mockResolvedValue({ data: {} });
    FETCHERS.slice(1).forEach((name) => vi.spyOn(portfolioApi, name).mockResolvedValue({ data: {} }));

    renderWithClient(
      <PortfolioFilterProvider>
        <PortfolioPage />
      </PortfolioFilterProvider>
    );

    await waitFor(() => expect(summarySpy).toHaveBeenCalledTimes(1));
    expect(screen.queryByText('Configure seu token DLP')).not.toBeInTheDocument();
  });
});
