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

  it('falha em /dlp/token-status mostra erro com nova tentativa, não skeleton infinito', async () => {
    const err = new Error('bad gateway');
    err.response = { status: 502, data: { code: 'upstream', message: 'Servidor indisponível' } };
    dlpApi.getTokenStatus.mockRejectedValueOnce(err).mockResolvedValue({ configured: false });

    renderWithClient(
      <PortfolioFilterProvider>
        <PortfolioPage />
      </PortfolioFilterProvider>
    );

    expect(await screen.findByText('Servidor indisponível')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Tentar novamente' }));
    expect(await screen.findByText('Configure seu token DLP')).toBeInTheDocument();
  });

  it('MigrationPage também mostra erro quando /dlp/token-status falha', async () => {
    const err = new Error('bad gateway');
    err.response = { status: 502, data: { code: 'upstream', message: 'Servidor indisponível' } };
    dlpApi.getTokenStatus.mockRejectedValue(err);

    renderWithClient(<MigrationPage />);

    expect(await screen.findByText('Servidor indisponível')).toBeInTheDocument();
  });

  it('token rejeitado mostra um único estado e não abre o modal', async () => {
    dlpApi.getTokenStatus.mockResolvedValue({ configured: true });
    const err = new Error('Token rejeitado');
    err.response = {
      status: 422,
      data: { code: 'dlp_token_rejected', message: 'Token DLP rejeitado pela API externa.' },
    };
    FETCHERS.forEach((name) => vi.spyOn(portfolioApi, name).mockRejectedValue(err));

    const listener = vi.fn();
    window.addEventListener('open-dlp-modal', listener);

    renderWithClient(
      <PortfolioFilterProvider>
        <PortfolioPage />
      </PortfolioFilterProvider>
    );

    expect(await screen.findByText(/Token DLP rejeitado/)).toBeInTheDocument();
    expect(screen.getAllByText(/Token DLP rejeitado/).length).toBe(1);
    expect(screen.queryByRole('dialog')).toBeNull();

    fireEvent.click(screen.getByRole('button', { name: /Atualizar token/i }));
    expect(listener).toHaveBeenCalledTimes(1);
    expect(listener.mock.calls[0][0].detail.reason).toBe('dlp_token_rejected');

    window.removeEventListener('open-dlp-modal', listener);
  });

  it('PeriodNotice some sem token configurado', async () => {
    dlpApi.getTokenStatus.mockResolvedValue({ configured: false });
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    qc.setQueryData(['portfolio', 'summary', { start: null, end: null, ativos: [], classes: [] }], {
      data: {},
      meta: { start: '2025-01-01', end: '2026-01-01' },
    });

    render(
      <QueryClientProvider client={qc}>
        <PortfolioFilterProvider>
          <PortfolioPage />
        </PortfolioFilterProvider>
      </QueryClientProvider>
    );

    expect(await screen.findByText('Configure seu token DLP')).toBeInTheDocument();
    expect(screen.queryByText(/Período analisado/)).not.toBeInTheDocument();
  });

  it('MigrationPage com token rejeitado mostra estado de token e não abre modal', async () => {
    dlpApi.getTokenStatus.mockResolvedValue({ configured: true });
    const err = new Error('Token rejeitado');
    err.response = {
      status: 422,
      data: { code: 'dlp_token_rejected', message: 'Token DLP rejeitado pela API externa.' },
    };
    vi.spyOn(migrationApi, 'getWallet').mockRejectedValue(err);

    const listener = vi.fn();
    window.addEventListener('open-dlp-modal', listener);

    renderWithClient(<MigrationPage />);

    expect(await screen.findByText(/Token DLP rejeitado/)).toBeInTheDocument();
    expect(screen.getAllByText(/Token DLP rejeitado/).length).toBe(1);
    expect(screen.queryByRole('dialog')).toBeNull();

    window.removeEventListener('open-dlp-modal', listener);
  });
});
