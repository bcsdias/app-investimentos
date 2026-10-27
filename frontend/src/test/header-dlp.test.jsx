import React from 'react';
import { render, screen, waitFor, fireEvent, act } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { MemoryRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

vi.mock('echarts-for-react', () => ({ default: () => <div data-testid="echarts-mock" /> }));

import { AuthProvider } from '../context/AuthContext';
import { authApi } from '../api/auth';
import { dlpApi } from '../api/dlp';
import { api } from '../api/client';
import { Header } from '../components/layout/Header';
import { PortfolioFilterProvider } from '../context/PortfolioFilterContext';
import { PortfolioPage } from '../pages/PortfolioPage';
import { portfolioApi } from '../api/portfolio';
import { benchmarksApi } from '../api/benchmarks';

function renderHeader(queryClient) {
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        <AuthProvider queryClient={queryClient}>
          <Header />
        </AuthProvider>
      </MemoryRouter>
    </QueryClientProvider>
  );
}

describe('Header, DlpTokenModal e Primitivas de UI', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.spyOn(authApi, 'getCsrf').mockResolvedValue({});
    vi.spyOn(authApi, 'getMe').mockResolvedValue({
      id: 1,
      username: 'bruno',
      email: 'bruno@example.com',
      has_dlp_token: true,
    });
  });

  it('exibe badge verde quando configured: true e amarelo quando configured: false', async () => {
    const statusSpy = vi
      .spyOn(dlpApi, 'getTokenStatus')
      .mockResolvedValueOnce({ configured: true, updated_at: '2026-10-26T10:00:00Z' });

    const qc1 = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const { unmount } = renderHeader(qc1);

    await waitFor(() => {
      expect(screen.getByTestId('dlp-status-badge')).toHaveTextContent(/DLP Conectado/i);
    });
    unmount();

    statusSpy.mockResolvedValueOnce({ configured: false, updated_at: null });
    const qc2 = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    renderHeader(qc2);

    await waitFor(() => {
      expect(screen.getByTestId('dlp-status-badge')).toHaveTextContent(/Token DLP Pendente|Não configurado/i);
    });
  });

  it('abre o modal de token ao clicar no badge ou ao receber o evento open-dlp-modal e salva o token', async () => {
    let isConfigured = false;
    vi.spyOn(dlpApi, 'getTokenStatus').mockImplementation(async () => ({
      configured: isConfigured,
      updated_at: isConfigured ? '2026-10-26T12:00:00Z' : null,
    }));
    const saveSpy = vi.spyOn(dlpApi, 'saveToken').mockImplementation(async () => {
      isConfigured = true;
      return { configured: true, updated_at: '2026-10-26T12:00:00Z' };
    });

    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    renderHeader(qc);

    await waitFor(() => {
      expect(screen.getByTestId('dlp-status-badge')).toBeInTheDocument();
    });

    // Dispara evento global de erro DLP 422
    act(() => {
      window.dispatchEvent(
        new CustomEvent('open-dlp-modal', {
          detail: { reason: 'rejected', message: 'Token DLP rejeitado pela API externa.' },
        })
      );
    });

    await waitFor(() => {
      expect(screen.getByRole('dialog')).toBeInTheDocument();
      expect(screen.getByText(/Token DLP rejeitado/i)).toBeInTheDocument();
    });

    const input = screen.getByPlaceholderText(/Cole seu token DLP/i);
    fireEvent.change(input, { target: { value: 'eyJ0b2tlbiI6IjEyMyJ9' } });

    const saveBtn = screen.getByRole('button', { name: /Salvar Token/i });
    fireEvent.click(saveBtn);

    await waitFor(() => {
      expect(saveSpy).toHaveBeenCalledWith('eyJ0b2tlbiI6IjEyMyJ9');
      expect(screen.getByTestId('dlp-status-badge')).toHaveTextContent(/DLP Conectado/i);
    });
  });

  it('botão Atualizar Dados chama POST /portfolio/refresh e invalida queries da carteira', async () => {
    vi.spyOn(dlpApi, 'getTokenStatus').mockResolvedValue({
      configured: true,
      updated_at: '2026-10-26T10:00:00Z',
    });
    const postSpy = vi.spyOn(api, 'post').mockResolvedValue({ status: 204, data: null });

    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const invalidateSpy = vi.spyOn(qc, 'invalidateQueries');

    renderHeader(qc);

    const refreshBtn = await screen.findByRole('button', { name: /Atualizar Dados/i });
    fireEvent.click(refreshBtn);

    await waitFor(() => {
      expect(postSpy).toHaveBeenCalledWith('/portfolio/refresh');
      expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: ['portfolio'] });
    });
  });

  it('trata erro em POST /portfolio/refresh sem vazar promise rejeitada', async () => {
    vi.spyOn(dlpApi, 'getTokenStatus').mockResolvedValue({
      configured: false,
      updated_at: null,
    });
    vi.spyOn(api, 'post').mockRejectedValueOnce(new Error('DLP Token Missing'));

    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    renderHeader(qc);

    const refreshBtn = await screen.findByRole('button', { name: /Atualizar Dados/i });
    fireEvent.click(refreshBtn);

    await waitFor(() => {
      expect(refreshBtn).not.toBeDisabled();
    });
  });

  it('salvar o primeiro token dispara cada análise da carteira uma única vez', async () => {
    let isConfigured = false;
    // latência de rede: o refetch do status não resolve no mesmo tick
    vi.spyOn(dlpApi, 'getTokenStatus').mockImplementation(
      () =>
        new Promise((resolve) =>
          setTimeout(() => resolve({ configured: isConfigured, updated_at: null }), 20)
        )
    );
    vi.spyOn(dlpApi, 'saveToken').mockImplementation(async () => {
      isConfigured = true;
      return { configured: true, updated_at: '2026-10-27T12:00:00Z' };
    });
    vi.spyOn(portfolioApi, 'getFilters').mockResolvedValue({ ativos: [], classes: [] });
    vi.spyOn(benchmarksApi, 'getCatalog').mockResolvedValue({
      groups: [],
      defaults: [],
      synthetic: { bases: [], operadores: [] },
    });
    const slow = () => new Promise((resolve) => setTimeout(() => resolve({ data: {} }), 30));
    const summarySpy = vi.spyOn(portfolioApi, 'fetchSummary').mockImplementation(slow);
    [
      'fetchTwrEvolution',
      'fetchDrawdown',
      'fetchRollingMetrics',
      'fetchRiskReturn',
      'fetchShadow',
      'fetchYearlySummary',
      'fetchIrrEvolution',
    ].forEach((name) => vi.spyOn(portfolioApi, name).mockImplementation(slow));

    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={qc}>
        <MemoryRouter>
          <AuthProvider queryClient={qc}>
            <PortfolioFilterProvider>
              <Header />
              <PortfolioPage />
            </PortfolioFilterProvider>
          </AuthProvider>
        </MemoryRouter>
      </QueryClientProvider>
    );

    fireEvent.click(await screen.findByRole('button', { name: 'Configurar token' }));
    fireEvent.change(await screen.findByPlaceholderText(/Cole seu token DLP/i), {
      target: { value: 'tok-novo' },
    });
    fireEvent.click(screen.getByRole('button', { name: /Salvar Token/i }));

    await waitFor(() => expect(summarySpy).toHaveBeenCalled());
    await new Promise((r) => setTimeout(r, 200));
    expect(summarySpy).toHaveBeenCalledTimes(1);
  });
});
