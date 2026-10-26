import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('echarts-for-react', () => ({
  default: () => <div data-testid="echarts-mock" />,
}));

import { authApi } from '../api/auth';
import { dlpApi } from '../api/dlp';
import { portfolioApi } from '../api/portfolio';
import { benchmarksApi } from '../api/benchmarks';
import { migrationApi } from '../api/migration';
import { App } from '../App';

describe('Roteamento Principal, LoginPage e Fluxos de Tela', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    window.history.pushState({}, '', '/');

    vi.spyOn(authApi, 'getCsrf').mockResolvedValue({});
    vi.spyOn(dlpApi, 'getTokenStatus').mockResolvedValue({
      configured: true,
      updated_at: '2026-10-26T10:00:00Z',
    });
    vi.spyOn(portfolioApi, 'getFilters').mockResolvedValue({ ativos: [], classes: [] });
    vi.spyOn(benchmarksApi, 'getCatalog').mockResolvedValue({
      groups: [],
      defaults: ['CDI', 'IBOV'],
      synthetic: { bases: ['CDI', 'IPCA'], operadores: ['+', '%'] },
    });
    vi.spyOn(portfolioApi, 'fetchSummary').mockResolvedValue({
      data: { patrimonio: 100000, twr_total: 10, xirr: 12 },
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
    vi.spyOn(migrationApi, 'getWallet').mockResolvedValue({
      data: [
        {
          ativo: 'PETR4',
          classe: 'ACAO',
          qtd: 100,
          pm: 30,
          price: 38,
          saldo: 3800,
          saldo_source: 'market',
          tax_treatment: 'exempt_20k',
        },
      ],
      meta: { total: 3800, warnings: [] },
    });
  });

  it('redireciona para /login sem autenticação, realiza login, navega para /migracao-ir e faz logout', async () => {
    vi.spyOn(authApi, 'getMe').mockRejectedValueOnce({ response: { status: 401 } });
    vi.spyOn(authApi, 'login').mockResolvedValue({
      id: 1,
      username: 'investidor',
      email: 'inv@example.com',
    });
    vi.spyOn(authApi, 'logout').mockResolvedValue({});

    render(<App />);

    // 1. Sem sessão -> redireciona para /login
    await waitFor(() => {
      expect(screen.getByRole('heading', { name: /Acesse sua Conta/i })).toBeInTheDocument();
    });

    // 2. Preenche e submete o login
    fireEvent.change(screen.getByLabelText(/Usuário/i), {
      target: { value: 'investidor' },
    });
    fireEvent.change(screen.getByLabelText(/Senha/i), {
      target: { value: 'segredo123' },
    });
    fireEvent.click(screen.getByRole('button', { name: /Entrar/i }));

    // 3. Redireciona para / e renderiza Header + PortfolioPage
    await waitFor(() => {
      expect(
        screen.getByRole('heading', { name: /Dashboard Analítico da Carteira/i })
      ).toBeInTheDocument();
    });

    // 4. Navega pelo Header para /migracao-ir
    fireEvent.click(screen.getByRole('link', { name: /Planejador de IR/i }));

    await waitFor(() => {
      expect(
        screen.getByRole('heading', { name: /Planejador de Migração com Isenção de IR/i })
      ).toBeInTheDocument();
    });

    // 5. Clica em Sair -> limpa sessão e retorna para /login
    fireEvent.click(screen.getByRole('button', { name: /Sair/i }));

    await waitFor(() => {
      expect(screen.getByRole('heading', { name: /Acesse sua Conta/i })).toBeInTheDocument();
    });
  });

  it('exibe mensagem de erro quando login retorna 401', async () => {
    vi.spyOn(authApi, 'getMe').mockRejectedValue({ response: { status: 401 } });
    vi.spyOn(authApi, 'login').mockRejectedValue({
      response: {
        status: 401,
        data: { code: 'invalid_credentials', message: 'Usuário ou senha inválidos.' },
      },
    });

    render(<App />);

    await waitFor(() => {
      expect(screen.getByRole('heading', { name: /Acesse sua Conta/i })).toBeInTheDocument();
    });

    fireEvent.change(screen.getByLabelText(/Usuário/i), {
      target: { value: 'errado' },
    });
    fireEvent.change(screen.getByLabelText(/Senha/i), {
      target: { value: 'errado' },
    });
    fireEvent.click(screen.getByRole('button', { name: /Entrar/i }));

    await waitFor(() => {
      expect(screen.getByText(/Usuário ou senha inválidos/i)).toBeInTheDocument();
    });
  });
});
