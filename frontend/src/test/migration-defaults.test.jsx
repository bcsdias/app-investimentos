import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { migrationApi } from '../api/migration';
import { dlpApi } from '../api/dlp';
import { MigrationPage } from '../pages/MigrationPage';
import { isTargetListValid } from '../components/migration/TargetAllocationForm';

const WALLET = {
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
    {
      ativo: 'HGLG11',
      classe: 'FII',
      qtd: 10,
      pm: 150,
      price: 160,
      saldo: 1600,
      saldo_source: 'market',
      tax_treatment: 'taxed',
    },
  ],
  meta: { total: 5400, warnings: [] },
};

const EMPTY_PLAN = {
  data: { months: [], totals: { months: 0, sold: 0, bought: 0, ir_estimado: 0, ir_not_estimated: [] } },
  meta: { monthly_cap: 20000, disclaimer: '', warnings: [] },
};

function renderPage() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <MigrationPage />
    </QueryClientProvider>
  );
}

describe('Planejador: padrões e caixa inicial', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    window.sessionStorage.clear();
    vi.spyOn(dlpApi, 'getTokenStatus').mockResolvedValue({ configured: true });
    vi.spyOn(migrationApi, 'getWallet').mockResolvedValue(WALLET);
  });

  it('sem escolha do usuário, todos os papéis vão como ignore (ADR 0015)', async () => {
    const planSpy = vi.spyOn(migrationApi, 'calculatePlan').mockResolvedValue(EMPTY_PLAN);
    renderPage();
    fireEvent.click(await screen.findByRole('button', { name: /Calcular Plano/i }));
    await waitFor(() => expect(planSpy).toHaveBeenCalledTimes(1));
    const actions = planSpy.mock.calls[0][0].holdings.map((h) => h.action);
    expect(actions).toEqual(['ignore', 'ignore']);
  });

  it('HoldingsTable mostra Ignorar selecionado por padrão em papel com cotação', async () => {
    renderPage();
    const select = await screen.findByLabelText(/PETR4/);
    expect(select.value).toBe('ignore');
  });

  it('caixa inicial e apenas aportes entram no payload', async () => {
    const planSpy = vi.spyOn(migrationApi, 'calculatePlan').mockResolvedValue(EMPTY_PLAN);
    renderPage();
    fireEvent.change(await screen.findByLabelText(/Caixa disponível para aportes/), {
      target: { value: '5000' },
    });
    fireEvent.click(screen.getByLabelText(/Apenas aportes/));
    fireEvent.click(screen.getByRole('button', { name: /Calcular Plano/i }));
    await waitFor(() => expect(planSpy).toHaveBeenCalledTimes(1));
    const payload = planSpy.mock.calls[0][0];
    expect(payload.initial_cash).toBe(5000);
    expect(payload.contributions_only).toBe(true);
  });

  it('soma dos pesos usa a mesma tolerância do backend (1e-6)', () => {
    expect(
      isTargetListValid([
        { ticker: 'A', weight: 33.333 },
        { ticker: 'B', weight: 33.333 },
        { ticker: 'C', weight: 33.334 },
      ])
    ).toBe(true);
    expect(
      isTargetListValid([
        { ticker: 'A', weight: 33.333 },
        { ticker: 'B', weight: 33.333 },
        { ticker: 'C', weight: 33.333 },
      ])
    ).toBe(false);
  });
});
