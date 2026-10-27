import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { formatPlanWarning } from '../components/migration/planWarnings';
import { MonthlyPurchasesTable } from '../components/migration/MonthlyPurchasesTable';
import { migrationApi } from '../api/migration';
import { dlpApi } from '../api/dlp';
import { MigrationPage } from '../pages/MigrationPage';

function plan({ monthWarnings = [], globalWarnings = [], irNotEstimated = [] } = {}) {
  return {
    data: {
      months: [
        {
          month: 1,
          sales: [],
          purchases: [],
          exempt_sales_total: 0,
          taxed_sales_total: 0,
          ir_estimado_total: 0,
          warnings: monthWarnings,
        },
      ],
      totals: { months: 1, sold: 0, bought: 0, ir_estimado: 0, ir_not_estimated: irNotEstimated },
    },
    meta: { monthly_cap: 20000, disclaimer: '', warnings: globalWarnings },
  };
}

describe('Avisos do plano e plano desatualizado', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    window.sessionStorage.clear();
  });

  it('formatPlanWarning traduz códigos com e sem ativo', () => {
    expect(formatPlanWarning('fx_approx:VWRA11')).toMatch(/VWRA11.*PTAX/);
    expect(formatPlanWarning('exceeds_exemption')).toMatch(/R\$ 20\.000/);
    expect(formatPlanWarning('codigo_novo')).toBe('codigo_novo');
  });

  it('MonthlyPurchasesTable não mostra códigos crus e lista IR não estimado', () => {
    render(
      <MonthlyPurchasesTable
        plan={plan({
          monthWarnings: ['exceeds_exemption'],
          globalWarnings: ['cost_basis:CDB X'],
          irNotEstimated: ['FI_X'],
        })}
      />
    );
    expect(screen.queryByText(/exceeds_exemption/)).not.toBeInTheDocument();
    expect(screen.queryByText(/cost_basis/)).not.toBeInTheDocument();
    expect(screen.getByText(/CDB X: saldo calculado pelo custo/)).toBeInTheDocument();
    expect(screen.getByText(/IR não estimado para: FI_X/)).toBeInTheDocument();
  });

  it('mudar as entradas depois de calcular remove o plano exibido', async () => {
    vi.spyOn(dlpApi, 'getTokenStatus').mockResolvedValue({ configured: true });
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
    vi.spyOn(migrationApi, 'calculatePlan').mockResolvedValue(plan());
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={qc}>
        <MigrationPage />
      </QueryClientProvider>
    );

    fireEvent.click(await screen.findByRole('button', { name: /Calcular Plano/i }));
    expect(await screen.findByText(/Cronograma Mensal de Migração/)).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText(/Teto Mensal/), { target: { value: '15000' } });
    await waitFor(() =>
      expect(screen.queryByText(/Cronograma Mensal de Migração/)).not.toBeInTheDocument()
    );
  });
});
