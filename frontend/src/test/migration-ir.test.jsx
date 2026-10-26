import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { migrationApi } from '../api/migration';
import { HoldingsTable } from '../components/migration/HoldingsTable';
import { TargetAllocationForm } from '../components/migration/TargetAllocationForm';
import { MonthlyPurchasesTable } from '../components/migration/MonthlyPurchasesTable';
import { ExportPdfButton } from '../components/migration/ExportPdfButton';
import { MigrationPage } from '../pages/MigrationPage';

describe('Módulo de Migração de IR e Exportação PDF', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    window.sessionStorage.clear();
  });

  it('HoldingsTable renderiza badges fiscais para ACAO (exempt_20k) e FII (taxed) e permite saldo manual em saldo_source="cost"', () => {
    const items = [
      {
        ativo: 'PETR4',
        classe: 'ACAO',
        qtd: 100,
        pm: 30.0,
        price: 38.0,
        saldo: 3800.0,
        saldo_source: 'market',
        tax_treatment: 'exempt_20k',
      },
      {
        ativo: 'HGLG11',
        classe: 'FII',
        qtd: 20,
        pm: 155.0,
        price: 160.0,
        saldo: 3200.0,
        saldo_source: 'market',
        tax_treatment: 'taxed',
      },
      {
        ativo: 'CDB BANCO X',
        classe: 'RENDA FIXA',
        qtd: 1,
        pm: 5000.0,
        price: null,
        saldo: 5000.0,
        saldo_source: 'cost',
        tax_treatment: 'not_estimated',
      },
    ];

    const onChangeAction = vi.fn();
    const onChangeSaldo = vi.fn();

    render(
      <HoldingsTable
        holdings={items}
        targets={[{ ticker: 'BOVA11', weight: 100 }]}
        actionsMap={{}}
        onChangeAction={onChangeAction}
        onChangeManualSaldo={onChangeSaldo}
      />
    );

    expect(screen.getByText(/Isento R\$ 20k/i)).toBeInTheDocument();
    expect(screen.getByText(/Tributado/i)).toBeInTheDocument();
    expect(screen.getByText(/Não estimado/i)).toBeInTheDocument();

    const selectPetr = screen.getByLabelText('Ação para PETR4');
    fireEvent.change(selectPetr, { target: { value: 'count_as:BOVA11' } });
    expect(onChangeAction).toHaveBeenCalledWith('PETR4', 'count_as', 'BOVA11');

    const manualSaldoInput = screen.getByLabelText('Saldo manual para CDB BANCO X');
    fireEvent.change(manualSaldoInput, { target: { value: '5450' } });
    expect(onChangeSaldo).toHaveBeenCalledWith('CDB BANCO X', 5450);
  });

  it('MigrationPage renderiza EmptyState orientativo quando getWallet retorna 404 portfolio_empty', async () => {
    const err = new Error('Not found');
    err.response = {
      status: 404,
      data: { code: 'portfolio_empty', message: 'Nenhum papel com saldo encontrado.' },
    };
    vi.spyOn(migrationApi, 'getWallet').mockRejectedValue(err);

    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={qc}>
        <MigrationPage />
      </QueryClientProvider>
    );

    await waitFor(() => {
      expect(screen.getByTestId('empty-state')).toBeInTheDocument();
      expect(screen.getByText(/Nenhum papel encontrado na custódia/i)).toBeInTheDocument();
    });
  });

  it('TargetAllocationForm desabilita botão de simular se a soma dos pesos for diferente de 100%', () => {
    const onSimulate = vi.fn();
    const onUpdateTargets = vi.fn();

    const { rerender } = render(
      <TargetAllocationForm
        targets={[
          { ticker: 'IVVB11', weight: 60 },
          { ticker: 'BOVA11', weight: 20 },
        ]}
        onChangeTargets={onUpdateTargets}
        monthlyCap={20000}
        onChangeMonthlyCap={vi.fn()}
        alreadySold={0}
        onChangeAlreadySold={vi.fn()}
        onSimulate={onSimulate}
        isSimulating={false}
      />
    );

    const btnSimulate = screen.getByRole('button', { name: /Calcular Plano/i });
    expect(btnSimulate).toBeDisabled();
    expect(screen.getByText(/80,0%|80%/i)).toBeInTheDocument();

    rerender(
      <TargetAllocationForm
        targets={[
          { ticker: 'IVVB11', weight: 60 },
          { ticker: 'BOVA11', weight: 40 },
        ]}
        onChangeTargets={onUpdateTargets}
        monthlyCap={20000}
        onChangeMonthlyCap={vi.fn()}
        alreadySold={0}
        onChangeAlreadySold={vi.fn()}
        onSimulate={onSimulate}
        isSimulating={false}
      />
    );

    expect(screen.getByRole('button', { name: /Calcular Plano/i })).not.toBeDisabled();
  });

  it('MonthlyPurchasesTable exibe vendas agrupadas por mês e alertas fiscais', () => {
    const planData = {
      data: {
        totals: {
          months: 2,
          sold: 35000,
          bought: 35000,
          ir_estimado: 300,
          ir_not_estimated: [],
        },
        months: [
          {
            month: 1,
            exempt_sales_total: 19500,
            taxed_sales_total: 2000,
            cash_in: 21500,
            purchase_total: 21500,
            ir_estimado_total: 300,
            warnings: ['exceeds_exemption: Venda unitária acima do limite mensal'],
            sales: [
              {
                ativo: 'PETR4',
                classe: 'ACAO',
                qtd: 500,
                valor: 19500,
                tax_treatment: 'exempt_20k',
                lucro_estimado: 4000,
                ir_estimado: 0,
              },
              {
                ativo: 'HGLG11',
                classe: 'FII',
                qtd: 12,
                valor: 2000,
                tax_treatment: 'taxed',
                lucro_estimado: 1500,
                ir_estimado: 300,
              },
            ],
            purchases: [
              {
                ticker: 'IVVB11',
                valor: 21500,
                price: 300,
                qtd_sugerida: 71,
                saldo_final: 21300,
                participacao_final: 100,
              },
            ],
          },
        ],
      },
      meta: {
        monthly_cap: 20000,
        disclaimer: 'Simulação meramente informativa.',
        warnings: ['exceeds_cap: Ativo indivisível excedeu teto'],
      },
    };

    render(<MonthlyPurchasesTable plan={planData} />);

    expect(screen.getByText(/Mês 1/i)).toBeInTheDocument();
    expect(screen.getByText('PETR4')).toBeInTheDocument();
    expect(screen.getByText('IVVB11')).toBeInTheDocument();
    expect(screen.getByText(/exceeds_cap/i)).toBeInTheDocument();
    expect(screen.getByText(/exceeds_exemption/i)).toBeInTheDocument();
  });

  it('ExportPdfButton recebe blob de migrationApi.downloadPdf e dispara download no DOM', async () => {
    const fakeBlob = new Blob(['%PDF-1.4 mock'], { type: 'application/pdf' });
    const pdfSpy = vi.spyOn(migrationApi, 'downloadPdf').mockResolvedValue(fakeBlob);
    const createUrlSpy = vi
      .spyOn(window.URL, 'createObjectURL')
      .mockReturnValue('blob:mock-pdf-url');
    const revokeUrlSpy = vi.spyOn(window.URL, 'revokeObjectURL').mockImplementation(() => {});
    const clickSpy = vi
      .spyOn(HTMLAnchorElement.prototype, 'click')
      .mockImplementation(() => {});

    const payload = {
      targets: [{ ticker: 'IVVB11', weight: 100 }],
      holdings: [],
      monthly_cap: 20000,
    };

    render(<ExportPdfButton payload={payload} disabled={false} />);

    const btn = screen.getByRole('button', { name: /Baixar Relatório em PDF/i });
    fireEvent.click(btn);

    await waitFor(() => {
      expect(pdfSpy).toHaveBeenCalledWith(payload);
      expect(createUrlSpy).toHaveBeenCalledWith(fakeBlob);
      expect(clickSpy).toHaveBeenCalledTimes(1);
      expect(revokeUrlSpy).toHaveBeenCalledWith('blob:mock-pdf-url');
    });
  });

  it('alinha planPayload com validate_plan_input (alvo na carteira vira ignore, sem cotação vira ignore e saldo manual só vai em count_as)', async () => {
    vi.spyOn(migrationApi, 'getWallet').mockResolvedValue({
      data: [
        {
          ativo: 'BOVA11',
          classe: 'ETF',
          qtd: 10,
          pm: 100,
          price: 110,
          saldo: 1100,
          saldo_source: 'market',
          tax_treatment: 'taxed',
        },
        {
          ativo: 'CDB BANCO X',
          classe: 'RENDA FIXA',
          qtd: 1,
          pm: 5000,
          price: null,
          saldo: 5000,
          saldo_source: 'cost',
          tax_treatment: 'not_estimated',
        },
      ],
      meta: { total: 6100, warnings: [] },
    });
    const planSpy = vi.spyOn(migrationApi, 'calculatePlan').mockResolvedValue({
      data: { months: [], totals: { months: 0, sold: 0, bought: 0, ir_estimado: 0 } },
      meta: { monthly_cap: 20000, disclaimer: '', warnings: [] },
    });

    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={qc}>
        <MigrationPage />
      </QueryClientProvider>
    );

    const manualInput = await screen.findByLabelText('Saldo manual para CDB BANCO X');
    fireEvent.change(manualInput, { target: { value: '5500' } });

    const calcBtn = screen.getByRole('button', { name: /Calcular Plano/i });
    fireEvent.click(calcBtn);

    await waitFor(() => {
      expect(planSpy).toHaveBeenCalledTimes(1);
    });

    const sentPayload = planSpy.mock.calls[0][0];
    const bovaHolding = sentPayload.holdings.find((h) => h.ativo === 'BOVA11');
    const cdbHolding = sentPayload.holdings.find((h) => h.ativo === 'CDB BANCO X');

    expect(bovaHolding).toEqual({
      ativo: 'BOVA11',
      action: 'ignore',
      target: null,
      saldo: null,
    });
    expect(cdbHolding).toEqual({
      ativo: 'CDB BANCO X',
      action: 'ignore',
      target: null,
      saldo: null,
    });
  });

  it('TargetAllocationForm bloqueia simulação quando há ticker vazio, peso <= 0 ou tickers duplicados mesmo somando 100%', () => {
    const { rerender } = render(
      <TargetAllocationForm
        targets={[
          { ticker: 'IVVB11', weight: 100 },
          { ticker: '', weight: 0 },
        ]}
        onChangeTargets={vi.fn()}
        onSimulate={vi.fn()}
      />
    );
    expect(screen.getByRole('button', { name: /Calcular Plano/i })).toBeDisabled();

    rerender(
      <TargetAllocationForm
        targets={[
          { ticker: 'IVVB11', weight: 50 },
          { ticker: 'IVVB11', weight: 50 },
        ]}
        onChangeTargets={vi.fn()}
        onSimulate={vi.fn()}
      />
    );
    expect(screen.getByRole('button', { name: /Calcular Plano/i })).toBeDisabled();
  });

  it('ExportPdfButton captura erro de downloadPdf e exibe alerta sem unhandled rejection', async () => {
    vi.spyOn(migrationApi, 'downloadPdf').mockRejectedValueOnce(new Error('Falha no PDF'));

    render(
      <ExportPdfButton
        payload={{ targets: [{ ticker: 'IVVB11', weight: 100 }], holdings: [] }}
        disabled={false}
      />
    );

    fireEvent.click(screen.getByRole('button', { name: /Baixar Relatório em PDF/i }));

    await waitFor(() => {
      expect(screen.getByRole('alert')).toHaveTextContent(/Erro ao gerar PDF/i);
    });
  });
});

