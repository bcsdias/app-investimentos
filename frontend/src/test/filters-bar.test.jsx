import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import {
  PortfolioFilterProvider,
  usePortfolioFilters,
  toLocalISODate,
} from '../context/PortfolioFilterContext';
import { FiltersBar } from '../components/portfolio/FiltersBar';
import { benchmarksApi } from '../api/benchmarks';
import { assetsApi } from '../api/assets';
import { portfolioApi } from '../api/portfolio';

function FilterStateInspector() {
  const { period, start, end, benchmarks, synthetics, ativo, classe } = usePortfolioFilters();
  return (
    <div data-testid="filter-inspector">
      <span data-testid="period-val">{period}</span>
      <span data-testid="start-val">{start ?? 'null'}</span>
      <span data-testid="end-val">{end ?? 'null'}</span>
      <span data-testid="benchmarks-val">{JSON.stringify(benchmarks)}</span>
      <span data-testid="synthetics-val">{JSON.stringify(synthetics)}</span>
      <span data-testid="ativo-val">{ativo ?? 'null'}</span>
      <span data-testid="classe-val">{classe ?? 'null'}</span>
    </div>
  );
}

function renderFiltersBar() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <PortfolioFilterProvider>
        <FiltersBar />
        <FilterStateInspector />
      </PortfolioFilterProvider>
    </QueryClientProvider>
  );
}

describe('PortfolioFilterContext, FiltersBar e BenchmarkSelector', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.spyOn(portfolioApi, 'getFilters').mockResolvedValue({
      ativos: ['PETR4', 'VALE3'],
      classes: ['ACAO', 'FII'],
    });
    vi.spyOn(benchmarksApi, 'getCatalog').mockResolvedValue({
      groups: [
        { name: 'Renda Fixa / Índices BCB', items: ['CDI', 'IPCA'] },
        { name: 'Índices B3', items: ['IBOV', 'IFIX'] },
      ],
      defaults: ['CDI', 'IBOV'],
      synthetic: { bases: ['IPCA', 'CDI'], operadores: ['+', '%'] },
    });
    vi.spyOn(assetsApi, 'searchAssets').mockResolvedValue([
      {
        sigla: 'AAPL',
        nome: 'Apple Inc.',
        classe: 'STOCK',
        mercado: 'US',
        yf_ticker: 'AAPL',
        brl_variant: true,
      },
    ]);
  });

  it('usa benchmarks padrão válidos no catálogo do backend (CDI e BOVA11, sem IBOV inválido)', () => {
    renderFiltersBar();
    const benchmarksList = JSON.parse(screen.getByTestId('benchmarks-val').textContent);
    expect(benchmarksList).toEqual(['CDI', 'BOVA11']);
    expect(benchmarksList).not.toContain('IBOV');
  });

  it('calcula start de 365 dias atrás e end atual ao selecionar 1A, e 1900-01-01 ao selecionar Tudo', async () => {
    renderFiltersBar();

    const btn1A = screen.getByRole('button', { name: '1A' });
    fireEvent.click(btn1A);

    const today = new Date();
    const expectedEnd = toLocalISODate(today);
    const past = new Date(today);
    past.setDate(past.getDate() - 365);
    const expectedStart = toLocalISODate(past);

    expect(screen.getByTestId('period-val')).toHaveTextContent('1A');
    expect(screen.getByTestId('start-val')).toHaveTextContent(expectedStart);
    expect(screen.getByTestId('end-val')).toHaveTextContent(expectedEnd);

    const btnTudo = screen.getByRole('button', { name: 'Tudo' });
    fireEvent.click(btnTudo);

    expect(screen.getByTestId('period-val')).toHaveTextContent('Tudo');
    expect(screen.getByTestId('start-val')).toHaveTextContent('1900-01-01');
    expect(screen.getByTestId('end-val')).toHaveTextContent('null');
  });

  it('adiciona benchmark do catálogo à lista benchmarks', async () => {
    renderFiltersBar();

    const openBtn = screen.getByRole('button', { name: /Comparadores|Benchmarks/i });
    fireEvent.click(openBtn);

    const ifixCheckbox = await screen.findByLabelText('IFIX');
    fireEvent.click(ifixCheckbox);

    const benchmarksList = JSON.parse(screen.getByTestId('benchmarks-val').textContent);
    expect(benchmarksList).toContain('IFIX');
  });

  it('busca ativo e adiciona com sufixo BRL quando Versão em BRL está marcada', async () => {
    renderFiltersBar();

    const openBtn = screen.getByRole('button', { name: /Comparadores|Benchmarks/i });
    fireEvent.click(openBtn);

    const tabTicker = screen.getByRole('button', { name: /Buscar Ticker/i });
    fireEvent.click(tabTicker);

    const searchInput = screen.getByPlaceholderText(/Digite ticker ou nome/i);
    fireEvent.change(searchInput, { target: { value: 'AAPL' } });

    await waitFor(() => {
      expect(assetsApi.searchAssets).toHaveBeenCalledWith('AAPL');
    });

    const brlCheckbox = await screen.findByLabelText(/Versão em BRL/i);
    fireEvent.click(brlCheckbox);

    const addAssetBtn = await screen.findByRole('button', { name: /Adicionar AAPL/i });
    fireEvent.click(addAssetBtn);

    const benchmarksList = JSON.parse(screen.getByTestId('benchmarks-val').textContent);
    expect(benchmarksList).toContain('AAPL BRL');
  });

  it('adiciona índice sintético estruturado {base: "CDI", operador: "+", valor: 2}', async () => {
    renderFiltersBar();

    const openBtn = screen.getByRole('button', { name: /Comparadores|Benchmarks/i });
    fireEvent.click(openBtn);

    const tabSynthetic = screen.getByRole('button', { name: /Sintético/i });
    fireEvent.click(tabSynthetic);

    const baseSelect = screen.getByLabelText(/Índice Base/i);
    fireEvent.change(baseSelect, { target: { value: 'CDI' } });

    const opSelect = screen.getByLabelText(/Operador/i);
    fireEvent.change(opSelect, { target: { value: '+' } });

    const valInput = screen.getByLabelText(/Taxa \/ Percentual/i);
    fireEvent.change(valInput, { target: { value: '2' } });

    const addSynthBtn = screen.getByRole('button', { name: /Adicionar Sintético/i });
    fireEvent.click(addSynthBtn);

    const syntheticsList = JSON.parse(screen.getByTestId('synthetics-val').textContent);
    expect(syntheticsList).toContainEqual({ base: 'CDI', operador: '+', valor: 2 });
  });
});

