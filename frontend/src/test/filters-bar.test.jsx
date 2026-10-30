import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';
import {
  PortfolioFilterProvider,
  usePortfolioFilters,
  toLocalISODate,
  parsePeriodParams,
  periodToParams,
} from '../context/PortfolioFilterContext';
import { FiltersBar } from '../components/portfolio/FiltersBar';
import { DateInput } from '../components/common/DateInput';
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

function renderFiltersBar({ initialUrl = '/' } = {}) {
  window.history.pushState({}, '', initialUrl);
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <PortfolioFilterProvider>
          <FiltersBar />
          <FilterStateInspector />
        </PortfolioFilterProvider>
      </BrowserRouter>
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

  it('painel de comparadores fecha com clique fora e com Escape', async () => {
    renderFiltersBar();
    const openBtn = screen.getByRole('button', { name: /Comparadores|Benchmarks/i });

    fireEvent.click(openBtn);
    await screen.findByLabelText('IFIX');
    fireEvent.mouseDown(document.body);
    expect(screen.queryByLabelText('IFIX')).not.toBeInTheDocument();

    fireEvent.click(openBtn);
    await screen.findByLabelText('IFIX');
    fireEvent.mouseDown(screen.getByLabelText('IFIX'));
    expect(screen.getByLabelText('IFIX')).toBeInTheDocument();
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(screen.queryByLabelText('IFIX')).not.toBeInTheDocument();
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

  describe('parsePeriodParams e periodToParams', () => {
    it('mapeia períodos predefinidos e cai em Tudo para valores desconhecidos', () => {
      expect(parsePeriodParams('?period=1a').period).toBe('1A');
      expect(parsePeriodParams('?period=2a').period).toBe('2A');
      expect(parsePeriodParams('?period=5a').period).toBe('5A');
      expect(parsePeriodParams('?period=tudo').period).toBe('Tudo');
      expect(parsePeriodParams('?period=desconhecido').period).toBe('Tudo');
      expect(parsePeriodParams('').period).toBe('Tudo');
    });

    it('faz parse de custom com datas válidas e rejeita datas malformadas ou invertidas', () => {
      const ok = parsePeriodParams('?period=custom&start=2024-01-01&end=2024-12-31');
      expect(ok).toEqual({
        period: 'Personalizado',
        start: '2024-01-01',
        end: '2024-12-31',
      });

      const badStart = parsePeriodParams('?period=custom&start=banana&end=2024-12-31');
      expect(badStart.period).toBe('Tudo');

      const badFormat = parsePeriodParams('?period=custom&start=2024-1-1&end=2024-12-31');
      expect(badFormat.period).toBe('Tudo');

      const inverted = parsePeriodParams('?period=custom&start=2024-12-31&end=2024-01-01');
      expect(inverted.period).toBe('Tudo');

      const missingEnd = parsePeriodParams('?period=custom&start=2024-01-01');
      expect(missingEnd.period).toBe('Tudo');
    });

    it('converte estado de período para URLSearchParams', () => {
      expect(periodToParams({ period: '1A' }).toString()).toBe('period=1a');
      expect(periodToParams({ period: '2A' }).toString()).toBe('period=2a');
      expect(periodToParams({ period: '5A' }).toString()).toBe('period=5a');
      expect(periodToParams({ period: 'Tudo' }).toString()).toBe('period=tudo');
      expect(
        periodToParams({
          period: 'Personalizado',
          start: '2024-01-01',
          end: '2024-12-31',
        }).toString()
      ).toBe('period=custom&start=2024-01-01&end=2024-12-31');
    });
  });

  it('lê period=custom com datas da URL', async () => {
    renderFiltersBar({ initialUrl: '/?period=custom&start=2024-01-01&end=2024-12-31' });

    expect(screen.getByTestId('period-val')).toHaveTextContent('Personalizado');
    expect(screen.getByTestId('start-val')).toHaveTextContent('2024-01-01');
    expect(screen.getByTestId('end-val')).toHaveTextContent('2024-12-31');

    const startInput = screen.getByLabelText(/Data inicial/i);
    const endInput = screen.getByLabelText(/Data final/i);
    expect(startInput).toHaveValue('2024-01-01');
    expect(endInput).toHaveValue('2024-12-31');
  });

  it('period inválido cai em Tudo', async () => {
    renderFiltersBar({ initialUrl: '/?period=xyz&start=banana' });

    expect(screen.getByTestId('period-val')).toHaveTextContent('Tudo');
    expect(screen.getByTestId('start-val')).toHaveTextContent('1900-01-01');
    expect(screen.getByTestId('end-val')).toHaveTextContent('null');
  });

  it('trocar preset atualiza a URL', async () => {
    renderFiltersBar({ initialUrl: '/' });

    const btn1A = screen.getByRole('button', { name: '1A' });
    fireEvent.click(btn1A);

    expect(window.location.search).toContain('period=1a');

    const btn2A = screen.getByRole('button', { name: '2A' });
    fireEvent.click(btn2A);

    expect(window.location.search).toContain('period=2a');

    const btnTudo = screen.getByRole('button', { name: 'Tudo' });
    fireEvent.click(btnTudo);

    expect(window.location.search).toContain('period=tudo');
  });

  it('Personalizado após Tudo não mostra 01/01/1900', async () => {
    renderFiltersBar({ initialUrl: '/' });

    const btnCustom = screen.getByRole('button', { name: 'Personalizado' });
    fireEvent.click(btnCustom);

    const startInput = screen.getByLabelText(/Data inicial/i);
    expect(startInput).toHaveValue('');
    expect(startInput.value).not.toBe('1900-01-01');
    expect(screen.getByTestId('start-val')).toHaveTextContent('1900-01-01');

    fireEvent.change(startInput, { target: { value: '2024-03-01' } });
    expect(screen.getByTestId('start-val')).toHaveTextContent('2024-03-01');
    expect(startInput).toHaveValue('2024-03-01');
  });

  describe('DateInput nos campos personalizados (A7)', () => {
    it('campo de data mostra botão de calendário', () => {
      renderFiltersBar({ initialUrl: '/?period=custom&start=2024-01-01&end=2024-12-31' });

      const calendarButtons = screen.getAllByRole('button', { name: 'Abrir calendário' });
      expect(calendarButtons).toHaveLength(2);
    });

    it('clicar no ícone chama showPicker', () => {
      const showPickerMock = vi.fn();
      HTMLInputElement.prototype.showPicker = showPickerMock;

      try {
        renderFiltersBar({ initialUrl: '/?period=custom&start=2024-01-01&end=2024-12-31' });

        const calendarButtons = screen.getAllByRole('button', { name: 'Abrir calendário' });
        fireEvent.click(calendarButtons[0]);
        expect(showPickerMock).toHaveBeenCalledTimes(1);

        const startInput = screen.getByLabelText(/Data inicial/i);
        fireEvent.click(startInput);
        expect(showPickerMock).toHaveBeenCalledTimes(2);
      } finally {
        delete HTMLInputElement.prototype.showPicker;
      }
    });

    it('showPicker ausente não quebra', () => {
      delete HTMLInputElement.prototype.showPicker;

      renderFiltersBar({ initialUrl: '/?period=custom&start=2024-01-01&end=2024-12-31' });

      const calendarButtons = screen.getAllByRole('button', { name: 'Abrir calendário' });
      const startInput = screen.getByLabelText(/Data inicial/i);

      expect(() => fireEvent.click(calendarButtons[0])).not.toThrow();
      expect(() => fireEvent.click(startInput)).not.toThrow();
    });

    it('onClick externo não substitui a abertura do calendário', () => {
      const showPickerMock = vi.fn();
      const onClick = vi.fn();
      HTMLInputElement.prototype.showPicker = showPickerMock;
      try {
        render(
          <DateInput aria-label="Data externa" value="2024-01-01" onChange={() => {}} onClick={onClick} />
        );
        fireEvent.click(screen.getByLabelText('Data externa'));
        expect(onClick).toHaveBeenCalledTimes(1);
        expect(showPickerMock).toHaveBeenCalledTimes(1);
      } finally {
        delete HTMLInputElement.prototype.showPicker;
      }
    });

    it('mudar a data mantém ISO no filtro', () => {
      renderFiltersBar({ initialUrl: '/?period=custom&start=2024-01-01&end=2024-12-31' });

      const startInput = screen.getByLabelText(/Data inicial/i);
      const endInput = screen.getByLabelText(/Data final/i);

      fireEvent.change(startInput, { target: { value: '2024-02-15' } });
      expect(screen.getByTestId('start-val')).toHaveTextContent('2024-02-15');

      fireEvent.change(endInput, { target: { value: '2024-11-20' } });
      expect(screen.getByTestId('end-val')).toHaveTextContent('2024-11-20');
    });
  });
});

