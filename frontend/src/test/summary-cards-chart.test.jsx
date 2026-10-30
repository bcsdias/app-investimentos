import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';

let latestEchartsOption = null;

vi.mock('echarts-for-react', () => ({
  default: ({ option, style }) => {
    latestEchartsOption = option;
    return (
      <div
        data-testid="echarts-mock"
        data-option={JSON.stringify(option)}
        style={style}
      />
    );
  },
}));

import { SummaryCards } from '../components/portfolio/SummaryCards';
import { BaseChart } from '../components/portfolio/charts/BaseChart';
import { IrrEvolutionChart } from '../components/portfolio/charts/IrrEvolutionChart';

describe('SummaryCards e BaseChart', () => {
  it('exibe skeletons quando isLoading: true', () => {
    render(<SummaryCards data={null} isLoading={true} isError={false} />);
    const skeletons = screen.getAllByTestId('skeleton');
    expect(skeletons.length).toBeGreaterThanOrEqual(5);
  });

  it('mensagem de primeira carga durante o loading do summary sem dados', () => {
    render(<SummaryCards data={null} isLoading={true} isError={false} />);
    expect(
      screen.getByText('Buscando sua carteira na DLP — a primeira carga pode levar até um minuto.')
    ).toBeInTheDocument();
  });

  it('formata valores monetários em BRL e percentuais com sinal (+ verde, - vermelho)', () => {
    const sampleData = {
      pat_fim: 150000,
      aportes: 120000,
      ganho: 30000,
      twr: 0.154,
      xirr: -0.021,
    };

    render(<SummaryCards data={sampleData} isLoading={false} isError={false} />);

    expect(screen.getByText(/R\$\s*150\.000,00/)).toBeInTheDocument();
    expect(screen.getByText(/R\$\s*120\.000,00/)).toBeInTheDocument();
    expect(screen.getByText(/R\$\s*30\.000,00/)).toBeInTheDocument();

    const twrEl = screen.getByText('+15,40%');
    expect(twrEl).toBeInTheDocument();
    expect(twrEl.className).toMatch(/emerald/);

    const xirrEl = screen.getByText('-2,10%');
    expect(xirrEl).toBeInTheDocument();
    expect(xirrEl.className).toMatch(/red/);
  });

  it('suporta envelope SummaryOut do backend (data.patrimonio, data.twr_total, etc.)', () => {
    const backendEnvelope = {
      data: {
        patrimonio: 250000.5,
        aportes_periodo: 200000,
        ganho_periodo: 50000.5,
        twr_total: 25.0,
        xirr: 14.25,
      },
      meta: { start: '2025-01-01', end: '2026-01-01', missing: [] },
    };

    render(<SummaryCards data={backendEnvelope} isLoading={false} isError={false} />);
    expect(screen.getByText(/R\$\s*250\.000,50/)).toBeInTheDocument();
    expect(screen.getByText('+25,00%')).toBeInTheDocument();
    expect(screen.getByText('+14,25%')).toBeInTheDocument();
  });

  it('BaseChart repassa as opções aplicando o tema escuro e paleta padrão ao ECharts', () => {
    const customOption = {
      xAxis: { type: 'time' },
      yAxis: { type: 'value' },
      series: [{ name: 'Carteira', type: 'line', data: [['2026-01-01', 100]] }],
    };

    render(<BaseChart option={customOption} height={360} isLoading={false} />);

    const chartEl = screen.getByTestId('echarts-mock');
    expect(chartEl).toBeInTheDocument();

    const mergedOption = JSON.parse(chartEl.getAttribute('data-option'));
    expect(mergedOption.backgroundColor).toBe('transparent');
    expect(mergedOption.color).toContain('#10b981');
    expect(mergedOption.tooltip.backgroundColor).toBe('#18181b');
    expect(mergedOption.series[0].name).toBe('Carteira');
  });

  it('card mostra TIR no período quando xirr_is_annual é false', () => {
    const data = {
      data: {
        patrimonio: 100000,
        aportes_periodo: 90000,
        ganho_periodo: 10000,
        twr_total: 5.0,
        xirr: -32.0,
        xirr_period: -3.0,
        xirr_is_annual: false,
      },
    };
    render(<SummaryCards data={data} isLoading={false} isError={false} />);
    expect(screen.getByText('TIR no período')).toBeInTheDocument();
    expect(screen.queryByText('TIR (XIRR a.a.)')).not.toBeInTheDocument();
    expect(screen.getByText('-3,00%')).toBeInTheDocument();
  });

  it('card mantém TIR a.a. quando o campo não vem (resposta antiga)', () => {
    const data = {
      data: {
        patrimonio: 100000,
        xirr: 14.25,
      },
    };
    render(<SummaryCards data={data} isLoading={false} isError={false} />);
    expect(screen.getByText('TIR (XIRR a.a.)')).toBeInTheDocument();
    expect(screen.queryByText('TIR no período')).not.toBeInTheDocument();
    expect(screen.getByText('+14,25%')).toBeInTheDocument();
  });

  it('card mostra TIR a.a. quando xirr_is_annual é true', () => {
    const data = {
      data: {
        patrimonio: 100000,
        xirr: 14.25,
        xirr_period: 25.0,
        xirr_is_annual: true,
      },
    };
    render(<SummaryCards data={data} isLoading={false} isError={false} />);
    expect(screen.getByText('TIR (XIRR a.a.)')).toBeInTheDocument();
    expect(screen.queryByText('TIR no período')).not.toBeInTheDocument();
    expect(screen.getByText('+14,25%')).toBeInTheDocument();
  });

  it('IrrEvolutionChart combina período antes de annual_from e série depois com markLine', () => {
    const chartData = {
      data: {
        series: [{ name: 'TIR', data: [['2024-02-29', -32.0], ['2025-01-31', 12.0]] }],
        period: [{ name: 'TIR no período', data: [['2024-02-29', -3.0], ['2025-01-31', 11.5]] }],
        annual_from: '2025-01-31',
      },
    };

    render(<IrrEvolutionChart data={chartData} isLoading={false} />);
    expect(screen.getByText('Evolução Histórica da TIR')).toBeInTheDocument();
    expect(screen.queryByText('Evolução Histórica da TIR (XIRR a.a.)')).not.toBeInTheDocument();

    const chartEl = screen.getByTestId('echarts-mock');
    const option = JSON.parse(chartEl.getAttribute('data-option'));
    expect(option.series).toHaveLength(1);
    expect(option.series[0].data).toEqual([
      ['2024-02-29', -3.0],
      ['2025-01-31', 12.0],
    ]);
    expect(option.series[0].markLine).toBeDefined();
    expect(option.series[0].markLine.data).toEqual([{ xAxis: '2025-01-31' }]);
    expect(option.series[0].markLine.label.formatter).toMatch(/a\.a\. a partir daqui/);

    // Test tooltip formatter
    const tooltipFn = latestEchartsOption.tooltip.formatter;
    const resPeriod = tooltipFn([{ axisValueLabel: '2024-02-29', value: ['2024-02-29', -3.0], seriesName: 'TIR', marker: '●' }]);
    expect(resPeriod).toContain('(no período)');
    const resAnnual = tooltipFn([{ axisValueLabel: '2025-01-31', value: ['2025-01-31', 12.0], seriesName: 'TIR', marker: '●' }]);
    expect(resAnnual).toContain('(a.a.)');
  });

  it('IrrEvolutionChart usa período para a linha toda quando não há annual_from', () => {
    const chartData = {
      data: {
        series: [{ name: 'TIR', data: [['2024-02-29', -32.0], ['2024-03-31', -15.0]] }],
        period: [{ name: 'TIR no período', data: [['2024-02-29', -3.0], ['2024-03-31', -2.5]] }],
        annual_from: null,
      },
    };

    render(<IrrEvolutionChart data={chartData} isLoading={false} />);
    const chartEl = screen.getByTestId('echarts-mock');
    const option = JSON.parse(chartEl.getAttribute('data-option'));
    expect(option.series).toHaveLength(1);
    expect(option.series[0].data).toEqual([
      ['2024-02-29', -3.0],
      ['2024-03-31', -2.5],
    ]);
    expect(option.series[0].markLine).toBeUndefined();

    const tooltipFn = latestEchartsOption.tooltip.formatter;
    const resPeriod = tooltipFn([{ axisValueLabel: '2024-02-29', value: ['2024-02-29', -3.0], seriesName: 'TIR', marker: '●' }]);
    expect(resPeriod).toContain('(no período)');
  });
});

