import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';

vi.mock('echarts-for-react', () => ({
  default: ({ option, style }) => (
    <div
      data-testid="echarts-mock"
      data-option={JSON.stringify(option)}
      style={style}
    />
  ),
}));

import { SummaryCards } from '../components/portfolio/SummaryCards';
import { BaseChart } from '../components/portfolio/charts/BaseChart';

describe('SummaryCards e BaseChart', () => {
  it('exibe skeletons quando isLoading: true', () => {
    render(<SummaryCards data={null} isLoading={true} isError={false} />);
    const skeletons = screen.getAllByTestId('skeleton');
    expect(skeletons.length).toBeGreaterThanOrEqual(5);
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
});

