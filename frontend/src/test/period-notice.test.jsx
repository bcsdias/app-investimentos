import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import {
  ALL_TIME_START,
  computePresetDates,
  toLocalISODate,
} from '../context/PortfolioFilterContext';
import { PeriodNotice } from '../components/portfolio/PeriodNotice';

describe('Preset Tudo, datas locais e PeriodNotice', () => {
  it('Tudo envia start 1900-01-01 e end null', () => {
    expect(ALL_TIME_START).toBe('1900-01-01');
    expect(computePresetDates('Tudo')).toEqual({ start: '1900-01-01', end: null });
  });

  it('1A às 22:30 locais usa a data local de hoje', () => {
    const base = new Date(2026, 9, 27, 22, 30);
    expect(computePresetDates('1A', base)).toEqual({ start: '2025-10-27', end: '2026-10-27' });
  });

  it('toLocalISODate usa ano/mês/dia locais com zero à esquerda', () => {
    expect(toLocalISODate(new Date(2026, 0, 5, 23, 59))).toBe('2026-01-05');
  });

  it('PeriodNotice mostra período efetivo e benchmarks sem dados', () => {
    render(
      <PeriodNotice meta={{ start: '2019-03-04', end: '2026-10-26', missing: ['IFIX', 'IPCA + 6%'] }} />
    );
    expect(screen.getByText(/04\/03\/2019 a 26\/10\/2026/)).toBeInTheDocument();
    expect(screen.getByText(/Sem dados no período: IFIX, IPCA \+ 6%/)).toBeInTheDocument();
  });

  it('PeriodNotice sem meta não renderiza nada', () => {
    const { container } = render(<PeriodNotice meta={undefined} />);
    expect(container).toBeEmptyDOMElement();
  });

  it('PeriodNotice lista lacunas de benchmark', () => {
    render(
      <PeriodNotice
        meta={{
          start: '2024-01-01',
          end: '2024-06-30',
          missing: [],
          gaps: [{ series: 'IMA-S', start: '2024-03-01', end: '2024-04-15' }],
        }}
      />
    );
    expect(
      screen.getByText('Lacuna nos dados de IMA-S de 01/03/2024 a 15/04/2024 (último valor repetido)')
    ).toBeInTheDocument();
  });

  it('PeriodNotice sem gaps não quebra', () => {
    render(<PeriodNotice meta={{ start: '2024-01-01', end: '2024-06-30', missing: [] }} />);
    expect(screen.queryByText(/Lacuna nos dados/)).toBeNull();
  });

  it('PeriodNotice com gaps não-array trata como array vazio', () => {
    render(
      <PeriodNotice
        meta={{
          start: '2024-01-01',
          end: '2024-06-30',
          missing: [],
          gaps: 'invalid',
        }}
      />
    );
    expect(screen.queryByText(/Lacuna nos dados/)).toBeNull();
  });
});
