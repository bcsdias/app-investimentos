import React, { useMemo } from 'react';
import { Card } from '../../common/Card';
import { BaseChart } from './BaseChart';

function toVolPercent(val) {
  if (val === null || val === undefined || Number.isNaN(Number(val))) return null;
  const num = Number(val);
  return Math.abs(num) <= 2 ? num * 100 : num;
}

export function RollingSharpeChart({ data, isLoading = false }) {
  const payload = data?.data || data || {};
  const sharpeSeries = payload.sharpe || [];
  const volSeries = payload.volatility || [];

  const option = useMemo(() => {
    const combinedSeries = [
      ...sharpeSeries.map((s) => ({
        name: `${s.name} (Sharpe)`,
        type: 'line',
        yAxisIndex: 0,
        showSymbol: false,
        connectNulls: false,
        lineStyle: { width: s.name === 'Carteira' ? 2.25 : 1.5 },
        data: (s.data || []).map(([dt, val]) => [
          dt,
          val === null || val === undefined || Number.isNaN(Number(val)) ? null : Number(val),
        ]),
      })),
      ...volSeries.map((s) => ({
        name: `${s.name} (Vol %)`,
        type: 'line',
        yAxisIndex: 1,
        showSymbol: false,
        connectNulls: false,
        lineStyle: { width: 1.5, type: 'dashed' },
        data: (s.data || []).map(([dt, val]) => [dt, toVolPercent(val)]),
      })),
    ];

    return {
      grid: { top: 44, left: 56, right: 56, bottom: 48 },
      xAxis: {
        type: 'time',
        axisLine: { lineStyle: { color: '#3f3f46' } },
        axisLabel: { color: '#a1a1aa' },
      },
      yAxis: [
        {
          type: 'value',
          name: 'Sharpe',
          scale: true,
          axisLine: { lineStyle: { color: '#3f3f46' } },
          splitLine: { lineStyle: { color: '#27272a', type: 'dashed' } },
          axisLabel: {
            color: '#a1a1aa',
            formatter: (val) => Number(val).toFixed(1),
          },
        },
        {
          type: 'value',
          name: 'Volatilidade (%)',
          scale: true,
          axisLine: { lineStyle: { color: '#3f3f46' } },
          splitLine: { show: false },
          axisLabel: {
            color: '#a1a1aa',
            formatter: (val) => `${Number(val).toFixed(0)}%`,
          },
        },
      ],
      series: combinedSeries,
    };
  }, [sharpeSeries, volSeries]);

  return (
    <Card
      title="Sharpe e Volatilidade Móvel (252 d.u.)"
      subtitle="Evolução da relação risco-retorno anualizada em janela móvel de 1 ano"
    >
      <BaseChart option={option} height={360} isLoading={isLoading} />
    </Card>
  );
}

export default RollingSharpeChart;

