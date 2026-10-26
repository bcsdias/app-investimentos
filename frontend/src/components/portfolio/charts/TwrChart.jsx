import React, { useMemo } from 'react';
import { Card } from '../../common/Card';
import { BaseChart } from './BaseChart';

export function TwrChart({ data, isLoading = false }) {
  const seriesList = data?.data?.series || data?.series || [];

  const option = useMemo(() => {
    return {
      grid: { top: 40, left: 56, right: 24, bottom: 72 },
      xAxis: {
        type: 'time',
        axisLine: { lineStyle: { color: '#3f3f46' } },
        axisLabel: { color: '#a1a1aa' },
      },
      yAxis: {
        type: 'value',
        scale: true,
        axisLine: { lineStyle: { color: '#3f3f46' } },
        splitLine: { lineStyle: { color: '#27272a', type: 'dashed' } },
        axisLabel: {
          color: '#a1a1aa',
          formatter: (val) => `${Number(val).toFixed(0)}`,
        },
      },
      tooltip: {
        trigger: 'axis',
        formatter: (params) => {
          if (!Array.isArray(params) || params.length === 0) return '';
          const dateLabel = params[0]?.axisValueLabel || params[0]?.value?.[0] || '';
          const lines = [`<div style="font-weight:600;margin-bottom:4px">${dateLabel}</div>`];
          params.forEach((p) => {
            const rawVal = Array.isArray(p.value) ? p.value[1] : p.value;
            if (rawVal === null || rawVal === undefined || Number.isNaN(Number(rawVal))) return;
            const num = Number(rawVal);
            const pct = num - 100;
            const sign = pct >= 0 ? '+' : '';
            lines.push(
              `<div>${p.marker} ${p.seriesName}: <b>${num.toFixed(2)}</b> (${sign}${pct.toFixed(2)}%)</div>`
            );
          });
          return lines.join('');
        },
      },
      dataZoom: [
        {
          type: 'inside',
          start: 0,
          end: 100,
        },
        {
          type: 'slider',
          bottom: 12,
          height: 22,
          borderColor: '#27272a',
          backgroundColor: '#18181b',
          fillerColor: 'rgba(16, 185, 129, 0.15)',
          textStyle: { color: '#a1a1aa' },
        },
      ],
      series: seriesList.map((s) => ({
        name: s.name,
        type: 'line',
        showSymbol: false,
        connectNulls: false,
        lineStyle: { width: s.name === 'Carteira' ? 2.5 : 1.75 },
        data: s.data || [],
      })),
    };
  }, [seriesList]);

  return (
    <Card
      title="Evolução da Cota (TWR — Base 100)"
      subtitle="Rentabilidade ponderada pelo tempo comparada aos benchmarks selecionados"
    >
      <BaseChart option={option} height={380} isLoading={isLoading} />
    </Card>
  );
}

export default TwrChart;

