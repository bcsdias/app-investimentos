import React, { useMemo } from 'react';
import { Card } from '../../common/Card';
import { BaseChart } from './BaseChart';

export function IrrEvolutionChart({ data, isLoading = false }) {
  const seriesList = data?.data?.series || data?.series || [];

  const option = useMemo(() => {
    return {
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
          formatter: (val) => `${Number(val).toFixed(1)}%`,
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
            lines.push(
              `<div>${p.marker} ${p.seriesName}: <b>${Number(rawVal).toFixed(2)}% a.a.</b></div>`
            );
          });
          return lines.join('');
        },
      },
      series: seriesList.map((s) => ({
        name: s.name || 'TIR',
        type: 'line',
        showSymbol: false,
        connectNulls: false,
        lineStyle: { width: 2.25 },
        data: s.data || [],
      })),
    };
  }, [seriesList]);

  return (
    <Card
      title="Evolução Histórica da TIR (XIRR a.a.)"
      subtitle="Taxa interna de retorno anualizada acumulada mês a mês"
    >
      <BaseChart option={option} height={340} isLoading={isLoading} />
    </Card>
  );
}

export default IrrEvolutionChart;

