import React, { useMemo } from 'react';
import { Card } from '../../common/Card';
import { BaseChart } from './BaseChart';
import { formatCurrencyBRL } from '../SummaryCards';

export function ShadowChart({ data, isLoading = false }) {
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
          formatter: (val) =>
            new Intl.NumberFormat('pt-BR', {
              notation: 'compact',
              style: 'currency',
              currency: 'BRL',
            }).format(Number(val)),
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
            lines.push(`<div>${p.marker} ${p.seriesName}: <b>${formatCurrencyBRL(rawVal)}</b></div>`);
          });
          return lines.join('');
        },
      },
      series: seriesList.map((s) => ({
        name: s.name,
        type: 'line',
        showSymbol: false,
        connectNulls: false,
        lineStyle: {
          width: s.name === 'Carteira Real' ? 2.5 : 1.75,
          type: s.name === 'Total Investido' ? 'dashed' : 'solid',
        },
        data: s.data || [],
      })),
    };
  }, [seriesList]);

  return (
    <Card
      title="Shadow Portfolio (Evolução Patrimonial R$)"
      subtitle="Simulação dos mesmos aportes aplicados diretamente nos benchmarks"
    >
      <BaseChart option={option} height={340} isLoading={isLoading} />
    </Card>
  );
}

export default ShadowChart;
