import React, { useMemo } from 'react';
import { Card } from '../../common/Card';
import { BaseChart } from './BaseChart';

export function toPercentPoint(val) {
  if (val === null || val === undefined || Number.isNaN(Number(val))) return null;
  const num = Number(val);
  return num * 100;
}

export function DrawdownChart({ data, isLoading = false, error = null, onRetry }) {
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
        max: 0,
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
              `<div>${p.marker} ${p.seriesName}: <b>${Number(rawVal).toFixed(2)}%</b></div>`
            );
          });
          return lines.join('');
        },
      },
      series: seriesList.map((s) => {
        const isPortfolio = s.name === 'Carteira';
        return {
          name: s.name,
          type: 'line',
          showSymbol: false,
          connectNulls: false,
          lineStyle: {
            width: isPortfolio ? 2.25 : 1.5,
            color: isPortfolio ? '#ef4444' : undefined,
          },
          itemStyle: isPortfolio ? { color: '#ef4444' } : undefined,
          areaStyle: isPortfolio
            ? {
                color: 'rgba(239, 68, 68, 0.25)',
              }
            : undefined,
          data: (s.data || []).map(([date, val]) => [date, toPercentPoint(val)]),
        };
      }),
    };
  }, [seriesList]);

  return (
    <Card
      title="Drawdown Subaquático (%)"
      subtitle="Queda percentual em relação à máxima histórica da cota"
    >
      <BaseChart option={option} height={340} isLoading={isLoading} error={error} onRetry={onRetry} />
    </Card>
  );
}

export default DrawdownChart;

