import React, { useMemo } from 'react';
import { Card } from '../../common/Card';
import { BaseChart } from './BaseChart';

export function IrrEvolutionChart({ data, isLoading = false, error = null, onRetry }) {
  const payload = data?.data || {};
  const seriesList = payload.series || [];
  const periodList = payload.period || [];
  const annualFrom = payload.annual_from || null;

  const option = useMemo(() => {
    const seriesPoints = seriesList[0]?.data || [];
    const periodPoints = periodList[0]?.data || [];

    let mergedData = [];
    if (periodPoints.length === 0 && seriesPoints.length > 0) {
      mergedData = seriesPoints;
    } else if (!annualFrom) {
      mergedData = periodPoints.length > 0 ? periodPoints : seriesPoints;
    } else {
      const periodMap = new Map(periodPoints.map(([d, v]) => [d, v]));
      const seriesMap = new Map(seriesPoints.map(([d, v]) => [d, v]));
      const allDates = (seriesPoints.length > 0 ? seriesPoints : periodPoints).map(([d]) => d);

      mergedData = allDates.map((d) => {
        const val =
          d < annualFrom
            ? periodMap.has(d)
              ? periodMap.get(d)
              : seriesMap.get(d)
            : seriesMap.has(d)
            ? seriesMap.get(d)
            : periodMap.get(d);
        return [d, val];
      });
    }

    const markLine = annualFrom
      ? {
          symbol: ['none', 'none'],
          lineStyle: { type: 'dashed', color: '#71717a' },
          label: {
            show: true,
            // Ao longo da linha, dentro do gráfico: no topo e horizontal, o rótulo saía
            // cortado quando annual_from cai no último ponto (janela de 12 meses exatos).
            position: 'insideEndTop',
            formatter: 'a.a. a partir daqui',
            color: '#a1a1aa',
            fontSize: 11,
          },
          data: [{ xAxis: annualFrom }],
        }
      : undefined;

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
            const ptDate = Array.isArray(p.value) ? p.value[0] : (p.name || dateLabel);
            const isAnnual =
              annualFrom && ptDate
                ? String(ptDate).slice(0, 10) >= annualFrom.slice(0, 10)
                : false;
            const suffix = isAnnual ? '(a.a.)' : '(no período)';
            lines.push(
              `<div>${p.marker} ${p.seriesName}: <b>${Number(rawVal).toFixed(2)}% ${suffix}</b></div>`
            );
          });
          return lines.join('');
        },
      },
      series: [
        {
          name: 'TIR',
          type: 'line',
          showSymbol: false,
          connectNulls: false,
          lineStyle: { width: 2.25 },
          data: mergedData,
          markLine,
        },
      ],
    };
  }, [seriesList, periodList, annualFrom]);

  return (
    <Card
      title="Evolução Histórica da TIR"
      subtitle="Taxa interna de retorno acumulada mês a mês"
    >
      <BaseChart option={option} height={340} isLoading={isLoading} error={error} onRetry={onRetry} />
    </Card>
  );
}

export default IrrEvolutionChart;

