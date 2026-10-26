import React, { useMemo } from 'react';
import { Card } from '../../common/Card';
import { BaseChart } from './BaseChart';

function toPct(val) {
  if (val === null || val === undefined || Number.isNaN(Number(val))) return 0;
  const num = Number(val);
  return Math.abs(num) <= 2 ? num * 100 : num;
}

export function RiskReturnChart({ data, isLoading = false }) {
  const rows = data?.data?.rows || data?.rows || [];

  const option = useMemo(() => {
    const validRows = rows.filter(
      (r) => r.volatilidade !== null && r.retorno_cagr !== null
    );

    const avgVol =
      validRows.length > 0
        ? validRows.reduce((acc, r) => acc + toPct(r.volatilidade), 0) / validRows.length
        : 0;
    const avgCagr =
      validRows.length > 0
        ? validRows.reduce((acc, r) => acc + toPct(r.retorno_cagr), 0) / validRows.length
        : 0;

    return {
      xAxis: {
        type: 'value',
        name: 'Volatilidade a.a. (%)',
        nameLocation: 'middle',
        nameGap: 28,
        scale: true,
        axisLine: { lineStyle: { color: '#3f3f46' } },
        splitLine: { lineStyle: { color: '#27272a', type: 'dashed' } },
        axisLabel: {
          color: '#a1a1aa',
          formatter: (val) => `${Number(val).toFixed(1)}%`,
        },
      },
      yAxis: {
        type: 'value',
        name: 'Retorno CAGR (%)',
        scale: true,
        axisLine: { lineStyle: { color: '#3f3f46' } },
        splitLine: { lineStyle: { color: '#27272a', type: 'dashed' } },
        axisLabel: {
          color: '#a1a1aa',
          formatter: (val) => `${Number(val).toFixed(1)}%`,
        },
      },
      tooltip: {
        trigger: 'item',
        formatter: (param) => {
          const [vol, cagr, sharpe, label] = param.value || [];
          return [
            `<div style="font-weight:600;margin-bottom:4px">${label || param.seriesName}</div>`,
            `<div>Volatilidade: <b>${Number(vol).toFixed(2)}%</b></div>`,
            `<div>CAGR: <b>${Number(cagr).toFixed(2)}%</b></div>`,
            sharpe !== null && sharpe !== undefined
              ? `<div>Sharpe: <b>${Number(sharpe).toFixed(2)}</b></div>`
              : '',
          ].join('');
        },
      },
      series: validRows.map((row, idx) => {
        const volPct = toPct(row.volatilidade);
        const cagrPct = toPct(row.retorno_cagr);
        const isPortfolio = row.name === 'Carteira';
        return {
          name: row.name,
          type: 'scatter',
          symbolSize: isPortfolio ? 18 : 13,
          data: [[volPct, cagrPct, row.sharpe ?? null, row.name]],
          label: {
            show: true,
            position: 'top',
            formatter: () => row.name,
            color: '#e4e4e7',
            fontSize: 11,
          },
          markLine:
            idx === 0 && validRows.length > 1
              ? {
                  silent: true,
                  symbol: 'none',
                  lineStyle: { color: '#52525b', type: 'dotted' },
                  data: [{ xAxis: avgVol }, { yAxis: avgCagr }],
                }
              : undefined,
        };
      }),
    };
  }, [rows]);

  return (
    <Card
      title="Dispersão Risco × Retorno (Volatilidade vs CAGR)"
      subtitle="Comparação de eficiência anualizada e quadrantes médios do período"
    >
      <BaseChart option={option} height={360} isLoading={isLoading} />
    </Card>
  );
}

export default RiskReturnChart;
