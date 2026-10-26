import React from 'react';
import { Card } from '../../common/Card';
import { Skeleton } from '../../common/Skeleton';

function formatYearReturn(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return { text: '—', colorClass: 'text-zinc-500' };
  }
  const pct = Number(value) * 100;
  const sign = pct > 0 ? '+' : '';
  const text = `${sign}${pct.toFixed(2).replace('.', ',')}%`;
  const colorClass =
    pct > 0
      ? 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20'
      : pct < 0
      ? 'text-red-400 bg-red-500/10 border-red-500/20'
      : 'text-zinc-300 bg-zinc-800/50 border-zinc-700';
  return { text, colorClass };
}

export function YearlyTable({ data, isLoading = false }) {
  if (isLoading) {
    return (
      <Card title="Rentabilidade Ano a Ano" subtitle="Retorno anual por série e acumulado no período">
        <Skeleton className="h-48 w-full" />
      </Card>
    );
  }

  const payload = data?.data || data || {};
  const years = payload.years || [];
  const rows = payload.rows || [];

  return (
    <Card
      title="Rentabilidade Ano a Ano"
      subtitle="Comparativo anual percentual da carteira frente aos indexadores"
    >
      {rows.length === 0 ? (
        <p className="text-sm text-zinc-400 py-6 text-center">
          Sem dados anuais disponíveis para o período selecionado.
        </p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-sm">
            <thead>
              <tr className="border-b border-zinc-800 text-xs uppercase text-zinc-400">
                <th className="py-3 px-3 font-semibold">Série</th>
                {years.map((yr) => (
                  <th key={yr} className="py-3 px-3 font-semibold text-right">
                    {yr}
                  </th>
                ))}
                <th className="py-3 px-3 font-semibold text-right">Total Acum.</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-800/60">
              {rows.map((row) => {
                const totalFormatted = formatYearReturn(row.total);
                return (
                  <tr
                    key={row.name}
                    className={`hover:bg-zinc-800/30 transition-colors ${
                      row.name === 'Carteira' ? 'bg-zinc-900/60 font-medium' : ''
                    }`}
                  >
                    <td className="py-3 px-3 text-zinc-100 whitespace-nowrap">{row.name}</td>
                    {years.map((yr) => {
                      const val = row.values?.[String(yr)] ?? row.values?.[yr];
                      const formatted = formatYearReturn(val);
                      return (
                        <td key={yr} className="py-3 px-3 text-right whitespace-nowrap">
                          <span
                            className={`inline-block px-2 py-0.5 rounded text-xs font-medium border ${formatted.colorClass}`}
                          >
                            {formatted.text}
                          </span>
                        </td>
                      );
                    })}
                    <td className="py-3 px-3 text-right whitespace-nowrap">
                      <span
                        className={`inline-block px-2.5 py-0.5 rounded text-xs font-semibold border ${totalFormatted.colorClass}`}
                      >
                        {totalFormatted.text}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}

export default YearlyTable;

