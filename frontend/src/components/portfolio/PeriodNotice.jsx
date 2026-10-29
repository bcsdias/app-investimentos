import React from 'react';
import { CalendarRange, AlertTriangle } from 'lucide-react';

function formatBRDate(iso) {
  const [y, m, d] = String(iso || '').split('-');
  return y && m && d ? `${d}/${m}/${y}` : iso;
}

export function PeriodNotice({ meta }) {
  if (!meta || !meta.start || !meta.end) return null;
  const missing = Array.isArray(meta.missing) ? meta.missing : [];
  const gaps = Array.isArray(meta.gaps) ? meta.gaps : [];

  return (
    <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-xs text-zinc-400">
      <span className="inline-flex items-center gap-1.5">
        <CalendarRange className="w-3.5 h-3.5 text-zinc-500" />
        Período analisado: {formatBRDate(meta.start)} a {formatBRDate(meta.end)}
      </span>
      {missing.length > 0 && (
        <span className="inline-flex items-center gap-1.5 text-amber-300">
          <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
          Sem dados no período: {missing.join(', ')}
        </span>
      )}
      {gaps.map((g, idx) => (
        <span
          key={`${g.series || ''}-${g.start || ''}-${g.end || ''}-${idx}`}
          className="inline-flex items-center gap-1.5 text-amber-300"
        >
          <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
          Lacuna nos dados de {g.series} de {formatBRDate(g.start)} a {formatBRDate(g.end)} (último valor repetido)
        </span>
      ))}
    </div>
  );
}

export default PeriodNotice;
