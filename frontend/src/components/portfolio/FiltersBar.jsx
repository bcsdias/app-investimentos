import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { X, RotateCcw } from 'lucide-react';
import {
  usePortfolioFilters,
  formatSyntheticLabel,
} from '../../context/PortfolioFilterContext';
import { portfolioApi } from '../../api/portfolio';
import { useDlpTokenConfigured } from '../../hooks/useDlpTokenStatus';
import { BenchmarkSelector } from './BenchmarkSelector';
import { Badge } from '../common/Badge';

const PRESET_PERIODS = ['1A', '2A', '5A', 'Tudo', 'Personalizado'];

export function FiltersBar() {
  const {
    period,
    setPeriod,
    start,
    end,
    setCustomDates,
    benchmarks,
    removeBenchmark,
    synthetics,
    removeSynthetic,
    ativo,
    setAtivo,
    classe,
    setClasse,
    resetFilters,
  } = usePortfolioFilters();
  const tokenConfigured = useDlpTokenConfigured();

  const { data: filterOptions } = useQuery({
    queryKey: ['portfolio', 'filters'],
    queryFn: ({ signal }) => portfolioApi.getFilters(signal),
    enabled: tokenConfigured === true,
    staleTime: 5 * 60 * 1000,
  });

  const ativos = filterOptions?.ativos || [];
  const classes = filterOptions?.classes || [];

  return (
    <div className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-4 space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        {/* Seletor de Período */}
        <div className="flex flex-wrap items-center gap-1.5">
          {PRESET_PERIODS.map((preset) => {
            const active = period === preset;
            return (
              <button
                key={preset}
                type="button"
                onClick={() => setPeriod(preset)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                  active
                    ? 'bg-emerald-600 text-white shadow-sm'
                    : 'bg-zinc-950 text-zinc-400 hover:text-zinc-100 border border-zinc-800'
                }`}
              >
                {preset}
              </button>
            );
          })}

          {period === 'Personalizado' && (
            <div className="flex items-center gap-2 ml-2">
              <input
                type="date"
                aria-label="Data inicial"
                value={start || ''}
                onChange={(e) => setCustomDates(e.target.value, end)}
                className="bg-zinc-950 border border-zinc-700 rounded-lg px-2.5 py-1 text-xs text-zinc-100"
              />
              <span className="text-xs text-zinc-500">até</span>
              <input
                type="date"
                aria-label="Data final"
                value={end || ''}
                onChange={(e) => setCustomDates(start, e.target.value)}
                className="bg-zinc-950 border border-zinc-700 rounded-lg px-2.5 py-1 text-xs text-zinc-100"
              />
            </div>
          )}
        </div>

        {/* Filtros de Custódia e Comparadores */}
        <div className="flex flex-wrap items-center gap-2">
          <select
            aria-label="Filtrar por Classe"
            value={classe || ''}
            onChange={(e) => setClasse(e.target.value || null)}
            className="bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-1.5 text-xs text-zinc-200 focus:outline-none focus:ring-2 focus:ring-emerald-500"
          >
            <option value="">Todas as Classes</option>
            {classes.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>

          <select
            aria-label="Filtrar por Ativo"
            value={ativo || ''}
            onChange={(e) => setAtivo(e.target.value || null)}
            className="bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-1.5 text-xs text-zinc-200 focus:outline-none focus:ring-2 focus:ring-emerald-500"
          >
            <option value="">Todos os Ativos</option>
            {ativos.map((a) => (
              <option key={a} value={a}>
                {a}
              </option>
            ))}
          </select>

          <BenchmarkSelector />

          <button
            type="button"
            onClick={resetFilters}
            title="Restaurar filtros padrão"
            className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800 transition-colors"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Limpar</span>
          </button>
        </div>
      </div>

      {/* Badges de Comparadores Ativos */}
      {(benchmarks.length > 0 || synthetics.length > 0 || ativo || classe) && (
        <div className="flex flex-wrap items-center gap-1.5 pt-2 border-t border-zinc-800/70">
          <span className="text-[11px] text-zinc-500 mr-1">Comparadores ativos:</span>
          {benchmarks.map((bm) => (
            <Badge key={bm} variant="neutral" className="pr-1.5">
              <span>{bm}</span>
              <button
                type="button"
                aria-label={`Remover ${bm}`}
                onClick={() => removeBenchmark(bm)}
                className="text-zinc-400 hover:text-zinc-100"
              >
                <X className="w-3 h-3" />
              </button>
            </Badge>
          ))}
          {synthetics.map((s) => {
            const label = formatSyntheticLabel(s);
            return (
              <Badge key={label} variant="info" className="pr-1.5">
                <span>{label}</span>
                <button
                  type="button"
                  aria-label={`Remover ${label}`}
                  onClick={() => removeSynthetic(s)}
                  className="text-blue-300 hover:text-white"
                >
                  <X className="w-3 h-3" />
                </button>
              </Badge>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default FiltersBar;

