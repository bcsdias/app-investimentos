import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Plus, Search, SlidersHorizontal, Check } from 'lucide-react';
import { usePortfolioFilters } from '../../context/PortfolioFilterContext';
import { benchmarksApi } from '../../api/benchmarks';
import { assetsApi } from '../../api/assets';
import { Button } from '../common/Button';

export function BenchmarkSelector() {
  const { benchmarks, addBenchmark, toggleBenchmark, addSynthetic } = usePortfolioFilters();
  const [isOpen, setIsOpen] = useState(false);
  const [activeTab, setActiveTab] = useState('catalog');

  // Ticker search state
  const [searchQuery, setSearchQuery] = useState('');
  const [debouncedQuery, setDebouncedQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [useBrlVariant, setUseBrlVariant] = useState(false);

  // Synthetic state
  const [synthBase, setSynthBase] = useState('CDI');
  const [synthOp, setSynthOp] = useState('+');
  const [synthVal, setSynthVal] = useState('2');
  const [synthError, setSynthError] = useState('');

  const { data: catalog } = useQuery({
    queryKey: ['benchmarks', 'catalog'],
    queryFn: () => benchmarksApi.getCatalog(),
    staleTime: 10 * 60 * 1000,
  });

  useEffect(() => {
    const trimmed = searchQuery.trim();
    if (trimmed.length < 2) {
      setDebouncedQuery('');
      setSearchResults([]);
      return undefined;
    }
    const timer = setTimeout(() => {
      setDebouncedQuery(trimmed);
    }, 150);
    return () => clearTimeout(timer);
  }, [searchQuery]);

  useEffect(() => {
    let active = true;
    if (debouncedQuery.length >= 2) {
      assetsApi
        .searchAssets(debouncedQuery)
        .then((res) => {
          if (active) setSearchResults(Array.isArray(res) ? res : []);
        })
        .catch(() => {
          if (active) setSearchResults([]);
        });
    }
    return () => {
      active = false;
    };
  }, [debouncedQuery]);

  const handleSelectAsset = (asset) => {
    const suffix = useBrlVariant && asset.brl_variant ? ' BRL' : '';
    addBenchmark(`${asset.yf_ticker}${suffix}`);
  };

  const handleAddSynthetic = (e) => {
    e.preventDefault();
    const numeric = Number(synthVal);
    if (!Number.isFinite(numeric) || numeric <= 0) {
      setSynthError('Informe um valor numérico positivo.');
      return;
    }
    if (synthOp === '+' && numeric > 50) {
      setSynthError('Para taxa adicional (+), o valor máximo é 50%.');
      return;
    }
    if (synthOp === '%' && numeric > 500) {
      setSynthError('Para percentual do índice (%), o valor máximo é 500%.');
      return;
    }
    setSynthError('');
    addSynthetic({ base: synthBase, operador: synthOp, valor: numeric });
  };

  const groups = catalog?.groups || [
    { name: 'Renda Fixa / Índices BCB', items: ['CDI', 'IPCA', 'SELIC'] },
    { name: 'Índices B3', items: ['IBOV', 'IFIX', 'IDIV', 'SMLL'] },
  ];

  return (
    <div className="relative">
      <Button
        type="button"
        variant="secondary"
        size="sm"
        onClick={() => setIsOpen((prev) => !prev)}
      >
        <SlidersHorizontal className="w-3.5 h-3.5 text-emerald-400" />
        <span>Comparadores ({benchmarks.length})</span>
      </Button>

      {isOpen && (
        <div className="absolute right-0 mt-2 w-80 sm:w-96 z-30 bg-zinc-900 border border-zinc-800 rounded-xl shadow-2xl p-4 space-y-3">
          <div className="flex items-center gap-1 p-1 bg-zinc-950 rounded-lg border border-zinc-800">
            <button
              type="button"
              onClick={() => setActiveTab('catalog')}
              className={`flex-1 py-1.5 text-xs font-medium rounded-md transition-colors ${
                activeTab === 'catalog'
                  ? 'bg-emerald-600 text-white'
                  : 'text-zinc-400 hover:text-zinc-200'
              }`}
            >
              Catálogo
            </button>
            <button
              type="button"
              onClick={() => setActiveTab('ticker')}
              className={`flex-1 py-1.5 text-xs font-medium rounded-md transition-colors ${
                activeTab === 'ticker'
                  ? 'bg-emerald-600 text-white'
                  : 'text-zinc-400 hover:text-zinc-200'
              }`}
            >
              Buscar Ticker
            </button>
            <button
              type="button"
              onClick={() => setActiveTab('synthetic')}
              className={`flex-1 py-1.5 text-xs font-medium rounded-md transition-colors ${
                activeTab === 'synthetic'
                  ? 'bg-emerald-600 text-white'
                  : 'text-zinc-400 hover:text-zinc-200'
              }`}
            >
              Sintético
            </button>
          </div>

          {activeTab === 'catalog' && (
            <div className="max-h-60 overflow-y-auto space-y-3 pr-1">
              {groups.map((grp) => (
                <div key={grp.name}>
                  <p className="text-[11px] font-semibold uppercase tracking-wider text-zinc-400 mb-1.5">
                    {grp.name}
                  </p>
                  <div className="grid grid-cols-2 gap-1.5">
                    {grp.items.map((item) => {
                      const checked = benchmarks.includes(item);
                      return (
                        <label
                          key={item}
                          className={`flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-xs cursor-pointer border transition-colors ${
                            checked
                              ? 'bg-emerald-500/15 border-emerald-500/40 text-emerald-300'
                              : 'bg-zinc-950 border-zinc-800 text-zinc-300 hover:border-zinc-700'
                          }`}
                        >
                          <input
                            type="checkbox"
                            checked={checked}
                            onChange={() => toggleBenchmark(item)}
                            className="sr-only"
                          />
                          <span
                            className={`w-3.5 h-3.5 rounded flex items-center justify-center border ${
                              checked
                                ? 'bg-emerald-500 border-emerald-500 text-zinc-950'
                                : 'border-zinc-600'
                            }`}
                          >
                            {checked && <Check className="w-2.5 h-2.5 stroke-[3]" />}
                          </span>
                          <span className="truncate">{item}</span>
                        </label>
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>
          )}

          {activeTab === 'ticker' && (
            <div className="space-y-3">
              <div className="relative">
                <Search className="w-4 h-4 text-zinc-500 absolute left-3 top-1/2 -translate-y-1/2" />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Digite ticker ou nome (ex: PETR4, AAPL)..."
                  className="w-full bg-zinc-950 border border-zinc-700 rounded-lg pl-9 pr-3 py-1.5 text-xs text-zinc-100 placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-emerald-500"
                />
              </div>

              <label className="flex items-center gap-2 text-xs text-zinc-300 cursor-pointer">
                <input
                  type="checkbox"
                  checked={useBrlVariant}
                  onChange={(e) => setUseBrlVariant(e.target.checked)}
                  className="rounded border-zinc-700 bg-zinc-950 text-emerald-500 focus:ring-emerald-500"
                />
                <span>Versão em BRL (para ativos internacionais)</span>
              </label>

              <div className="max-h-48 overflow-y-auto space-y-1.5">
                {searchResults.map((asset) => (
                  <div
                    key={asset.yf_ticker}
                    className="flex items-center justify-between gap-2 p-2 rounded-lg bg-zinc-950 border border-zinc-800"
                  >
                    <div className="min-w-0">
                      <p className="text-xs font-semibold text-zinc-100 truncate">
                        {asset.sigla}{' '}
                        <span className="text-[11px] font-normal text-zinc-400">
                          ({asset.yf_ticker})
                        </span>
                      </p>
                      <p className="text-[11px] text-zinc-400 truncate">{asset.nome}</p>
                    </div>
                    <Button
                      type="button"
                      size="sm"
                      variant="secondary"
                      aria-label={`Adicionar ${asset.sigla}`}
                      onClick={() => handleSelectAsset(asset)}
                    >
                      <Plus className="w-3 h-3" />
                      Adicionar
                    </Button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {activeTab === 'synthetic' && (
            <form onSubmit={handleAddSynthetic} className="space-y-3">
              <div className="grid grid-cols-3 gap-2">
                <div>
                  <label
                    htmlFor="synth-base"
                    className="block text-[11px] text-zinc-400 mb-1"
                  >
                    Índice Base
                  </label>
                  <select
                    id="synth-base"
                    value={synthBase}
                    onChange={(e) => setSynthBase(e.target.value)}
                    className="w-full bg-zinc-950 border border-zinc-700 rounded-lg px-2 py-1.5 text-xs text-zinc-100"
                  >
                    {(catalog?.synthetic?.bases || ['CDI', 'IPCA']).map((b) => (
                      <option key={b} value={b}>
                        {b}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label
                    htmlFor="synth-op"
                    className="block text-[11px] text-zinc-400 mb-1"
                  >
                    Operador
                  </label>
                  <select
                    id="synth-op"
                    value={synthOp}
                    onChange={(e) => setSynthOp(e.target.value)}
                    className="w-full bg-zinc-950 border border-zinc-700 rounded-lg px-2 py-1.5 text-xs text-zinc-100"
                  >
                    {(catalog?.synthetic?.operadores || ['+', '%']).map((op) => (
                      <option key={op} value={op}>
                        {op === '+' ? '+ Taxa a.a.' : '% do Índice'}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label
                    htmlFor="synth-val"
                    className="block text-[11px] text-zinc-400 mb-1"
                  >
                    Taxa / Percentual
                  </label>
                  <input
                    id="synth-val"
                    type="number"
                    step="0.1"
                    value={synthVal}
                    onChange={(e) => setSynthVal(e.target.value)}
                    className="w-full bg-zinc-950 border border-zinc-700 rounded-lg px-2 py-1.5 text-xs text-zinc-100"
                  />
                </div>
              </div>

              {synthError && <p className="text-xs text-red-400">{synthError}</p>}

              <Button type="submit" size="sm" variant="primary" className="w-full">
                <Plus className="w-3.5 h-3.5" />
                Adicionar Sintético
              </Button>
            </form>
          )}
        </div>
      )}
    </div>
  );
}

export default BenchmarkSelector;
