import React, { useState } from 'react';
import { Plus, Trash2, Calculator } from 'lucide-react';
import { Card } from '../common/Card';
import { Button } from '../common/Button';
import { Badge } from '../common/Badge';
import { assetsApi } from '../../api/assets';

// Mesma tolerância de `_WEIGHT_TOL` em backend/apps/migration_ir/planner.py.
const WEIGHT_TOLERANCE = 1e-6;

export function isTargetListValid(targets = []) {
  if (!Array.isArray(targets) || targets.length === 0) return false;
  const totalWeight = targets.reduce((acc, t) => acc + (Number(t.weight) || 0), 0);
  if (Math.abs(totalWeight - 100) > WEIGHT_TOLERANCE) return false;

  const seen = new Set();
  for (const t of targets) {
    const ticker = String(t.ticker || '').trim().toUpperCase();
    const weight = Number(t.weight);
    if (!ticker || !Number.isFinite(weight) || weight <= 0 || seen.has(ticker)) {
      return false;
    }
    seen.add(ticker);
  }
  return true;
}

export function TargetAllocationForm({
  targets = [],
  onChangeTargets,
  monthlyCap = 20000,
  onChangeMonthlyCap,
  alreadySold = 0,
  onChangeAlreadySold,
  initialCash = 0,
  onChangeInitialCash,
  contributionsOnly = false,
  onChangeContributionsOnly,
  onSimulate,
  isSimulating = false,
  actionSlot = null,
}) {
  const [suggestions, setSuggestions] = useState({});

  const totalWeight = targets.reduce((acc, t) => acc + (Number(t.weight) || 0), 0);
  const isTotalValid = isTargetListValid(targets);
  const diff = 100 - totalWeight;

  const handleUpdateRow = (index, field, value) => {
    if (!onChangeTargets) return;
    const next = targets.map((item, idx) =>
      idx === index ? { ...item, [field]: value } : item
    );
    onChangeTargets(next);

    if (field === 'ticker') {
      const term = String(value || '').trim();
      if (term.length >= 2) {
        assetsApi
          .searchAssets(term, 5)
          .then((res) => {
            setSuggestions((prev) => ({
              ...prev,
              [index]: Array.isArray(res) ? res : [],
            }));
          })
          .catch(() => {});
      } else {
        setSuggestions((prev) => ({ ...prev, [index]: [] }));
      }
    }
  };

  const handleSelectSuggestion = (index, sigla) => {
    if (!onChangeTargets) return;
    const next = targets.map((item, idx) =>
      idx === index ? { ...item, ticker: sigla.toUpperCase() } : item
    );
    onChangeTargets(next);
    setSuggestions((prev) => ({ ...prev, [index]: [] }));
  };

  const handleAddRow = () => {
    if (!onChangeTargets) return;
    onChangeTargets([...targets, { ticker: '', weight: Math.max(0, Number(diff.toFixed(2))) }]);
  };

  const handleRemoveRow = (index) => {
    if (!onChangeTargets) return;
    onChangeTargets(targets.filter((_, idx) => idx !== index));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (isTotalValid && onSimulate) {
      onSimulate();
    }
  };

  return (
    <Card
      title="2. Alocação-Alvo e Limite Mensal"
      subtitle="Defina os ativos de destino e seus pesos percentuais (soma obrigatória de 100%)"
      action={
        <Badge variant={isTotalValid ? 'success' : 'danger'}>
          Soma: {totalWeight.toFixed(1).replace('.', ',')}%
          {!isTotalValid &&
            ` (${diff > 0 ? `faltam ${diff.toFixed(1)}%` : `excesso de ${Math.abs(diff).toFixed(1)}%`})`}
        </Badge>
      }
    >
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="space-y-2">
          {targets.map((target, idx) => {
            const rowSuggestions = suggestions[idx] || [];
            return (
              <div key={idx} className="flex items-center gap-3">
                <div className="flex-1 relative">
                  <input
                    type="text"
                    aria-label={`Ticker alvo ${idx + 1}`}
                    placeholder="Ticker alvo (ex: IVVB11, BOVA11)"
                    value={target.ticker}
                    onChange={(e) =>
                      handleUpdateRow(idx, 'ticker', e.target.value.toUpperCase())
                    }
                    className="w-full bg-zinc-950 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-100 placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-emerald-500"
                  />
                  {rowSuggestions.length > 0 && (
                    <div className="absolute left-0 right-0 mt-1 z-20 bg-zinc-900 border border-zinc-700 rounded-lg shadow-xl max-h-40 overflow-y-auto">
                      {rowSuggestions.map((asset) => (
                        <button
                          key={asset.yf_ticker}
                          type="button"
                          onClick={() => handleSelectSuggestion(idx, asset.sigla)}
                          className="w-full text-left px-3 py-1.5 text-xs text-zinc-200 hover:bg-zinc-800 flex items-center justify-between"
                        >
                          <span className="font-semibold">{asset.sigla}</span>
                          <span className="text-zinc-400 truncate ml-2">{asset.nome}</span>
                        </button>
                      ))}
                    </div>
                  )}
                </div>
                <div className="w-32 relative">
                  <input
                    type="number"
                    step="0.1"
                    aria-label={`Peso alvo ${idx + 1}`}
                    value={target.weight}
                    onChange={(e) =>
                      handleUpdateRow(idx, 'weight', Number(e.target.value))
                    }
                    className="w-full bg-zinc-950 border border-zinc-700 rounded-lg pl-3 pr-7 py-2 text-sm text-right text-zinc-100 focus:outline-none focus:ring-2 focus:ring-emerald-500"
                  />
                  <span className="absolute right-2.5 top-1/2 -translate-y-1/2 text-xs text-zinc-400">
                    %
                  </span>
                </div>
                {targets.length > 1 && (
                  <button
                    type="button"
                    aria-label={`Remover alvo ${idx + 1}`}
                    onClick={() => handleRemoveRow(idx)}
                    className="p-2 text-zinc-400 hover:text-red-400 rounded-lg hover:bg-zinc-800 transition-colors"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                )}
              </div>
            );
          })}
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2 border-t border-zinc-800/70">
          <div>
            <label
              htmlFor="monthly-cap-input"
              className="block text-xs font-medium text-zinc-400 mb-1"
            >
              Teto Mensal de Vendas de Ações (R$)
            </label>
            <input
              id="monthly-cap-input"
              type="number"
              step="100"
              value={monthlyCap}
              onChange={(e) =>
                onChangeMonthlyCap && onChangeMonthlyCap(Number(e.target.value))
              }
              className="w-full bg-zinc-950 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-100"
            />
          </div>
          <div>
            <label
              htmlFor="already-sold-input"
              className="block text-xs font-medium text-zinc-400 mb-1"
            >
              Vendas de Ações Já Realizadas no Mês Atual (R$)
            </label>
            <input
              id="already-sold-input"
              type="number"
              step="100"
              value={alreadySold}
              onChange={(e) =>
                onChangeAlreadySold && onChangeAlreadySold(Number(e.target.value))
              }
              className="w-full bg-zinc-950 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-100"
            />
          </div>
          <div>
            <label
              htmlFor="initial-cash-input"
              className="block text-xs font-medium text-zinc-400 mb-1"
            >
              Caixa disponível para aportes (R$)
            </label>
            <input
              id="initial-cash-input"
              type="number"
              min="0"
              step="100"
              value={initialCash}
              onChange={(e) =>
                onChangeInitialCash && onChangeInitialCash(Number(e.target.value))
              }
              className="w-full bg-zinc-950 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-100"
            />
          </div>
          <div className="flex items-end">
            <label className="inline-flex items-center gap-2 text-sm text-zinc-300 pb-2">
              <input
                type="checkbox"
                checked={contributionsOnly}
                onChange={(e) =>
                  onChangeContributionsOnly && onChangeContributionsOnly(e.target.checked)
                }
                className="accent-emerald-500"
              />
              Apenas aportes (não vender nada)
            </label>
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3 pt-2">
          <Button type="button" variant="secondary" size="sm" onClick={handleAddRow}>
            <Plus className="w-4 h-4" />
            Adicionar Ativo-Alvo
          </Button>

          <div className="flex items-center gap-3">
            {actionSlot}
            <Button
              type="submit"
              variant="primary"
              disabled={!isTotalValid}
              isLoading={isSimulating}
            >
              <Calculator className="w-4 h-4" />
              Calcular Plano
            </Button>
          </div>
        </div>
      </form>
    </Card>
  );
}

export default TargetAllocationForm;

