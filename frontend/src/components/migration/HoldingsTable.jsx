import React from 'react';
import { Card } from '../common/Card';
import { Badge } from '../common/Badge';
import { formatCurrencyBRL } from '../portfolio/SummaryCards';
import { defaultActionFor } from '../../hooks/useMigrationPlan';

function getTaxBadge(taxTreatment) {
  switch (taxTreatment) {
    case 'exempt_20k':
      return { label: 'Isento R$ 20k', variant: 'success' };
    case 'taxed':
      return { label: 'Tributado', variant: 'warning' };
    default:
      return { label: 'Não estimado', variant: 'neutral' };
  }
}

export function HoldingsTable({
  holdings = [],
  targets = [],
  actionsMap = {},
  onChangeAction,
  onChangeManualSaldo,
}) {
  const validTargets = targets.filter((t) => (t.ticker || '').trim().length > 0);

  const handleSelectChange = (ativo, rawValue) => {
    if (!onChangeAction) return;
    if (rawValue.startsWith('count_as:')) {
      const targetTicker = rawValue.slice('count_as:'.length);
      onChangeAction(ativo, 'count_as', targetTicker);
    } else {
      onChangeAction(ativo, rawValue, null);
    }
  };

  return (
    <Card
      title="1. Custódia Atual e Tratamento por Papel"
      subtitle="Defina se cada ativo atual será vendido gradualmente, mantido como parte de um alvo ou ignorado"
    >
      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse text-sm">
          <thead>
            <tr className="border-b border-zinc-800 text-xs uppercase text-zinc-400">
              <th className="py-3 px-3 font-semibold">Ativo</th>
              <th className="py-3 px-3 font-semibold">Classe</th>
              <th className="py-3 px-3 font-semibold text-right">Qtd</th>
              <th className="py-3 px-3 font-semibold text-right">Preço Médio</th>
              <th className="py-3 px-3 font-semibold text-right">Cotação</th>
              <th className="py-3 px-3 font-semibold text-right">Saldo</th>
              <th className="py-3 px-3 font-semibold">Regra Fiscal</th>
              <th className="py-3 px-3 font-semibold">Ação no Plano</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-800/60">
            {holdings.map((h) => {
              const badge = getTaxBadge(h.tax_treatment);
              const cfg = actionsMap[h.ativo] || {};
              const isTargetHolding = validTargets.some(
                (t) => t.ticker.trim().toUpperCase() === (h.ativo || '').toUpperCase()
              );
              const hasPrice = h.price !== null && h.price !== undefined;
              const defaultAction = defaultActionFor(h, isTargetHolding);
              const selectVal = isTargetHolding
                ? 'ignore'
                : cfg.action === 'count_as' && cfg.target
                ? `count_as:${cfg.target}`
                : cfg.action || defaultAction;

              return (
                <tr key={h.ativo} className="hover:bg-zinc-800/30 transition-colors">
                  <td className="py-3 px-3 font-semibold text-zinc-100 whitespace-nowrap">
                    {h.ativo}
                  </td>
                  <td className="py-3 px-3 text-zinc-300 whitespace-nowrap">{h.classe}</td>
                  <td className="py-3 px-3 text-right text-zinc-300 whitespace-nowrap">
                    {Number(h.qtd).toLocaleString('pt-BR')}
                  </td>
                  <td className="py-3 px-3 text-right text-zinc-300 whitespace-nowrap">
                    {h.pm !== null && h.pm !== undefined ? formatCurrencyBRL(h.pm) : '—'}
                  </td>
                  <td className="py-3 px-3 text-right text-zinc-300 whitespace-nowrap">
                    {hasPrice ? formatCurrencyBRL(h.price) : '—'}
                  </td>
                  <td className="py-3 px-3 text-right whitespace-nowrap">
                    {h.saldo_source === 'cost' ? (
                      <input
                        type="number"
                        step="0.01"
                        aria-label={`Saldo manual para ${h.ativo}`}
                        defaultValue={cfg.saldo ?? h.saldo}
                        onChange={(e) =>
                          onChangeManualSaldo &&
                          onChangeManualSaldo(h.ativo, Number(e.target.value))
                        }
                        className="w-28 bg-zinc-950 border border-amber-500/40 rounded-lg px-2 py-1 text-right text-xs text-zinc-100 focus:outline-none focus:ring-2 focus:ring-emerald-500"
                      />
                    ) : (
                      <span className="font-medium text-zinc-100">
                        {formatCurrencyBRL(h.saldo)}
                      </span>
                    )}
                  </td>
                  <td className="py-3 px-3 whitespace-nowrap">
                    <Badge variant={badge.variant}>{badge.label}</Badge>
                  </td>
                  <td className="py-3 px-3 whitespace-nowrap">
                    <select
                      aria-label={`Ação para ${h.ativo}`}
                      value={selectVal}
                      disabled={isTargetHolding}
                      onChange={(e) => handleSelectChange(h.ativo, e.target.value)}
                      className="bg-zinc-950 border border-zinc-700 rounded-lg px-2.5 py-1.5 text-xs text-zinc-100 focus:outline-none focus:ring-2 focus:ring-emerald-500 disabled:opacity-60"
                    >
                      {hasPrice && !isTargetHolding && <option value="sell">Vender</option>}
                      {!isTargetHolding &&
                        validTargets.map((t) => (
                          <option key={t.ticker} value={`count_as:${t.ticker}`}>
                            Contar como {t.ticker}
                          </option>
                        ))}
                      <option value="ignore">
                        {isTargetHolding ? 'Ativo-Alvo (Saldo Direto)' : 'Ignorar'}
                      </option>
                    </select>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

export default HoldingsTable;

