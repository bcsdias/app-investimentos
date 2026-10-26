import React from 'react';
import { AlertTriangle, Calendar, ArrowUpRight, ArrowDownLeft } from 'lucide-react';
import { Card } from '../common/Card';
import { Badge } from '../common/Badge';
import { formatCurrencyBRL } from '../portfolio/SummaryCards';

export function MonthlyPurchasesTable({ plan }) {
  if (!plan) return null;

  const data = plan.data || plan;
  const meta = plan.meta || {};
  const months = data.months || [];
  const totals = data.totals || {};
  const globalWarnings = meta.warnings || [];

  return (
    <div className="space-y-4">
      {globalWarnings.length > 0 && (
        <div className="p-4 rounded-xl bg-amber-500/10 border border-amber-500/30 space-y-1.5">
          <div className="flex items-center gap-2 text-amber-300 font-semibold text-sm">
            <AlertTriangle className="w-4 h-4 text-amber-400" />
            <span>Alertas do Planejamento Fiscal</span>
          </div>
          <ul className="list-disc list-inside text-xs text-amber-200/90 space-y-1">
            {globalWarnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </div>
      )}

      <Card
        title="3. Cronograma Mensal de Migração"
        subtitle={`Plano estimado em ${totals.months || months.length} mes(es) • Total vendido: ${formatCurrencyBRL(totals.sold || 0)} • DARF total estimado: ${formatCurrencyBRL(totals.ir_estimado || 0)}`}
      >
        <div className="space-y-6">
          {months.map((m) => (
            <div
              key={m.month}
              className="p-4 rounded-xl bg-zinc-950 border border-zinc-800 space-y-4"
            >
              <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-zinc-800">
                <div className="flex items-center gap-2">
                  <Calendar className="w-4 h-4 text-emerald-400" />
                  <h4 className="font-semibold text-zinc-100">Mês {m.month}</h4>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <Badge variant="success">
                    Vendas Isentas (Ações): {formatCurrencyBRL(m.exempt_sales_total)}
                  </Badge>
                  {m.taxed_sales_total > 0 && (
                    <Badge variant="warning">
                      Vendas Tributadas: {formatCurrencyBRL(m.taxed_sales_total)}
                    </Badge>
                  )}
                  {m.ir_estimado_total > 0 && (
                    <Badge variant="danger">
                      DARF Estimado: {formatCurrencyBRL(m.ir_estimado_total)}
                    </Badge>
                  )}
                </div>
              </div>

              {Array.isArray(m.warnings) && m.warnings.length > 0 && (
                <div className="p-2.5 rounded-lg bg-amber-500/10 border border-amber-500/30 text-xs text-amber-300 space-y-1">
                  {m.warnings.map((mw, idx) => (
                    <p key={idx}>• {mw}</p>
                  ))}
                </div>
              )}

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {/* Vendas do mês */}
                <div className="space-y-2">
                  <p className="text-xs font-semibold uppercase tracking-wider text-zinc-400 flex items-center gap-1.5">
                    <ArrowUpRight className="w-3.5 h-3.5 text-amber-400" />
                    Ordens de Venda ({m.sales?.length || 0})
                  </p>
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs text-left">
                      <thead>
                        <tr className="border-b border-zinc-800 text-zinc-400">
                          <th className="py-1.5 px-2">Ativo</th>
                          <th className="py-1.5 px-2 text-right">Qtd</th>
                          <th className="py-1.5 px-2 text-right">Valor</th>
                          <th className="py-1.5 px-2 text-right">IR Est.</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-zinc-800/50">
                        {(m.sales || []).map((s, idx) => (
                          <tr key={`${s.ativo}-${idx}`}>
                            <td className="py-1.5 px-2 font-medium text-zinc-200">{s.ativo}</td>
                            <td className="py-1.5 px-2 text-right text-zinc-300">
                              {Number(s.qtd).toLocaleString('pt-BR')}
                            </td>
                            <td className="py-1.5 px-2 text-right text-zinc-200">
                              {formatCurrencyBRL(s.valor)}
                            </td>
                            <td className="py-1.5 px-2 text-right text-zinc-400">
                              {s.ir_estimado ? formatCurrencyBRL(s.ir_estimado) : 'R$ 0,00'}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Compras do mês */}
                <div className="space-y-2">
                  <p className="text-xs font-semibold uppercase tracking-wider text-zinc-400 flex items-center gap-1.5">
                    <ArrowDownLeft className="w-3.5 h-3.5 text-emerald-400" />
                    Compras Sugeridas ({m.purchases?.length || 0})
                  </p>
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs text-left">
                      <thead>
                        <tr className="border-b border-zinc-800 text-zinc-400">
                          <th className="py-1.5 px-2">Ticker Alvo</th>
                          <th className="py-1.5 px-2 text-right">Qtd Sugerida</th>
                          <th className="py-1.5 px-2 text-right">Valor Alocado</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-zinc-800/50">
                        {(m.purchases || []).map((p, idx) => (
                          <tr key={`${p.ticker}-${idx}`}>
                            <td className="py-1.5 px-2 font-medium text-emerald-400">
                              {p.ticker}
                            </td>
                            <td className="py-1.5 px-2 text-right text-zinc-300">
                              {Number(p.qtd_sugerida).toLocaleString('pt-BR')}
                            </td>
                            <td className="py-1.5 px-2 text-right text-zinc-200">
                              {formatCurrencyBRL(p.valor)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        {meta.disclaimer && (
          <p className="text-[11px] text-zinc-500 mt-4 pt-3 border-t border-zinc-800">
            {meta.disclaimer}
          </p>
        )}
      </Card>
    </div>
  );
}

export default MonthlyPurchasesTable;
