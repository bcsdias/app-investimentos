// Códigos emitidos por backend/apps/migration_ir/{planner,prices,wallet}.py.
const SIMPLE = {
  exceeds_exemption: 'Vendas de ações do mês passam de R$ 20.000: o lucro do mês é tributado.',
  already_over_exemption: 'As vendas já feitas neste mês passam de R$ 20.000.',
  exceeds_cap: 'Uma única unidade ultrapassa o teto mensal informado.',
};

const WITH_ATIVO = {
  cost_basis: (ativo) => `${ativo}: saldo calculado pelo custo (sem cotação de mercado).`,
  fx_approx: (ativo) => `${ativo}: cotação convertida pela PTAX (aproximada).`,
  not_estimated: (ativo) => `${ativo}: IR não estimado para esta classe.`,
  mixed_class: (ativo) => `${ativo}: aparece com mais de uma classe na carteira.`,
};

export function formatPlanWarning(code) {
  const text = String(code ?? '');
  if (SIMPLE[text]) return SIMPLE[text];
  const sep = text.indexOf(':');
  if (sep > 0) {
    const format = WITH_ATIVO[text.slice(0, sep)];
    if (format) return format(text.slice(sep + 1));
  }
  return text;
}
