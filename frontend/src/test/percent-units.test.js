import { describe, it, expect } from 'vitest';
import { toPct } from '../components/portfolio/charts/RiskReturnChart';
import { toVolPercent } from '../components/portfolio/charts/RollingSharpeChart';
import { toPercentPoint } from '../components/portfolio/charts/DrawdownChart';

// Contrato (ADR 0013): volatilidade, retorno_cagr, vol. móvel e drawdown são frações.
describe('conversão fração → % nos gráficos de risco', () => {
  it('toPct converte sempre, inclusive CAGR acima de 200%', () => {
    expect(toPct(2.14)).toBeCloseTo(214);
    expect(toPct(0.15)).toBeCloseTo(15);
    expect(toPct(null)).toBe(0);
  });

  it('toVolPercent converte sempre e mantém nulo', () => {
    expect(toVolPercent(2.5)).toBeCloseTo(250);
    expect(toVolPercent(null)).toBeNull();
  });

  it('toPercentPoint converte sempre', () => {
    expect(toPercentPoint(-0.25)).toBeCloseTo(-25);
    expect(toPercentPoint(1.5)).toBeCloseTo(150);
  });
});
