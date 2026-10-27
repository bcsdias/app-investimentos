import React, { createContext, useContext, useState, useCallback, useMemo } from 'react';

const PortfolioFilterContext = createContext(null);

const PERIOD_DAYS = {
  '1A': 365,
  '2A': 730,
  '5A': 1825,
};

// "Tudo": data bem antiga; o backend recorta pelo início real e devolve em meta.start.
export const ALL_TIME_START = '1900-01-01';

export function toLocalISODate(date) {
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, '0');
  const d = String(date.getDate()).padStart(2, '0');
  return `${y}-${m}-${d}`;
}

export function computePresetDates(periodPreset, baseDate = new Date()) {
  if (periodPreset === 'Tudo') {
    return { start: ALL_TIME_START, end: null };
  }
  const days = PERIOD_DAYS[periodPreset];
  if (!days) {
    return { start: null, end: null };
  }
  const endObj = new Date(baseDate);
  const startObj = new Date(baseDate);
  startObj.setDate(startObj.getDate() - days);

  return {
    start: toLocalISODate(startObj),
    end: toLocalISODate(endObj),
  };
}

export function formatSyntheticLabel(synth) {
  if (!synth) return '';
  if (synth.operador === '+') {
    return `${synth.base} + ${synth.valor}%`;
  }
  return `${synth.valor}% do ${synth.base}`;
}

export function PortfolioFilterProvider({
  children,
  initialPeriod = 'Tudo',
  initialBenchmarks = ['CDI', 'BOVA11'],
}) {
  const initialDates = computePresetDates(initialPeriod);
  const [period, setPeriodState] = useState(initialPeriod);
  const [start, setStart] = useState(initialDates.start);
  const [end, setEnd] = useState(initialDates.end);
  const [benchmarks, setBenchmarks] = useState(initialBenchmarks);
  const [synthetics, setSynthetics] = useState([]);
  const [ativo, setAtivo] = useState(null);
  const [classe, setClasse] = useState(null);

  const setPeriod = useCallback((nextPeriod) => {
    setPeriodState(nextPeriod);
    if (nextPeriod !== 'Personalizado') {
      const dates = computePresetDates(nextPeriod);
      setStart(dates.start);
      setEnd(dates.end);
    }
  }, []);

  const setCustomDates = useCallback((customStart, customEnd) => {
    setPeriodState('Personalizado');
    setStart(customStart || null);
    setEnd(customEnd || null);
  }, []);

  const addBenchmark = useCallback((bm) => {
    const trimmed = (bm || '').trim();
    if (!trimmed) return;
    setBenchmarks((prev) => (prev.includes(trimmed) ? prev : [...prev, trimmed]));
  }, []);

  const removeBenchmark = useCallback((bm) => {
    setBenchmarks((prev) => prev.filter((item) => item !== bm));
  }, []);

  const toggleBenchmark = useCallback((bm) => {
    const trimmed = (bm || '').trim();
    if (!trimmed) return;
    setBenchmarks((prev) =>
      prev.includes(trimmed) ? prev.filter((item) => item !== trimmed) : [...prev, trimmed]
    );
  }, []);

  const addSynthetic = useCallback((synth) => {
    if (!synth || !synth.base || !synth.operador || !(Number(synth.valor) > 0)) return;
    const normalized = {
      base: synth.base,
      operador: synth.operador,
      valor: Number(synth.valor),
    };
    setSynthetics((prev) => {
      const exists = prev.some(
        (item) =>
          item.base === normalized.base &&
          item.operador === normalized.operador &&
          item.valor === normalized.valor
      );
      return exists ? prev : [...prev, normalized];
    });
  }, []);

  const removeSynthetic = useCallback((synthToRemove) => {
    setSynthetics((prev) =>
      prev.filter(
        (item) =>
          !(
            item.base === synthToRemove.base &&
            item.operador === synthToRemove.operador &&
            item.valor === synthToRemove.valor
          )
      )
    );
  }, []);

  const resetFilters = useCallback(() => {
    const allDates = computePresetDates('Tudo');
    setPeriodState('Tudo');
    setStart(allDates.start);
    setEnd(allDates.end);
    setBenchmarks(initialBenchmarks);
    setSynthetics([]);
    setAtivo(null);
    setClasse(null);
  }, [initialBenchmarks]);

  const value = useMemo(
    () => ({
      period,
      setPeriod,
      start,
      end,
      setCustomDates,
      benchmarks,
      addBenchmark,
      removeBenchmark,
      toggleBenchmark,
      synthetics,
      addSynthetic,
      removeSynthetic,
      ativo,
      setAtivo,
      classe,
      setClasse,
      resetFilters,
    }),
    [
      period,
      setPeriod,
      start,
      end,
      setCustomDates,
      benchmarks,
      addBenchmark,
      removeBenchmark,
      toggleBenchmark,
      synthetics,
      addSynthetic,
      removeSynthetic,
      ativo,
      classe,
      resetFilters,
    ]
  );

  return (
    <PortfolioFilterContext.Provider value={value}>{children}</PortfolioFilterContext.Provider>
  );
}

export function usePortfolioFilters() {
  const context = useContext(PortfolioFilterContext);
  if (!context) {
    throw new Error('usePortfolioFilters deve ser utilizado dentro de um PortfolioFilterProvider');
  }
  return context;
}

export default PortfolioFilterContext;

