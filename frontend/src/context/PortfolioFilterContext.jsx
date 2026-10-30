import React, {
  createContext,
  useContext,
  useState,
  useCallback,
  useMemo,
  useEffect,
  useRef,
} from 'react';
import { useSearchParams, useInRouterContext } from 'react-router-dom';

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

export function isValidISODate(str) {
  if (typeof str !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(str)) {
    return false;
  }
  const [yStr, mStr, dStr] = str.split('-');
  const y = Number(yStr);
  const m = Number(mStr);
  const d = Number(dStr);
  if (m < 1 || m > 12 || d < 1 || d > 31) {
    return false;
  }
  const date = new Date(y, m - 1, d);
  return (
    date.getFullYear() === y &&
    date.getMonth() === m - 1 &&
    date.getDate() === d
  );
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

export function parsePeriodParams(search) {
  const defaultFallback = {
    period: 'Tudo',
    ...computePresetDates('Tudo'),
  };

  if (!search) {
    return defaultFallback;
  }

  let params;
  if (search instanceof URLSearchParams) {
    params = search;
  } else if (typeof search === 'string') {
    const clean = search.startsWith('?') ? search.slice(1) : search;
    params = new URLSearchParams(clean);
  } else {
    return defaultFallback;
  }

  const rawPeriod = params.get('period');
  if (!rawPeriod) {
    return defaultFallback;
  }

  const normalized = rawPeriod.toLowerCase();
  if (normalized === '1a') {
    return { period: '1A', ...computePresetDates('1A') };
  }
  if (normalized === '2a') {
    return { period: '2A', ...computePresetDates('2A') };
  }
  if (normalized === '5a') {
    return { period: '5A', ...computePresetDates('5A') };
  }
  if (normalized === 'tudo') {
    return { period: 'Tudo', ...computePresetDates('Tudo') };
  }
  if (normalized === 'custom') {
    const start = params.get('start');
    const end = params.get('end');
    if (isValidISODate(start) && isValidISODate(end) && start <= end) {
      return { period: 'Personalizado', start, end };
    }
    return defaultFallback;
  }

  return defaultFallback;
}

export function periodToParams(periodState) {
  const params = new URLSearchParams();
  if (!periodState) {
    params.set('period', 'tudo');
    return params;
  }

  const { period, start, end } = periodState;
  const PERIOD_MAP = {
    '1A': '1a',
    '2A': '2a',
    '5A': '5a',
    'Tudo': 'tudo',
    'Personalizado': 'custom',
  };

  const code = PERIOD_MAP[period] || 'tudo';
  params.set('period', code);

  if (
    code === 'custom' &&
    isValidISODate(start) &&
    isValidISODate(end) &&
    start !== ALL_TIME_START &&
    start <= end
  ) {
    params.set('start', start);
    params.set('end', end);
  }

  return params;
}

function usePortfolioFilterState({
  initialPeriod = 'Tudo',
  initialBenchmarks = ['CDI', 'BOVA11'],
  initialPeriodState,
  onPeriodChange,
}) {
  const initialDates = useMemo(
    () => initialPeriodState || { period: initialPeriod, ...computePresetDates(initialPeriod) },
    [initialPeriodState, initialPeriod]
  );

  const [period, setPeriodState] = useState(initialDates.period);
  const [start, setStart] = useState(initialDates.start);
  const [end, setEnd] = useState(initialDates.end);
  const [benchmarks, setBenchmarks] = useState(initialBenchmarks);
  const [synthetics, setSynthetics] = useState([]);
  const [ativo, setAtivo] = useState(null);
  const [classe, setClasse] = useState(null);

  const setPeriod = useCallback(
    (nextPeriod) => {
      setPeriodState(nextPeriod);
      let nextStart = start;
      let nextEnd = end;
      if (nextPeriod !== 'Personalizado') {
        const dates = computePresetDates(nextPeriod);
        nextStart = dates.start;
        nextEnd = dates.end;
        setStart(nextStart);
        setEnd(nextEnd);
      }
      onPeriodChange?.({ period: nextPeriod, start: nextStart, end: nextEnd });
    },
    [start, end, onPeriodChange]
  );

  const setCustomDates = useCallback(
    (customStart, customEnd) => {
      setPeriodState('Personalizado');
      const nextStart = customStart || null;
      const nextEnd = customEnd || null;
      setStart(nextStart);
      setEnd(nextEnd);
      onPeriodChange?.({ period: 'Personalizado', start: nextStart, end: nextEnd });
    },
    [onPeriodChange]
  );

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
    onPeriodChange?.({ period: 'Tudo', start: allDates.start, end: allDates.end });
  }, [initialBenchmarks, onPeriodChange]);

  const setExternalPeriod = useCallback((newPeriod, newStart, newEnd) => {
    setPeriodState(newPeriod);
    setStart(newStart);
    setEnd(newEnd);
  }, []);

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
      _setExternalPeriod: setExternalPeriod,
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
      setExternalPeriod,
    ]
  );

  return value;
}

function PortfolioFilterProviderRouted({
  children,
  initialPeriod = 'Tudo',
  initialBenchmarks = ['CDI', 'BOVA11'],
}) {
  const [searchParams, setSearchParams] = useSearchParams();
  const lastSyncedParamsRef = React.useRef(searchParams.toString());

  const initialPeriodState = useMemo(() => {
    if (searchParams.has('period')) {
      return parsePeriodParams(searchParams);
    }
    return null;
  }, []);

  const onPeriodChange = useCallback(
    ({ period, start, end }) => {
      setSearchParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          const params = periodToParams({ period, start, end });
          next.delete('period');
          next.delete('start');
          next.delete('end');
          for (const [k, v] of params.entries()) {
            next.set(k, v);
          }
          lastSyncedParamsRef.current = next.toString();
          return next;
        },
        { replace: true }
      );
    },
    [setSearchParams]
  );

  const filterState = usePortfolioFilterState({
    initialPeriod,
    initialBenchmarks,
    initialPeriodState,
    onPeriodChange,
  });

  const { _setExternalPeriod } = filterState;

  useEffect(() => {
    const currentStr = searchParams.toString();
    if (currentStr === lastSyncedParamsRef.current) {
      return;
    }
    lastSyncedParamsRef.current = currentStr;
    if (searchParams.has('period')) {
      const parsed = parsePeriodParams(searchParams);
      _setExternalPeriod(parsed.period, parsed.start, parsed.end);
    }
  }, [searchParams, _setExternalPeriod]);

  return (
    <PortfolioFilterContext.Provider value={filterState}>
      {children}
    </PortfolioFilterContext.Provider>
  );
}

function PortfolioFilterProviderBase({
  children,
  initialPeriod = 'Tudo',
  initialBenchmarks = ['CDI', 'BOVA11'],
}) {
  const filterState = usePortfolioFilterState({
    initialPeriod,
    initialBenchmarks,
  });

  return (
    <PortfolioFilterContext.Provider value={filterState}>
      {children}
    </PortfolioFilterContext.Provider>
  );
}

export function PortfolioFilterProvider(props) {
  const inRouter = useInRouterContext?.() ?? false;
  if (inRouter) {
    return <PortfolioFilterProviderRouted {...props} />;
  }
  return <PortfolioFilterProviderBase {...props} />;
}

export function usePortfolioFilters() {
  const context = useContext(PortfolioFilterContext);
  if (!context) {
    throw new Error('usePortfolioFilters deve ser utilizado dentro de um PortfolioFilterProvider');
  }
  return context;
}

export default PortfolioFilterContext;
