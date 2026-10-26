import { useMemo } from 'react';
import { useQueries } from '@tanstack/react-query';
import { usePortfolioFilters } from '../context/PortfolioFilterContext';
import { portfolioApi } from '../api/portfolio';

const STALE_TIME_MS = 5 * 60 * 1000;

export function usePortfolioQueries() {
  const { start, end, benchmarks, synthetics, ativo, classe } = usePortfolioFilters();

  const payload = useMemo(
    () => ({
      start: start || null,
      end: end || null,
      ativo: ativo || null,
      classe: classe || null,
      benchmarks: benchmarks || [],
      synthetics: synthetics || [],
    }),
    [start, end, ativo, classe, benchmarks, synthetics]
  );

  const results = useQueries({
    queries: [
      {
        queryKey: ['portfolio', 'summary', payload],
        queryFn: ({ signal }) => portfolioApi.fetchSummary(payload, signal),
        staleTime: STALE_TIME_MS,
      },
      {
        queryKey: ['portfolio', 'twr-evolution', payload],
        queryFn: ({ signal }) => portfolioApi.fetchTwrEvolution(payload, signal),
        staleTime: STALE_TIME_MS,
      },
      {
        queryKey: ['portfolio', 'drawdown', payload],
        queryFn: ({ signal }) => portfolioApi.fetchDrawdown(payload, signal),
        staleTime: STALE_TIME_MS,
      },
      {
        queryKey: ['portfolio', 'rolling-metrics', payload],
        queryFn: ({ signal }) => portfolioApi.fetchRollingMetrics(payload, signal),
        staleTime: STALE_TIME_MS,
      },
      {
        queryKey: ['portfolio', 'risk-return', payload],
        queryFn: ({ signal }) => portfolioApi.fetchRiskReturn(payload, signal),
        staleTime: STALE_TIME_MS,
      },
      {
        queryKey: ['portfolio', 'shadow', payload],
        queryFn: ({ signal }) => portfolioApi.fetchShadow(payload, signal),
        staleTime: STALE_TIME_MS,
      },
      {
        queryKey: ['portfolio', 'yearly-summary', payload],
        queryFn: ({ signal }) => portfolioApi.fetchYearlySummary(payload, signal),
        staleTime: STALE_TIME_MS,
      },
      {
        queryKey: ['portfolio', 'irr-evolution', payload],
        queryFn: ({ signal }) => portfolioApi.fetchIrrEvolution(payload, signal),
        staleTime: STALE_TIME_MS,
      },
    ],
  });

  return {
    summary: results[0],
    twr: results[1],
    drawdown: results[2],
    rolling: results[3],
    riskReturn: results[4],
    shadow: results[5],
    yearly: results[6],
    irr: results[7],
  };
}

export default usePortfolioQueries;
