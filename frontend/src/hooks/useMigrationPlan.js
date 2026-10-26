import { useState, useEffect, useCallback, useMemo } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { migrationApi } from '../api/migration';

const STORAGE_KEY = 'migration_ir_draft_v1';

function loadDraft() {
  if (typeof window === 'undefined' || !window.sessionStorage) return null;
  try {
    const raw = window.sessionStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

export function useMigrationPlan() {
  const saved = useMemo(() => loadDraft(), []);

  const [targets, setTargets] = useState(
    saved?.targets || [{ ticker: 'BOVA11', weight: 100 }]
  );
  const [actionsMap, setActionsMap] = useState(saved?.actionsMap || {});
  const [monthlyCap, setMonthlyCap] = useState(saved?.monthlyCap ?? 20000);
  const [initialCash, setInitialCash] = useState(saved?.initialCash ?? 0);
  const [alreadySold, setAlreadySold] = useState(saved?.alreadySold ?? 0);
  const [contributionsOnly, setContributionsOnly] = useState(
    saved?.contributionsOnly ?? false
  );

  useEffect(() => {
    if (typeof window === 'undefined' || !window.sessionStorage) return;
    try {
      window.sessionStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({
          targets,
          actionsMap,
          monthlyCap,
          initialCash,
          alreadySold,
          contributionsOnly,
        })
      );
    } catch {
      // ignore storage quota errors
    }
  }, [targets, actionsMap, monthlyCap, initialCash, alreadySold, contributionsOnly]);

  const walletQuery = useQuery({
    queryKey: ['migration', 'wallet'],
    queryFn: ({ signal }) => migrationApi.getWallet(signal),
    staleTime: 5 * 60 * 1000,
  });

  const holdings = walletQuery.data?.data || [];

  const handleActionChange = useCallback((ativo, action, target = null) => {
    setActionsMap((prev) => ({
      ...prev,
      [ativo]: {
        ...(prev[ativo] || {}),
        action,
        target: action === 'count_as' ? target : null,
      },
    }));
  }, []);

  const handleManualSaldoChange = useCallback((ativo, saldoValue) => {
    setActionsMap((prev) => ({
      ...prev,
      [ativo]: {
        ...(prev[ativo] || {}),
        saldo:
          saldoValue === '' || saldoValue === null || Number.isNaN(Number(saldoValue))
            ? null
            : Number(saldoValue),
      },
    }));
  }, []);

  const planPayload = useMemo(() => {
    const validTargets = targets
      .filter((t) => (t.ticker || '').trim().length > 0 && Number(t.weight) > 0)
      .map((t) => ({
        ticker: t.ticker.trim().toUpperCase(),
        weight: Number(t.weight),
      }));

    const targetSet = new Set(validTargets.map((t) => t.ticker));

    const holdingsPayload = holdings.map((h) => {
      const cfg = actionsMap[h.ativo] || {};
      const isTargetHolding = targetSet.has((h.ativo || '').toUpperCase());

      if (isTargetHolding) {
        return {
          ativo: h.ativo,
          action: 'ignore',
          target: null,
          saldo: null,
        };
      }

      const defaultAction = h.price === null || h.price === undefined ? 'ignore' : 'sell';
      let resolvedAction = cfg.action || defaultAction;
      if (resolvedAction === 'sell' && (h.price === null || h.price === undefined)) {
        resolvedAction = 'ignore';
      }

      const resolvedTarget = resolvedAction === 'count_as' ? cfg.target || null : null;
      const resolvedSaldo =
        resolvedAction === 'count_as' && cfg.saldo !== undefined && cfg.saldo !== null
          ? Number(cfg.saldo)
          : null;

      return {
        ativo: h.ativo,
        action: resolvedAction,
        target: resolvedTarget,
        saldo: resolvedSaldo,
      };
    });

    return {
      targets: validTargets,
      holdings: holdingsPayload,
      monthly_cap: Number(monthlyCap) || 20000,
      initial_cash: Number(initialCash) || 0,
      already_sold_this_month: Number(alreadySold) || 0,
      contributions_only: Boolean(contributionsOnly),
    };
  }, [targets, holdings, actionsMap, monthlyCap, initialCash, alreadySold, contributionsOnly]);

  const planMutation = useMutation({
    mutationFn: (customPayload) =>
      migrationApi.calculatePlan(customPayload || planPayload),
  });

  return {
    walletQuery,
    holdings,
    targets,
    setTargets,
    actionsMap,
    handleActionChange,
    handleManualSaldoChange,
    monthlyCap,
    setMonthlyCap,
    initialCash,
    setInitialCash,
    alreadySold,
    setAlreadySold,
    contributionsOnly,
    setContributionsOnly,
    planPayload,
    planMutation,
  };
}

export default useMigrationPlan;

