import React from 'react';
import { Wallet, ArrowDownToLine, TrendingUp, Percent, Activity } from 'lucide-react';
import { Skeleton } from '../common/Skeleton';
import { ErrorState } from '../common/ErrorState';

export function formatCurrencyBRL(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return '—';
  }
  return new Intl.NumberFormat('pt-BR', {
    style: 'currency',
    currency: 'BRL',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(Number(value));
}

export function formatSignedPercent(pctValue) {
  if (pctValue === null || pctValue === undefined || Number.isNaN(Number(pctValue))) {
    return '—';
  }
  const num = Number(pctValue);
  const sign = num > 0 ? '+' : '';
  return `${sign}${num.toFixed(2).replace('.', ',')}%`;
}

export function SummaryCards({ data, isLoading = false, isError = false, error = null, onRetry }) {
  if (isLoading) {
    const hasData = Boolean(data?.data);
    return (
      <div className="space-y-3">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
          {Array.from({ length: 5 }).map((_, idx) => (
            <div
              key={idx}
              className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-4 space-y-3"
            >
              <Skeleton className="h-4 w-28" />
              <Skeleton className="h-7 w-36" />
            </div>
          ))}
        </div>
        {!hasData && (
          <p className="text-xs text-zinc-400 text-center sm:text-left">
            Buscando sua carteira na DLP — a primeira carga pode levar até um minuto.
          </p>
        )}
      </div>
    );
  }

  if (isError || error) {
    return <ErrorState error={error} onRetry={onRetry} />;
  }

  const raw = data?.data || {};

  const patrimonio = raw.patrimonio ?? null;
  const aportes = raw.aportes_periodo ?? null;
  const ganho = raw.ganho_periodo ?? null;

  const twrPercent =
    raw.twr_total !== undefined && raw.twr_total !== null
      ? Number(raw.twr_total)
      : null;

  const isAnnual = raw.xirr_is_annual !== false;
  const tirLabel = isAnnual ? 'TIR (XIRR a.a.)' : 'TIR no período';
  const targetTir =
    !isAnnual && raw.xirr_period !== undefined && raw.xirr_period !== null
      ? raw.xirr_period
      : raw.xirr;

  const xirrPercent =
    targetTir !== undefined && targetTir !== null
      ? Number(targetTir)
      : null;

  const cards = [
    {
      label: 'Patrimônio Atual',
      value: formatCurrencyBRL(patrimonio),
      icon: Wallet,
      colorClass: 'text-zinc-100',
    },
    {
      label: 'Total Aportado',
      value: formatCurrencyBRL(aportes),
      icon: ArrowDownToLine,
      colorClass: 'text-zinc-100',
    },
    {
      label: 'Ganho de Capital',
      value: formatCurrencyBRL(ganho),
      icon: TrendingUp,
      colorClass:
        ganho === null
          ? 'text-zinc-400'
          : Number(ganho) >= 0
          ? 'text-emerald-400'
          : 'text-red-400',
    },
    {
      label: 'Rentabilidade TWR',
      value: formatSignedPercent(twrPercent),
      icon: Percent,
      colorClass:
        twrPercent === null
          ? 'text-zinc-400'
          : twrPercent >= 0
          ? 'text-emerald-400'
          : 'text-red-400',
    },
    {
      label: tirLabel,
      value: formatSignedPercent(xirrPercent),
      icon: Activity,
      colorClass:
        xirrPercent === null
          ? 'text-zinc-400'
          : xirrPercent >= 0
          ? 'text-emerald-400'
          : 'text-red-400',
    },
  ];

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
      {cards.map((card) => {
        const Icon = card.icon;
        return (
          <div
            key={card.label}
            className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-4 flex flex-col justify-between shadow-lg shadow-black/20"
          >
            <div className="flex items-center justify-between text-xs text-zinc-400 mb-2">
              <span className="font-medium">{card.label}</span>
              <Icon className="w-4 h-4 text-zinc-500" />
            </div>
            <div className={`text-xl font-bold tracking-tight ${card.colorClass}`}>
              {card.value}
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default SummaryCards;

