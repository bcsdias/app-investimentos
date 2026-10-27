import React from 'react';
import { useMigrationPlan } from '../hooks/useMigrationPlan';
import { HoldingsTable } from '../components/migration/HoldingsTable';
import {
  TargetAllocationForm,
  isTargetListValid,
} from '../components/migration/TargetAllocationForm';
import { MonthlyPurchasesTable } from '../components/migration/MonthlyPurchasesTable';
import { ExportPdfButton } from '../components/migration/ExportPdfButton';
import { Skeleton } from '../components/common/Skeleton';
import { EmptyState } from '../components/common/EmptyState';
import { DlpTokenRequiredState } from '../components/dlp/DlpTokenRequiredState';

export function MigrationPage() {
  const {
    tokenConfigured,
    walletQuery,
    holdings,
    targets,
    setTargets,
    actionsMap,
    handleActionChange,
    handleManualSaldoChange,
    monthlyCap,
    setMonthlyCap,
    alreadySold,
    setAlreadySold,
    planPayload,
    planMutation,
  } = useMigrationPlan();

  const isPortfolioEmpty =
    walletQuery.error?.response?.status === 404 &&
    walletQuery.error?.response?.data?.code === 'portfolio_empty';

  const isTotalValid = isTargetListValid(targets);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-zinc-100">
          Planejador de Migração com Isenção de IR
        </h1>
        <p className="text-sm text-zinc-400">
          Simule a transição gradual da sua custódia atual para uma carteira-alvo respeitando o limite mensal de R$ 20.000 em vendas de ações
        </p>
      </div>

      {tokenConfigured === false ? (
        <DlpTokenRequiredState />
      ) : tokenConfigured === undefined || walletQuery.isLoading ? (
        <div className="space-y-4">
          <Skeleton className="h-64 w-full" />
          <Skeleton className="h-48 w-full" />
        </div>
      ) : isPortfolioEmpty ? (
        <EmptyState
          title="Nenhum papel encontrado na custódia"
          description="Sua carteira não possui ativos com saldo positivo na DLP Invest. Cadastre suas movimentações ou verifique o token DLP configurado."
        />
      ) : (
        <>
          <HoldingsTable
            holdings={holdings}
            targets={targets}
            actionsMap={actionsMap}
            onChangeAction={handleActionChange}
            onChangeManualSaldo={handleManualSaldoChange}
          />

          <TargetAllocationForm
            targets={targets}
            onChangeTargets={setTargets}
            monthlyCap={monthlyCap}
            onChangeMonthlyCap={setMonthlyCap}
            alreadySold={alreadySold}
            onChangeAlreadySold={setAlreadySold}
            onSimulate={() => planMutation.mutate(planPayload)}
            isSimulating={planMutation.isPending}
            actionSlot={
              <ExportPdfButton
                payload={planPayload}
                disabled={!isTotalValid || holdings.length === 0}
              />
            }
          />

          {planMutation.isError && (
            <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/30 text-sm text-red-300">
              {planMutation.error?.response?.data?.message ||
                'Não foi possível calcular o plano de migração. Verifique os dados informados.'}
            </div>
          )}

          {planMutation.data && <MonthlyPurchasesTable plan={planMutation.data} />}
        </>
      )}
    </div>
  );
}

export default MigrationPage;

