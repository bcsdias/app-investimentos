import React, { useState, useEffect } from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { useQueryClient } from '@tanstack/react-query';
import {
  TrendingUp,
  RefreshCw,
  KeyRound,
  LogOut,
  CheckCircle2,
  AlertTriangle,
  Calculator,
  LineChart,
} from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import { useDlpTokenStatus, DLP_TOKEN_STATUS_KEY } from '../../hooks/useDlpTokenStatus';
import { api } from '../../api/client';
import { Button } from '../common/Button';
import { DlpTokenModal } from '../dlp/DlpTokenModal';

export function Header() {
  const { user, logout } = useAuth();
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [alertInfo, setAlertInfo] = useState({ reason: null, message: null });
  const [isRefreshing, setIsRefreshing] = useState(false);

  const { data: tokenStatus, refetch: refetchTokenStatus } = useDlpTokenStatus();

  // Sair é explícito: vai para /login sem `state.from`, para o próximo login começar
  // nos filtros padrão. Sessão expirada ou link sem sessão passam pelo RequireAuth,
  // que guarda a rota de origem (com ?period=) para o LoginPage restaurar.
  const handleLogout = async () => {
    await logout();
    navigate('/login', { replace: true });
  };

  useEffect(() => {
    const handleOpenDlpModal = (event) => {
      const detail = event.detail || {};
      setAlertInfo({ reason: detail.reason || 'missing', message: detail.message || '' });
      setIsModalOpen(true);
    };
    window.addEventListener('open-dlp-modal', handleOpenDlpModal);
    return () => window.removeEventListener('open-dlp-modal', handleOpenDlpModal);
  }, []);

  const handleRefreshPortfolio = async () => {
    setIsRefreshing(true);
    try {
      await api.post('/portfolio/refresh');
      await queryClient.invalidateQueries({ queryKey: ['portfolio'] });
      await queryClient.invalidateQueries({ queryKey: ['migration'] });
    } catch {
      // Erros 401/409/422 já são interceptados globalmente em api/client.js
    } finally {
      setIsRefreshing(false);
    }
  };

  const handleStatusUpdated = async (newStatus) => {
    queryClient.setQueryData(DLP_TOKEN_STATUS_KEY, newStatus);
    setAlertInfo({ reason: null, message: null });
    await refetchTokenStatus();
    await queryClient.invalidateQueries({ queryKey: ['portfolio'] });
    await queryClient.invalidateQueries({ queryKey: ['migration'] });
  };

  const configured = Boolean(tokenStatus?.configured);

  return (
    <>
      <header className="sticky top-0 z-40 bg-zinc-950/90 backdrop-blur border-b border-zinc-800">
        <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between gap-4">
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2.5">
              <div className="p-2 rounded-lg bg-emerald-500/15 border border-emerald-500/30 text-emerald-400">
                <TrendingUp className="w-5 h-5" />
              </div>
              <span className="font-bold text-base tracking-tight text-zinc-100">
                App Investimentos
              </span>
            </div>

            <nav className="flex items-center gap-1" aria-label="Navegação principal">
              <NavLink
                to="/"
                end
                className={({ isActive }) =>
                  `inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                    isActive
                      ? 'bg-zinc-800 text-emerald-400'
                      : 'text-zinc-400 hover:text-zinc-100 hover:bg-zinc-900'
                  }`
                }
              >
                <LineChart className="w-4 h-4" />
                Carteira
              </NavLink>
              <NavLink
                to="/migracao-ir"
                className={({ isActive }) =>
                  `inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                    isActive
                      ? 'bg-zinc-800 text-emerald-400'
                      : 'text-zinc-400 hover:text-zinc-100 hover:bg-zinc-900'
                  }`
                }
              >
                <Calculator className="w-4 h-4" />
                Planejador de IR
              </NavLink>
            </nav>
          </div>

          <div className="flex items-center gap-3">
            <button
              type="button"
              data-testid="dlp-status-badge"
              onClick={() => {
                setAlertInfo({ reason: null, message: null });
                setIsModalOpen(true);
              }}
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium border transition-colors ${
                configured
                  ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30 hover:bg-emerald-500/25'
                  : 'bg-amber-500/15 text-amber-300 border-amber-500/30 hover:bg-amber-500/25'
              }`}
            >
              {configured ? (
                <>
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  <span>DLP Conectado</span>
                </>
              ) : (
                <>
                  <AlertTriangle className="w-3.5 h-3.5" />
                  <span>Token DLP Pendente</span>
                </>
              )}
            </button>

            <Button
              type="button"
              variant="secondary"
              size="sm"
              onClick={handleRefreshPortfolio}
              disabled={isRefreshing}
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isRefreshing ? 'animate-spin' : ''}`} />
              <span>Atualizar Dados</span>
            </Button>

            {user && (
              <span className="hidden md:inline-block text-xs text-zinc-400 px-2">
                {user.username}
              </span>
            )}

            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={handleLogout}
              aria-label="Sair"
              title="Sair"
            >
              <LogOut className="w-4 h-4" />
              <span className="sr-only">Sair</span>
            </Button>
          </div>
        </div>
      </header>

      <DlpTokenModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        statusData={tokenStatus}
        alertReason={alertInfo.reason}
        alertMessage={alertInfo.message}
        onStatusUpdated={handleStatusUpdated}
      />
    </>
  );
}

export default Header;

