import React, { useState, useEffect } from 'react';
import { Eye, EyeOff, KeyRound, Trash2, AlertTriangle, CheckCircle2 } from 'lucide-react';
import { Modal } from '../common/Modal';
import { Button } from '../common/Button';
import { dlpApi } from '../../api/dlp';

export function DlpTokenModal({
  isOpen,
  onClose,
  statusData,
  alertReason,
  alertMessage,
  onStatusUpdated,
}) {
  const [token, setToken] = useState('');
  const [showToken, setShowToken] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [error, setError] = useState('');

  const configured = Boolean(statusData?.configured);

  const resetState = () => {
    setToken('');
    setShowToken(false);
    setError('');
  };

  const handleClose = () => {
    resetState();
    if (onClose) onClose();
  };

  useEffect(() => {
    if (!isOpen) {
      resetState();
    }
  }, [isOpen]);

  const handleSave = async (e) => {
    e.preventDefault();
    const trimmed = token.trim();
    if (!trimmed) {
      setError('Informe um token DLP válido.');
      return;
    }
    setError('');
    setIsSaving(true);
    try {
      const updated = await dlpApi.saveToken(trimmed);
      resetState();
      if (onStatusUpdated) {
        await onStatusUpdated(updated);
      }
      if (onClose) onClose();
    } catch (err) {
      setError(err.response?.data?.message || 'Falha ao salvar o token DLP.');
    } finally {
      setIsSaving(false);
    }
  };

  const handleDelete = async () => {
    setError('');
    setIsDeleting(true);
    try {
      await dlpApi.deleteToken();
      if (onStatusUpdated) {
        await onStatusUpdated({ configured: false, updated_at: null });
      }
    } catch (err) {
      setError(err.response?.data?.message || 'Falha ao remover o token DLP.');
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={handleClose} title="Configuração do Token DLP Invest">
      <div className="space-y-4">
        {alertReason && (
          <div className="flex items-start gap-3 p-3 rounded-lg bg-amber-500/10 border border-amber-500/30 text-amber-200 text-sm">
            <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
            <div>
              <p className="font-medium">
                {alertReason === 'rejected' || alertReason === 'dlp_token_rejected'
                  ? 'Credencial recusada pela DLP'
                  : 'Credencial DLP ausente'}
              </p>
              {alertMessage && <p className="text-xs text-amber-300/90 mt-0.5">{alertMessage}</p>}
            </div>
          </div>
        )}

        <div className="flex items-center justify-between p-3 rounded-lg bg-zinc-950 border border-zinc-800">
          <div className="flex items-center gap-2.5 text-sm">
            {configured ? (
              <>
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                <span className="text-zinc-200 font-medium">Token DLP configurado</span>
              </>
            ) : (
              <>
                <KeyRound className="w-4 h-4 text-amber-400" />
                <span className="text-zinc-300">Nenhum token DLP ativo</span>
              </>
            )}
          </div>
          {statusData?.updated_at && (
            <span className="text-xs text-zinc-400">
              Atualizado em {new Date(statusData.updated_at).toLocaleDateString('pt-BR')}
            </span>
          )}
        </div>

        <form onSubmit={handleSave} className="space-y-4">
          <div>
            <label
              htmlFor="dlp-token-input"
              className="block text-xs font-medium text-zinc-300 mb-1.5"
            >
              Token de Acesso DLP
            </label>
            <div className="relative">
              <input
                id="dlp-token-input"
                type={showToken ? 'text' : 'password'}
                value={token}
                onChange={(e) => setToken(e.target.value)}
                placeholder="Cole seu token DLP Invest aqui..."
                className="w-full bg-zinc-950 border border-zinc-700 rounded-lg pl-3 pr-10 py-2 text-sm text-zinc-100 placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-emerald-500"
              />
              <button
                type="button"
                onClick={() => setShowToken((prev) => !prev)}
                aria-label={showToken ? 'Ocultar token' : 'Revelar token'}
                className="absolute right-2.5 top-1/2 -translate-y-1/2 text-zinc-400 hover:text-zinc-200"
              >
                {showToken ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
          </div>

          {error && <p className="text-xs text-red-400">{error}</p>}

          <div className="flex items-center justify-between pt-2">
            {configured ? (
              <Button
                type="button"
                variant="danger"
                size="sm"
                onClick={handleDelete}
                isLoading={isDeleting}
              >
                <Trash2 className="w-3.5 h-3.5" />
                Remover Token
              </Button>
            ) : (
              <div />
            )}
            <div className="flex items-center gap-2">
              <Button type="button" variant="ghost" size="sm" onClick={handleClose}>
                Cancelar
              </Button>
              <Button type="submit" variant="primary" size="sm" isLoading={isSaving}>
                Salvar Token
              </Button>
            </div>
          </div>
        </form>
      </div>
    </Modal>
  );
}

export default DlpTokenModal;
