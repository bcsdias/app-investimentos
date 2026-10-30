import React from 'react';
import { AlertTriangle } from 'lucide-react';
import { Button } from '../common/Button';

export const DLP_TOKEN_ERROR_CODES = ['dlp_token_rejected', 'dlp_token_missing'];

export function dlpTokenErrorCode(error) {
  if (!error) return null;
  if (typeof error === 'string') {
    return DLP_TOKEN_ERROR_CODES.includes(error) ? error : null;
  }
  const code = error?.response?.data?.code || error?.code || null;
  return DLP_TOKEN_ERROR_CODES.includes(code) ? code : null;
}

export function DlpTokenErrorState({ code }) {
  const isRejected = code === 'dlp_token_rejected';
  const title = isRejected ? 'Token DLP rejeitado' : 'Não foi possível ler seu token DLP';
  const description = isRejected
    ? 'Sua credencial foi recusada pela DLP Invest. O token configurado pode ter expirado ou sido revogado.'
    : 'Token DLP não configurado ou ilegível. Configure seu token pessoal para visualizar a carteira.';

  const handleOpenModal = () => {
    window.dispatchEvent(
      new CustomEvent('open-dlp-modal', {
        detail: {
          reason: code,
          message: isRejected
            ? 'Token DLP rejeitado pela API externa.'
            : 'Token DLP não configurado.',
        },
      })
    );
  };

  return (
    <div
      role="alert"
      data-testid="dlp-token-error-state"
      className="flex flex-col items-center justify-center text-center p-8 rounded-xl bg-amber-500/10 border border-amber-500/30 space-y-3"
    >
      <div className="p-3 rounded-full bg-amber-500/20 text-amber-400">
        <AlertTriangle className="w-6 h-6" />
      </div>
      <h3 className="text-base font-semibold text-amber-200">{title}</h3>
      <p className="text-sm text-amber-300/90 max-w-md">{description}</p>
      <div className="pt-2">
        <Button type="button" onClick={handleOpenModal}>
          Atualizar token
        </Button>
      </div>
    </div>
  );
}

export default DlpTokenErrorState;
