import React from 'react';
import { KeyRound } from 'lucide-react';
import { EmptyState } from '../common/EmptyState';
import { Button } from '../common/Button';

export function openDlpTokenModal() {
  window.dispatchEvent(
    new CustomEvent('open-dlp-modal', {
      detail: { reason: 'missing', message: 'Cadastre seu token DLP para carregar a carteira.' },
    })
  );
}

export function DlpTokenRequiredState() {
  return (
    <EmptyState
      icon={KeyRound}
      title="Configure seu token DLP"
      description="A carteira é lida da DLP Invest com o seu token pessoal. Cadastre o token para ver as análises."
      action={
        <Button type="button" onClick={openDlpTokenModal}>
          Configurar token
        </Button>
      }
    />
  );
}

export default DlpTokenRequiredState;
