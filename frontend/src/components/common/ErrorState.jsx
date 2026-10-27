import React from 'react';
import { AlertCircle, RotateCcw } from 'lucide-react';
import { Button } from './Button';
import { getApiError } from '../../api/errors';

const FALLBACK_MESSAGE = 'Não foi possível carregar estes dados.';

export function ErrorState({ error, onRetry, className = '' }) {
  const message = getApiError(error).message || FALLBACK_MESSAGE;
  return (
    <div
      role="alert"
      className={`flex flex-col items-center justify-center text-center gap-3 p-6 rounded-xl bg-red-500/10 border border-red-500/30 ${className}`}
    >
      <div className="flex items-center gap-2 text-sm text-red-300">
        <AlertCircle className="w-4 h-4 text-red-400 shrink-0" />
        <span>{message}</span>
      </div>
      {onRetry && (
        <Button type="button" variant="secondary" size="sm" onClick={() => onRetry()}>
          <RotateCcw className="w-3.5 h-3.5" />
          <span>Tentar novamente</span>
        </Button>
      )}
    </div>
  );
}

export default ErrorState;
