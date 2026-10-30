import React, { useState } from 'react';
import { FileDown } from 'lucide-react';
import { migrationApi } from '../../api/migration';
import { Button } from '../common/Button';

export function ExportPdfButton({ payload, disabled = false }) {
  const [isDownloading, setIsDownloading] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');

  const handleDownload = async () => {
    if (!payload || disabled) return;
    setErrorMsg('');
    setIsDownloading(true);
    try {
      const result = await migrationApi.downloadPdf(payload);
      const blob = result?.blob || result;
      const filename =
        result?.filename ||
        `plano_migracao_ir_${new Date().toISOString().slice(0, 10).replace(/-/g, '')}.pdf`;
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', filename);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      setTimeout(() => {
        window.URL.revokeObjectURL(url);
      }, 0);
    } catch (err) {
      setErrorMsg(err?.response?.data?.message || 'Erro ao gerar PDF do plano.');
    } finally {
      setIsDownloading(false);
    }
  };

  return (
    <div className="inline-flex flex-col items-end gap-1">
      <Button
        type="button"
        variant="secondary"
        disabled={disabled}
        isLoading={isDownloading}
        onClick={handleDownload}
      >
        <FileDown className="w-4 h-4 text-emerald-400" />
        <span>Baixar Relatório em PDF</span>
      </Button>
      {errorMsg && (
        <span role="alert" className="text-xs text-red-400">
          {errorMsg}
        </span>
      )}
    </div>
  );
}

export default ExportPdfButton;
