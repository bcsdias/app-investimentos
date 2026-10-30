import { api } from './client';

export function extractFilename(dispositionHeader) {
  if (!dispositionHeader || typeof dispositionHeader !== 'string') {
    const today = new Date().toISOString().slice(0, 10).replace(/-/g, '');
    return `plano_migracao_ir_${today}.pdf`;
  }

  // RFC 5987 / RFC 6266 filename*=
  const starMatch = dispositionHeader.match(/filename\*=(?:UTF-8''|utf-8'')?([^;]+)/i);
  if (starMatch && starMatch[1]) {
    try {
      const cleaned = starMatch[1].trim().replace(/^["']|["']$/g, '');
      return decodeURIComponent(cleaned);
    } catch {
      return starMatch[1].trim().replace(/^["']|["']$/g, '');
    }
  }

  // Standard filename="..." or filename=...
  const match = dispositionHeader.match(/filename=(?:["']([^"']+)["']|([^;\s]+))/i);
  if (match) {
    const name = match[1] || match[2];
    if (name) return name.trim();
  }

  const today = new Date().toISOString().slice(0, 10).replace(/-/g, '');
  return `plano_migracao_ir_${today}.pdf`;
}

export const migrationApi = {
  async getWallet(signal) {
    const response = await api.get('/migration/wallet', { signal });
    return response.data;
  },

  async calculatePlan(payload) {
    const response = await api.post('/migration/plan', payload);
    return response.data;
  },

  async downloadPdf(payload) {
    const response = await api.post('/migration/plan.pdf', payload, {
      responseType: 'blob',
    });
    const header =
      (typeof response.headers?.get === 'function'
        ? response.headers.get('content-disposition')
        : response.headers?.['content-disposition']) || '';
    const filename = extractFilename(header);
    return { blob: response.data, filename };
  },
};

export default migrationApi;
