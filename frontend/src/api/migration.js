import { api } from './client';

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
    return response.data;
  },
};

export default migrationApi;
