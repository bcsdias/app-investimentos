import { api } from './client';

export const portfolioApi = {
  async getFilters(signal) {
    const response = await api.get('/portfolio/filters', { signal });
    return response.data;
  },

  async refreshPortfolio() {
    const response = await api.post('/portfolio/refresh');
    return response.data;
  },

  async fetchSummary(payload, signal) {
    const response = await api.post('/portfolio/summary', payload, { signal });
    return response.data;
  },

  async fetchTwrEvolution(payload, signal) {
    const response = await api.post('/portfolio/twr-evolution', payload, { signal });
    return response.data;
  },

  async fetchDrawdown(payload, signal) {
    const response = await api.post('/portfolio/drawdown', payload, { signal });
    return response.data;
  },

  async fetchRollingMetrics(payload, signal) {
    const response = await api.post('/portfolio/rolling-metrics', payload, { signal });
    return response.data;
  },

  async fetchRiskReturn(payload, signal) {
    const response = await api.post('/portfolio/risk-return', payload, { signal });
    return response.data;
  },

  async fetchShadow(payload, signal) {
    const response = await api.post('/portfolio/shadow', payload, { signal });
    return response.data;
  },

  async fetchYearlySummary(payload, signal) {
    const response = await api.post('/portfolio/yearly-summary', payload, { signal });
    return response.data;
  },

  async fetchIrrEvolution(payload, signal) {
    const response = await api.post('/portfolio/irr-evolution', payload, { signal });
    return response.data;
  },
};

export default portfolioApi;

