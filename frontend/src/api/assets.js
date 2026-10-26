import { api } from './client';

export const assetsApi = {
  async searchAssets(query, limit = 20) {
    const term = (query || '').trim();
    if (term.length < 2) return [];
    const response = await api.get(
      `/assets/search?q=${encodeURIComponent(term)}&limit=${encodeURIComponent(limit)}`
    );
    return response.data;
  },
};

export default assetsApi;
