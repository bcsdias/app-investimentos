import { api } from './client';

export const benchmarksApi = {
  async getCatalog() {
    const response = await api.get('/benchmarks/catalog');
    return response.data;
  },
};

export default benchmarksApi;
