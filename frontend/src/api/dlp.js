import { api } from './client';

export const dlpApi = {
  async getTokenStatus() {
    const response = await api.get('/dlp/token-status');
    return response.data;
  },

  async saveToken(token) {
    const response = await api.post('/dlp/token', { token });
    return response.data;
  },

  async deleteToken() {
    const response = await api.delete('/dlp/token');
    return response.data;
  },
};

export default dlpApi;
