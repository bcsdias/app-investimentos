import { api } from './client';

export const authApi = {
  async getCsrf() {
    const response = await api.get('/auth/csrf');
    return response.data;
  },

  async login({ username, password }) {
    await this.getCsrf();
    const response = await api.post('/auth/login', { username, password });
    return response.data;
  },

  async logout() {
    const response = await api.post('/auth/logout');
    return response.data;
  },

  async getMe() {
    const response = await api.get('/auth/me');
    return response.data;
  },
};

export default authApi;

