import axios from 'axios';

export function getCsrfTokenFromCookie() {
  if (typeof document === 'undefined' || !document.cookie) return null;
  const row = document.cookie
    .split('; ')
    .find((item) => item.startsWith('csrftoken='));
  return row ? decodeURIComponent(row.split('=')[1]) : null;
}

export const api = axios.create({
  baseURL: '/api/v1',
  withCredentials: true,
  headers: {
    'Content-Type': 'application/json',
  },
});

api.interceptors.request.use((config) => {
  const csrfToken = getCsrfTokenFromCookie();
  const method = (config.method || 'get').toLowerCase();
  if (csrfToken && ['post', 'put', 'patch', 'delete'].includes(method)) {
    config.headers = config.headers || {};
    config.headers['X-CSRFToken'] = csrfToken;
  }
  return config;
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    const status = error.response?.status;
    const code = error.response?.data?.code;
    const message = error.response?.data?.message;

    if (typeof window !== 'undefined') {
      const requestUrl = error.config?.url || '';
      const isAuthCheckOrLogin =
        requestUrl.includes('/auth/me') || requestUrl.includes('/auth/login');

      if (status === 401 && !isAuthCheckOrLogin && window.location.pathname !== '/login') {
        window.dispatchEvent(new CustomEvent('auth-unauthorized'));
        try {
          window.location.assign('/login');
        } catch {
          // jsdom pode não implementar navigation completa
        }
      } else if (status === 409 && code === 'dlp_token_missing') {
        window.dispatchEvent(
          new CustomEvent('open-dlp-modal', {
            detail: { reason: 'missing', message: message || 'Token DLP não configurado.' },
          })
        );
      } else if (status === 422 && code === 'dlp_token_rejected') {
        window.dispatchEvent(
          new CustomEvent('open-dlp-modal', {
            detail: { reason: 'rejected', message: message || 'Token DLP rejeitado pela API externa.' },
          })
        );
      }
    }

    return Promise.reject(error);
  }
);

export default api;
