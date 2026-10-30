import React from 'react';
import { render, screen, waitFor, act } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { MemoryRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { api } from '../api/client';
import { authApi } from '../api/auth';
import { AuthProvider, useAuth } from '../context/AuthContext';
import { RequireAuth } from '../components/common/RequireAuth';

function AuthConsumer() {
  const { user, isAuthenticated, isLoading, login, logout } = useAuth();
  if (isLoading) return <div>Carregando...</div>;
  return (
    <div>
      <div data-testid="auth-status">{isAuthenticated ? `Logado: ${user.username}` : 'Deslogado'}</div>
      <button onClick={() => login({ username: 'bruno', password: '123' })}>Fazer Login</button>
      <button onClick={() => logout()}>Sair</button>
    </div>
  );
}

describe('Cliente Axios, Interceptores CSRF/DLP e AuthContext', () => {
  beforeEach(() => {
    document.cookie = 'csrftoken=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/';
    vi.restoreAllMocks();
  });

  it('injeta X-CSRFToken em requisições POST quando o cookie csrftoken existe', async () => {
    document.cookie = 'csrftoken=test-csrf-token-123';
    let capturedConfig = null;

    const adapter = async (config) => {
      capturedConfig = config;
      return {
        data: { ok: true },
        status: 200,
        statusText: 'OK',
        headers: {},
        config,
      };
    };

    await api.post('/dlp/token', { token: 'abc' }, { adapter });
    expect(capturedConfig.headers['X-CSRFToken']).toBe('test-csrf-token-123');
  });

  it('não dispara evento open-dlp-modal no interceptor ao receber HTTP 409 dlp_token_missing', async () => {
    const handler = vi.fn();
    window.addEventListener('open-dlp-modal', handler);

    const adapter = async (config) => {
      const error = new Error('Request failed with status code 409');
      error.config = config;
      error.response = {
        status: 409,
        data: { code: 'dlp_token_missing', message: 'Token não configurado' },
      };
      throw error;
    };

    await expect(api.get('/portfolio/filters', { adapter })).rejects.toThrow();
    expect(handler).not.toHaveBeenCalled();

    window.removeEventListener('open-dlp-modal', handler);
  });

  it('não dispara evento open-dlp-modal no interceptor ao receber HTTP 422 dlp_token_rejected', async () => {
    const handler = vi.fn();
    window.addEventListener('open-dlp-modal', handler);

    const adapter = async (config) => {
      const error = new Error('Request failed with status code 422');
      error.config = config;
      error.response = {
        status: 422,
        data: { code: 'dlp_token_rejected', message: 'Token inválido na DLP' },
      };
      throw error;
    };

    await expect(api.post('/portfolio/summary', {}, { adapter })).rejects.toThrow();
    expect(handler).not.toHaveBeenCalled();

    window.removeEventListener('open-dlp-modal', handler);
  });

  it('AuthContext atualiza user ao chamar login() com sucesso e limpa ao chamar logout()', async () => {
    vi.spyOn(authApi, 'getCsrf').mockResolvedValue({});
    vi.spyOn(authApi, 'getMe').mockRejectedValue({ response: { status: 401 } });
    vi.spyOn(authApi, 'login').mockResolvedValue({
      id: 1,
      username: 'bruno',
      email: 'bruno@example.com',
    });
    vi.spyOn(authApi, 'logout').mockResolvedValue({});

    const queryClient = new QueryClient();

    render(
      <QueryClientProvider client={queryClient}>
        <AuthProvider>
          <AuthConsumer />
        </AuthProvider>
      </QueryClientProvider>
    );

    await waitFor(() => {
      expect(screen.getByTestId('auth-status')).toHaveTextContent('Deslogado');
    });

    await act(async () => {
      screen.getByText('Fazer Login').click();
    });

    await waitFor(() => {
      expect(screen.getByTestId('auth-status')).toHaveTextContent('Logado: bruno');
    });

    await act(async () => {
      screen.getByText('Sair').click();
    });

    await waitFor(() => {
      expect(screen.getByTestId('auth-status')).toHaveTextContent('Deslogado');
    });
  });

  it('RequireAuth redireciona para /login quando não autenticado', async () => {
    vi.spyOn(authApi, 'getCsrf').mockResolvedValue({});
    vi.spyOn(authApi, 'getMe').mockRejectedValue({ response: { status: 401 } });

    render(
      <MemoryRouter initialEntries={['/']}>
        <AuthProvider>
          <Routes>
            <Route path="/login" element={<div>Tela de Login</div>} />
            <Route element={<RequireAuth />}>
              <Route path="/" element={<div>Dashboard Protegido</div>} />
            </Route>
          </Routes>
        </AuthProvider>
      </MemoryRouter>
    );

    await waitFor(() => {
      expect(screen.getByText('Tela de Login')).toBeInTheDocument();
    });
  });
});

