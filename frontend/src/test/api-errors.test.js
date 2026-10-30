import { describe, it, expect, vi, afterEach } from 'vitest';
import { AxiosError } from 'axios';
import { api } from '../api/client';
import { getApiError, parseBlobErrorData, shouldRetry } from '../api/errors';

function jsonBlob(obj) {
  return new Blob([JSON.stringify(obj)], { type: 'application/json' });
}

describe('helpers de erro da API', () => {
  afterEach(() => vi.restoreAllMocks());

  it('getApiError lê code/message do envelope ApiError', () => {
    const err = { response: { status: 422, data: { code: 'invalid_period', message: 'Período inválido' } } };
    expect(getApiError(err)).toEqual({ status: 422, code: 'invalid_period', message: 'Período inválido' });
  });

  it('getApiError traduz detail do Ninja para invalid_request', () => {
    const err = { response: { status: 422, data: { detail: [{ msg: 'campo obrigatório' }] } } };
    expect(getApiError(err)).toEqual({ status: 422, code: 'invalid_request', message: 'campo obrigatório' });
  });

  it('getApiError sem response vira network_error', () => {
    expect(getApiError(new Error('boom'))).toEqual({
      status: null,
      code: 'network_error',
      message: 'Falha de conexão com o servidor.',
    });
  });

  it('parseBlobErrorData converte Blob JSON em objeto', async () => {
    const blob = jsonBlob({ code: 'nothing_to_plan', message: 'Nada a planejar' });
    expect(await parseBlobErrorData(blob)).toEqual({ code: 'nothing_to_plan', message: 'Nada a planejar' });
  });

  it('parseBlobErrorData mantém dados que não são Blob JSON', async () => {
    const data = { code: 'x' };
    expect(await parseBlobErrorData(data)).toBe(data);
  });

  it('shouldRetry: não repete 4xx, repete uma vez 5xx/rede', () => {
    expect(shouldRetry(0, { response: { status: 409 } })).toBe(false);
    expect(shouldRetry(0, { response: { status: 502 } })).toBe(true);
    expect(shouldRetry(1, { response: { status: 502 } })).toBe(false);
    expect(shouldRetry(0, {})).toBe(true);
  });

  it('interceptor parseia Blob em erro 409 sem abrir modal automaticamente', async () => {
    const listener = vi.fn();
    window.addEventListener('open-dlp-modal', listener);
    const adapter = async (config) => {
      const response = {
        data: jsonBlob({ code: 'dlp_token_missing', message: 'Sem token' }),
        status: 409,
        statusText: 'Conflict',
        headers: {},
        config,
      };
      throw new AxiosError('Conflict', AxiosError.ERR_BAD_REQUEST, config, null, response);
    };
    const err = await api
      .post('/migration/plan.pdf', {}, { adapter, responseType: 'blob' })
      .catch((e) => e);
    window.removeEventListener('open-dlp-modal', listener);
    expect(err.response.data).toEqual({ code: 'dlp_token_missing', message: 'Sem token' });
    expect(listener).not.toHaveBeenCalled();
  });
});
