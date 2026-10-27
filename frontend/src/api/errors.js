const NETWORK_ERROR = {
  status: null,
  code: 'network_error',
  message: 'Falha de conexão com o servidor.',
};

export function getApiError(err) {
  const response = err?.response;
  if (!response) return NETWORK_ERROR;

  const data = response.data || {};
  if (Array.isArray(data.detail)) {
    return {
      status: response.status ?? null,
      code: 'invalid_request',
      message: data.detail[0]?.msg || 'Requisição inválida.',
    };
  }
  return {
    status: response.status ?? null,
    code: data.code || null,
    message: data.message || null,
  };
}

export async function parseBlobErrorData(data) {
  if (typeof Blob === 'undefined' || !(data instanceof Blob)) return data;
  if (!(data.type || '').includes('json')) return data;
  try {
    const text =
      typeof data.text === 'function'
        ? await data.text()
        : await new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = () => reject(reader.error);
            reader.readAsText(data);
          });
    return JSON.parse(text);
  } catch {
    return data;
  }
}

export function shouldRetry(failureCount, err) {
  if (failureCount >= 1) return false;
  const status = err?.response?.status;
  return !err?.response || status >= 500;
}
