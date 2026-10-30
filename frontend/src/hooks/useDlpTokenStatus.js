import { useQuery } from '@tanstack/react-query';
import { dlpApi } from '../api/dlp';

export const DLP_TOKEN_STATUS_KEY = ['dlp', 'token-status'];

export function useDlpTokenStatus() {
  return useQuery({
    queryKey: DLP_TOKEN_STATUS_KEY,
    queryFn: () => dlpApi.getTokenStatus(),
    staleTime: 60 * 1000,
  });
}

// true/false quando o status chegou; undefined enquanto carrega ou se falhou
// (as páginas tratam a falha via useDlpTokenStatus().isError).
export function useDlpTokenConfigured() {
  const { data } = useDlpTokenStatus();
  return data ? Boolean(data.configured) : undefined;
}

export default useDlpTokenStatus;
