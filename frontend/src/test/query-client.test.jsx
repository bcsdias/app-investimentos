import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { QueryClientProvider, useQuery } from '@tanstack/react-query';
import { createQueryClient } from '../App';

function Probe({ queryFn }) {
  const q = useQuery({ queryKey: ['probe'], queryFn });
  return <div data-testid="status">{q.status}</div>;
}

describe('createQueryClient', () => {
  it('não repete requisição com erro 4xx', async () => {
    const queryFn = vi.fn().mockRejectedValue({ response: { status: 409 } });
    render(
      <QueryClientProvider client={createQueryClient()}>
        <Probe queryFn={queryFn} />
      </QueryClientProvider>
    );
    await waitFor(() => expect(screen.getByTestId('status')).toHaveTextContent('error'), {
      timeout: 2500,
    });
    expect(queryFn).toHaveBeenCalledTimes(1);
  });
});
