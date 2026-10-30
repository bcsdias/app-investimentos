import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';

describe('Frontend Environment Smoke Test', () => {
  it('renders a simple element into jsdom with tailwind class', () => {
    render(<div className="bg-zinc-900 text-white">App Investimentos</div>);
    const element = screen.getByText('App Investimentos');
    expect(element).toBeInTheDocument();
    expect(element).toHaveClass('bg-zinc-900');
  });
});

