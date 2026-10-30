import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

// Fuso do usuário: testes de data local só pegam bugs de UTC fora de UTC.
process.env.TZ = 'America/Sao_Paulo';

export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './src/test/setup.js',
  },
});

