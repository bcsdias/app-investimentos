import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // changeOrigin: false preserva o Host do navegador (localhost:5173): o CSRF do Django
    // compara Origin com Host, como fará atrás do nginx na mesma origem (ADR 0005).
    proxy: {
      '/api/v1': {
        target: 'http://localhost:8000',
        changeOrigin: false,
      },
      '/admin': {
        target: 'http://localhost:8000',
        changeOrigin: false,
      },
    },
  },
});

