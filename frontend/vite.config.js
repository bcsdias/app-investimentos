import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        // ECharts e zrender (motor de desenho dele) em chunks próprios: nenhum passa de
        // 500 kB e o navegador os reaproveita entre deploys que não os mudam (ADR 0022).
        manualChunks(id) {
          if (/node_modules[\\/]zrender[\\/]/.test(id)) return 'zrender';
          if (/node_modules[\\/]echarts[\\/]/.test(id)) return 'echarts';
        },
      },
    },
  },
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

