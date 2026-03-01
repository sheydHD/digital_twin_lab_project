import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
// BACKEND_URL is a server-side (Node/Vite process) env var — safe to use here.
// In Docker it is set to http://backend:8000 via docker-compose.
// Outside Docker it falls back to http://localhost:8000.
const backendUrl = process.env.BACKEND_URL ?? 'http://localhost:8000';

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    proxy: {
      '/api': { target: backendUrl, changeOrigin: true },
      '/static': { target: backendUrl, changeOrigin: true },
    },
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './src/setupTests.ts',
  },
})
