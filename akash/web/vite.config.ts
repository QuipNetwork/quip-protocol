import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { nodePolyfills } from 'vite-plugin-node-polyfills'
import fs from 'fs'
import path from 'path'
import { providerProxyPlugin } from './vite-provider-proxy'

const certPath = path.resolve(__dirname, 'localhost.pem')

export default defineConfig({
  plugins: [
    react(),
    nodePolyfills({
      include: ['buffer', 'crypto', 'stream', 'util', 'process'],
      globals: {
        Buffer: true,
        global: true,
        process: true,
      },
      // Use browser crypto.subtle when available
      overrides: {
        crypto: 'crypto-browserify',
      },
    }),
    // Provider proxy for mTLS manifest submission
    providerProxyPlugin(),
  ],

  // Development server with HTTPS and API proxy
  server: {
    https: fs.existsSync(certPath) ? {
      key: fs.readFileSync(certPath),
      cert: fs.readFileSync(certPath)
    } : undefined,
    host: true,
    port: 8000,
    open: true,
    // Proxy API requests to avoid CORS issues
    proxy: {
      // Console API proxy (manifest submission, provider list)
      '/api/console': {
        target: 'https://console-api.akash.network',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/console/, ''),
        secure: true
      },
      // Akash REST API proxy (blockchain queries)
      '/api/akash': {
        target: 'https://rest.cosmos.directory/akash',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/akash/, ''),
        secure: true
      }
    }
  },

  // Build configuration
  build: {
    outDir: 'dist',
    sourcemap: false,
    rollupOptions: {
      output: {
        manualChunks: {
          'cosmos': ['@cosmjs/stargate', '@cosmjs/proto-signing'],
          'react': ['react', 'react-dom']
        }
      }
    }
  },

  // Required for WalletConnect
  define: {
    global: 'globalThis',
  },

  // Handle potential polyfill issues
  optimizeDeps: {
    esbuildOptions: {
      define: { global: 'globalThis' }
    }
  }
})
