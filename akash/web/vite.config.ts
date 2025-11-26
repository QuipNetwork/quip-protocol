import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { nodePolyfills } from 'vite-plugin-node-polyfills'
import fs from 'fs'
import path from 'path'

const certPath = path.resolve(__dirname, 'localhost.pem')

export default defineConfig({
  plugins: [
    react(),
    nodePolyfills({
      // Include specific polyfills needed by cosmos-kit
      include: ['buffer', 'crypto', 'stream', 'util', 'process'],
      globals: {
        Buffer: true,
        global: true,
        process: true,
      },
    }),
  ],

  // Development server with HTTPS
  server: {
    https: fs.existsSync(certPath) ? {
      key: fs.readFileSync(certPath),
      cert: fs.readFileSync(certPath)
    } : undefined,
    host: true,
    port: 8000,
    open: true
  },

  // Build configuration
  build: {
    outDir: 'dist',
    sourcemap: false,
    // Optimize for smaller bundle
    rollupOptions: {
      output: {
        manualChunks: {
          'cosmos': [
            '@cosmos-kit/core',
            '@cosmos-kit/react',
            '@cosmjs/stargate'
          ],
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
