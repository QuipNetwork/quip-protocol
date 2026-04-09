// WalletConnect Project ID
// Get your own at https://cloud.walletconnect.com/
export const WALLETCONNECT_PROJECT_ID = '8f26f1d68dd60fa28cd4369dc06cdb79'

// Detect development mode
const isDev = import.meta.env.DEV

// Akash Network endpoints
// In development: use Vite proxy to avoid CORS issues
// In production: use direct URLs (requires proper CORS or same-origin deployment)
export const AKASH_RPC = 'https://akash-rpc.polkachu.com'
export const AKASH_REST = isDev ? '/api/akash' : 'https://akash-api.polkachu.com'
export const AKASH_CONSOLE_API = isDev ? '/api/console' : 'https://console-api.akash.network'
export const AKASH_CHAIN_ID = 'akashnet-2'
export const AKASH_DENOM = 'uakt'
export const AKASH_DECIMALS = 6

// Docker images (must match Docker Hub repository names)
export const DOCKER_IMAGES = {
  cpu: 'carback1/quip-protocol-cpu-miner:v0.1.0',
  cuda: 'carback1/quip-protocol-cuda-miner:v0.1.0',
}

// Resource limits per deployment
// Most Akash providers have 1-8 GPUs max, so requesting more won't get bids
// Start at max realistic size - adaptive reduction will halve to 4, 2, 1 if no bids
export const MAX_GPU_PER_DEPLOYMENT = 8  // Start high, reduce adaptively if no bids
export const MAX_CPU_PER_DEPLOYMENT = 32 // Most providers have 32-64 cores

// Mining defaults
export const DEFAULTS = {
  minerType: 'cpu' as const,
  fleetSize: 10,
  duration: '90m',
  difficulty: -14900,
  diversity: 0.15,
  minSolutions: 5,
}
