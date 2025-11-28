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
  cpu: 'carback1/quip-protocol-cpu-miner:v0.0.4',
  cuda: 'carback1/quip-protocol-cuda-miner:v0.0.4',
}

// Mining defaults
export const DEFAULTS = {
  minerType: 'cpu' as const,
  fleetSize: 10,
  duration: '90m',
  difficulty: -14900,
  diversity: 0.15,
  minSolutions: 5,
}
