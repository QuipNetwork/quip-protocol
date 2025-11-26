// WalletConnect Project ID
// Get your own at https://cloud.walletconnect.com/
export const WALLETCONNECT_PROJECT_ID = '8f26f1d68dd60fa28cd4369dc06cdb79'

// Akash Network - using CORS-enabled endpoints
// Note: Many public RPC endpoints don't have CORS enabled for browser requests
// These endpoints are known to work with browser-based apps
export const AKASH_RPC = 'https://rpc.cosmos.directory/akash'
export const AKASH_REST = 'https://rest.cosmos.directory/akash'
export const AKASH_CHAIN_ID = 'akashnet-2'
export const AKASH_DENOM = 'uakt'
export const AKASH_DECIMALS = 6

// Docker images
export const DOCKER_IMAGES = {
  cpu: 'carback1/quip-miner:cpu-latest',
  cuda: 'carback1/quip-miner:cuda-latest',
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
