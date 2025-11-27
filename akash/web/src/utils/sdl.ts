import { DOCKER_IMAGES } from '../config/constants'

/**
 * SDL configuration for single deployment
 * Used by DeploymentForm for manual deployments
 */
export interface SDLConfig {
  minerType: 'cpu' | 'cuda'
  fleetSize: number  // Total CPUs/GPUs wanted (becomes instance count)
  miningDuration: string
  difficultyEnergy: number
  minDiversity: number
  minSolutions: number
  // Optional: dynamic resource allocation for fleet deployments
  cpuUnits?: number    // Override default CPU units per instance
  gpuUnits?: number    // Override default GPU units per instance (for CUDA)
  memoryGi?: number    // Override default memory in GiB per instance
  instanceCount?: number  // Number of container instances to run (default: 1)
  // Optional: IPFS configuration for result uploads
  ipfsNode?: string    // IPFS node API endpoint
  ipfsApiKey?: string  // Bearer token for IPFS API authentication
  ipfsPin?: boolean    // Whether to pin files permanently (default: true)
}

/**
 * Generate SDL configuration for Akash deployment
 *
 * Format matches Akash Console's Hello World SDL for proper events/logs visibility
 *
 * Resource allocation:
 * - Default: 1 CPU, 2GiB memory for CPU miner; 2 CPU, 1 GPU, 4GiB for CUDA
 * - Fleet mode: Use cpuUnits/gpuUnits/memoryGi to request specific resources
 */
export function generateSDL(config: SDLConfig): object {
  const isCuda = config.minerType === 'cuda'
  const image = isCuda ? DOCKER_IMAGES.cuda : DOCKER_IMAGES.cpu

  // Determine resource allocation
  // For fleet deployments, use explicit values; otherwise use defaults
  const cpuUnits = config.cpuUnits ?? (isCuda ? 2 : 1)
  const memoryGi = config.memoryGi ?? (isCuda ? 4 : 2)
  const gpuUnits = config.gpuUnits ?? (isCuda ? 1 : 0)
  const storageGi = isCuda ? 10 : 5

  // Build resources object
  const resources: Record<string, unknown> = {
    cpu: { units: cpuUnits },
    memory: { size: `${memoryGi}Gi` },
    storage: { size: `${storageGi}Gi` }
  }

  // Add GPU resources for CUDA deployments
  if (gpuUnits > 0) {
    resources.gpu = {
      units: gpuUnits,
      attributes: { vendor: { nvidia: {} } }
    }
  }

  // Service name that shows up in Akash Console
  const serviceName = `quip-${config.minerType}-miner`

  // Instance count - how many containers to run
  const instanceCount = config.instanceCount ?? 1

  // Calculate pricing based on resources PER INSTANCE
  // Base: 10000 uakt/block for 1 CPU
  // Scale with CPU count, add premium for GPU
  const basePricePerCpu = 10000
  const gpuPremium = 50000  // Additional per GPU
  const pricePerInstance = (cpuUnits * basePricePerCpu) + (gpuUnits * gpuPremium)
  // Total price scales with instance count
  const priceAmount = pricePerInstance * instanceCount

  return {
    version: '2.0',
    services: {
      [serviceName]: {
        image,
        // Don't restart when container exits - this is a job-style workload
        // Container will exit after mining completes and IPFS upload succeeds
        restart: 'never',
        expose: [
          {
            port: 8080,
            as: 80,
            to: [{ global: true }]
          }
        ],
        env: [
          `MINER_TYPE=${config.minerType}`,
          `MINING_DURATION=${config.miningDuration}`,
          `DIFFICULTY_ENERGY=${config.difficultyEnergy}`,
          `MIN_DIVERSITY=${config.minDiversity}`,
          `MIN_SOLUTIONS=${config.minSolutions}`,
          // Pass resource info so container can auto-detect
          `REQUESTED_CPUS=${cpuUnits}`,
          `REQUESTED_GPUS=${gpuUnits}`,
          // IPFS configuration (only if configured)
          ...(config.ipfsNode ? [`IPFS_NODE=${config.ipfsNode}`] : []),
          ...(config.ipfsApiKey ? [`IPFS_API_KEY=${config.ipfsApiKey}`] : []),
          ...(config.ipfsNode ? [`IPFS_PIN=${config.ipfsPin ?? true}`] : [])
        ]
      }
    },
    profiles: {
      compute: {
        [serviceName]: {
          resources
        }
      },
      placement: {
        dcloud: {
          pricing: {
            [serviceName]: {
              denom: 'uakt',
              amount: priceAmount
            }
          }
        }
      }
    },
    deployment: {
      [serviceName]: {
        dcloud: {
          profile: serviceName,
          count: instanceCount  // Number of container instances to run
        }
      }
    }
  }
}

/**
 * Generate SDL for a specific provider allocation
 * Used by fleet manager to create optimized SDLs per provider
 */
export function generateSDLForAllocation(
  baseConfig: Omit<SDLConfig, 'cpuUnits' | 'gpuUnits' | 'memoryGi'>,
  allocation: {
    cpuUnits: number
    gpuUnits: number
    memoryGi: number
  }
): object {
  return generateSDL({
    ...baseConfig,
    cpuUnits: allocation.cpuUnits,
    gpuUnits: allocation.gpuUnits,
    memoryGi: allocation.memoryGi
  })
}

/**
 * Estimate deployment cost in AKT (micro AKT per block converted to AKT)
 * Based on pricing in SDL
 * Akash blocks are ~6 seconds, so ~10 blocks/minute, ~600 blocks/hour
 *
 * @param minerType - cpu or cuda
 * @param resourceUnits - number of CPUs (for cpu) or GPUs (for cuda)
 * @param durationMinutes - deployment duration in minutes
 */
export function estimateCostAKT(
  minerType: 'cpu' | 'cuda',
  resourceUnits: number,
  durationMinutes: number
): number {
  const basePricePerCpu = 10000
  const gpuPremium = 50000

  // Calculate price per block based on resources
  let uaktPerBlock: number
  if (minerType === 'cuda') {
    // For CUDA: 2 CPUs base + GPU premium per GPU
    uaktPerBlock = (2 * basePricePerCpu) + (resourceUnits * gpuPremium)
  } else {
    // For CPU: price scales with CPU count
    uaktPerBlock = resourceUnits * basePricePerCpu
  }

  const blocksPerMinute = 10
  const totalBlocks = durationMinutes * blocksPerMinute
  const totalUakt = uaktPerBlock * totalBlocks

  // Convert uakt to AKT (1 AKT = 1,000,000 uakt)
  return totalUakt / 1_000_000
}

/**
 * Estimate fleet deployment cost
 *
 * @param minerType - cpu or cuda
 * @param totalResources - total CPUs or GPUs wanted across fleet
 * @param durationMinutes - deployment duration in minutes
 * @param deploymentsNeeded - number of deployments in the fleet
 */
export function estimateFleetCostAKT(
  minerType: 'cpu' | 'cuda',
  totalResources: number,
  durationMinutes: number,
  deploymentsNeeded: number
): {
  perDeploymentCost: number
  totalCost: number
  costPerResource: number
} {
  // Average resources per deployment
  const resourcesPerDeployment = Math.ceil(totalResources / deploymentsNeeded)
  const perDeploymentCost = estimateCostAKT(minerType, resourcesPerDeployment, durationMinutes)
  const totalCost = perDeploymentCost * deploymentsNeeded
  const costPerResource = totalCost / totalResources

  return {
    perDeploymentCost,
    totalCost,
    costPerResource
  }
}
