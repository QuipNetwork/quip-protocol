import { DOCKER_IMAGES } from '../config/constants'

/**
 * SDL configuration for single deployment
 * Used by DeploymentForm for manual deployments
 */
// Available GPU models for Akash deployments
export const GPU_MODELS = {
  any: { label: 'Any NVIDIA GPU', value: '' },
  rtx3060: { label: 'RTX 3060', value: 'rtx3060' },
  rtx3060ti: { label: 'RTX 3060 Ti', value: 'rtx3060ti' },
  rtx3070: { label: 'RTX 3070', value: 'rtx3070' },
  rtx3070ti: { label: 'RTX 3070 Ti', value: 'rtx3070ti' },
  rtx3080: { label: 'RTX 3080', value: 'rtx3080' },
  rtx3080ti: { label: 'RTX 3080 Ti', value: 'rtx3080ti' },
  rtx3090: { label: 'RTX 3090', value: 'rtx3090' },
  rtx3090ti: { label: 'RTX 3090 Ti', value: 'rtx3090ti' },
  rtx4060: { label: 'RTX 4060', value: 'rtx4060' },
  rtx4060ti: { label: 'RTX 4060 Ti', value: 'rtx4060ti' },
  rtx4070: { label: 'RTX 4070', value: 'rtx4070' },
  rtx4070ti: { label: 'RTX 4070 Ti', value: 'rtx4070ti' },
  rtx4080: { label: 'RTX 4080', value: 'rtx4080' },
  rtx4090: { label: 'RTX 4090', value: 'rtx4090' },
  a10: { label: 'A10', value: 'a10' },
  a10g: { label: 'A10G', value: 'a10g' },
  a100: { label: 'A100', value: 'a100' },
  h100: { label: 'H100', value: 'h100' },
  t4: { label: 'T4', value: 't4' },
  v100: { label: 'V100', value: 'v100' },
} as const

export type GpuModelKey = keyof typeof GPU_MODELS

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
  // Optional: GPU model selection for CUDA deployments
  gpuModel?: GpuModelKey  // Specific GPU model to request
  // Optional: CPU performance requirement for CPU deployments
  minCpuScore?: number  // Minimum CPU score (0=any, 2000=entry, 3000=good, 4000=high)
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
 * - Default: 1 CPU, 2GiB memory for CPU miner; 1 CPU per GPU, 4GiB for CUDA
 * - Fleet mode: Use cpuUnits/gpuUnits/memoryGi to request specific resources
 */
export function generateSDL(config: SDLConfig): object {
  const isCuda = config.minerType === 'cuda'
  const image = isCuda ? DOCKER_IMAGES.cuda : DOCKER_IMAGES.cpu

  // Determine resource allocation
  // For fleet deployments, use explicit values; otherwise use defaults
  // For CUDA: CPU units match GPU units (minimal CPU needed for GPU coordination)
  const gpuUnits = config.gpuUnits ?? (isCuda ? 1 : 0)
  const cpuUnits = config.cpuUnits ?? (isCuda ? gpuUnits : 1)
  const memoryGi = config.memoryGi ?? (isCuda ? 4 : 2)
  const storageGi = isCuda ? 10 : 5

  // Build resources object
  const resources: Record<string, unknown> = {
    cpu: { units: cpuUnits },
    memory: { size: `${memoryGi}Gi` },
    storage: { size: `${storageGi}Gi` }
  }

  // Add GPU resources for CUDA deployments
  if (gpuUnits > 0) {
    // Build GPU attributes - always require nvidia vendor
    // Akash SDL expects GPU models as an ARRAY, not an object
    // Format: nvidia: [{ model: "rtx3080" }] or nvidia: [{ model: "rtx4090" }, ...]
    //
    // NOTE: Empty array [] does NOT work in practice - providers don't bid on it.
    // When no specific model is requested, we include all known GPU models.
    const gpuModelValue = config.gpuModel ? GPU_MODELS[config.gpuModel]?.value : ''

    // If no specific model, use ALL available GPU models
    // This ensures any provider with a supported GPU can bid
    const ALL_GPU_MODELS = Object.values(GPU_MODELS)
      .filter(gpu => gpu.value !== '')  // Exclude 'any' which has empty value
      .map(gpu => ({ model: gpu.value }))

    const nvidiaModels: Array<{ model: string }> = gpuModelValue
      ? [{ model: gpuModelValue }]
      : ALL_GPU_MODELS

    resources.gpu = {
      units: gpuUnits,
      attributes: { vendor: { nvidia: nvidiaModels } }
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
    // For CUDA: 1 CPU per GPU + GPU premium per GPU
    uaktPerBlock = (resourceUnits * basePricePerCpu) + (resourceUnits * gpuPremium)
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
