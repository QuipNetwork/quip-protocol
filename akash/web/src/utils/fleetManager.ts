/**
 * Fleet Manager - Adaptive fleet deployment across Akash providers
 *
 * Strategy (Iterative Consolidation):
 * 1. Start with full request (e.g., 10 CPUs with count=10, or 10 GPUs)
 * 2. Solicit bids from all providers
 * 3. If no bids received, reduce count/GPUs and try again
 * 4. User selects provider, deployment is created
 * 5. Repeat for remaining resources until all allocated
 *
 * For CPUs: Use instance count (count=N) for parallel containers
 * For GPUs: Request N GPUs in a single deployment
 *
 * Benefits:
 * - Consolidates where possible (fewer deployments = less overhead)
 * - Adapts to actual provider capacity via bid responses
 * - User maintains provider selection control
 * - Falls back gracefully when large requests can't be filled
 */

import type { SigningStargateClient } from '@cosmjs/stargate'
import {
  fetchAllProviders,
  filterProvidersByCapacity,
  getTotalCapacity,
  createDeployment,
  waitForBids,
  acceptBid,
  sendManifestViaConsole,
  type ProviderCapacity,
  type Bid
} from './akashApi'
import { generateSDL, type SDLConfig } from './sdl'

// ============================================================================
// Types
// ============================================================================

/**
 * Capacity check result - ensures network can fill the request
 */
export interface CapacityCheck {
  // Request
  requestedCpu: number
  requestedGpu: number

  // Network capacity
  totalCpu: number           // Total CPUs available across all providers
  totalGpu: number           // Total GPUs available
  providerCount: number      // Number of suitable providers

  // Providers that could potentially fill bids (top 10 by capacity)
  topProviders: ProviderCapacity[]

  // Can we likely fill the request?
  canFill: boolean
  capacityRatio: number      // Available / Requested (>1.5 is good)

  // Initial deployment plan - will adapt based on bids
  // For CPU: start with count=N instances
  // For GPU: start with N GPUs per deployment
  initialRequestSize: number  // How many CPUs/GPUs to request initially
}

/**
 * Status of a single deployment in the fleet
 */
export type FleetDeploymentStatus =
  | 'pending'
  | 'creating'
  | 'waiting-bids'
  | 'selecting-bid'
  | 'accepting-bid'
  | 'sending-manifest'
  | 'active'
  | 'failed'

/**
 * State of a single deployment in the fleet
 * For CPU: requestedCpu=1, instanceCount=N (N parallel containers)
 * For GPU: requestedGpu=N, instanceCount=1 (1 container with N GPUs)
 */
export interface FleetDeploymentItem {
  index: number
  status: FleetDeploymentStatus
  // Resources per instance
  requestedCpu: number       // CPUs per instance (1 for CPU miners, 2 for GPU miners)
  requestedGpu: number       // GPUs per instance (0 for CPU, N for GPU)
  requestedMemoryGi: number  // Memory per instance
  instanceCount: number      // Number of container instances (N for CPU, 1 for GPU)
  // Deployment info
  dseq?: string
  transactionHash?: string
  manifestJson?: string
  // Bid info
  availableBids?: Bid[]
  selectedBid?: Bid
  selectedProviderName?: string
  // Result
  serviceUrl?: string
  error?: string
  startedAt?: Date
  completedAt?: Date
}

/**
 * Overall fleet deployment state
 */
export interface FleetState {
  config: SDLConfig
  capacityCheck: CapacityCheck
  deployments: FleetDeploymentItem[]
  status: 'checking' | 'ready' | 'deploying' | 'selecting-bids' | 'completed' | 'cancelled'
  startedAt: Date
  completedAt?: Date
  // Tracking for adaptive deployment
  remainingResources: number  // CPUs or GPUs still to be allocated
  currentRequestSize: number  // Current deployment request size (may decrease if no bids)
  stats: {
    totalRequested: number    // Total CPUs/GPUs requested
    totalAllocated: number    // CPUs/GPUs successfully allocated
    deploymentsCreated: number
    deploymentsActive: number
    deploymentsFailed: number
  }
}

/**
 * Callback for fleet deployment progress updates
 */
export type FleetProgressCallback = (state: FleetState) => void

// ============================================================================
// Capacity Check
// ============================================================================

/**
 * Check network capacity before starting fleet deployment
 *
 * Returns info about available capacity and whether the request can likely be filled.
 * We want at least 5-10 providers with capacity to ensure bids will come in.
 */
export async function checkNetworkCapacity(
  targetCpu: number,
  targetGpu: number,
  options: {
    minProvidersRequired?: number  // Minimum providers needed (default: 5)
    capacityMultiplier?: number    // How much excess capacity we want (default: 1.5x)
  } = {}
): Promise<CapacityCheck> {
  const {
    minProvidersRequired = 5,
    capacityMultiplier = 1.5
  } = options

  const isGpuRequest = targetGpu > 0

  // Fetch all providers
  const allProviders = await fetchAllProviders()

  // Filter by resource type
  const providers = filterProvidersByCapacity(allProviders, {
    minCpu: isGpuRequest ? 0 : 1,
    minGpu: isGpuRequest ? 1 : 0,
    onlyOnline: true
  })

  console.log(`Found ${providers.length} suitable providers out of ${allProviders.length} total`)

  // Get total capacity
  const capacity = getTotalCapacity(providers)
  console.log('Network capacity:', capacity)

  // Get top providers by capacity (for display)
  const sortedProviders = [...providers].sort((a, b) => {
    if (isGpuRequest) {
      return b.availableGpu - a.availableGpu
    }
    return b.availableCpu - a.availableCpu
  })
  const topProviders = sortedProviders.slice(0, 10)

  // Calculate if we can likely fill the request
  const requestedAmount = isGpuRequest ? targetGpu : targetCpu
  const availableAmount = isGpuRequest ? capacity.totalGpu : capacity.totalCpu
  const capacityRatio = requestedAmount > 0 ? availableAmount / requestedAmount : 0

  const canFill =
    providers.length >= minProvidersRequired &&
    capacityRatio >= capacityMultiplier

  // Start with full request - will adapt based on bids received
  // For CPU: request count=N instances (parallel containers on one provider)
  // For GPU: request N GPUs (single container with multiple GPUs)
  const initialRequestSize = requestedAmount

  return {
    requestedCpu: targetCpu,
    requestedGpu: targetGpu,
    totalCpu: capacity.totalCpu,
    totalGpu: capacity.totalGpu,
    providerCount: providers.length,
    topProviders,
    canFill,
    capacityRatio,
    initialRequestSize
  }
}

// ============================================================================
// Fleet Deployment Execution
// ============================================================================

/**
 * Initialize fleet state for adaptive deployment
 * Starts with full request, will create deployments dynamically based on bids
 */
export function initializeFleetState(
  config: SDLConfig,
  capacityCheck: CapacityCheck
): FleetState {
  const isGpu = config.minerType === 'cuda'
  const totalRequested = isGpu ? capacityCheck.requestedGpu : capacityCheck.requestedCpu

  return {
    config,
    capacityCheck,
    deployments: [],  // Will be populated as deployments are created
    status: 'ready',
    startedAt: new Date(),
    remainingResources: totalRequested,
    currentRequestSize: capacityCheck.initialRequestSize,
    stats: {
      totalRequested,
      totalAllocated: 0,
      deploymentsCreated: 0,
      deploymentsActive: 0,
      deploymentsFailed: 0
    }
  }
}

/**
 * Create the next deployment item for the fleet
 * For CPU: requests N CPUs in a single container (miner auto-parallelizes)
 * For GPU: requests N GPUs in a single container
 */
export function createNextDeploymentItem(
  state: FleetState,
  requestSize: number
): FleetDeploymentItem {
  const isGpu = state.config.minerType === 'cuda'

  // For CPU: N CPUs per instance, 1 instance (miner uses all available cores)
  // For GPU: N GPUs per instance, 1 instance
  const deployment: FleetDeploymentItem = {
    index: state.deployments.length,
    status: 'pending',
    requestedCpu: isGpu ? 2 : requestSize,  // N CPUs for CPU miner, 2 for GPU
    requestedGpu: isGpu ? requestSize : 0,  // N GPUs for GPU deployment
    requestedMemoryGi: isGpu ? 4 * requestSize : 2 * requestSize,  // Scale memory with resources
    instanceCount: 1  // Always 1 container - miner handles parallelization internally
  }

  return deployment
}

/**
 * Create a single deployment and wait for bids
 * Returns the deployment with available bids for user selection
 */
export async function createDeploymentAndGetBids(
  signingClient: SigningStargateClient,
  owner: string,
  config: SDLConfig,
  deployment: FleetDeploymentItem,
  onUpdate: (update: Partial<FleetDeploymentItem>) => void
): Promise<Bid[]> {
  // Generate SDL for this deployment
  // For CPU: 1 container with N CPUs (miner uses all cores)
  // For GPU: 1 container with N GPUs
  onUpdate({ status: 'creating', startedAt: new Date() })

  const isGpu = config.minerType === 'cuda'

  const sdl = generateSDL({
    ...config,
    fleetSize: isGpu ? deployment.requestedGpu : deployment.requestedCpu,
    cpuUnits: deployment.requestedCpu,
    gpuUnits: isGpu ? deployment.requestedGpu : 0,
    memoryGi: deployment.requestedMemoryGi,
    instanceCount: 1  // Always 1 container - miner handles parallelization
  })

  // Create deployment
  const deployResult = await createDeployment(signingClient, owner, sdl)
  onUpdate({
    dseq: deployResult.dseq,
    transactionHash: deployResult.transactionHash,
    manifestJson: deployResult.manifestJson,
    status: 'waiting-bids'
  })

  // Wait for bids (2 minutes, polling every 5 seconds)
  const bids = await waitForBids(owner, deployResult.dseq, 120000, 5000)

  onUpdate({
    availableBids: bids,
    status: bids.length > 0 ? 'selecting-bid' : 'failed',
    error: bids.length === 0 ? 'No bids received' : undefined
  })

  return bids
}

/**
 * Accept a bid and send manifest
 */
export async function acceptBidAndDeploy(
  signingClient: SigningStargateClient,
  owner: string,
  deployment: FleetDeploymentItem,
  selectedBid: Bid,
  providerName: string,
  onUpdate: (update: Partial<FleetDeploymentItem>) => void
): Promise<void> {
  if (!deployment.dseq || !deployment.manifestJson) {
    throw new Error('Deployment not properly initialized')
  }

  onUpdate({
    selectedBid,
    selectedProviderName: providerName,
    status: 'accepting-bid'
  })

  // Accept the bid
  await acceptBid(signingClient, selectedBid)
  onUpdate({ status: 'sending-manifest' })

  // Send manifest
  const manifestResult = await sendManifestViaConsole(
    owner,
    deployment.dseq,
    selectedBid.bid.id.provider,
    deployment.manifestJson
  )

  if (!manifestResult.success) {
    throw new Error(`Manifest failed: ${manifestResult.error}`)
  }

  onUpdate({
    status: 'active',
    completedAt: new Date()
  })
}

/**
 * Deploy fleet with automatic bid selection (lowest price)
 *
 * Adaptive deployment strategy:
 * 1. Start with full request (e.g., 10 CPUs or 10 GPUs)
 * 2. If no bids, halve the request and retry
 * 3. Continue until all resources allocated or minimum reached
 * 4. Auto-select lowest price bid for each deployment
 */
export async function deployFleetAutomatic(
  signingClient: SigningStargateClient,
  owner: string,
  config: SDLConfig,
  capacityCheck: CapacityCheck,
  onProgress?: FleetProgressCallback
): Promise<FleetState> {
  const state = initializeFleetState(config, capacityCheck)
  state.status = 'deploying'
  const isGpu = config.minerType === 'cuda'

  const updateStats = () => {
    state.stats = {
      totalRequested: isGpu ? capacityCheck.requestedGpu : capacityCheck.requestedCpu,
      totalAllocated: state.deployments
        .filter(d => d.status === 'active')
        .reduce((sum, d) => sum + (isGpu ? d.requestedGpu : d.requestedCpu), 0),
      deploymentsCreated: state.deployments.length,
      deploymentsActive: state.deployments.filter(d => d.status === 'active').length,
      deploymentsFailed: state.deployments.filter(d => d.status === 'failed').length
    }
    onProgress?.(state)
  }

  updateStats()

  // Adaptive deployment loop
  while (state.remainingResources > 0) {
    // Create deployment for current request size
    const requestSize = Math.min(state.currentRequestSize, state.remainingResources)
    const deployment = createNextDeploymentItem(state, requestSize)
    state.deployments.push(deployment)
    updateStats()

    try {
      // Create deployment and get bids
      const bids = await createDeploymentAndGetBids(
        signingClient,
        owner,
        config,
        deployment,
        (update) => {
          Object.assign(deployment, update)
          updateStats()
        }
      )

      if (bids.length === 0) {
        // No bids - reduce request size and retry
        deployment.status = 'failed'
        deployment.error = `No bids for ${requestSize} ${isGpu ? 'GPU' : 'CPU'}${requestSize > 1 ? 's' : ''}`
        updateStats()

        // Halve the request size (minimum 1)
        if (state.currentRequestSize > 1) {
          state.currentRequestSize = Math.max(1, Math.floor(state.currentRequestSize / 2))
          console.log(`No bids received, reducing request to ${state.currentRequestSize}`)
          continue
        } else {
          // Already at minimum, give up on remaining
          console.log('No bids at minimum size, stopping')
          break
        }
      }

      // Auto-select lowest price bid
      const sortedBids = [...bids].sort(
        (a, b) => parseFloat(a.bid.price.amount) - parseFloat(b.bid.price.amount)
      )
      const selectedBid = sortedBids[0]
      const providerName = selectedBid.bid.id.provider.slice(0, 12) + '...'

      // Accept bid and deploy
      await acceptBidAndDeploy(
        signingClient,
        owner,
        deployment,
        selectedBid,
        providerName,
        (update) => {
          Object.assign(deployment, update)
          updateStats()
        }
      )

      // Successfully allocated - update remaining
      const allocated = isGpu ? deployment.requestedGpu : deployment.requestedCpu
      state.remainingResources -= allocated
      console.log(`Allocated ${allocated} ${isGpu ? 'GPU' : 'CPU'}(s), remaining: ${state.remainingResources}`)

    } catch (error) {
      deployment.status = 'failed'
      deployment.error = error instanceof Error ? error.message : String(error)
      updateStats()
      // Continue trying with remaining resources
    }
  }

  state.status = state.stats.deploymentsActive === 0 ? 'cancelled' : 'completed'
  state.completedAt = new Date()
  updateStats()

  return state
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Cancel pending deployments in a fleet
 */
export function cancelFleet(state: FleetState): FleetState {
  return {
    ...state,
    status: 'cancelled',
    completedAt: new Date(),
    deployments: state.deployments.map(d => ({
      ...d,
      status: ['pending', 'selecting-bid'].includes(d.status) ? 'failed' : d.status,
      error: ['pending', 'selecting-bid'].includes(d.status) ? 'Cancelled by user' : d.error
    })) as FleetDeploymentItem[]
  }
}

/**
 * Get summary of fleet deployment results
 */
export function getFleetSummary(state: FleetState): {
  totalDeployments: number
  activeDeployments: number
  failedDeployments: number
  totalCpuDeployed: number
  totalGpuDeployed: number
  totalInstancesDeployed: number
  dseqs: string[]
} {
  const activeDeployments = state.deployments.filter(d => d.status === 'active')
  const isGpu = state.config.minerType === 'cuda'

  return {
    totalDeployments: state.deployments.length,
    activeDeployments: activeDeployments.length,
    failedDeployments: state.deployments.filter(d => d.status === 'failed').length,
    // For CPU: count requestedCpu, for GPU: count requestedGpu
    totalCpuDeployed: activeDeployments.reduce((sum, d) => sum + d.requestedCpu, 0),
    totalGpuDeployed: activeDeployments.reduce((sum, d) => sum + d.requestedGpu, 0),
    totalInstancesDeployed: activeDeployments.reduce((sum, d) => sum + d.instanceCount, 0),
    dseqs: activeDeployments.map(d => d.dseq!).filter(Boolean)
  }
}

// Legacy exports for backward compatibility
export type { ProviderCapacity }
export interface AllocationPlan {
  allocations: { provider: ProviderCapacity; requestedCpu: number; requestedGpu: number; requestedMemoryGi: number }[]
  totalCpu: number
  totalGpu: number
  deploymentsNeeded: number
  requestedCpu: number
  requestedGpu: number
  shortfall: { cpu: number; gpu: number }
}

export async function planFleetAllocation(
  targetCpu: number,
  targetGpu: number,
  _options: Record<string, unknown> = {}
): Promise<AllocationPlan> {
  const check = await checkNetworkCapacity(targetCpu, targetGpu)
  return {
    allocations: check.topProviders.map(p => ({
      provider: p,
      requestedCpu: targetGpu > 0 ? 2 : 1,
      requestedGpu: targetGpu > 0 ? 1 : 0,
      requestedMemoryGi: targetGpu > 0 ? 4 : 2
    })),
    totalCpu: check.totalCpu,
    totalGpu: check.totalGpu,
    deploymentsNeeded: check.deploymentsNeeded,
    requestedCpu: targetCpu,
    requestedGpu: targetGpu,
    shortfall: {
      cpu: Math.max(0, targetCpu - check.totalCpu),
      gpu: Math.max(0, targetGpu - check.totalGpu)
    }
  }
}
