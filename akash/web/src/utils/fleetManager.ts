/**
 * Fleet Manager - Intelligent allocation of deployments across Akash providers
 *
 * This module handles:
 * 1. Querying provider capacity
 * 2. Planning optimal resource allocation across providers
 * 3. Executing fleet deployments
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
import { generateSDLForAllocation, type SDLConfig } from './sdl'

// ============================================================================
// Types
// ============================================================================

/**
 * Allocation for a single provider in the fleet
 */
export interface ProviderAllocation {
  provider: ProviderCapacity
  requestedCpu: number    // CPUs to request
  requestedGpu: number    // GPUs to request
  requestedMemoryGi: number  // Memory in GiB
}

/**
 * Complete allocation plan for a fleet deployment
 */
export interface AllocationPlan {
  allocations: ProviderAllocation[]
  totalCpu: number
  totalGpu: number
  deploymentsNeeded: number
  // What we wanted vs what we can get
  requestedCpu: number
  requestedGpu: number
  shortfall: {
    cpu: number
    gpu: number
  }
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
 */
export interface FleetDeploymentItem {
  index: number
  allocation: ProviderAllocation
  status: FleetDeploymentStatus
  dseq?: string
  transactionHash?: string
  selectedBid?: Bid
  serviceUrl?: string
  error?: string
  startedAt?: Date
  completedAt?: Date
}

/**
 * Overall fleet deployment state
 */
export interface FleetState {
  config: Omit<SDLConfig, 'cpuUnits' | 'gpuUnits' | 'memoryGi'>
  plan: AllocationPlan
  deployments: FleetDeploymentItem[]
  status: 'planning' | 'deploying' | 'completed' | 'cancelled'
  startedAt: Date
  completedAt?: Date
  stats: {
    total: number
    active: number
    failed: number
    pending: number
  }
}

/**
 * Callback for fleet deployment progress updates
 */
export type FleetProgressCallback = (state: FleetState) => void

// ============================================================================
// Allocation Algorithm
// ============================================================================

/**
 * Plan fleet allocation across available providers
 *
 * Strategy: Greedy allocation starting with largest providers
 * - Prefer fewer, larger deployments (less transaction overhead)
 * - Prioritize reliable providers (have active leases)
 * - Respect minimum resource requirements
 */
export async function planFleetAllocation(
  targetCpu: number,
  targetGpu: number,
  options: {
    minCpuPerDeployment?: number
    minGpuPerDeployment?: number
    memoryPerCpuGi?: number  // Memory per CPU in GiB
    onlyReliable?: boolean
  } = {}
): Promise<AllocationPlan> {
  const {
    minCpuPerDeployment = 1,
    minGpuPerDeployment = 1,
    memoryPerCpuGi = 2,
    onlyReliable = false
  } = options

  // Fetch all providers with capacity
  const allProviders = await fetchAllProviders()

  // Filter by requirements
  const providers = filterProvidersByCapacity(allProviders, {
    minCpu: targetGpu > 0 ? 0 : minCpuPerDeployment,
    minGpu: targetGpu > 0 ? minGpuPerDeployment : 0,
    onlyReliable,
    onlyOnline: true
  })

  console.log(`Found ${providers.length} suitable providers out of ${allProviders.length} total`)

  // Get total capacity for reference
  const totalCapacity = getTotalCapacity(providers)
  console.log('Total available capacity:', totalCapacity)

  const allocations: ProviderAllocation[] = []
  let remainingCpu = targetCpu * 1000  // Convert to millicores
  let remainingGpu = targetGpu

  // Sort providers by capacity (largest first for CPU, or by GPU for GPU deployments)
  const sortedProviders = [...providers].sort((a, b) => {
    if (targetGpu > 0) {
      // For GPU deployments, sort by GPU capacity
      return b.availableGpu - a.availableGpu
    }
    // For CPU deployments, sort by CPU capacity
    return b.availableCpu - a.availableCpu
  })

  for (const provider of sortedProviders) {
    if (targetGpu > 0) {
      // GPU allocation
      if (remainingGpu <= 0) break
      if (provider.availableGpu <= 0) continue

      const gpusToAllocate = Math.min(provider.availableGpu, remainingGpu)
      const cpusForGpus = gpusToAllocate * 2  // 2 CPUs per GPU
      const memoryForGpus = gpusToAllocate * 4  // 4GiB per GPU

      allocations.push({
        provider,
        requestedCpu: cpusForGpus,
        requestedGpu: gpusToAllocate,
        requestedMemoryGi: memoryForGpus
      })

      remainingGpu -= gpusToAllocate
    } else {
      // CPU allocation
      if (remainingCpu <= 0) break
      if (provider.availableCpu < minCpuPerDeployment * 1000) continue

      const cpusToAllocate = Math.min(provider.availableCpu, remainingCpu)
      const cpuUnits = Math.floor(cpusToAllocate / 1000)
      const memoryGi = cpuUnits * memoryPerCpuGi

      allocations.push({
        provider,
        requestedCpu: cpuUnits,
        requestedGpu: 0,
        requestedMemoryGi: memoryGi
      })

      remainingCpu -= cpuUnits * 1000
    }
  }

  // Calculate totals
  const totalAllocatedCpu = allocations.reduce((sum, a) => sum + a.requestedCpu, 0)
  const totalAllocatedGpu = allocations.reduce((sum, a) => sum + a.requestedGpu, 0)

  return {
    allocations,
    totalCpu: totalAllocatedCpu,
    totalGpu: totalAllocatedGpu,
    deploymentsNeeded: allocations.length,
    requestedCpu: targetCpu,
    requestedGpu: targetGpu,
    shortfall: {
      cpu: Math.max(0, targetCpu - totalAllocatedCpu),
      gpu: Math.max(0, targetGpu - totalAllocatedGpu)
    }
  }
}

// ============================================================================
// Fleet Deployment Execution
// ============================================================================

/**
 * Execute a fleet deployment plan
 *
 * @param signingClient - Cosmos signing client
 * @param owner - Wallet address
 * @param config - Base SDL configuration (without resource overrides)
 * @param plan - Allocation plan from planFleetAllocation
 * @param onProgress - Callback for progress updates
 */
export async function deployFleet(
  signingClient: SigningStargateClient,
  owner: string,
  config: Omit<SDLConfig, 'cpuUnits' | 'gpuUnits' | 'memoryGi'>,
  plan: AllocationPlan,
  onProgress?: FleetProgressCallback
): Promise<FleetState> {
  // Initialize fleet state
  const state: FleetState = {
    config,
    plan,
    deployments: plan.allocations.map((allocation, index) => ({
      index,
      allocation,
      status: 'pending' as FleetDeploymentStatus
    })),
    status: 'deploying',
    startedAt: new Date(),
    stats: {
      total: plan.allocations.length,
      active: 0,
      failed: 0,
      pending: plan.allocations.length
    }
  }

  const updateState = () => {
    state.stats = {
      total: state.deployments.length,
      active: state.deployments.filter(d => d.status === 'active').length,
      failed: state.deployments.filter(d => d.status === 'failed').length,
      pending: state.deployments.filter(d =>
        !['active', 'failed'].includes(d.status)
      ).length
    }
    onProgress?.(state)
  }

  updateState()

  // Deploy each allocation sequentially
  // (Could be parallelized in batches, but sequential is safer for wallet interactions)
  for (const deployment of state.deployments) {
    try {
      await deployAllocation(
        signingClient,
        owner,
        config,
        deployment,
        (updatedDeployment) => {
          Object.assign(deployment, updatedDeployment)
          updateState()
        }
      )
    } catch (error) {
      deployment.status = 'failed'
      deployment.error = error instanceof Error ? error.message : String(error)
      updateState()
      // Continue with other deployments even if one fails
    }
  }

  state.status = state.stats.failed === state.stats.total ? 'cancelled' : 'completed'
  state.completedAt = new Date()
  updateState()

  return state
}

/**
 * Deploy a single allocation
 */
async function deployAllocation(
  signingClient: SigningStargateClient,
  owner: string,
  config: Omit<SDLConfig, 'cpuUnits' | 'gpuUnits' | 'memoryGi'>,
  deployment: FleetDeploymentItem,
  onUpdate: (deployment: Partial<FleetDeploymentItem>) => void
): Promise<void> {
  const { allocation } = deployment

  // Step 1: Generate SDL for this allocation
  onUpdate({ status: 'creating', startedAt: new Date() })

  const sdl = generateSDLForAllocation(config, {
    cpuUnits: allocation.requestedCpu,
    gpuUnits: allocation.requestedGpu,
    memoryGi: allocation.requestedMemoryGi
  })

  // Step 2: Create deployment
  const deployResult = await createDeployment(signingClient, owner, sdl)
  onUpdate({
    dseq: deployResult.dseq,
    transactionHash: deployResult.transactionHash,
    status: 'waiting-bids'
  })

  // Step 3: Wait for bids
  const bids = await waitForBids(owner, deployResult.dseq, 120000, 5000)

  if (bids.length === 0) {
    throw new Error('No bids received')
  }

  onUpdate({ status: 'selecting-bid' })

  // Step 4: Select best bid
  // Prefer bid from our target provider, otherwise take lowest price
  let selectedBid = bids.find(
    b => b.bid.id.provider === allocation.provider.address
  )

  if (!selectedBid) {
    // Fallback: select lowest price
    selectedBid = [...bids].sort(
      (a, b) => parseFloat(a.bid.price.amount) - parseFloat(b.bid.price.amount)
    )[0]
  }

  onUpdate({ selectedBid, status: 'accepting-bid' })

  // Step 5: Accept bid
  await acceptBid(signingClient, selectedBid)
  onUpdate({ status: 'sending-manifest' })

  // Step 6: Send manifest
  const manifestResult = await sendManifestViaConsole(
    owner,
    deployResult.dseq,
    selectedBid.bid.id.provider,
    sdl
  )

  if (!manifestResult.success) {
    throw new Error(`Manifest failed: ${manifestResult.error}`)
  }

  onUpdate({
    status: 'active',
    completedAt: new Date()
  })
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
      status: d.status === 'pending' ? 'failed' : d.status,
      error: d.status === 'pending' ? 'Cancelled by user' : d.error
    }))
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
  dseqs: string[]
} {
  const activeDeployments = state.deployments.filter(d => d.status === 'active')

  return {
    totalDeployments: state.deployments.length,
    activeDeployments: activeDeployments.length,
    failedDeployments: state.deployments.filter(d => d.status === 'failed').length,
    totalCpuDeployed: activeDeployments.reduce(
      (sum, d) => sum + d.allocation.requestedCpu,
      0
    ),
    totalGpuDeployed: activeDeployments.reduce(
      (sum, d) => sum + d.allocation.requestedGpu,
      0
    ),
    dseqs: activeDeployments.map(d => d.dseq!).filter(Boolean)
  }
}
