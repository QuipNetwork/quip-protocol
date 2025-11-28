/**
 * Deployment Monitor Utilities
 *
 * Monitors active deployments to detect when containers have exited,
 * enabling automatic lease closure to stop billing and recover escrow.
 */

import { AKASH_CONSOLE_API } from '../config/constants'

/**
 * Container state as reported by the provider
 */
export type ContainerState = 'running' | 'exited' | 'error' | 'pending' | 'unknown'

/**
 * Deployment status with container health info
 */
export interface DeploymentStatus {
  dseq: string
  owner: string
  deploymentState: 'active' | 'closed' | 'unknown'
  leaseState: 'active' | 'closed' | 'unknown'
  containerState: ContainerState
  exitCode?: number
  exitReason?: string
  isComplete: boolean // True if container exited normally (exit code 0)
  isError: boolean    // True if container exited with error
  lastChecked: Date
  provider?: string
  providerUri?: string
}

/**
 * Service status from provider
 */
interface ServiceStatus {
  name: string
  available: number
  total: number
  uris?: string[]
  ready_replicas?: number
  available_replicas?: number
}

/**
 * Lease status response from Console API
 */
interface LeaseStatusResponse {
  services?: Record<string, ServiceStatus>
  forwarded_ports?: Array<{
    host: string
    port: number
    externalPort: number
    proto: string
    name: string
  }>
}

/**
 * Check deployment status via Console API
 *
 * Returns detailed status including whether the container is still running
 */
export async function checkDeploymentStatus(
  owner: string,
  dseq: string
): Promise<DeploymentStatus> {
  const now = new Date()
  const baseStatus: DeploymentStatus = {
    dseq,
    owner,
    deploymentState: 'unknown',
    leaseState: 'unknown',
    containerState: 'unknown',
    isComplete: false,
    isError: false,
    lastChecked: now
  }

  try {
    // First, get deployment details from Console API
    const deploymentResponse = await fetch(
      `${AKASH_CONSOLE_API}/v1/deployment/${owner}/${dseq}`,
      { signal: AbortSignal.timeout(10000) }
    )

    if (!deploymentResponse.ok) {
      if (deploymentResponse.status === 404) {
        return {
          ...baseStatus,
          deploymentState: 'closed',
          leaseState: 'closed',
          containerState: 'exited',
          isComplete: true
        }
      }
      throw new Error(`Deployment API returned ${deploymentResponse.status}`)
    }

    const deploymentData = await deploymentResponse.json()

    // Extract deployment and lease state
    const deploymentState = deploymentData.status === 'active' ? 'active' : 'closed'
    const hasActiveLease = deploymentData.leases?.some(
      (l: { status: string }) => l.status === 'active'
    )
    const leaseState = hasActiveLease ? 'active' : 'closed'

    // Get provider info from first active lease
    const activeLease = deploymentData.leases?.find(
      (l: { status: string }) => l.status === 'active'
    )
    const provider = activeLease?.provider?.address
    const providerUri = activeLease?.provider?.hostUri

    // Update base status
    baseStatus.deploymentState = deploymentState
    baseStatus.leaseState = leaseState
    baseStatus.provider = provider
    baseStatus.providerUri = providerUri

    // If no active lease, consider it closed
    if (!hasActiveLease || deploymentState === 'closed') {
      return {
        ...baseStatus,
        containerState: 'exited',
        isComplete: true
      }
    }

    // Try to get lease status for container state
    if (activeLease) {
      const leaseStatus = await checkLeaseStatus(
        owner,
        dseq,
        activeLease.gseq || 1,
        activeLease.oseq || 1
      )

      if (leaseStatus) {
        // Analyze service status to determine container state
        const containerStatus = analyzeServiceStatus(leaseStatus)
        return {
          ...baseStatus,
          containerState: containerStatus.state,
          exitCode: containerStatus.exitCode,
          exitReason: containerStatus.reason,
          isComplete: containerStatus.state === 'exited' && containerStatus.exitCode === 0,
          isError: containerStatus.state === 'error' || (containerStatus.state === 'exited' && containerStatus.exitCode !== 0)
        }
      }
    }

    // If we can't determine container state, assume running
    return {
      ...baseStatus,
      containerState: 'running'
    }

  } catch (error) {
    console.error(`Error checking deployment ${dseq}:`, error)
    return {
      ...baseStatus,
      containerState: 'unknown',
      exitReason: error instanceof Error ? error.message : 'Unknown error'
    }
  }
}

/**
 * Check lease status from Console API
 */
async function checkLeaseStatus(
  owner: string,
  dseq: string,
  gseq: number,
  oseq: number
): Promise<LeaseStatusResponse | null> {
  try {
    const response = await fetch(
      `${AKASH_CONSOLE_API}/v1/lease/${owner}/${dseq}/${gseq}/${oseq}/status`,
      { signal: AbortSignal.timeout(10000) }
    )

    if (!response.ok) {
      return null
    }

    return await response.json()
  } catch {
    return null
  }
}

/**
 * Analyze service status to determine container state
 */
function analyzeServiceStatus(leaseStatus: LeaseStatusResponse): {
  state: ContainerState
  exitCode?: number
  reason?: string
} {
  if (!leaseStatus.services) {
    return { state: 'unknown', reason: 'No service status available' }
  }

  const services = Object.values(leaseStatus.services)

  if (services.length === 0) {
    return { state: 'unknown', reason: 'No services found' }
  }

  // Check if any service is not fully available
  for (const service of services) {
    const available = service.available_replicas ?? service.available ?? 0
    const total = service.total ?? 1

    if (available < total) {
      // Service not fully available - could be starting, crashed, or exited
      if (available === 0) {
        // No replicas available - likely exited or error
        return {
          state: 'exited',
          reason: `Service ${service.name} has 0/${total} replicas available`
        }
      }
      return {
        state: 'pending',
        reason: `Service ${service.name} has ${available}/${total} replicas`
      }
    }
  }

  // All services have all replicas available
  return { state: 'running' }
}

/**
 * Check if a deployment should be auto-closed
 *
 * Returns true if:
 * - Container has exited (any exit code)
 * - Deployment is in error state
 * - Lease is closed but deployment is still active
 */
export function shouldAutoClose(status: DeploymentStatus): boolean {
  // Already closed
  if (status.deploymentState === 'closed') {
    return false
  }

  // Container exited (success or failure)
  if (status.containerState === 'exited' || status.containerState === 'error') {
    return true
  }

  // Lease closed but deployment still active
  if (status.leaseState === 'closed' && status.deploymentState === 'active') {
    return true
  }

  return false
}

/**
 * Get human-readable status message
 */
export function getStatusMessage(status: DeploymentStatus): string {
  if (status.deploymentState === 'closed') {
    return 'Deployment closed'
  }

  if (status.leaseState === 'closed') {
    return 'Lease closed - no active provider'
  }

  switch (status.containerState) {
    case 'running':
      return 'Container running'
    case 'exited':
      if (status.exitCode === 0) {
        return 'Container completed successfully'
      }
      return `Container exited (code: ${status.exitCode ?? 'unknown'})`
    case 'error':
      return `Container error: ${status.exitReason || 'Unknown error'}`
    case 'pending':
      return 'Container starting...'
    case 'unknown':
      return status.exitReason || 'Status unknown'
    default:
      return 'Unknown state'
  }
}

/**
 * Get status color for UI display
 */
export function getStatusColor(status: DeploymentStatus): string {
  if (status.deploymentState === 'closed') {
    return '#718096' // gray
  }

  switch (status.containerState) {
    case 'running':
      return '#48bb78' // green
    case 'exited':
      return status.exitCode === 0 ? '#48bb78' : '#fc8181' // green or red
    case 'error':
      return '#fc8181' // red
    case 'pending':
      return '#f6ad55' // orange
    case 'unknown':
      return '#a0aec0' // light gray
    default:
      return '#a0aec0'
  }
}

/**
 * Estimate remaining escrow balance
 *
 * Based on initial deposit, price per block, and blocks elapsed
 */
export function estimateRemainingEscrow(
  initialDepositUakt: number,
  pricePerBlockUakt: number,
  startBlock: number,
  currentBlock: number
): number {
  const blocksElapsed = Math.max(0, currentBlock - startBlock)
  const spent = blocksElapsed * pricePerBlockUakt
  return Math.max(0, initialDepositUakt - spent)
}

/**
 * Format escrow amount for display
 */
export function formatEscrow(uakt: number): string {
  const akt = uakt / 1_000_000
  return `${akt.toFixed(4)} AKT`
}
