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
  // Mining completion status (detected via HTTP done.json endpoint)
  miningComplete?: boolean  // True if mining finished and results uploaded
  ipfsCid?: string          // IPFS CID of results manifest (if available)
  deploymentUri?: string    // HTTP URI for the deployment
}

/**
 * Load certificate from localStorage for mTLS provider queries
 */
function loadCertificateFromStorage(owner: string): { certPem: string; keyPem: string } | null {
  const key = `akash_cert_${owner}`
  const stored = localStorage.getItem(key)
  if (!stored) return null

  try {
    const parsed = JSON.parse(stored)
    if (parsed.certPem && parsed.privateKeyPem) {
      return {
        certPem: parsed.certPem,
        keyPem: parsed.privateKeyPem
      }
    }
    return null
  } catch {
    return null
  }
}

/**
 * Get base deployment status via Console API (internal helper)
 */
async function getBaseDeploymentStatus(
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

    // Try to get deployment URI by querying the provider via mTLS proxy
    if (activeLease?.provider?.hostUri) {
      const providerHostUri = activeLease.provider.hostUri
      const gseq = activeLease.gseq || 1
      const oseq = activeLease.oseq || 1

      // Load certificate from localStorage
      const cert = loadCertificateFromStorage(owner)
      if (cert) {
        console.log(`Querying provider ${providerHostUri} for lease status via mTLS proxy...`)

        try {
          // Query the provider's lease status endpoint via the mTLS proxy
          const providerStatusUrl = `${providerHostUri}/lease/${dseq}/${gseq}/${oseq}/status`
          const proxyResponse = await fetch('/api/provider-proxy', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              url: providerStatusUrl,
              method: 'GET',
              certPem: cert.certPem,
              keyPem: cert.keyPem,
              timeout: 10000
            }),
            signal: AbortSignal.timeout(15000)
          })

          if (proxyResponse.ok) {
            const proxyResult = await proxyResponse.json()
            console.log(`Provider proxy response:`, proxyResult)

            if (proxyResult.success && proxyResult.body) {
              const leaseStatus = JSON.parse(proxyResult.body)
              console.log(`Provider lease status:`, leaseStatus)

              // Extract service URIs from the provider's response
              if (leaseStatus.services) {
                const services = Object.values(leaseStatus.services) as Array<{ uris?: string[] }>
                const firstUri = services[0]?.uris?.[0]
                if (firstUri) {
                  baseStatus.deploymentUri = firstUri.startsWith('http') ? firstUri : `http://${firstUri}`
                  console.log(`Deployment URI from provider: ${baseStatus.deploymentUri}`)
                }
              }
            }
          } else {
            console.log(`Provider proxy returned ${proxyResponse.status}`)
          }
        } catch (error) {
          console.log(`Could not query provider for lease status:`, error)
        }
      } else {
        console.log(`No certificate found for ${owner} - cannot query provider`)
      }
    }

    // If no active lease, consider it closed
    if (!hasActiveLease || deploymentState === 'closed') {
      return {
        ...baseStatus,
        containerState: 'exited',
        isComplete: true
      }
    }

    // Container is running if we have an active lease
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

  // Mining complete (detected via HTTP done.json) - safe to close
  if (status.miningComplete) {
    return true
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

  // Mining completion takes priority over container state
  if (status.miningComplete) {
    if (status.ipfsCid) {
      return 'Mining complete - results on IPFS'
    }
    return 'Mining complete - ready to close'
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

  // Mining complete is always green (success)
  if (status.miningComplete) {
    return '#48bb78' // green
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

/**
 * Response from the container's done.json endpoint
 */
interface DoneResponse {
  status: 'complete' | 'error'
  exit_code: number
  ipfs_cid?: string
  timestamp: string
  message: string
}

/**
 * Check if mining is complete by fetching the done.json endpoint from the container
 *
 * The container writes done.json when mining is finished and results are uploaded.
 * This allows the monitor to detect completion even though the container stays running
 * (because Akash providers don't support restart:never).
 *
 * @param deploymentUri - The HTTP URI for the deployment (e.g., from lease status)
 * @returns Mining completion status with IPFS CID if available
 */
export async function checkMiningComplete(
  deploymentUri: string
): Promise<{ complete: boolean; ipfsCid?: string; exitCode?: number; message?: string }> {
  try {
    // Ensure URI has protocol (use https since Akash ingress uses self-signed certs)
    let uri = deploymentUri
    if (!uri.startsWith('http://') && !uri.startsWith('https://')) {
      uri = `https://${uri}`
    }

    const targetUrl = `${uri}/done.json`
    console.log(`Checking done.json via proxy: ${targetUrl}`)

    // Use the fetch proxy to bypass CORS and certificate issues
    const proxyUrl = `/api/fetch-proxy?url=${encodeURIComponent(targetUrl)}`
    const response = await fetch(proxyUrl, {
      signal: AbortSignal.timeout(15000)
    })

    if (!response.ok) {
      console.log(`Fetch proxy returned ${response.status}`)
      return { complete: false }
    }

    const proxyResult = await response.json()
    console.log(`Fetch proxy result:`, proxyResult)

    if (!proxyResult.success) {
      // 404 means mining not complete yet, other errors are transient
      console.log(`done.json fetch failed:`, proxyResult.error || proxyResult.status)
      return { complete: false }
    }

    // Parse the response body
    const data: DoneResponse = JSON.parse(proxyResult.body)
    console.log(`done.json response:`, data)

    if (data.status === 'complete') {
      return {
        complete: true,
        ipfsCid: data.ipfs_cid,
        exitCode: data.exit_code,
        message: data.message
      }
    }

    return { complete: false }
  } catch (error) {
    // Network errors, timeouts, etc. - assume not complete
    console.log(`Could not fetch done.json:`, error)
    return { complete: false }
  }
}

/**
 * Check deployment status via Console API and HTTP done.json endpoint
 *
 * Returns detailed status including whether the container is still running
 * and whether mining has completed (detected via done.json).
 */
export async function checkDeploymentStatus(
  owner: string,
  dseq: string
): Promise<DeploymentStatus> {
  // Get the base deployment status (includes deploymentUri from lease status)
  const status = await getBaseDeploymentStatus(owner, dseq)

  // If deployment is not active or we don't have a deployment URI, return as-is
  if (status.deploymentState !== 'active' || !status.deploymentUri) {
    return status
  }

  // Check the done.json endpoint to detect mining completion
  try {
    const miningStatus = await checkMiningComplete(status.deploymentUri)

    if (miningStatus.complete) {
      status.miningComplete = true
      status.ipfsCid = miningStatus.ipfsCid
      status.isComplete = true

      // Update exit code if available
      if (miningStatus.exitCode !== undefined) {
        status.exitCode = miningStatus.exitCode
        status.isError = miningStatus.exitCode !== 0
      }
    }
  } catch (error) {
    console.debug(`Could not check mining status for ${dseq}:`, error)
    // Continue with base status - mining check is optional
  }

  return status
}
