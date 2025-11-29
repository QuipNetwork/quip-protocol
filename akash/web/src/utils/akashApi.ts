import { AKASH_REST, AKASH_DENOM, AKASH_CONSOLE_API } from '../config/constants'
import type { SigningStargateClient } from '@cosmjs/stargate'
import Long from 'long'
import { signAndBroadcastWithSession } from './sessionSigning'

// Import v1beta4 message types from chain-sdk (the current Akash mainnet version)
// Using chain-sdk ensures we stay compatible with API upgrades
import { MsgCreateDeployment, MsgCloseDeployment } from '@akashnetwork/chain-sdk/private-types/akash.v1beta4'
// Market messages are in v1beta5
import { MsgCreateLease } from '@akashnetwork/chain-sdk/private-types/akash.v1beta5'
// Source enum and escrow deposit message are in the v1 base types
import { Source, MsgAccountDeposit } from '@akashnetwork/chain-sdk/private-types/akash.v1'

// SDL parsing uses chain-sdk/web (browser-compatible version)
// chain-sdk's manifestSortedJSON() omits null values like Go's omitempty,
// while akashjs includes them, causing hash mismatches with providers
import { SDL } from '@akashnetwork/chain-sdk/web'

/**
 * Fetch the current block height from Akash
 */
export async function fetchBlockHeight(): Promise<number> {
  try {
    const response = await fetch(`${AKASH_REST}/cosmos/base/tendermint/v1beta1/blocks/latest`)
    const data = await response.json()
    return parseInt(data.block?.header?.height || '0', 10)
  } catch (error) {
    console.error('Failed to fetch block height:', error)
    return 0
  }
}

/**
 * Fetch balance for an Akash address
 */
export async function fetchBalance(address: string): Promise<string> {
  try {
    const response = await fetch(
      `${AKASH_REST}/cosmos/bank/v1beta1/balances/${address}`
    )
    const data = await response.json()

    const aktBalance = data.balances?.find(
      (b: { denom: string; amount: string }) => b.denom === AKASH_DENOM
    )
    return aktBalance?.amount || '0'
  } catch (error) {
    console.error('Failed to fetch balance:', error)
    return '0'
  }
}

/**
 * Fetch deployments for an address (using v1beta4 API)
 */
export async function fetchDeployments(address: string): Promise<Deployment[]> {
  try {
    const response = await fetch(
      `${AKASH_REST}/akash/deployment/v1beta4/deployments/list?filters.owner=${address}`
    )
    const data = await response.json()
    return data.deployments || []
  } catch (error) {
    console.error('Failed to fetch deployments:', error)
    return []
  }
}

/**
 * Fetch lease info for a deployment (using v1beta5 market API)
 */
export async function fetchLeases(
  owner: string,
  dseq: string
): Promise<Lease[]> {
  try {
    const response = await fetch(
      `${AKASH_REST}/akash/market/v1beta5/leases/list?filters.owner=${owner}&filters.dseq=${dseq}`
    )
    const data = await response.json()
    return data.leases || []
  } catch (error) {
    console.error('Failed to fetch leases:', error)
    return []
  }
}

// Types for v1beta4 API responses
export interface Deployment {
  deployment: {
    id: {
      owner: string
      dseq: string
    }
    state: string
    hash: string
    created_at: string
  }
  groups: Array<{
    id: {
      owner: string
      dseq: string
      gseq: number
    }
    state: string
    group_spec: {
      name: string
      requirements: unknown
      resources: unknown[]
    }
    created_at: string
  }>
  escrow_account: {
    id: {
      scope: string
      xid: string
    }
    state: {
      owner: string
      state: string
      transferred: Array<{ denom: string; amount: string }>
      settled_at: string
      funds: Array<{ denom: string; amount: string }>
      deposits: Array<{
        owner: string
        height: string
        source: string
        balance: { denom: string; amount: string }
      }>
    }
  }
}

// Lease type for v1beta5 market API
export interface Lease {
  lease: {
    id: {
      owner: string
      dseq: string
      gseq: number
      oseq: number
      provider: string
      bseq?: number
    }
    state: string
    price: {
      denom: string
      amount: string
    }
    created_at: string
    closed_on: string
    reason?: string
  }
  escrow_payment: {
    id: {
      aid: {
        scope: string
        xid: string
      }
      xid: string
    }
    state: {
      owner: string
      state: string
      rate: {
        denom: string
        amount: string
      }
      balance: {
        denom: string
        amount: string
      }
      unsettled?: {
        denom: string
        amount: string
      }
      withdrawn: {
        denom: string
        amount: string
      }
    }
  }
}

// Bid type for v1beta5 market API
export interface Bid {
  bid: {
    id: {
      owner: string
      dseq: string
      gseq: number
      oseq: number
      provider: string
    }
    state: string
    price: {
      denom: string
      amount: string
    }
    created_at: string
  }
}

/**
 * Fetch bids for a deployment (using v1beta5 market API)
 */
export async function fetchBids(owner: string, dseq: string): Promise<Bid[]> {
  try {
    const response = await fetch(
      `${AKASH_REST}/akash/market/v1beta5/bids/list?filters.owner=${owner}&filters.dseq=${dseq}`
    )
    const data = await response.json()
    return data.bids || []
  } catch (error) {
    console.error('Failed to fetch bids:', error)
    return []
  }
}

/**
 * Wait for bids on a deployment with polling
 */
export async function waitForBids(
  owner: string,
  dseq: string,
  maxWaitMs = 60000,
  pollIntervalMs = 3000
): Promise<Bid[]> {
  const startTime = Date.now()

  while (Date.now() - startTime < maxWaitMs) {
    const bids = await fetchBids(owner, dseq)
    const openBids = bids.filter(b => b.bid.state === 'open')

    if (openBids.length > 0) {
      return openBids
    }

    // Wait before polling again
    await new Promise(resolve => setTimeout(resolve, pollIntervalMs))
  }

  return []
}

/**
 * Select the best bid (lowest price)
 * In the future, this could also consider provider uptime/reputation
 */
export function selectBestBid(bids: Bid[]): Bid | null {
  if (bids.length === 0) return null

  // Sort by price (lowest first)
  const sorted = [...bids].sort((a, b) => {
    const priceA = parseFloat(a.bid.price.amount)
    const priceB = parseFloat(b.bid.price.amount)
    return priceA - priceB
  })

  return sorted[0]
}

/**
 * Accept a bid by creating a lease
 * Uses session signing when available for automatic approval
 */
export async function acceptBid(
  signingClient: SigningStargateClient,
  bid: Bid
): Promise<{ transactionHash: string }> {
  const { owner, dseq, gseq, oseq, provider } = bid.bid.id

  const msg = MsgCreateLease.fromPartial({
    bidId: {
      owner,
      dseq: Long.fromString(dseq, true),
      gseq,
      oseq,
      provider
    }
  })

  const typeUrl = `/${MsgCreateLease.$type}`

  // Use session signing if available
  const result = await signAndBroadcastWithSession(
    signingClient,
    owner,
    [{ typeUrl, value: msg }],
    'auto',
    'Accept bid via Quip Protocol'
  )

  if (result.code !== 0) {
    throw new Error(`Failed to accept bid: ${result.rawLog}`)
  }

  return { transactionHash: result.transactionHash }
}

/**
 * Remove null and empty values from manifest to match Go's `omitempty` JSON behavior
 *
 * The Akash Go code uses `json:",omitempty"` tags which OMIT null/empty values from JSON.
 * But akashjs's manifestSortedJSON() includes explicit `null` values.
 * This causes hash mismatch because:
 *   Go produces:   {"count":1,"env":[...],...}
 *   akashjs produces: {"args":null,"command":null,"count":1,...}
 *
 * We must strip nulls/empty values to match Go's output for hash consistency.
 */
function stripEmptyValues(obj: unknown, key?: string): unknown {
  // null/undefined -> stripped (Go's omitempty strips nil)
  if (obj === null || obj === undefined) {
    return undefined
  }

  // Empty string -> stripped (Go's omitempty strips "")
  if (typeof obj === 'string' && obj === '') {
    return undefined
  }

  // Convert 'val' fields to strings (Go's ResourceValue.val is string type)
  // This must happen BEFORE other processing
  if (key === 'val' && typeof obj === 'number') {
    return String(obj)
  }

  if (Array.isArray(obj)) {
    // Process array elements, filtering out undefined results
    const filtered = obj.map(item => stripEmptyValues(item)).filter(v => v !== undefined)
    // Empty array -> stripped (Go's omitempty strips [])
    return filtered.length > 0 ? filtered : undefined
  }

  if (typeof obj === 'object') {
    const result: Record<string, unknown> = {}
    for (const [k, value] of Object.entries(obj as Record<string, unknown>)) {
      // Pass the key name so we can detect 'val' fields
      const stripped = stripEmptyValues(value, k)
      // Only include non-undefined values (matching Go's omitempty)
      if (stripped !== undefined) {
        result[k] = stripped
      }
    }
    // Empty object -> stripped (Go's omitempty strips {})
    return Object.keys(result).length > 0 ? result : undefined
  }

  // Numbers (including 0) and booleans (including false) pass through
  // Go's omitempty DOES strip 0 and false, but those are meaningful in manifests
  // so we keep them to match expected behavior
  return obj
}

/**
 * Sort object keys recursively for deterministic JSON output
 * This matches Go's yaml.YAMLToJSON behavior which sorts keys
 */
function sortObjectKeys(obj: unknown): unknown {
  if (obj === null || obj === undefined) {
    return obj
  }

  if (Array.isArray(obj)) {
    return obj.map(sortObjectKeys)
  }

  if (typeof obj === 'object') {
    const sorted: Record<string, unknown> = {}
    const keys = Object.keys(obj as Record<string, unknown>).sort()
    for (const key of keys) {
      sorted[key] = sortObjectKeys((obj as Record<string, unknown>)[key])
    }
    return sorted
  }

  return obj
}

/**
 * Transform manifest to match Go's JSON output:
 * 1. Strip null/empty values (omitempty behavior)
 * 2. Sort object keys (yaml.YAMLToJSON behavior)
 */
function transformManifestForGo(manifest: unknown): string {
  const stripped = stripEmptyValues(manifest)
  const sorted = sortObjectKeys(stripped)
  return JSON.stringify(sorted)
}

/**
 * Send manifest to a provider via local mTLS proxy
 *
 * In development, the Vite server provides a /api/provider-proxy endpoint
 * that handles mTLS authentication to the provider.
 *
 * The browser sends: certificate + manifest to local proxy
 * The proxy sends: mTLS request to provider
 */
export interface ManifestSubmissionResult {
  success: boolean
  timestamp: string
  durationMs: number
  request: {
    url: string
    owner: string
    dseq: string
    provider: string
    manifestSize: number
  }
  response: {
    status: number
    statusText: string
    body: string
  }
  error?: string
}

/**
 * Load certificate from localStorage
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

export async function sendManifestViaConsole(
  owner: string,
  dseq: string,
  provider: string,
  manifestJson: string  // Pre-computed manifest JSON from createDeployment
): Promise<ManifestSubmissionResult> {
  const startTime = Date.now()
  const timestamp = new Date().toISOString()

  console.log('=== MANIFEST SUBMISSION START ===')
  console.log('Timestamp:', timestamp)
  console.log('Owner:', owner)
  console.log('DSEQ:', dseq)
  console.log('Provider:', provider)

  // Load certificate from localStorage
  const cert = loadCertificateFromStorage(owner)
  if (!cert) {
    throw new Error('No certificate found in localStorage. Please create a certificate first.')
  }
  console.log('Certificate loaded from localStorage')

  // Get provider host URI
  const providerInfo = await fetchProvider(provider)
  if (!providerInfo?.host_uri) {
    throw new Error(`Could not fetch provider host URI for ${provider}`)
  }
  const providerHost = providerInfo.host_uri.replace(/\/$/, '') // Remove trailing slash
  console.log('Provider host:', providerHost)

  // Use the pre-computed manifest JSON from deployment creation
  // CRITICAL: This must be the exact same string used to compute the deployment hash
  console.log('Using pre-computed manifest:', manifestJson)
  console.log('Manifest size:', manifestJson.length, 'bytes')

  // Provider manifest endpoint
  const providerUrl = `${providerHost}/deployment/${dseq}/manifest`
  console.log('Provider URL:', providerUrl)

  // Use local mTLS proxy
  const proxyUrl = '/api/provider-proxy'
  const proxyRequest = {
    url: providerUrl,
    method: 'PUT' as const,
    body: manifestJson,
    certPem: cert.certPem,
    keyPem: cert.keyPem,
    timeout: 30000
  }

  console.log('Sending to local mTLS proxy...')

  let response: Response
  let proxyResponse: {
    success: boolean
    status?: number
    statusText?: string
    body?: string
    error?: string
    durationMs: number
  }

  try {
    response = await fetch(proxyUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(proxyRequest)
    })

    proxyResponse = await response.json()
    const durationMs = Date.now() - startTime

    console.log('Proxy response:', proxyResponse)
    console.log('Duration:', durationMs, 'ms')

    const result: ManifestSubmissionResult = {
      success: proxyResponse.success,
      timestamp,
      durationMs,
      request: {
        url: providerUrl,
        owner,
        dseq,
        provider,
        manifestSize: manifestJson.length
      },
      response: {
        status: proxyResponse.status || 0,
        statusText: proxyResponse.statusText || '',
        body: proxyResponse.body || proxyResponse.error || ''
      }
    }

    if (!proxyResponse.success) {
      result.error = proxyResponse.error || `HTTP ${proxyResponse.status}: ${proxyResponse.body}`
      console.error('=== MANIFEST SUBMISSION FAILED ===')
      console.error('Error:', result.error)
      throw new Error(`Failed to send manifest: ${result.error}`)
    }

    console.log('=== MANIFEST SUBMISSION SUCCESS ===')
    return result

  } catch (fetchError) {
    const durationMs = Date.now() - startTime
    const errorMessage = fetchError instanceof Error ? fetchError.message : String(fetchError)

    console.error('=== MANIFEST SUBMISSION ERROR ===')
    console.error('Fetch error:', errorMessage)
    console.error('Duration:', durationMs, 'ms')

    // Re-throw with more context
    throw new Error(`Manifest submission failed: ${errorMessage}`)
  }
}

/**
 * Alternative: Send manifest directly to provider (requires mTLS - won't work in browser)
 * Kept for reference / non-browser use
 */
export async function sendManifestDirect(
  providerUri: string,
  dseq: string,
  manifest: unknown
): Promise<void> {
  const url = `${providerUri}/deployment/${dseq}/manifest`

  const response = await fetch(url, {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(manifest)
  })

  if (!response.ok) {
    throw new Error(`Failed to send manifest: ${response.statusText}`)
  }
}

/**
 * Fetch provider info
 */
export async function fetchProvider(providerAddress: string): Promise<{
  host_uri: string
  attributes: Array<{ key: string; value: string }>
} | null> {
  try {
    const response = await fetch(
      `${AKASH_REST}/akash/provider/v1beta4/providers/${providerAddress}`
    )
    const data = await response.json()
    return data.provider || null
  } catch (error) {
    console.error('Failed to fetch provider:', error)
    return null
  }
}

/**
 * Provider health status with detailed reliability info
 */
export interface ProviderHealthStatus {
  isOnline: boolean
  isReliable: boolean  // More stringent check
  activeLeaseCount: number
  isAudited: boolean
  uptime1d?: number
  uptime7d?: number
  uptime30d?: number
  lastCheckDate?: string
  error?: string
}

/**
 * Check if a provider is healthy/accessible via Console API
 * Returns detailed status info for reliability assessment
 *
 * Note: If Console API fails or returns unexpected format, we default to
 * treating the provider as "available" since they submitted a bid
 */
export async function checkProviderHealth(providerAddress: string): Promise<ProviderHealthStatus> {
  try {
    // Use Console API to check provider status
    const response = await fetch(
      `${AKASH_CONSOLE_API}/v1/providers/${providerAddress}`,
      { signal: AbortSignal.timeout(8000) } // 8 second timeout
    )

    if (!response.ok) {
      // If Console API doesn't know about provider, they may still be valid
      // (they submitted a bid, so they're at least registered)
      console.log(`Console API returned ${response.status} for ${providerAddress.slice(0, 12)}...`)
      return {
        isOnline: true,  // Assume online since they submitted a bid
        isReliable: false,
        activeLeaseCount: 0,
        isAudited: false,
        error: `API returned ${response.status}`
      }
    }

    const data = await response.json()
    console.log(`Provider ${providerAddress.slice(0, 12)}... status:`, data)

    // Handle various response formats - Console API may use different field names
    // Check for both camelCase and snake_case variants
    const isOnline = data.isOnline === true || data.is_online === true ||
                     data.online === true || data.status === 'online' ||
                     data.isActive === true || data.is_active === true
    const activeLeaseCount = data.activeLeaseCount || data.active_lease_count ||
                            data.leaseCount || data.lease_count || 0
    const isAudited = data.isAudited === true || data.is_audited === true ||
                      data.audited === true

    // Provider is "reliable" if:
    // 1. Marked as online by Console API
    // 2. Has at least 1 active lease (proves they're actually working)
    // 3. OR is audited (verified by Akash)
    const isReliable = isOnline && (activeLeaseCount > 0 || isAudited)

    return {
      isOnline,
      isReliable,
      activeLeaseCount,
      isAudited,
      uptime1d: data.uptime1d || data.uptime_1d,
      uptime7d: data.uptime7d || data.uptime_7d,
      uptime30d: data.uptime30d || data.uptime_30d,
      lastCheckDate: data.lastCheckDate || data.last_check_date
    }
  } catch (error) {
    // On any error (timeout, network, CORS), assume provider is available
    // since they submitted a bid - let user decide
    const message = error instanceof Error ? error.message : 'Unknown error'
    console.log(`Health check failed for ${providerAddress.slice(0, 12)}...: ${message}`)
    return {
      isOnline: true,  // Assume online since they submitted a bid
      isReliable: false,
      activeLeaseCount: 0,
      isAudited: false,
      error: message
    }
  }
}

/**
 * Check multiple providers in parallel and return only healthy ones
 */
export async function filterHealthyProviders<T extends { providerAddress: string }>(
  items: T[]
): Promise<T[]> {
  const healthChecks = await Promise.all(
    items.map(async (item) => {
      const health = await checkProviderHealth(item.providerAddress)
      return { item, isOnline: health.isOnline }
    })
  )

  return healthChecks
    .filter(({ isOnline }) => isOnline)
    .map(({ item }) => item)
}

// Console API types for deployment details
export interface ForwardedPort {
  host: string
  port: number
  externalPort: number
  proto: string
  name: string
  available?: number
}

export interface ConsoleDeploymentLease {
  owner: string
  dseq: string
  gseq: number
  oseq: number
  provider: {
    address: string
    hostUri: string
    organization?: string
    email?: string
    website?: string
  }
  status: string
  createdHeight: number
  cpuUnits: number
  gpuUnits: number
  memoryQuantity: number
  storageQuantity: number
  price: number
}

export interface ConsoleDeploymentDetails {
  owner: string
  dseq: string
  status: string
  createdHeight: number
  balance: number
  leases: ConsoleDeploymentLease[]
  forwardedPorts?: ForwardedPort[]
}

/**
 * Fetch deployment details from Akash Console API
 * This provides forwarded ports and more detailed lease info
 */
export async function fetchDeploymentDetails(
  owner: string,
  dseq: string
): Promise<ConsoleDeploymentDetails | null> {
  try {
    const response = await fetch(
      `${AKASH_CONSOLE_API}/v1/deployment/${owner}/${dseq}`
    )
    if (!response.ok) {
      console.error('Console API returned:', response.status)
      return null
    }
    const data = await response.json()
    return data
  } catch (error) {
    console.error('Failed to fetch deployment details from console API:', error)
    return null
  }
}

/**
 * Fetch lease status including forwarded ports from Akash Console API
 * The lease status endpoint returns actual service URLs
 */
export async function fetchLeaseStatus(
  owner: string,
  dseq: string,
  gseq: number,
  oseq: number
): Promise<{ services?: Record<string, { uris?: string[]; ports?: ForwardedPort[] }> } | null> {
  try {
    const response = await fetch(
      `${AKASH_CONSOLE_API}/v1/lease/${owner}/${dseq}/${gseq}/${oseq}/status`
    )
    if (!response.ok) {
      console.error('Lease status API returned:', response.status)
      return null
    }
    const data = await response.json()
    console.log('Lease status response:', data)
    return data
  } catch (error) {
    console.error('Failed to fetch lease status:', error)
    return null
  }
}

/**
 * Extract service URIs from lease status response
 */
export function extractServiceUrls(leaseStatus: { services?: Record<string, { uris?: string[]; ports?: ForwardedPort[] }> } | null): string[] {
  if (!leaseStatus?.services) return []

  const urls: string[] = []
  for (const [, service] of Object.entries(leaseStatus.services)) {
    if (service.uris) {
      urls.push(...service.uris)
    }
  }
  return urls
}

/**
 * Compute manifest hash using browser-native Web Crypto API
 * This replaces sdl.manifestVersion() which uses node:crypto (not available in browser)
 */
async function computeManifestHash(sdl: SDL): Promise<Uint8Array> {
  // Get the sorted JSON string (this method doesn't use crypto)
  const jsonStr = sdl.manifestSortedJSON()
  const encoder = new TextEncoder()
  const data = encoder.encode(jsonStr)

  // Use browser's native Web Crypto API
  const hashBuffer = await crypto.subtle.digest('SHA-256', data)
  return new Uint8Array(hashBuffer)
}

/**
 * Create an Akash deployment from SDL
 *
 * Uses v1beta4 message format which is the current Akash mainnet version.
 * The message structure differs from v1beta3:
 * - Uses `hash` instead of `version`
 * - Uses `deposit.sources` to specify funding sources (grant, balance)
 */
// Certificate type for on-chain certificate query
export interface OnChainCertificate {
  owner: string
  serial: string
  state: string
  cert: string // base64 encoded
  pubkey: string // base64 encoded
}

/**
 * Fetch certificate from blockchain (using v1 API)
 * Returns the first valid certificate found
 */
export async function fetchCertificateFromChain(owner: string): Promise<OnChainCertificate | null> {
  try {
    const response = await fetch(
      `${AKASH_REST}/akash/cert/v1/certificates/list?filter.owner=${owner}&filter.state=valid`
    )
    const data = await response.json()

    if (data.certificates && data.certificates.length > 0) {
      const cert = data.certificates[0].certificate
      console.log('On-chain certificate data:', cert)
      console.log('Total certificates on-chain:', data.certificates.length)

      // Decode the on-chain pubkey to see its format
      if (cert.cert?.pubkey) {
        try {
          const decodedPubkey = atob(cert.cert.pubkey)
          console.log('On-chain pubkey (decoded first 100 chars):', decodedPubkey.substring(0, 100))
          console.log('On-chain pubkey starts with BEGIN:', decodedPubkey.includes('BEGIN'))
        } catch (e) {
          console.log('Could not decode on-chain pubkey as base64')
        }
      }

      return {
        owner: cert.cert?.owner || owner,
        serial: cert.cert?.serial || '',
        state: cert.cert?.state || 'valid',
        cert: cert.cert?.cert || '',
        pubkey: cert.cert?.pubkey || ''
      }
    }
    return null
  } catch (error) {
    console.error('Failed to fetch certificate from chain:', error)
    return null
  }
}

/**
 * Check if a specific local certificate pubkey is registered on-chain
 * Searches through ALL certificates, not just the first one
 */
export async function isLocalCertRegistered(owner: string, localPubkeyPem: string): Promise<boolean> {
  try {
    const response = await fetch(
      `${AKASH_REST}/akash/cert/v1/certificates/list?filter.owner=${owner}&filter.state=valid`
    )
    const data = await response.json()

    if (!data.certificates || data.certificates.length === 0) {
      console.log('No certificates found on-chain')
      return false
    }

    console.log(`Checking ${data.certificates.length} on-chain certificates for match...`)

    // Normalize local pubkey for comparison (strip all whitespace and headers)
    const normalizedLocal = localPubkeyPem.replace(/\s/g, '')
      .replace(/-----BEGIN[^-]+-----/g, '')
      .replace(/-----END[^-]+-----/g, '')

    console.log('Local pubkey (normalized, first 50):', normalizedLocal.substring(0, 50))

    for (let i = 0; i < data.certificates.length; i++) {
      const cert = data.certificates[i].certificate
      // The certificate object has pubkey directly, not nested under .cert
      const pubkeyBase64 = cert.pubkey || cert.cert?.pubkey
      if (!pubkeyBase64) {
        console.log(`Certificate ${i + 1} has no pubkey, skipping`)
        continue
      }

      try {
        // Decode on-chain pubkey from base64
        const decodedPubkey = atob(pubkeyBase64)
        console.log(`Certificate ${i + 1} pubkey (first 60):`, decodedPubkey.substring(0, 60))

        // Normalize on-chain pubkey (strip whitespace and headers)
        const normalizedOnChain = decodedPubkey.replace(/\s/g, '')
          .replace(/-----BEGIN[^-]+-----/g, '')
          .replace(/-----END[^-]+-----/g, '')

        console.log(`Certificate ${i + 1} normalized (first 50):`, normalizedOnChain.substring(0, 50))

        if (normalizedOnChain === normalizedLocal) {
          console.log(`Match found! Certificate ${i + 1} matches local cert`)
          return true
        }
      } catch (e) {
        console.log(`Could not decode certificate ${i + 1} pubkey:`, e)
      }
    }

    console.log('No matching certificate found on-chain')
    return false
  } catch (error) {
    console.error('Failed to check certificates:', error)
    return false
  }
}

/**
 * Publish certificate to blockchain (using v1 API)
 *
 * Uses MsgCreateCertificate from chain-sdk which is registered in the signing client's registry
 */
export async function publishCertificateToChain(
  signingClient: SigningStargateClient,
  owner: string,
  certPem: string,
  pubkeyPem: string
): Promise<{ transactionHash: string }> {
  console.log('=== Publishing certificate to blockchain ===')
  console.log('Owner:', owner)
  console.log('Cert PEM length:', certPem.length)
  console.log('Cert PEM starts with:', certPem.substring(0, 60))
  console.log('Pubkey PEM length:', pubkeyPem.length)
  console.log('Pubkey PEM:', pubkeyPem)

  // Convert PEM strings to bytes
  const certBytes = new TextEncoder().encode(certPem)
  const pubkeyBytes = new TextEncoder().encode(pubkeyPem)

  console.log('Cert bytes length:', certBytes.length)
  console.log('Pubkey bytes length:', pubkeyBytes.length)

  // Create the message using the v1 type URL
  // The MsgCreateCertificate is registered in the registry via KeplrContext
  const msg = {
    typeUrl: '/akash.cert.v1.MsgCreateCertificate',
    value: {
      owner,
      cert: certBytes,
      pubkey: pubkeyBytes
    }
  }

  console.log('Broadcasting MsgCreateCertificate...')

  // Sign and broadcast using session if available
  const result = await signAndBroadcastWithSession(
    signingClient,
    owner,
    [msg],
    'auto',
    'Create certificate via Quip Protocol'
  )

  console.log('Broadcast result code:', result.code)
  console.log('Broadcast result rawLog:', result.rawLog)

  if (result.code !== 0) {
    throw new Error(`Failed to publish certificate: ${result.rawLog}`)
  }

  return { transactionHash: result.transactionHash }
}

/**
 * Revoke certificate on blockchain (using v1 API)
 *
 * This is needed when:
 * - Certificate exists on-chain but we don't have the private key (created elsewhere)
 * - We want to replace an existing certificate with a new one
 */
export async function revokeCertificateOnChain(
  signingClient: SigningStargateClient,
  owner: string,
  serial: string
): Promise<{ transactionHash: string }> {
  // MsgRevokeCertificate requires the certificate ID (owner + serial)
  const msg = {
    typeUrl: '/akash.cert.v1.MsgRevokeCertificate',
    value: {
      id: {
        owner,
        serial
      }
    }
  }

  // Sign and broadcast using session if available
  const result = await signAndBroadcastWithSession(
    signingClient,
    owner,
    [msg],
    'auto',
    'Revoke certificate via Quip Protocol'
  )

  if (result.code !== 0) {
    throw new Error(`Failed to revoke certificate: ${result.rawLog}`)
  }

  return { transactionHash: result.transactionHash }
}

/**
 * Calculate deposit amount for a deployment
 * @param sdlConfig - SDL configuration object
 * @param durationMinutes - Deployment duration in minutes
 * @returns Deposit amount in uAKT (with 4x safety buffer for initial deposit)
 */
function calculateDeposit(sdlConfig: object, durationMinutes: number): string {
  // Extract pricing from SDL
  const config = sdlConfig as Record<string, unknown>
  const profiles = config.profiles as Record<string, unknown> | undefined
  const placement = profiles?.placement as Record<string, unknown> | undefined
  const dcloud = placement?.dcloud as Record<string, unknown> | undefined
  const pricing = dcloud?.pricing as Record<string, { amount: number }> | undefined

  // Get the first service's pricing (should be only one)
  let pricePerBlock = 10000 // Default fallback
  if (pricing) {
    const firstService = Object.values(pricing)[0]
    if (firstService?.amount) {
      pricePerBlock = firstService.amount
    }
  }

  // Calculate: blocks needed * price per block * 1.5 (safety buffer)
  // Akash blocks are ~6 seconds, so ~10 blocks/minute
  const blocksPerMinute = 10
  const totalBlocks = durationMinutes * blocksPerMinute
  const depositUakt = pricePerBlock * totalBlocks * 1.5

  // Minimum 5 AKT, maximum 50 AKT
  const minDeposit = 5_000_000  // 5 AKT
  const maxDeposit = 50_000_000 // 50 AKT
  const finalDeposit = Math.max(minDeposit, Math.min(maxDeposit, depositUakt))

  console.log(`Deposit calculation: ${pricePerBlock} uakt/block × ${totalBlocks} blocks × 1.5 = ${depositUakt} uakt (clamped to ${finalDeposit})`)

  return finalDeposit.toString()
}

export async function createDeployment(
  signingClient: SigningStargateClient,
  owner: string,
  sdlConfig: object,
  durationMinutes: number = 90  // Default 90 minutes if not specified
): Promise<{ transactionHash: string; dseq: string; manifestJson: string }> {
  // Convert SDL object to YAML string for akashjs
  const yaml = await import('yaml')
  const sdlYaml = yaml.stringify(sdlConfig)

  // Parse SDL using akashjs (still uses beta3 internally for parsing)
  const sdl = SDL.fromString(sdlYaml, 'beta3')

  // Get deployment sequence number (current block height)
  const blockHeight = await signingClient.getHeight()
  const dseq = Long.fromNumber(blockHeight, true) // unsigned long

  // Use akashjs's manifestSortedJSON() directly - it's designed to match the provider's expectations:
  // 1. Uses manifest(true) for string values
  // 2. Replaces "quantity" with "size" for memory/storage
  // 3. Uses json-stable-stringify for deterministic key ordering
  // 4. Escapes HTML characters (<, >, &) via escapeHtml()
  //
  // Note: manifestSortedJSON includes null values. We previously tried stripping them
  // to match Go's omitempty, but that didn't work. Trying with nulls intact.
  const manifestJson = sdl.manifestSortedJSON()
  console.log('manifestSortedJSON:', manifestJson)
  console.log('Has null values:', manifestJson.includes(':null'))
  console.log('Manifest size:', manifestJson.length, 'bytes')

  // Compute hash from the canonical manifest JSON
  const encoder = new TextEncoder()
  const data = encoder.encode(manifestJson)
  const hashBuffer = await crypto.subtle.digest('SHA-256', data)
  const hash = new Uint8Array(hashBuffer)
  console.log('Computed manifest hash:', Array.from(hash.slice(0, 8)).map(b => b.toString(16).padStart(2, '0')).join(''))

  // Calculate deposit based on duration and pricing (with 2x safety buffer)
  const depositAmount = calculateDeposit(sdlConfig, durationMinutes)

  // Create v1beta4 deployment message
  // Note: v1beta4 uses 'hash' instead of 'version', and 'deposit' has 'sources'
  const groups = sdl.groups()
  const msg = MsgCreateDeployment.fromPartial({
    id: {
      owner,
      dseq
    },
    groups,
    hash,
    deposit: {
      amount: {
        denom: AKASH_DENOM,
        amount: depositAmount
      },
      // Use both grant and balance as funding sources
      sources: [Source.grant, Source.balance]
    }
  })

  // Get type URL from the message's $type property
  const typeUrl = `/${MsgCreateDeployment.$type}`

  // Sign and broadcast transaction using session if available
  const result = await signAndBroadcastWithSession(
    signingClient,
    owner,
    [{ typeUrl, value: msg }],
    'auto',
    'Created via Quip Protocol'
  )

  if (result.code !== 0) {
    throw new Error(`Transaction failed: ${result.rawLog}`)
  }

  return {
    transactionHash: result.transactionHash,
    dseq: dseq.toString(),
    manifestJson  // Return the exact manifest JSON used for hashing
  }
}

// ============================================================================
// Provider Capacity Types and Functions (for Fleet Deployment)
// ============================================================================

/**
 * Provider capacity information from Console API
 * Used for intelligent fleet allocation
 */
export interface ProviderCapacity {
  address: string
  name: string
  hostUri?: string
  // Resource stats in Console API units
  availableCpu: number      // millicores (1000 = 1 CPU)
  availableGpu: number      // GPU units
  availableMemory: number   // bytes
  availableStorage: number  // bytes (ephemeral)
  // Status flags
  isOnline: boolean
  isReliable: boolean
  isAudited: boolean
  activeLeaseCount: number
  // Additional info
  uptime7d?: number
  organization?: string
  region?: string
  // GPU details if available
  gpuModels?: string[]
}

/**
 * Raw provider response from Console API
 */
interface ConsoleProviderResponse {
  owner: string
  name?: string
  hostUri?: string
  isOnline?: boolean
  isActive?: boolean
  isAudited?: boolean
  activeStats?: {
    cpu: number
    memory: number
    storage: number
    gpu: number
  }
  pendingStats?: {
    cpu: number
    memory: number
    storage: number
    gpu: number
  }
  availableStats?: {
    cpu: number
    memory: number
    storage: number
    gpu: number
  }
  // Alternative field names
  stats?: {
    cpu?: { active?: number; available?: number; pending?: number }
    gpu?: { active?: number; available?: number; pending?: number }
    memory?: { active?: number; available?: number; pending?: number }
    storage?: { ephemeral?: { available?: number } }
  }
  leaseCount?: number
  activeLeaseCount?: number
  uptime7d?: number
  uptime_7d?: number
  attributes?: Array<{ key: string; value: string }>
  gpuModels?: string[]
}

/**
 * Fetch all providers with their capacity from Console API
 * Returns providers sorted by available capacity
 */
export async function fetchAllProviders(): Promise<ProviderCapacity[]> {
  try {
    const response = await fetch(
      `${AKASH_CONSOLE_API}/v1/providers`,
      { signal: AbortSignal.timeout(15000) }
    )

    if (!response.ok) {
      console.error('Failed to fetch providers:', response.status)
      return []
    }

    const data: ConsoleProviderResponse[] = await response.json()
    console.log(`Fetched ${data.length} providers from Console API`)

    // Parse and filter providers
    const providers: ProviderCapacity[] = data
      .map((p): ProviderCapacity | null => {
        // Extract available resources - handle both response formats
        let availableCpu = 0
        let availableGpu = 0
        let availableMemory = 0
        let availableStorage = 0

        if (p.availableStats) {
          availableCpu = p.availableStats.cpu || 0
          availableGpu = p.availableStats.gpu || 0
          availableMemory = p.availableStats.memory || 0
          availableStorage = p.availableStats.storage || 0
        } else if (p.stats) {
          availableCpu = p.stats.cpu?.available || 0
          availableGpu = p.stats.gpu?.available || 0
          availableMemory = p.stats.memory?.available || 0
          availableStorage = p.stats.storage?.ephemeral?.available || 0
        }

        const isOnline = p.isOnline === true || p.isActive === true
        const activeLeaseCount = p.activeLeaseCount || p.leaseCount || 0
        const isReliable = isOnline && (activeLeaseCount > 0 || p.isAudited === true)

        // Extract organization from attributes
        const organization = p.attributes?.find(a => a.key === 'organization')?.value
        const region = p.attributes?.find(a => a.key === 'region')?.value

        return {
          address: p.owner,
          name: p.name || organization || p.owner.slice(0, 12) + '...',
          hostUri: p.hostUri,
          availableCpu,
          availableGpu,
          availableMemory,
          availableStorage,
          isOnline,
          isReliable,
          isAudited: p.isAudited === true,
          activeLeaseCount,
          uptime7d: p.uptime7d || p.uptime_7d,
          organization,
          region,
          gpuModels: p.gpuModels
        }
      })
      .filter((p): p is ProviderCapacity => p !== null)

    // Sort by reliability then capacity
    providers.sort((a, b) => {
      // Reliable providers first
      if (a.isReliable !== b.isReliable) return a.isReliable ? -1 : 1
      // Then by CPU capacity
      return b.availableCpu - a.availableCpu
    })

    return providers
  } catch (error) {
    console.error('Failed to fetch providers:', error)
    return []
  }
}

/**
 * Filter providers by minimum resource requirements
 */
export function filterProvidersByCapacity(
  providers: ProviderCapacity[],
  options: {
    minCpu?: number        // minimum CPUs (not millicores)
    minGpu?: number        // minimum GPUs
    minMemoryGi?: number   // minimum memory in GiB
    onlyReliable?: boolean // only include reliable providers
    onlyOnline?: boolean   // only include online providers
  }
): ProviderCapacity[] {
  const {
    minCpu = 0,
    minGpu = 0,
    minMemoryGi = 0,
    onlyReliable = false,
    onlyOnline = true
  } = options

  const minCpuMillicores = minCpu * 1000
  const minMemoryBytes = minMemoryGi * 1024 * 1024 * 1024

  return providers.filter(p => {
    if (onlyOnline && !p.isOnline) return false
    if (onlyReliable && !p.isReliable) return false
    if (p.availableCpu < minCpuMillicores) return false
    if (p.availableGpu < minGpu) return false
    if (p.availableMemory < minMemoryBytes) return false
    return true
  })
}

/**
 * Get total available capacity across all providers
 */
export function getTotalCapacity(providers: ProviderCapacity[]): {
  totalCpu: number       // in full CPU units
  totalGpu: number
  totalMemoryGi: number
  providerCount: number
} {
  const totals = providers.reduce(
    (acc, p) => ({
      cpu: acc.cpu + p.availableCpu,
      gpu: acc.gpu + p.availableGpu,
      memory: acc.memory + p.availableMemory
    }),
    { cpu: 0, gpu: 0, memory: 0 }
  )

  return {
    totalCpu: Math.floor(totals.cpu / 1000),
    totalGpu: totals.gpu,
    totalMemoryGi: Math.floor(totals.memory / (1024 * 1024 * 1024)),
    providerCount: providers.length
  }
}

/**
 * Close a deployment
 * This releases all resources and stops billing
 * Uses session signing when available for automatic approval
 */
export async function closeDeployment(
  signingClient: SigningStargateClient,
  owner: string,
  dseq: string
): Promise<{ transactionHash: string }> {
  const msg = MsgCloseDeployment.fromPartial({
    id: {
      owner,
      dseq: Long.fromString(dseq, true)
    }
  })

  const typeUrl = `/${MsgCloseDeployment.$type}`

  // Use session signing if available
  const result = await signAndBroadcastWithSession(
    signingClient,
    owner,
    [{ typeUrl, value: msg }],
    'auto',
    'Closed via Quip Protocol'
  )

  if (result.code !== 0) {
    throw new Error(`Failed to close deployment: ${result.rawLog}`)
  }

  return { transactionHash: result.transactionHash }
}

/**
 * Fetch current escrow balance for a deployment
 */
async function fetchEscrowBalance(owner: string, dseq: string): Promise<number> {
  try {
    const response = await fetch(
      `${AKASH_REST}/akash/deployment/v1beta4/deployments/list?filters.owner=${owner}&filters.dseq=${dseq}`
    )
    const data = await response.json()

    if (data.deployments && data.deployments.length > 0) {
      const deployment = data.deployments[0]
      // Sum all deposit balances
      const deposits = deployment.escrow_account?.state?.deposits || []
      let totalBalance = 0
      for (const deposit of deposits) {
        if (deposit.balance?.denom === AKASH_DENOM) {
          totalBalance += parseInt(deposit.balance.amount, 10)
        }
      }
      return totalBalance
    }
    return 0
  } catch (error) {
    console.error('Failed to fetch escrow balance:', error)
    return 0
  }
}

/**
 * Top up a deployment's escrow based on actual bid price
 *
 * After accepting a bid, we know the real cost per block. This function
 * calculates if additional deposit is needed and adds it.
 *
 * @param signingClient - Signing client for transactions
 * @param owner - Deployment owner address
 * @param dseq - Deployment sequence number
 * @param bidPriceUakt - Actual bid price per block in uAKT
 * @param durationMinutes - Mining duration in minutes
 * @returns Transaction result or null if no top-up needed
 */
export async function topUpDeployment(
  signingClient: SigningStargateClient,
  owner: string,
  dseq: string,
  bidPriceUakt: number,
  durationMinutes: number
): Promise<{ transactionHash: string; depositAmount: string } | null> {
  // Calculate required escrow: bid price × blocks × 1.5 (safety buffer)
  const blocksPerMinute = 10
  const totalBlocks = durationMinutes * blocksPerMinute
  const requiredBalance = Math.ceil(bidPriceUakt * totalBlocks * 1.5)

  // Fetch current escrow balance
  const currentBalance = await fetchEscrowBalance(owner, dseq)

  console.log(`Escrow check: current=${currentBalance} uAKT, required=${requiredBalance} uAKT (${bidPriceUakt}/block × ${totalBlocks} blocks × 1.5)`)

  // If we have enough, no top-up needed
  if (currentBalance >= requiredBalance) {
    console.log(`Escrow sufficient, no top-up needed`)
    return null
  }

  // Calculate additional deposit needed
  const additionalNeeded = requiredBalance - currentBalance

  // Add a small buffer (10%) to avoid edge cases
  const depositAmount = Math.ceil(additionalNeeded * 1.1)

  console.log(`Topping up escrow: adding ${depositAmount} uAKT (${(depositAmount / 1_000_000).toFixed(4)} AKT)`)

  // Create deposit message using MsgAccountDeposit (escrow deposit for v1)
  // For deployment escrow: scope = 1 (deployment), xid = "owner/dseq"
  const msg = MsgAccountDeposit.fromPartial({
    signer: owner,
    id: {
      scope: 1,  // 1 = deployment
      xid: `${owner}/${dseq}`
    },
    deposit: {
      amount: {
        denom: AKASH_DENOM,
        amount: depositAmount.toString()
      },
      sources: [Source.balance]
    }
  })

  const typeUrl = `/${MsgAccountDeposit.$type}`

  // Sign and broadcast using session if available
  const result = await signAndBroadcastWithSession(
    signingClient,
    owner,
    [{ typeUrl, value: msg }],
    'auto',
    'Top up deposit via Quip Protocol'
  )

  if (result.code !== 0) {
    throw new Error(`Failed to top up deployment: ${result.rawLog}`)
  }

  return {
    transactionHash: result.transactionHash,
    depositAmount: depositAmount.toString()
  }
}

/**
 * Fetch provider hardware info string from Console API.
 *
 * Scans ALL provider attributes and concatenates their values
 * into a single string. This string is then matched against
 * CPU performance patterns in cpuPerformance.ts to identify
 * the provider's CPU type and score it.
 *
 * @param providerAddress - Provider's Akash address
 * @returns Concatenated hardware info string, or null on error
 */
export async function fetchProviderCpuInfo(providerAddress: string): Promise<string | null> {
  try {
    const response = await fetch(
      `${AKASH_CONSOLE_API}/v1/providers/${providerAddress}`,
      { signal: AbortSignal.timeout(5000) }
    )

    if (!response.ok) {
      return null
    }

    const data = await response.json()
    const attributes: Array<{ key: string; value: string }> = data.attributes || []

    if (attributes.length === 0) {
      return null
    }

    // Concatenate ALL attribute values into a single searchable string
    // Include both keys and values to catch patterns in either place
    // Example: "hardware-cpu: Intel Xeon E5-2660 | region: us-west | ..."
    const hardwareString = attributes
      .map(attr => `${attr.key}: ${attr.value}`)
      .join(' | ')

    // Return the full string - getCpuScore() will find CPU patterns within it
    return hardwareString || null
  } catch (error) {
    console.debug('Failed to fetch provider hardware info:', error)
    return null
  }
}

/**
 * Batch fetch CPU info for multiple providers.
 * More efficient than calling fetchProviderCpuInfo individually.
 *
 * @param providerAddresses - Array of provider addresses
 * @returns Map of provider address to CPU info string
 */
export async function fetchProvidersCpuInfo(
  providerAddresses: string[]
): Promise<Map<string, string | null>> {
  const results = new Map<string, string | null>()

  // Fetch in parallel with concurrency limit
  const batchSize = 5
  for (let i = 0; i < providerAddresses.length; i += batchSize) {
    const batch = providerAddresses.slice(i, i + batchSize)
    const promises = batch.map(async (addr) => {
      const cpuInfo = await fetchProviderCpuInfo(addr)
      results.set(addr, cpuInfo)
    })
    await Promise.all(promises)
  }

  return results
}
