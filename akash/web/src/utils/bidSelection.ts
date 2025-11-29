/**
 * Bid Selection Utilities
 *
 * Shared logic for selecting the best bid from available providers.
 * Used by both single deployments (with auto-deploy) and fleet deployments.
 */

import type { Bid } from './akashApi'
import { getCpuScoreNormalized, getCpuTier, type CpuTier } from '../config/cpuPerformance'

/**
 * Bid with enriched provider information
 */
export interface BidWithProvider {
  bid: Bid
  providerName: string
  providerOrg?: string
  isOnline: boolean
  isReliable: boolean
  isAudited: boolean
  activeLeaseCount: number
  uptime7d?: number
  /** CPU model string from provider attributes */
  cpuInfo?: string | null
  /** Normalized CPU performance score (0-100) */
  cpuScore?: number
  /** CPU performance tier */
  cpuTier?: CpuTier
}

/**
 * Options for bid selection algorithm
 */
export interface BidSelectionOptions {
  /** Prefer reliable providers (with active leases) over cheapest */
  preferReliable: boolean
  /** Maximum price multiplier vs lowest bid (e.g., 1.2 = accept up to 20% more for reliability) */
  maxPriceMultiplier: number
  /** Provider addresses to exclude from selection */
  excludeProviders: string[]
  /** Prefer high-performance CPUs (enables CPU scoring in bid selection) */
  preferHighPerfCpu: boolean
  /** Weight for CPU score in ranking (0-1, default 0.3 when enabled) */
  cpuScoreWeight: number
}

/**
 * Default bid selection options
 */
export const DEFAULT_BID_OPTIONS: BidSelectionOptions = {
  preferReliable: true,
  maxPriceMultiplier: 1.2, // Accept up to 20% more for a reliable provider
  excludeProviders: [],
  preferHighPerfCpu: false, // Disabled by default (price-first behavior)
  cpuScoreWeight: 0.3 // When enabled, 30% weight for CPU score
}

/**
 * Ranked bid with score breakdown
 */
export interface RankedBid {
  bid: BidWithProvider
  rank: number
  score: number
  priceScore: number
  reliabilityScore: number
  cpuScore: number
  reason: string
}

/**
 * Parse bid price to number
 */
function getBidPrice(bid: BidWithProvider): number {
  return parseFloat(bid.bid.bid.price.amount)
}

/**
 * Select the best bid based on price, reliability, and optionally CPU performance
 *
 * Algorithm:
 * 1. Filter out excluded providers
 * 2. If preferReliable, filter to reliable providers (if any exist)
 * 3. If preferHighPerfCpu, sort by CPU score first, then price
 * 4. Otherwise, sort by price (lowest first)
 * 5. Within price threshold, prefer high-CPU/audited/reliable providers
 *
 * @param bids - Available bids with provider info
 * @param options - Selection options
 * @returns Best bid or null if none available
 */
export function selectBestBid(
  bids: BidWithProvider[],
  options: Partial<BidSelectionOptions> = {}
): BidWithProvider | null {
  const opts = { ...DEFAULT_BID_OPTIONS, ...options }

  if (bids.length === 0) return null

  // 1. Filter out excluded providers
  let candidates = bids.filter(
    b => !opts.excludeProviders.includes(b.bid.bid.id.provider)
  )

  if (candidates.length === 0) return null

  // 2. If preferReliable, try to filter to reliable providers
  if (opts.preferReliable) {
    const reliable = candidates.filter(b => b.isReliable)
    if (reliable.length > 0) {
      candidates = reliable
    }
    // If no reliable providers, fall through to use all candidates
  }

  // 3. Get lowest price for threshold calculation
  candidates.sort((a, b) => getBidPrice(a) - getBidPrice(b))
  const lowestPrice = getBidPrice(candidates[0])
  const threshold = lowestPrice * opts.maxPriceMultiplier

  // Filter to bids within price threshold
  const withinThreshold = candidates.filter(b => getBidPrice(b) <= threshold)

  // 4. If preferHighPerfCpu, prioritize by CPU score within threshold
  if (opts.preferHighPerfCpu) {
    // Ensure CPU scores are calculated for all bids
    for (const bid of withinThreshold) {
      if (bid.cpuScore === undefined && bid.cpuInfo) {
        bid.cpuScore = getCpuScoreNormalized(bid.cpuInfo)
        bid.cpuTier = getCpuTier(bid.cpuScore * 40 + 1000) // Convert back to raw score
      }
    }

    // Sort by CPU score (highest first), then by price (lowest first)
    withinThreshold.sort((a, b) => {
      const cpuDiff = (b.cpuScore ?? 0) - (a.cpuScore ?? 0)
      if (Math.abs(cpuDiff) > 10) return cpuDiff // Significant CPU difference
      return getBidPrice(a) - getBidPrice(b) // Similar CPU, prefer cheaper
    })

    // Return best CPU within threshold
    if (withinThreshold.length > 0) {
      return withinThreshold[0]
    }
  }

  // 5. Standard selection: prefer audited > reliable > online > cheapest
  const audited = withinThreshold.find(b => b.isAudited)
  if (audited) return audited

  const reliable = withinThreshold.find(b => b.isReliable)
  if (reliable) return reliable

  const online = withinThreshold.find(b => b.isOnline)
  if (online) return online

  // Return cheapest
  return withinThreshold[0]
}

/**
 * Rank all bids with detailed scoring
 *
 * Useful for displaying bid options with explanations
 *
 * @param bids - Available bids with provider info
 * @param options - Selection options
 * @returns Ranked bids with scores
 */
export function rankBids(
  bids: BidWithProvider[],
  options: Partial<BidSelectionOptions> = {}
): RankedBid[] {
  const opts = { ...DEFAULT_BID_OPTIONS, ...options }

  if (bids.length === 0) return []

  // Filter out excluded providers
  const candidates = bids.filter(
    b => !opts.excludeProviders.includes(b.bid.bid.id.provider)
  )

  if (candidates.length === 0) return []

  // Find price range for normalization
  const prices = candidates.map(getBidPrice)
  const minPrice = Math.min(...prices)
  const maxPrice = Math.max(...prices)
  const priceRange = maxPrice - minPrice || 1 // Avoid division by zero

  // Score each bid
  const scored = candidates.map(bid => {
    const price = getBidPrice(bid)

    // Price score: 100 for lowest, 0 for highest
    const priceScore = 100 * (1 - (price - minPrice) / priceRange)

    // Reliability score: 0-100 based on flags
    let reliabilityScore = 0
    if (bid.isOnline) reliabilityScore += 25
    if (bid.isReliable) reliabilityScore += 35
    if (bid.isAudited) reliabilityScore += 25
    if (bid.activeLeaseCount > 0) {
      reliabilityScore += Math.min(15, bid.activeLeaseCount) // Up to 15 points for active leases
    }

    // CPU score: 0-100 based on CPU performance
    // Calculate if not already set
    let cpuScore = bid.cpuScore ?? 0
    if (cpuScore === 0 && bid.cpuInfo) {
      cpuScore = getCpuScoreNormalized(bid.cpuInfo)
      // Update bid with calculated score
      bid.cpuScore = cpuScore
      bid.cpuTier = getCpuTier(cpuScore * 40 + 1000)
    }

    // Combined score (weighted based on preferences)
    let score: number
    if (opts.preferHighPerfCpu) {
      // CPU-focused: price 40%, reliability 30%, CPU 30%
      const cpuWeight = opts.cpuScoreWeight
      const reliabilityWeight = opts.preferReliable ? 0.3 : 0.1
      const priceWeight = 1 - cpuWeight - reliabilityWeight
      score = priceScore * priceWeight + reliabilityScore * reliabilityWeight + cpuScore * cpuWeight
    } else if (opts.preferReliable) {
      // Reliability-focused: price 60%, reliability 40%
      score = priceScore * 0.6 + reliabilityScore * 0.4
    } else {
      // Price-focused: price 90%, reliability 10%
      score = priceScore * 0.9 + reliabilityScore * 0.1
    }

    // Generate reason string
    const reasons: string[] = []
    if (price === minPrice) reasons.push('Lowest price')
    if (bid.isAudited) reasons.push('Audited')
    if (bid.isReliable) reasons.push('Verified reliable')
    if (bid.activeLeaseCount > 10) reasons.push(`${bid.activeLeaseCount} active leases`)
    if (bid.cpuTier === 'top') reasons.push('Top-tier CPU')
    else if (bid.cpuTier === 'high') reasons.push('High-perf CPU')
    if (bid.cpuInfo && opts.preferHighPerfCpu) reasons.push(bid.cpuInfo)
    const reason = reasons.length > 0 ? reasons.join(', ') : 'Available'

    return {
      bid,
      score,
      priceScore,
      reliabilityScore,
      cpuScore,
      reason,
      rank: 0 // Will be set after sorting
    }
  })

  // Sort by score (highest first)
  scored.sort((a, b) => b.score - a.score)

  // Assign ranks
  scored.forEach((item, index) => {
    item.rank = index + 1
  })

  return scored
}

/**
 * Format bid price for display
 */
export function formatBidPrice(bid: BidWithProvider): string {
  const price = getBidPrice(bid)
  return `${price.toLocaleString()} uakt/block`
}

/**
 * Estimate monthly cost from bid price
 * Assumes ~6 second blocks on Akash
 */
export function estimateMonthlyCost(bid: BidWithProvider): number {
  const pricePerBlock = getBidPrice(bid)
  const blocksPerMonth = (30 * 24 * 60 * 60) / 6 // ~432,000 blocks/month
  const uaktPerMonth = pricePerBlock * blocksPerMonth
  const aktPerMonth = uaktPerMonth / 1_000_000
  return aktPerMonth
}

/**
 * Get provider display name from bid
 */
export function getProviderDisplayName(bid: BidWithProvider): string {
  if (bid.providerOrg) return bid.providerOrg
  if (bid.providerName) return bid.providerName
  return bid.bid.bid.id.provider.slice(0, 12) + '...'
}

/**
 * Enrich bids with CPU information from provider attributes.
 *
 * Fetches CPU info for each provider and calculates performance scores.
 * Call this before using selectBestBid with preferHighPerfCpu enabled.
 *
 * @param bids - Bids to enrich
 * @param fetchCpuInfo - Function to fetch CPU info for a provider
 * @returns Bids with cpuInfo and cpuScore populated
 */
export async function enrichBidsWithCpuInfo(
  bids: BidWithProvider[],
  fetchCpuInfo: (providerAddress: string) => Promise<string | null>
): Promise<BidWithProvider[]> {
  // Fetch CPU info in parallel with concurrency limit
  const batchSize = 5
  for (let i = 0; i < bids.length; i += batchSize) {
    const batch = bids.slice(i, i + batchSize)
    await Promise.all(
      batch.map(async (bid) => {
        if (bid.cpuInfo === undefined) {
          const cpuInfo = await fetchCpuInfo(bid.bid.bid.id.provider)
          bid.cpuInfo = cpuInfo
          if (cpuInfo) {
            bid.cpuScore = getCpuScoreNormalized(cpuInfo)
            bid.cpuTier = getCpuTier(bid.cpuScore * 40 + 1000)
          } else {
            bid.cpuScore = getCpuScoreNormalized(null)
            bid.cpuTier = 'low'
          }
        }
      })
    )
  }

  return bids
}

/**
 * Get CPU tier emoji for display
 */
export function getCpuTierEmoji(tier: CpuTier | undefined): string {
  switch (tier) {
    case 'top': return '🔥'
    case 'high': return '⚡'
    case 'mid': return '✓'
    case 'low': return '📦'
    default: return ''
  }
}

/**
 * Format CPU info for display with tier badge
 */
export function formatCpuInfoWithTier(bid: BidWithProvider): string {
  if (!bid.cpuInfo) return 'Unknown CPU'
  const emoji = getCpuTierEmoji(bid.cpuTier)
  return `${bid.cpuInfo} ${emoji}`
}
