/**
 * Bid Selection Utilities
 *
 * Shared logic for selecting the best bid from available providers.
 * Used by both single deployments (with auto-deploy) and fleet deployments.
 */

import type { Bid } from './akashApi'

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
}

/**
 * Default bid selection options
 */
export const DEFAULT_BID_OPTIONS: BidSelectionOptions = {
  preferReliable: true,
  maxPriceMultiplier: 1.2, // Accept up to 20% more for a reliable provider
  excludeProviders: []
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
  reason: string
}

/**
 * Parse bid price to number
 */
function getBidPrice(bid: BidWithProvider): number {
  return parseFloat(bid.bid.bid.price.amount)
}

/**
 * Select the best bid based on price and reliability
 *
 * Algorithm:
 * 1. Filter out excluded providers
 * 2. If preferReliable, filter to reliable providers (if any exist)
 * 3. Sort by price (lowest first)
 * 4. Within price threshold, prefer reliable/audited providers
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

  // 3. Sort by price (lowest first)
  candidates.sort((a, b) => getBidPrice(a) - getBidPrice(b))

  // 4. Get lowest price and calculate threshold
  const lowestPrice = getBidPrice(candidates[0])
  const threshold = lowestPrice * opts.maxPriceMultiplier

  // Filter to bids within price threshold
  const withinThreshold = candidates.filter(b => getBidPrice(b) <= threshold)

  // Within threshold, prefer: audited > reliable > online > cheapest
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

    // Combined score (weighted)
    const score = opts.preferReliable
      ? priceScore * 0.6 + reliabilityScore * 0.4
      : priceScore * 0.9 + reliabilityScore * 0.1

    // Generate reason string
    const reasons: string[] = []
    if (price === minPrice) reasons.push('Lowest price')
    if (bid.isAudited) reasons.push('Audited')
    if (bid.isReliable) reasons.push('Verified reliable')
    if (bid.activeLeaseCount > 10) reasons.push(`${bid.activeLeaseCount} active leases`)
    const reason = reasons.length > 0 ? reasons.join(', ') : 'Available'

    return {
      bid,
      score,
      priceScore,
      reliabilityScore,
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
