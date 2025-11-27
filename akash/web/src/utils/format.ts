import { AKASH_DECIMALS } from '../config/constants'

/**
 * Format a uAKT amount to AKT with proper decimals
 */
export function formatAKT(uakt: string | number): string {
  const amount = typeof uakt === 'string' ? parseInt(uakt, 10) : uakt
  return (amount / Math.pow(10, AKASH_DECIMALS)).toFixed(6) + ' AKT'
}

/**
 * Truncate an address for display
 */
export function truncateAddress(address: string | undefined): string {
  if (!address) return ''
  if (address.length <= 20) return address
  return `${address.slice(0, 10)}...${address.slice(-8)}`
}

/**
 * Format a date for display
 * Note: Akash API returns block heights for created_at, not timestamps
 */
export function formatDate(date: Date | string | number): string {
  // If it's a number or numeric string (block height), display as block #
  if (typeof date === 'number' || (typeof date === 'string' && /^\d+$/.test(date))) {
    return `Block #${date}`
  }
  const d = typeof date === 'string' ? new Date(date) : date
  if (isNaN(d.getTime())) {
    return `Block #${date}`
  }
  return d.toLocaleString()
}

/**
 * Calculate uptime from block heights
 * Akash blocks are approximately 6 seconds apart
 */
export function calculateUptime(createdAtBlock: string | number, currentBlock: number): string {
  const created = typeof createdAtBlock === 'string' ? parseInt(createdAtBlock, 10) : createdAtBlock
  if (isNaN(created) || created <= 0) return 'Unknown'

  const blockDiff = currentBlock - created
  if (blockDiff < 0) return 'Not started'

  // ~6 seconds per block
  const totalSeconds = blockDiff * 6
  const days = Math.floor(totalSeconds / 86400)
  const hours = Math.floor((totalSeconds % 86400) / 3600)
  const minutes = Math.floor((totalSeconds % 3600) / 60)

  if (days > 0) {
    return `${days}d ${hours}h ${minutes}m`
  } else if (hours > 0) {
    return `${hours}h ${minutes}m`
  } else {
    return `${minutes}m`
  }
}

/**
 * Parse duration string (e.g., "90m", "2h") to minutes
 */
export function parseDurationToMinutes(duration: string): number {
  const match = duration.match(/^(\d+)(m|h|d)$/)
  if (!match) return 90 // default

  const value = parseInt(match[1], 10)
  const unit = match[2]

  switch (unit) {
    case 'm':
      return value
    case 'h':
      return value * 60
    case 'd':
      return value * 60 * 24
    default:
      return 90
  }
}
