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
 */
export function formatDate(date: Date | string): string {
  const d = typeof date === 'string' ? new Date(date) : date
  return d.toLocaleString()
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
