/**
 * CPU Performance Table
 *
 * Maps CPU model patterns to single-thread performance scores
 * based on PassMark benchmarks. Used to prioritize bids from
 * providers with higher-performance CPUs.
 *
 * Sources:
 * - https://www.cpubenchmark.net/singleThread.html
 * - https://www.cpubenchmark.net/cpu_list.php
 */

export type CpuTier = 'top' | 'high' | 'mid' | 'low'

export interface CpuPattern {
  match: string[]
  score: number
  tier: CpuTier
}

export interface TierInfo {
  min: number
  description: string
}

/**
 * CPU performance patterns ordered from most specific to least specific.
 * First match wins, so put specific models before generic vendor names.
 *
 * Scores are PassMark single-thread scores (higher = faster single-thread).
 * Top consumer CPUs hit 4500-5000, server CPUs typically 1500-2500.
 */
export const CPU_PATTERNS: CpuPattern[] = [
  // Top tier - Modern high-performance (4GHz+ boost, excellent single-thread)
  { match: ['9950x3d', '9950x'], score: 4700, tier: 'top' },
  { match: ['9900x3d', '9900x', '9800x3d'], score: 4500, tier: 'top' },
  { match: ['threadripper 7970', 'threadripper 7980', 'threadripper 79'], score: 4200, tier: 'top' },
  { match: ['i9-13900', 'i9-14900', 'core ultra 9', 'ultra 9'], score: 4600, tier: 'top' },
  { match: ['epyc 4'], score: 4500, tier: 'top' }, // EPYC 4000 series (Zen 4)

  // High tier - Recent consumer/workstation (good single-thread)
  { match: ['ryzen 9 7900', 'ryzen 7 7700', 'ryzen 5 7600', '7900x', '7700x', '7600x'], score: 3800, tier: 'high' },
  { match: ['5950x', '5900x', '5900'], score: 3450, tier: 'high' },
  { match: ['5800x3d', '5800x'], score: 3500, tier: 'high' },
  { match: ['i9-12900', 'i7-12700', 'i9-11900'], score: 3200, tier: 'high' },
  { match: ['i9-9900k', 'i9-9900', '9900k'], score: 2900, tier: 'high' },

  // Mid tier - Older desktop or server-class
  { match: ['3900x', '3950x', '3900'], score: 2700, tier: 'mid' },
  { match: ['threadripper pro 3975', 'threadripper 3970', '3975wx', '3970x'], score: 2650, tier: 'mid' },
  { match: ['threadripper 5975', '5975wx'], score: 2800, tier: 'mid' },
  { match: ['epyc 9755', 'epyc 97', 'epyc 9'], score: 2800, tier: 'mid' }, // EPYC 9000 series
  { match: ['epyc 7402', 'epyc 7502', 'epyc 7002', '7402p', '7502p'], score: 2200, tier: 'mid' },
  { match: ['epyc 7302', 'epyc rome', 'epyc-rome', '7302p'], score: 2100, tier: 'mid' },
  { match: ['xeon gold 6', 'xeon gold', 'gold 6'], score: 2400, tier: 'mid' },
  { match: ['xeon cascadelake', 'cascadelake'], score: 2300, tier: 'mid' },
  { match: ['i7-8700', 'i7-9700', '8700k', '9700k'], score: 2500, tier: 'mid' },
  { match: ['apple m2', 'm2 pro', 'm2 max'], score: 2600, tier: 'mid' },
  { match: ['apple m1', 'm1 pro', 'm1 max'], score: 2400, tier: 'mid' },

  // Low tier - Legacy or slower server CPUs
  { match: ['xeon e5-2699', 'e5-2699'], score: 2000, tier: 'low' },
  { match: ['xeon e5-2697 v4', 'e5-2697v4', 'e5-2697 v4'], score: 1900, tier: 'low' },
  { match: ['xeon e5-2697 v2', 'e5-2697v2', 'e5-2697 v2'], score: 1700, tier: 'low' },
  { match: ['xeon e5-2680', 'e5-2680'], score: 1800, tier: 'low' },
  { match: ['xeon e5-2670', 'e5-2670'], score: 1700, tier: 'low' },
  { match: ['xeon e5-2660 v4', 'e5-2660v4', 'e5-2660 v4'], score: 1650, tier: 'low' },  // 2.0GHz Broadwell
  { match: ['xeon e5-2660 v3', 'e5-2660v3', 'e5-2660 v3'], score: 1600, tier: 'low' },  // 2.6GHz Haswell
  { match: ['xeon e5-2660', 'e5-2660'], score: 1550, tier: 'low' },  // Generic E5-2660
  { match: ['xeon e5-2630', 'e5-2630'], score: 1600, tier: 'low' },
  { match: ['xeon e7', 'e7-4870'], score: 1500, tier: 'low' },
  { match: ['i5-', 'core i5'], score: 2000, tier: 'low' },

  // Generic fallbacks (least specific, must be last)
  { match: ['xeon'], score: 1800, tier: 'low' },
  { match: ['epyc'], score: 2000, tier: 'mid' },
  { match: ['ryzen 9'], score: 2800, tier: 'mid' },
  { match: ['ryzen 7'], score: 2500, tier: 'mid' },
  { match: ['ryzen 5'], score: 2200, tier: 'mid' },
  { match: ['ryzen'], score: 2300, tier: 'mid' },
  { match: ['threadripper'], score: 2500, tier: 'mid' },
  { match: ['intel'], score: 1800, tier: 'low' },
  { match: ['amd'], score: 2000, tier: 'mid' },
]

/**
 * Default score for unknown CPUs
 */
export const DEFAULT_CPU_SCORE = 1500

/**
 * Tier thresholds and descriptions
 */
export const CPU_TIERS: Record<CpuTier, TierInfo> = {
  top: { min: 4000, description: 'Modern high-perf (4GHz+ boost)' },
  high: { min: 3000, description: 'Recent consumer/workstation' },
  mid: { min: 2000, description: 'Older desktop or server-class' },
  low: { min: 0, description: 'Legacy or unknown' },
}

/**
 * CPU performance levels for dropdown selection.
 * Ordered from lowest to highest minimum requirement.
 * Estimated clock speeds are typical boost clocks for CPUs in each score range.
 */
export interface CpuPerformanceLevel {
  value: string  // Score threshold as string for flexibility
  label: string
  description: string
  minScore: number
  clockRange: string  // Estimated boost clock range
  examples: string
}

export const CPU_PERFORMANCE_LEVELS: CpuPerformanceLevel[] = [
  {
    value: 'any',
    label: 'Any CPU',
    description: 'No minimum requirement',
    minScore: 0,
    clockRange: '',
    examples: 'Accept any provider'
  },
  {
    value: '1500',
    label: 'Basic',
    description: 'Legacy server CPUs',
    minScore: 1500,
    clockRange: '~2.5 GHz',
    examples: 'Xeon E5-2670, older EPYC'
  },
  {
    value: '2000',
    label: 'Entry',
    description: 'Older server-class',
    minScore: 2000,
    clockRange: '~3.0 GHz',
    examples: 'EPYC 7002, Xeon E5-2699'
  },
  {
    value: '2500',
    label: 'Standard',
    description: 'Modern server / older desktop',
    minScore: 2500,
    clockRange: '~3.5 GHz',
    examples: 'EPYC 9000, Xeon Gold, Ryzen 3000'
  },
  {
    value: '3000',
    label: 'Good',
    description: 'Recent workstation-class',
    minScore: 3000,
    clockRange: '~4.0 GHz',
    examples: 'Ryzen 5000, i9-9900K'
  },
  {
    value: '3500',
    label: 'High',
    description: 'High-performance desktop',
    minScore: 3500,
    clockRange: '~4.5 GHz',
    examples: 'Ryzen 5800X3D, i9-12900'
  },
  {
    value: '4000',
    label: 'Very High',
    description: 'Latest gen high-end',
    minScore: 4000,
    clockRange: '~5.0 GHz',
    examples: 'Ryzen 7000, Threadripper 7000'
  },
  {
    value: '4500',
    label: 'Top Tier',
    description: 'Flagship consumer CPUs',
    minScore: 4500,
    clockRange: '~5.5 GHz',
    examples: 'Ryzen 9950X, i9-14900K'
  },
]

/**
 * Maximum possible score (for normalization)
 */
export const MAX_CPU_SCORE = 5000

/**
 * Get CPU performance score from a model string.
 * Uses fuzzy pattern matching - first match wins.
 *
 * @param cpuString - CPU model string from provider attributes
 * @returns Raw PassMark-like score (1500-5000 range)
 */
export function getCpuScore(cpuString: string | null | undefined): number {
  if (!cpuString) return DEFAULT_CPU_SCORE

  // Normalize: lowercase, remove special chars except spaces and dashes
  const normalized = cpuString
    .toLowerCase()
    .replace(/[^a-z0-9\s\-]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()

  // Check patterns in order (most specific first)
  for (const pattern of CPU_PATTERNS) {
    for (const match of pattern.match) {
      if (normalized.includes(match)) {
        return pattern.score
      }
    }
  }

  return DEFAULT_CPU_SCORE
}

/**
 * Get CPU performance score normalized to 0-100 range.
 * Suitable for integration with bid scoring system.
 *
 * @param cpuString - CPU model string from provider attributes
 * @returns Normalized score 0-100
 */
export function getCpuScoreNormalized(cpuString: string | null | undefined): number {
  const raw = getCpuScore(cpuString)
  // Scale from 1000-5000 range to 0-100
  // Score of 1000 = 0, Score of 5000 = 100
  const normalized = ((raw - 1000) / 4000) * 100
  return Math.max(0, Math.min(100, normalized))
}

/**
 * Get the performance tier for a CPU score.
 *
 * @param score - Raw CPU score
 * @returns Tier classification
 */
export function getCpuTier(score: number): CpuTier {
  if (score >= CPU_TIERS.top.min) return 'top'
  if (score >= CPU_TIERS.high.min) return 'high'
  if (score >= CPU_TIERS.mid.min) return 'mid'
  return 'low'
}

/**
 * Get tier description for display.
 *
 * @param tier - CPU tier
 * @returns Human-readable description
 */
export function getTierDescription(tier: CpuTier): string {
  return CPU_TIERS[tier].description
}

/**
 * Format CPU info for display with tier badge.
 *
 * @param cpuString - CPU model string
 * @returns Formatted string with tier indicator
 */
export function formatCpuInfo(cpuString: string | null | undefined): string {
  if (!cpuString) return 'Unknown CPU'

  const score = getCpuScore(cpuString)
  const tier = getCpuTier(score)
  const tierEmoji = tier === 'top' ? '🔥' : tier === 'high' ? '⚡' : tier === 'mid' ? '✓' : '📦'

  return `${cpuString} ${tierEmoji}`
}
