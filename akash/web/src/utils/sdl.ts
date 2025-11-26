import { DOCKER_IMAGES } from '../config/constants'

export interface SDLConfig {
  minerType: 'cpu' | 'cuda'
  fleetSize: number
  miningDuration: string
  difficultyEnergy: number
  minDiversity: number
  minSolutions: number
}

/**
 * Generate SDL configuration for Akash deployment
 */
export function generateSDL(config: SDLConfig): object {
  const isCuda = config.minerType === 'cuda'
  const image = isCuda ? DOCKER_IMAGES.cuda : DOCKER_IMAGES.cpu

  const resources = isCuda
    ? {
        cpu: { units: 2 },
        memory: { size: '4Gi' },
        storage: [{ size: '10Gi' }],
        gpu: { units: 1, attributes: { vendor: { nvidia: {} } } }
      }
    : {
        cpu: { units: 1 },
        memory: { size: '2Gi' },
        storage: [{ size: '5Gi' }]
      }

  return {
    version: '2.0',
    services: {
      miner: {
        image,
        expose: [
          {
            port: 8080,
            as: 80,
            to: [{ global: true }]
          }
        ],
        env: [
          `MINER_TYPE=${config.minerType}`,
          `MINING_DURATION=${config.miningDuration}`,
          `DIFFICULTY_ENERGY=${config.difficultyEnergy}`,
          `MIN_DIVERSITY=${config.minDiversity}`,
          `MIN_SOLUTIONS=${config.minSolutions}`
        ]
      }
    },
    profiles: {
      compute: {
        miner: {
          resources
        }
      },
      placement: {
        akash: {
          pricing: {
            miner: {
              denom: 'uakt',
              amount: isCuda ? 10000 : 1000
            }
          }
        }
      }
    },
    deployment: {
      miner: {
        akash: {
          profile: 'miner',
          count: config.fleetSize
        }
      }
    }
  }
}

/**
 * Estimate deployment cost in AKT (micro AKT per block converted to AKT)
 * Based on pricing in SDL: cuda = 10000 uakt/block, cpu = 1000 uakt/block
 * Akash blocks are ~6 seconds, so ~10 blocks/minute, ~600 blocks/hour
 */
export function estimateCostAKT(
  minerType: 'cpu' | 'cuda',
  fleetSize: number,
  durationMinutes: number
): number {
  const uaktPerBlock = minerType === 'cuda' ? 10000 : 1000
  const blocksPerMinute = 10
  const totalBlocks = durationMinutes * blocksPerMinute
  const totalUakt = uaktPerBlock * fleetSize * totalBlocks
  // Convert uakt to AKT (1 AKT = 1,000,000 uakt)
  return totalUakt / 1_000_000
}
