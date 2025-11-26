import { AKASH_REST, AKASH_DENOM } from '../config/constants'
import type { SigningStargateClient } from '@cosmjs/stargate'
import { MsgCreateDeployment } from '@akashnetwork/akash-api/akash/deployment/v1beta3'
import { SDL } from '@akashnetwork/akashjs/build/sdl'

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
 * Fetch deployments for an address
 */
export async function fetchDeployments(address: string): Promise<Deployment[]> {
  try {
    const response = await fetch(
      `${AKASH_REST}/akash/deployment/v1beta3/deployments/list?filters.owner=${address}`
    )
    const data = await response.json()
    return data.deployments || []
  } catch (error) {
    console.error('Failed to fetch deployments:', error)
    return []
  }
}

/**
 * Fetch lease info for a deployment
 */
export async function fetchLeases(
  owner: string,
  dseq: string
): Promise<Lease[]> {
  try {
    const response = await fetch(
      `${AKASH_REST}/akash/market/v1beta4/leases/list?filters.owner=${owner}&filters.dseq=${dseq}`
    )
    const data = await response.json()
    return data.leases || []
  } catch (error) {
    console.error('Failed to fetch leases:', error)
    return []
  }
}

// Types
export interface Deployment {
  deployment: {
    deployment_id: {
      owner: string
      dseq: string
    }
    state: string
    version: string
    created_at: string
  }
  groups: Array<{
    group_id: {
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
    owner: string
    state: string
    balance: {
      denom: string
      amount: string
    }
    transferred: {
      denom: string
      amount: string
    }
    settled_at: string
    depositor: string
    funds: {
      denom: string
      amount: string
    }
  }
}

export interface Lease {
  lease: {
    lease_id: {
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
    closed_on: string
  }
  escrow_payment: {
    account_id: {
      scope: string
      xid: string
    }
    payment_id: string
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
    withdrawn: {
      denom: string
      amount: string
    }
  }
}

/**
 * Create an Akash deployment from SDL
 */
export async function createDeployment(
  signingClient: SigningStargateClient,
  owner: string,
  sdlConfig: object
): Promise<{ transactionHash: string; dseq: string }> {
  // Convert SDL object to YAML string for akashjs
  const yaml = await import('yaml')
  const sdlYaml = yaml.stringify(sdlConfig)

  // Parse SDL using akashjs
  const sdl = SDL.fromString(sdlYaml, 'beta3')

  // Get deployment sequence number (current block height)
  const blockHeight = await signingClient.getHeight()
  const dseq = BigInt(blockHeight)

  // Create deployment message
  const groups = sdl.groups()
  const msg = MsgCreateDeployment.fromPartial({
    id: {
      owner,
      dseq
    },
    groups,
    version: await sdl.manifestVersion(),
    deposit: {
      denom: AKASH_DENOM,
      amount: '5000000' // 5 AKT deposit
    },
    depositor: owner
  })

  // Sign and broadcast transaction
  const result = await signingClient.signAndBroadcast(
    owner,
    [
      {
        typeUrl: '/akash.deployment.v1beta3.MsgCreateDeployment',
        value: msg
      }
    ],
    'auto',
    'Created via Quip Protocol'
  )

  if (result.code !== 0) {
    throw new Error(`Transaction failed: ${result.rawLog}`)
  }

  return {
    transactionHash: result.transactionHash,
    dseq: dseq.toString()
  }
}
