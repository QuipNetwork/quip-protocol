/**
 * Authz Utilities
 *
 * Query and manage Cosmos SDK authz grants for session signing.
 * Allows the web app to execute transactions on behalf of the main wallet
 * after receiving a one-time grant approval.
 */

import { AKASH_REST } from '../config/constants'

/**
 * Message types that we need authorization for
 */
export const REQUIRED_MSG_TYPES = [
  '/akash.deployment.v1beta4.MsgCreateDeployment',
  '/akash.deployment.v1beta4.MsgCloseDeployment',
  '/akash.deployment.v1beta4.MsgUpdateDeployment',
  '/akash.market.v1beta5.MsgCreateLease',
  '/akash.cert.v1.MsgCreateCertificate',
  '/akash.cert.v1.MsgRevokeCertificate'
] as const

export type RequiredMsgType = typeof REQUIRED_MSG_TYPES[number]

/**
 * Grant from the blockchain
 */
export interface Grant {
  granter: string
  grantee: string
  authorization: {
    '@type': string
    msg?: string // For GenericAuthorization
  }
  expiration: string // ISO timestamp
}

/**
 * Grants response from REST API
 */
interface GrantsResponse {
  grants: Grant[]
  pagination: {
    next_key: string | null
    total: string
  }
}

/**
 * Fetch all grants from granter to grantee
 */
export async function fetchGrants(
  granter: string,
  grantee: string
): Promise<Grant[]> {
  try {
    const url = `${AKASH_REST}/cosmos/authz/v1beta1/grants?granter=${granter}&grantee=${grantee}`
    const response = await fetch(url, { signal: AbortSignal.timeout(10000) })

    if (!response.ok) {
      if (response.status === 404) {
        return []
      }
      throw new Error(`Failed to fetch grants: ${response.status}`)
    }

    const data: GrantsResponse = await response.json()
    return data.grants || []
  } catch (error) {
    console.error('Error fetching grants:', error)
    return []
  }
}

/**
 * Check if a grant is valid for a specific message type
 */
export function isGrantValidForMsg(grant: Grant, msgType: string): boolean {
  // Check if expired
  const expiration = new Date(grant.expiration)
  if (expiration.getTime() < Date.now()) {
    return false
  }

  // Check authorization type
  const authType = grant.authorization['@type']

  // GenericAuthorization grants permission for a specific message type
  if (authType === '/cosmos.authz.v1beta1.GenericAuthorization') {
    return grant.authorization.msg === msgType
  }

  // Other authorization types (we only use GenericAuthorization for now)
  return false
}

/**
 * Get all valid grants for required message types
 */
export function getValidGrants(grants: Grant[]): Map<string, Grant> {
  const validGrants = new Map<string, Grant>()

  for (const msgType of REQUIRED_MSG_TYPES) {
    const grant = grants.find(g => isGrantValidForMsg(g, msgType))
    if (grant) {
      validGrants.set(msgType, grant)
    }
  }

  return validGrants
}

/**
 * Check if we have grants for all required message types
 */
export function hasAllRequiredGrants(grants: Grant[]): boolean {
  const validGrants = getValidGrants(grants)
  return REQUIRED_MSG_TYPES.every(msgType => validGrants.has(msgType))
}

/**
 * Get missing grant types
 */
export function getMissingGrantTypes(grants: Grant[]): string[] {
  const validGrants = getValidGrants(grants)
  return REQUIRED_MSG_TYPES.filter(msgType => !validGrants.has(msgType))
}

/**
 * Get earliest grant expiration from valid grants
 */
export function getEarliestExpiration(grants: Grant[]): Date | null {
  const validGrants = getValidGrants(grants)

  if (validGrants.size === 0) return null

  let earliest: Date | null = null

  for (const grant of validGrants.values()) {
    const expiration = new Date(grant.expiration)
    if (!earliest || expiration < earliest) {
      earliest = expiration
    }
  }

  return earliest
}

/**
 * MsgExec to execute messages on behalf of granter
 */
export interface MsgExec {
  typeUrl: '/cosmos.authz.v1beta1.MsgExec'
  value: {
    grantee: string
    msgs: Array<{
      typeUrl: string
      value: Uint8Array
    }>
  }
}

/**
 * MsgGrant to grant authorization
 */
export interface MsgGrant {
  typeUrl: '/cosmos.authz.v1beta1.MsgGrant'
  value: {
    granter: string
    grantee: string
    grant: {
      authorization: {
        typeUrl: string
        value: Uint8Array
      }
      expiration: {
        seconds: bigint
        nanos: number
      }
    }
  }
}

/**
 * Create a MsgGrant for GenericAuthorization
 *
 * This is what the mobile wallet needs to sign to grant authorization
 */
export function createMsgGrant(
  granter: string,
  grantee: string,
  msgType: string,
  expiration: Date
): MsgGrant {
  // GenericAuthorization proto encoding
  // We encode it as Any with typeUrl and value
  const genericAuth = {
    msg: msgType
  }

  // Simple encoding - in production use proper protobuf
  const authValue = new TextEncoder().encode(JSON.stringify(genericAuth))

  const expirationSeconds = BigInt(Math.floor(expiration.getTime() / 1000))

  return {
    typeUrl: '/cosmos.authz.v1beta1.MsgGrant',
    value: {
      granter,
      grantee,
      grant: {
        authorization: {
          typeUrl: '/cosmos.authz.v1beta1.GenericAuthorization',
          value: authValue
        },
        expiration: {
          seconds: expirationSeconds,
          nanos: 0
        }
      }
    }
  }
}

/**
 * Create all required MsgGrant messages
 */
export function createAllMsgGrants(
  granter: string,
  grantee: string,
  expiration: Date
): MsgGrant[] {
  return REQUIRED_MSG_TYPES.map(msgType =>
    createMsgGrant(granter, grantee, msgType, expiration)
  )
}

/**
 * Wrap messages in MsgExec for execution via session key
 *
 * @param grantee - The session address that will sign
 * @param msgs - The messages to execute (will be executed as if from granter)
 */
export function wrapInMsgExec(
  grantee: string,
  msgs: Array<{ typeUrl: string; value: unknown }>
): MsgExec {
  // Encode inner messages
  const encodedMsgs = msgs.map(msg => ({
    typeUrl: msg.typeUrl,
    value: new TextEncoder().encode(JSON.stringify(msg.value))
  }))

  return {
    typeUrl: '/cosmos.authz.v1beta1.MsgExec',
    value: {
      grantee,
      msgs: encodedMsgs
    }
  }
}

/**
 * Format message type for display
 */
export function formatMsgType(msgType: string): string {
  // Extract the message name from the full type URL
  const parts = msgType.split('.')
  const msgName = parts[parts.length - 1]

  // Convert from PascalCase to readable format
  return msgName
    .replace(/^Msg/, '')
    .replace(/([A-Z])/g, ' $1')
    .trim()
}

/**
 * Get human-readable description for message type
 */
export function getMsgTypeDescription(msgType: string): string {
  const descriptions: Record<string, string> = {
    '/akash.deployment.v1beta4.MsgCreateDeployment': 'Create new deployments',
    '/akash.deployment.v1beta4.MsgCloseDeployment': 'Close deployments',
    '/akash.deployment.v1beta4.MsgUpdateDeployment': 'Update deployments',
    '/akash.market.v1beta5.MsgCreateLease': 'Accept provider bids',
    '/akash.cert.v1.MsgCreateCertificate': 'Create certificates',
    '/akash.cert.v1.MsgRevokeCertificate': 'Revoke certificates'
  }

  return descriptions[msgType] || formatMsgType(msgType)
}

/**
 * Fee grant allowance from REST API
 */
export interface FeeAllowance {
  granter: string
  grantee: string
  allowance: {
    '@type': string
    spend_limit?: Array<{ denom: string; amount: string }>
    expiration?: string
  }
}

/**
 * Check if a fee grant exists from granter to grantee
 * Note: Some APIs return 500 instead of 404 when the allowance doesn't exist
 */
export async function checkFeeGrant(
  granter: string,
  grantee: string
): Promise<FeeAllowance | null> {
  try {
    const url = `${AKASH_REST}/cosmos/feegrant/v1beta1/allowance/${granter}/${grantee}`
    console.log('Checking fee grant at:', url)
    const response = await fetch(url, { signal: AbortSignal.timeout(10000) })

    if (!response.ok) {
      // Some chains/proxies return 500 when allowance doesn't exist
      if (response.status === 404 || response.status === 500) {
        // Try to get error details from response
        try {
          const errorData = await response.json()
          console.log('Fee grant not found, response:', errorData)
        } catch {
          console.log('Fee grant not found, status:', response.status)
        }
        return null
      }
      throw new Error(`Failed to fetch fee grant: ${response.status}`)
    }

    const data = await response.json()
    console.log('Fee grant found:', data)
    return data.allowance as FeeAllowance
  } catch (error) {
    // Network errors or timeouts - treat as "not found" but log the error
    console.error('Error checking fee grant (treating as not found):', error)
    return null
  }
}
