/**
 * Session Keys Utilities
 *
 * Generates and manages temporary keypairs for authz sessions.
 * The session key is stored encrypted in localStorage and used to
 * sign transactions on behalf of the main wallet via MsgExec.
 */

import { Secp256k1, sha256, ripemd160, Random } from '@cosmjs/crypto'
import { toBech32, fromHex, toHex } from '@cosmjs/encoding'
import { DirectSecp256k1Wallet } from '@cosmjs/proto-signing'

// Storage keys
const SESSION_KEY_STORAGE = 'akash_session_key'
const SESSION_EXPIRY_STORAGE = 'akash_session_expiry'
const SESSION_GRANTER_STORAGE = 'akash_session_granter'

/**
 * Session keypair with address
 */
export interface SessionKeypair {
  privateKey: Uint8Array
  publicKey: Uint8Array
  address: string
}

/**
 * Stored session info
 */
export interface StoredSession {
  privateKeyHex: string
  expiration: number // Unix timestamp
  granterAddress: string
}

/**
 * Generate a new session keypair
 *
 * Creates a secp256k1 keypair and derives the Akash bech32 address
 */
export async function generateSessionKeypair(): Promise<SessionKeypair> {
  // Generate random private key
  const privateKey = Random.getBytes(32)

  // Derive public key
  const keypair = await Secp256k1.makeKeypair(privateKey)
  const publicKey = Secp256k1.compressPubkey(keypair.pubkey)

  // Derive address from public key
  // Akash uses the same address derivation as Cosmos SDK:
  // address = ripemd160(sha256(pubkey))
  const pubKeyHash = ripemd160(sha256(publicKey))
  const address = toBech32('akash', pubKeyHash)

  return {
    privateKey,
    publicKey,
    address
  }
}

/**
 * Store session key in localStorage
 *
 * Note: In production, consider encrypting with a user password
 * or using WebCrypto for additional security
 */
export function storeSessionKey(
  privateKey: Uint8Array,
  expiration: Date,
  granterAddress: string
): void {
  const session: StoredSession = {
    privateKeyHex: toHex(privateKey),
    expiration: expiration.getTime(),
    granterAddress
  }

  localStorage.setItem(SESSION_KEY_STORAGE, session.privateKeyHex)
  localStorage.setItem(SESSION_EXPIRY_STORAGE, session.expiration.toString())
  localStorage.setItem(SESSION_GRANTER_STORAGE, session.granterAddress)
}

/**
 * Load session key from localStorage
 *
 * Returns null if no session exists or if expired
 */
export function loadSessionKey(): StoredSession | null {
  const privateKeyHex = localStorage.getItem(SESSION_KEY_STORAGE)
  const expirationStr = localStorage.getItem(SESSION_EXPIRY_STORAGE)
  const granterAddress = localStorage.getItem(SESSION_GRANTER_STORAGE)

  if (!privateKeyHex || !expirationStr || !granterAddress) {
    return null
  }

  const expiration = parseInt(expirationStr, 10)

  // Check if expired
  if (Date.now() > expiration) {
    clearSessionKey()
    return null
  }

  return {
    privateKeyHex,
    expiration,
    granterAddress
  }
}

/**
 * Clear session key from localStorage
 */
export function clearSessionKey(): void {
  localStorage.removeItem(SESSION_KEY_STORAGE)
  localStorage.removeItem(SESSION_EXPIRY_STORAGE)
  localStorage.removeItem(SESSION_GRANTER_STORAGE)
}

/**
 * Check if a valid session exists
 */
export function isSessionValid(): boolean {
  const session = loadSessionKey()
  if (!session) return false
  return Date.now() < session.expiration
}

/**
 * Get session address without loading full key
 */
export async function getSessionAddress(): Promise<string | null> {
  const session = loadSessionKey()
  if (!session) return null

  try {
    const privateKey = fromHex(session.privateKeyHex)
    const keypair = await Secp256k1.makeKeypair(privateKey)
    const publicKey = Secp256k1.compressPubkey(keypair.pubkey)
    // Cosmos SDK address derivation: ripemd160(sha256(pubkey))
    const pubKeyHash = ripemd160(sha256(publicKey))
    return toBech32('akash', pubKeyHash)
  } catch {
    return null
  }
}

/**
 * Get the granter address (main wallet) for current session
 */
export function getSessionGranter(): string | null {
  return localStorage.getItem(SESSION_GRANTER_STORAGE)
}

/**
 * Get session expiration time
 */
export function getSessionExpiration(): Date | null {
  const session = loadSessionKey()
  if (!session) return null
  return new Date(session.expiration)
}

/**
 * Get remaining session time in milliseconds
 */
export function getSessionTimeRemaining(): number {
  const session = loadSessionKey()
  if (!session) return 0
  return Math.max(0, session.expiration - Date.now())
}

/**
 * Format session time remaining for display
 */
export function formatSessionTimeRemaining(): string {
  const remaining = getSessionTimeRemaining()
  if (remaining === 0) return 'Expired'

  const hours = Math.floor(remaining / (1000 * 60 * 60))
  const minutes = Math.floor((remaining % (1000 * 60 * 60)) / (1000 * 60))

  if (hours > 0) {
    return `${hours}h ${minutes}m`
  }
  return `${minutes}m`
}

/**
 * Create a DirectSecp256k1Wallet from session key
 *
 * This wallet can be used to sign transactions locally
 */
export async function createSessionWallet(): Promise<DirectSecp256k1Wallet | null> {
  const session = loadSessionKey()
  if (!session) return null

  try {
    const privateKey = fromHex(session.privateKeyHex)
    return await DirectSecp256k1Wallet.fromKey(privateKey, 'akash')
  } catch (error) {
    console.error('Failed to create session wallet:', error)
    return null
  }
}

/**
 * Session duration presets
 */
export const SESSION_DURATIONS = {
  '1h': 60 * 60 * 1000,
  '8h': 8 * 60 * 60 * 1000,
  '24h': 24 * 60 * 60 * 1000,
  '7d': 7 * 24 * 60 * 60 * 1000
} as const

export type SessionDuration = keyof typeof SESSION_DURATIONS

/**
 * Get expiration date for a duration preset
 */
export function getExpirationForDuration(duration: SessionDuration): Date {
  return new Date(Date.now() + SESSION_DURATIONS[duration])
}
