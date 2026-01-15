/**
 * Session Signing Utility
 *
 * Provides the ability to sign transactions using a session key via MsgExec.
 * When a session is active, transactions are wrapped in MsgExec and signed
 * by the session wallet, executing on behalf of the main wallet (granter).
 */

import { SigningStargateClient, GasPrice, StargateClient } from '@cosmjs/stargate'
import { Registry, type EncodeObject } from '@cosmjs/proto-signing'
import { TxRaw } from 'cosmjs-types/cosmos/tx/v1beta1/tx'
import { MsgExec } from 'cosmjs-types/cosmos/authz/v1beta1/tx'
import { Any } from 'cosmjs-types/google/protobuf/any'
import { createSessionWallet, loadSessionKey, getSessionGranter } from './sessionKeys'
import { checkFeeGrant } from './authz'
import { AKASH_RPC, AKASH_DENOM, AKASH_CHAIN_ID } from '../config/constants'

// Import message types for registry
import {
  MsgCreateDeployment,
  MsgUpdateDeployment,
  MsgCloseDeployment
} from '@akashnetwork/chain-sdk/private-types/akash.v1beta4'
import { MsgCreateLease } from '@akashnetwork/chain-sdk/private-types/akash.v1beta5'
import { MsgCreateCertificate, MsgRevokeCertificate } from '@akashnetwork/chain-sdk/private-types/akash.v1'
import { MsgGrant, MsgRevoke } from 'cosmjs-types/cosmos/authz/v1beta1/tx'
import { GenericAuthorization } from 'cosmjs-types/cosmos/authz/v1beta1/authz'

/**
 * Message type to encoder mapping
 * Maps typeUrl to encoder function
 */
const MESSAGE_ENCODERS: Record<string, { encode: (msg: unknown) => { finish: () => Uint8Array } }> = {
  '/akash.deployment.v1beta4.MsgCreateDeployment': MsgCreateDeployment,
  '/akash.deployment.v1beta4.MsgCloseDeployment': MsgCloseDeployment,
  '/akash.deployment.v1beta4.MsgUpdateDeployment': MsgUpdateDeployment,
  '/akash.market.v1beta5.MsgCreateLease': MsgCreateLease,
  '/akash.cert.v1.MsgCreateCertificate': MsgCreateCertificate,
  '/akash.cert.v1.MsgRevokeCertificate': MsgRevokeCertificate,
}

/**
 * Create registry with all required message types
 */
function createSessionRegistry(): Registry {
  const registry = new Registry()

  // Register deployment messages (v1beta4)
  registry.register(`/${MsgCreateDeployment.$type}`, MsgCreateDeployment as never)
  registry.register(`/${MsgUpdateDeployment.$type}`, MsgUpdateDeployment as never)
  registry.register(`/${MsgCloseDeployment.$type}`, MsgCloseDeployment as never)

  // Register market messages (v1beta5)
  registry.register(`/${MsgCreateLease.$type}`, MsgCreateLease as never)

  // Register certificate messages (v1)
  registry.register(`/${MsgCreateCertificate.$type}`, MsgCreateCertificate as never)
  registry.register(`/${MsgRevokeCertificate.$type}`, MsgRevokeCertificate as never)

  // Register authz messages
  registry.register('/cosmos.authz.v1beta1.MsgGrant', MsgGrant as never)
  registry.register('/cosmos.authz.v1beta1.MsgRevoke', MsgRevoke as never)
  registry.register('/cosmos.authz.v1beta1.MsgExec', MsgExec as never)
  registry.register('/cosmos.authz.v1beta1.GenericAuthorization', GenericAuthorization as never)

  return registry
}

/**
 * Check if a valid session is available for signing
 */
export function isSessionAvailable(): boolean {
  const session = loadSessionKey()
  if (!session) return false

  // Check if expired
  if (Date.now() > session.expiration) return false

  return true
}

/**
 * Get the granter address (main wallet) if session is available
 */
export function getGranterAddress(): string | null {
  if (!isSessionAvailable()) return null
  return getSessionGranter()
}

/**
 * Create a signing client using the session wallet
 */
export async function createSessionSigningClient(): Promise<SigningStargateClient | null> {
  const sessionWallet = await createSessionWallet()
  if (!sessionWallet) {
    console.log('No session wallet available')
    return null
  }

  try {
    const client = await SigningStargateClient.connectWithSigner(
      AKASH_RPC,
      sessionWallet,
      {
        gasPrice: GasPrice.fromString(`0.025${AKASH_DENOM}`),
        registry: createSessionRegistry()
      }
    )
    return client
  } catch (error) {
    console.error('Failed to create session signing client:', error)
    return null
  }
}

/**
 * Encode a message to Any for MsgExec
 */
function encodeMessageToAny(msg: EncodeObject): Any {
  const encoder = MESSAGE_ENCODERS[msg.typeUrl]
  if (!encoder) {
    throw new Error(`Unknown message type: ${msg.typeUrl}. Cannot encode for MsgExec.`)
  }

  // Encode the message value
  const encoded = encoder.encode(msg.value).finish()

  return {
    typeUrl: msg.typeUrl,
    value: encoded
  }
}

/**
 * Sign and broadcast a transaction using the session wallet via MsgExec
 *
 * @param granterAddress - The main wallet address (granter)
 * @param messages - The messages to execute (will be wrapped in MsgExec)
 * @param memo - Transaction memo
 * @returns Transaction result
 */
export async function signWithSession(
  granterAddress: string,
  messages: EncodeObject[],
  memo: string = ''
): Promise<{ transactionHash: string; code: number; rawLog?: string }> {
  // Get session wallet
  const sessionWallet = await createSessionWallet()
  if (!sessionWallet) {
    throw new Error('No session wallet available')
  }

  // Get session address
  const [sessionAccount] = await sessionWallet.getAccounts()
  const sessionAddress = sessionAccount.address

  console.log('Signing with session:', {
    sessionAddress,
    granterAddress,
    messageCount: messages.length,
    messageTypes: messages.map(m => m.typeUrl)
  })

  // Verify granter matches stored session
  const storedGranter = getSessionGranter()
  if (storedGranter !== granterAddress) {
    throw new Error(`Session granter mismatch: expected ${storedGranter}, got ${granterAddress}`)
  }

  // Verify fee grant exists before attempting to use it
  const feeGrant = await checkFeeGrant(granterAddress, sessionAddress)
  if (!feeGrant) {
    throw new Error(
      `Fee grant not found from ${granterAddress} to ${sessionAddress}. ` +
      'Please create a new session to re-establish the fee grant.'
    )
  }
  console.log('Fee grant verified:', feeGrant)

  // Create signing client with session wallet
  const client = await SigningStargateClient.connectWithSigner(
    AKASH_RPC,
    sessionWallet,
    {
      gasPrice: GasPrice.fromString(`0.025${AKASH_DENOM}`),
      registry: createSessionRegistry()
    }
  )

  // Encode inner messages to Any
  const encodedMsgs = messages.map(encodeMessageToAny)

  // Create MsgExec wrapper
  const msgExec: EncodeObject = {
    typeUrl: '/cosmos.authz.v1beta1.MsgExec',
    value: MsgExec.fromPartial({
      grantee: sessionAddress,
      msgs: encodedMsgs
    })
  }

  // Sign and broadcast with session wallet
  // Note: The signer is the session address (grantee)
  // Use fee granter so the main wallet pays for gas
  //
  // We can't use signAndBroadcast() because it calls getSequence() which fails
  // for accounts that don't exist or have never sent a transaction.
  // Instead, we use sign() with explicit signerData and broadcast separately.

  // Try to get account info from chain, fall back to sequence 0 for new accounts
  let accountNumber = 0
  let sequence = 0

  try {
    // Use a read-only client to query account info
    const queryClient = await StargateClient.connect(AKASH_RPC)
    const account = await queryClient.getAccount(sessionAddress)
    if (account) {
      accountNumber = account.accountNumber
      sequence = account.sequence
      console.log('Session account found on chain:', { accountNumber, sequence })
    } else {
      console.log('Session account not found on chain, using sequence 0')
    }
  } catch (e) {
    // Account doesn't exist yet, use defaults
    console.log('Could not query session account, using sequence 0:', e)
  }

  // Sign with explicit signer data (bypasses getSequence call)
  const signerData = {
    accountNumber,
    sequence,
    chainId: AKASH_CHAIN_ID
  }

  // Estimate gas via simulation, with fallback to fixed limit
  // Simulation gives us actual gas needed, then we add a safety buffer
  const GAS_MULTIPLIER = 1.4  // 40% buffer for safety
  const FALLBACK_GAS_LIMIT = 1500000  // 1.5M gas fallback if simulation fails
  const MIN_GAS_LIMIT = 200000  // Minimum 200k gas

  let gasLimit: number

  try {
    // Simulate the transaction to estimate gas
    const simulatedGas = await client.simulate(sessionAddress, [msgExec], memo + ' (via session)')
    gasLimit = Math.max(MIN_GAS_LIMIT, Math.ceil(simulatedGas * GAS_MULTIPLIER))
    console.log('Gas estimation:', {
      simulated: simulatedGas,
      multiplier: GAS_MULTIPLIER,
      finalLimit: gasLimit,
      estimatedFee: `${Math.ceil(gasLimit * 0.025)} uakt (~${(gasLimit * 0.025 / 1_000_000).toFixed(4)} AKT)`
    })
  } catch (simError) {
    // Simulation can fail for new accounts or other reasons
    // Fall back to a safe fixed limit
    gasLimit = FALLBACK_GAS_LIMIT
    console.log('Gas simulation failed, using fallback:', {
      error: simError instanceof Error ? simError.message : String(simError),
      fallbackLimit: gasLimit
    })
  }

  const fee = {
    amount: [{ denom: AKASH_DENOM, amount: String(Math.ceil(gasLimit * 0.025)) }],
    gas: String(gasLimit),
    granter: granterAddress // Main wallet pays the fee
  }

  console.log('Signing with explicit signerData:', signerData)

  const txRaw = await client.sign(
    sessionAddress,
    [msgExec],
    fee,
    memo + ' (via session)',
    signerData
  )

  // Broadcast the signed transaction
  const txBytes = TxRaw.encode(txRaw).finish()
  const result = await client.broadcastTx(txBytes)

  console.log('Session signing result:', {
    code: result.code,
    transactionHash: result.transactionHash,
    rawLog: result.rawLog?.substring(0, 200)
  })

  return {
    transactionHash: result.transactionHash,
    code: result.code,
    rawLog: result.rawLog
  }
}

/**
 * Sign and broadcast - uses session if available, otherwise falls back to provided client
 *
 * This is a drop-in replacement for signingClient.signAndBroadcast that automatically
 * uses session signing when available.
 *
 * @param signingClient - The main wallet's signing client (used if no session)
 * @param signerAddress - The address that will sign (main wallet address)
 * @param messages - Messages to sign and broadcast
 * @param fee - Fee or 'auto'
 * @param memo - Transaction memo
 */
export async function signAndBroadcastWithSession(
  signingClient: SigningStargateClient,
  signerAddress: string,
  messages: EncodeObject[],
  fee: 'auto' | number,
  memo: string = ''
): Promise<{ transactionHash: string; code: number; rawLog?: string }> {
  // Check if session is available and matches the signer
  if (isSessionAvailable()) {
    const granterAddress = getGranterAddress()
    if (granterAddress === signerAddress) {
      console.log('Using session signing for', messages.map(m => m.typeUrl))
      try {
        return await signWithSession(signerAddress, messages, memo)
      } catch (error) {
        // If session signing fails, log and fall back to main wallet
        console.error('Session signing failed, falling back to main wallet:', error)
      }
    }
  }

  // Fall back to main wallet signing
  console.log('Using main wallet signing for', messages.map(m => m.typeUrl))
  const result = await signingClient.signAndBroadcast(
    signerAddress,
    messages,
    fee === 'auto' ? 'auto' : { amount: [{ denom: AKASH_DENOM, amount: String(fee) }], gas: '200000' },
    memo
  )

  return {
    transactionHash: result.transactionHash,
    code: result.code,
    rawLog: result.rawLog
  }
}
