/**
 * Session Context
 *
 * Manages authz session state for delegated signing.
 * Allows the web app to sign transactions locally after receiving
 * a one-time grant from the mobile wallet.
 */

import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  type ReactNode
} from 'react'
import { useKeplr } from './KeplrContext'
import { MsgGrant } from 'cosmjs-types/cosmos/authz/v1beta1/tx'
import { GenericAuthorization } from 'cosmjs-types/cosmos/authz/v1beta1/authz'
import { MsgGrantAllowance } from 'cosmjs-types/cosmos/feegrant/v1beta1/tx'
import { BasicAllowance } from 'cosmjs-types/cosmos/feegrant/v1beta1/feegrant'
import { Timestamp } from 'cosmjs-types/google/protobuf/timestamp'
import {
  generateSessionKeypair,
  storeSessionKey,
  loadSessionKey,
  clearSessionKey,
  isSessionValid,
  formatSessionTimeRemaining,
  createSessionWallet,
  SESSION_DURATIONS,
  getExpirationForDuration,
  type SessionDuration,
  type SessionKeypair
} from '../utils/sessionKeys'
import {
  fetchGrants,
  hasAllRequiredGrants,
  getMissingGrantTypes,
  getEarliestExpiration,
  REQUIRED_MSG_TYPES
} from '../utils/authz'

/**
 * Session state
 */
type SessionState =
  | 'none'           // No session
  | 'generating'     // Generating keypair
  | 'pending_grant'  // Waiting for mobile to grant
  | 'checking'       // Checking grants on chain
  | 'active'         // Session active with valid grants
  | 'expired'        // Session expired

/**
 * Session context value
 */
interface SessionContextValue {
  // State
  sessionState: SessionState
  sessionAddress: string | null
  sessionExpiration: Date | null
  timeRemaining: string
  grantedMsgTypes: string[]
  missingMsgTypes: string[]

  // Pending session (waiting for grant)
  pendingSession: SessionKeypair | null
  pendingDuration: SessionDuration | null

  // Actions
  requestSession: (duration: SessionDuration) => Promise<SessionKeypair>
  checkGrants: () => Promise<boolean>
  activateSession: () => Promise<boolean>
  clearSession: () => void

  // Error state
  error: string | null
}

const SessionContext = createContext<SessionContextValue | null>(null)

/**
 * Session provider component
 */
export function SessionProvider({ children }: { children: ReactNode }) {
  const { address, isConnected, getSigningClient } = useKeplr()

  // Session state
  const [sessionState, setSessionState] = useState<SessionState>('none')
  const [sessionAddress, setSessionAddress] = useState<string | null>(null)
  const [sessionExpiration, setSessionExpiration] = useState<Date | null>(null)
  const [grantedMsgTypes, setGrantedMsgTypes] = useState<string[]>([])
  const [missingMsgTypes, setMissingMsgTypes] = useState<string[]>([...REQUIRED_MSG_TYPES])
  const [error, setError] = useState<string | null>(null)

  // Pending session (before grants are confirmed)
  const [pendingSession, setPendingSession] = useState<SessionKeypair | null>(null)
  const [pendingDuration, setPendingDuration] = useState<SessionDuration | null>(null)

  // Time remaining display
  const [timeRemaining, setTimeRemaining] = useState<string>('')

  // Load existing session on mount
  useEffect(() => {
    const loadExistingSession = async () => {
      if (!isConnected || !address) {
        setSessionState('none')
        return
      }

      const session = loadSessionKey()
      if (!session) {
        setSessionState('none')
        return
      }

      // Check if session matches current wallet
      if (session.granterAddress !== address) {
        clearSessionKey()
        setSessionState('none')
        return
      }

      // Load session address
      const wallet = await createSessionWallet()
      if (!wallet) {
        clearSessionKey()
        setSessionState('none')
        return
      }

      const [account] = await wallet.getAccounts()
      setSessionAddress(account.address)
      setSessionExpiration(new Date(session.expiration))

      // Check grants
      setSessionState('checking')
      const grants = await fetchGrants(address, account.address)
      const hasAll = hasAllRequiredGrants(grants)

      if (hasAll) {
        setSessionState('active')
        setGrantedMsgTypes([...REQUIRED_MSG_TYPES])
        setMissingMsgTypes([])
      } else {
        setSessionState('expired')
        setMissingMsgTypes(getMissingGrantTypes(grants))
      }
    }

    loadExistingSession()
  }, [isConnected, address])

  // Update time remaining every minute
  useEffect(() => {
    if (sessionState !== 'active') {
      setTimeRemaining('')
      return
    }

    const updateTime = () => {
      setTimeRemaining(formatSessionTimeRemaining())

      // Check if expired
      if (!isSessionValid()) {
        setSessionState('expired')
      }
    }

    updateTime()
    const interval = setInterval(updateTime, 60000)

    return () => clearInterval(interval)
  }, [sessionState])

  // Request a new session
  const requestSession = useCallback(async (duration: SessionDuration): Promise<SessionKeypair> => {
    if (!address) {
      throw new Error('Wallet not connected')
    }

    setError(null)
    setSessionState('generating')

    try {
      // Generate new keypair
      const keypair = await generateSessionKeypair()

      // Calculate expiration
      const expiration = getExpirationForDuration(duration)

      // Store the session locally first
      storeSessionKey(keypair.privateKey, expiration, address)

      // Set pending state - will be confirmed after grants are received
      setPendingSession(keypair)
      setPendingDuration(duration)
      setSessionAddress(keypair.address)
      setSessionExpiration(expiration)

      // Get signing client to broadcast MsgGrant
      const signingClient = await getSigningClient()
      if (!signingClient) {
        throw new Error('Failed to get signing client. Please reconnect wallet.')
      }

      // Create MsgGrant messages for each required message type
      // This will trigger a signing request on the mobile wallet
      const expirationSeconds = Math.floor(expiration.getTime() / 1000)

      // Authz grants for each message type
      // Use MsgGrant.fromPartial() for proper protobuf encoding
      const authzGrants = REQUIRED_MSG_TYPES.map(msgType => ({
        typeUrl: '/cosmos.authz.v1beta1.MsgGrant',
        value: MsgGrant.fromPartial({
          granter: address,
          grantee: keypair.address,
          grant: {
            authorization: {
              typeUrl: '/cosmos.authz.v1beta1.GenericAuthorization',
              value: GenericAuthorization.encode(
                GenericAuthorization.fromPartial({ msg: msgType })
              ).finish()
            },
            expiration: Timestamp.fromPartial({
              seconds: BigInt(expirationSeconds),
              nanos: 0
            })
          }
        })
      }))

      // Fee grant - allows session to use main wallet's funds for gas
      // BasicAllowance with no spend limit, just expiration
      // Must use fromPartial() for proper protobuf encoding
      const feeGrant = {
        typeUrl: '/cosmos.feegrant.v1beta1.MsgGrantAllowance',
        value: MsgGrantAllowance.fromPartial({
          granter: address,
          grantee: keypair.address,
          allowance: {
            typeUrl: '/cosmos.feegrant.v1beta1.BasicAllowance',
            value: BasicAllowance.encode(
              BasicAllowance.fromPartial({
                spendLimit: [], // No limit - allows paying any fee amount
                expiration: Timestamp.fromPartial({
                  seconds: BigInt(expirationSeconds),
                  nanos: 0
                })
              })
            ).finish()
          }
        })
      }

      // Combine all grants into single transaction
      const grantMsgs = [...authzGrants, feeGrant]

      setSessionState('pending_grant')

      // Log detailed message info for debugging
      console.log('=== SESSION GRANT DEBUG ===')
      console.log('Granter (main wallet):', address)
      console.log('Grantee (session):', keypair.address)
      console.log('Total messages to broadcast:', grantMsgs.length)
      console.log('Message types:', grantMsgs.map(m => m.typeUrl))
      console.log('FeeGrant message:', JSON.stringify({
        typeUrl: feeGrant.typeUrl,
        granter: feeGrant.value.granter,
        grantee: feeGrant.value.grantee,
        allowanceTypeUrl: feeGrant.value.allowance?.typeUrl,
        allowanceValueLength: feeGrant.value.allowance?.value?.length
      }, null, 2))

      // Broadcast the grant transaction - this triggers mobile wallet approval
      console.log('Broadcasting MsgGrant transaction for session:', keypair.address)
      const result = await signingClient.signAndBroadcast(
        address,
        grantMsgs,
        'auto',
        `Grant session signing authority to ${keypair.address}`
      )

      console.log('Transaction result:', {
        code: result.code,
        transactionHash: result.transactionHash,
        rawLog: result.rawLog,
        gasUsed: result.gasUsed,
        gasWanted: result.gasWanted
      })

      if (result.code !== 0) {
        throw new Error(`Grant transaction failed: ${result.rawLog}`)
      }

      console.log('Grant transaction successful:', result.transactionHash)

      // Immediately verify feegrant was created
      const { checkFeeGrant } = await import('../utils/authz')
      const feeGrantCheck = await checkFeeGrant(address, keypair.address)
      console.log('Feegrant verification after tx:', feeGrantCheck ? 'EXISTS' : 'NOT FOUND')
      if (!feeGrantCheck) {
        console.warn('WARNING: Transaction succeeded but feegrant was not created!')
        console.warn('This may indicate an encoding issue with MsgGrantAllowance')
      }

      // Grants are now on-chain, verify and activate
      setSessionState('checking')
      const grants = await fetchGrants(address, keypair.address)
      const hasAll = hasAllRequiredGrants(grants)

      if (hasAll) {
        setSessionState('active')
        setGrantedMsgTypes([...REQUIRED_MSG_TYPES])
        setMissingMsgTypes([])
        setPendingSession(null)
        setPendingDuration(null)
      } else {
        // Some grants may have failed, show what's missing
        setMissingMsgTypes(getMissingGrantTypes(grants))
        setSessionState('pending_grant')
      }

      return keypair
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to generate session'
      setError(message)
      setSessionState('none')
      clearSessionKey()
      throw err
    }
  }, [address, getSigningClient])

  // Check if grants have been received
  const checkGrants = useCallback(async (): Promise<boolean> => {
    if (!address || !sessionAddress) {
      return false
    }

    setError(null)
    setSessionState('checking')

    try {
      const grants = await fetchGrants(address, sessionAddress)
      const hasAll = hasAllRequiredGrants(grants)

      if (hasAll) {
        setGrantedMsgTypes([...REQUIRED_MSG_TYPES])
        setMissingMsgTypes([])
        setSessionState('active')

        // Update expiration to earliest grant expiration
        const grantExpiration = getEarliestExpiration(grants)
        if (grantExpiration && sessionExpiration && grantExpiration < sessionExpiration) {
          setSessionExpiration(grantExpiration)
        }

        setPendingSession(null)
        setPendingDuration(null)
        return true
      } else {
        const missing = getMissingGrantTypes(grants)
        setMissingMsgTypes(missing)

        // Still update granted types
        const granted = REQUIRED_MSG_TYPES.filter(t => !missing.includes(t))
        setGrantedMsgTypes([...granted])

        setSessionState('pending_grant')
        return false
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to check grants'
      setError(message)
      setSessionState('pending_grant')
      return false
    }
  }, [address, sessionAddress, sessionExpiration])

  // Activate session after grants are confirmed
  const activateSession = useCallback(async (): Promise<boolean> => {
    const hasAll = await checkGrants()
    return hasAll
  }, [checkGrants])

  // Clear session
  const clearSession = useCallback(() => {
    clearSessionKey()
    setSessionState('none')
    setSessionAddress(null)
    setSessionExpiration(null)
    setGrantedMsgTypes([])
    setMissingMsgTypes([...REQUIRED_MSG_TYPES])
    setPendingSession(null)
    setPendingDuration(null)
    setError(null)
  }, [])

  // Clear session when wallet disconnects
  useEffect(() => {
    if (!isConnected) {
      clearSession()
    }
  }, [isConnected, clearSession])

  const value: SessionContextValue = {
    sessionState,
    sessionAddress,
    sessionExpiration,
    timeRemaining,
    grantedMsgTypes,
    missingMsgTypes,
    pendingSession,
    pendingDuration,
    requestSession,
    checkGrants,
    activateSession,
    clearSession,
    error
  }

  return (
    <SessionContext.Provider value={value}>
      {children}
    </SessionContext.Provider>
  )
}

/**
 * Hook to use session context
 */
export function useSession(): SessionContextValue {
  const context = useContext(SessionContext)
  if (!context) {
    throw new Error('useSession must be used within a SessionProvider')
  }
  return context
}
