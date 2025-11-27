/**
 * Keplr WalletConnect context for sharing wallet state across components
 *
 * Uses @akashnetwork/chain-sdk for future-proof message type handling.
 * The chain-sdk provides v1beta4 protobuf types for the current Akash mainnet.
 */

import { createContext, useContext, useState, useCallback, ReactNode } from 'react'
import { KeplrWalletConnectV2 } from '@keplr-wallet/wc-client'
import { SigningStargateClient, GasPrice } from '@cosmjs/stargate'
import { Registry } from '@cosmjs/proto-signing'
import { connectKeplrMobile, disconnectKeplr, getKeplrInstance, setQRUriCallback } from '../lib/keplrWalletConnect'
import { AKASH_RPC, AKASH_DENOM } from '../config/constants'

// Import v1beta4 message types from chain-sdk (current Akash mainnet version)
// When upgrading to a new API version, update these imports
import {
  MsgCreateDeployment,
  MsgUpdateDeployment,
  MsgCloseDeployment
} from '@akashnetwork/chain-sdk/private-types/akash.v1beta4'

// Market messages are in v1beta5
import { MsgCreateLease } from '@akashnetwork/chain-sdk/private-types/akash.v1beta5'

// Certificate messages are in v1
import { MsgCreateCertificate, MsgRevokeCertificate } from '@akashnetwork/chain-sdk/private-types/akash.v1'

// Certificate utilities
import {
  generateCertificate,
  storeCertificate,
  loadCertificate,
  removeCertificate,
  isCertificateValid,
  AkashCertificate
} from '../utils/certificate'
import { fetchCertificateFromChain, publishCertificateToChain, revokeCertificateOnChain, isLocalCertRegistered } from '../utils/akashApi'

// Create registry with Akash message types
function createAkashRegistry(): Registry {
  const registry = new Registry()

  // Register deployment messages (v1beta4)
  // The $type property contains the full protobuf type name
  registry.register(`/${MsgCreateDeployment.$type}`, MsgCreateDeployment as never)
  registry.register(`/${MsgUpdateDeployment.$type}`, MsgUpdateDeployment as never)
  registry.register(`/${MsgCloseDeployment.$type}`, MsgCloseDeployment as never)

  // Register market messages (v1beta5)
  registry.register(`/${MsgCreateLease.$type}`, MsgCreateLease as never)

  // Register certificate messages (v1)
  registry.register(`/${MsgCreateCertificate.$type}`, MsgCreateCertificate as never)
  registry.register(`/${MsgRevokeCertificate.$type}`, MsgRevokeCertificate as never)

  return registry
}

interface KeplrContextValue {
  isConnected: boolean
  isConnecting: boolean
  address: string | null
  keplr: KeplrWalletConnectV2 | null
  qrUri: string | null
  connect: () => Promise<void>
  disconnect: () => Promise<void>
  getSigningClient: () => Promise<SigningStargateClient | null>
  error: string | null
  // Certificate state
  certificate: AkashCertificate | null
  hasCertificateOnChain: boolean
  hasLocalCertificate: boolean
  certificateMismatch: boolean  // on-chain cert doesn't match local cert
  isCheckingCertificate: boolean
  isCertificateLoading: boolean
  checkAndLoadCertificate: () => Promise<void>
  createAndPublishCertificate: () => Promise<boolean>
  revokeAndRecreateCertificate: () => Promise<boolean>
  clearAndRegenerateCertificate: () => Promise<boolean>
}

const KeplrContext = createContext<KeplrContextValue | null>(null)

export function KeplrProvider({ children }: { children: ReactNode }) {
  const [isConnected, setIsConnected] = useState(false)
  const [isConnecting, setIsConnecting] = useState(false)
  const [address, setAddress] = useState<string | null>(null)
  const [keplr, setKeplr] = useState<KeplrWalletConnectV2 | null>(null)
  const [qrUri, setQrUri] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Certificate state
  const [certificate, setCertificate] = useState<AkashCertificate | null>(null)
  const [hasCertificateOnChain, setHasCertificateOnChain] = useState(false)
  const [hasLocalCertificate, setHasLocalCertificate] = useState(false)
  const [certificateMismatch, setCertificateMismatch] = useState(false)
  const [isCheckingCertificate, setIsCheckingCertificate] = useState(false)
  const [isCertificateLoading, setIsCertificateLoading] = useState(false)

  const connect = useCallback(async () => {
    setIsConnecting(true)
    setError(null)
    setQrUri(null)

    // Set up callback to receive QR URI
    setQRUriCallback((uri: string) => {
      console.log('QR URI callback received')
      setQrUri(uri)
    })

    try {
      const { keplr: keplrInstance } = await connectKeplrMobile(['akashnet-2'])

      // Clear the QR code once connected
      setQrUri(null)

      // Get the address
      const key = await keplrInstance.getKey('akashnet-2')

      setKeplr(keplrInstance)
      setAddress(key.bech32Address)
      setIsConnected(true)
    } catch (e) {
      const message = e instanceof Error ? e.message : 'Connection failed'
      console.error('Keplr connection error:', e)
      setError(message)
      setIsConnected(false)
      setQrUri(null)
    } finally {
      setIsConnecting(false)
    }
  }, [])

  const disconnect = useCallback(async () => {
    await disconnectKeplr()
    setKeplr(null)
    setAddress(null)
    setIsConnected(false)
  }, [])

  const getSigningClient = useCallback(async (): Promise<SigningStargateClient | null> => {
    const keplrInstance = getKeplrInstance()
    if (!keplrInstance) return null

    try {
      const offlineSigner = keplrInstance.getOfflineSigner('akashnet-2')
      const client = await SigningStargateClient.connectWithSigner(AKASH_RPC, offlineSigner, {
        gasPrice: GasPrice.fromString(`0.025${AKASH_DENOM}`),
        registry: createAkashRegistry()
      })
      return client
    } catch (e) {
      console.error('Error getting signing client:', e)
      return null
    }
  }, [])

  /**
   * Check for existing certificate on-chain and in localStorage
   * Detects if local certificate is registered on-chain
   */
  const checkAndLoadCertificate = useCallback(async () => {
    if (!address) return

    setIsCheckingCertificate(true)
    setCertificateMismatch(false)
    setHasLocalCertificate(false)

    try {
      // First, check localStorage for existing certificate
      const storedCert = loadCertificate(address)
      const hasValidLocalCert = storedCert && isCertificateValid(storedCert)

      if (hasValidLocalCert) {
        setCertificate(storedCert)
        setHasLocalCertificate(true)
        console.log('Loaded certificate from localStorage')
      } else {
        setCertificate(null)
        console.log('No valid local certificate found')
      }

      // Check if certificate exists on-chain
      const onChainCert = await fetchCertificateFromChain(address)
      setHasCertificateOnChain(!!onChainCert)

      if (onChainCert) {
        console.log('Certificate(s) found on-chain, first serial:', onChainCert.serial)

        // Check if our local cert is registered (could be any cert on-chain, not just the first)
        if (hasValidLocalCert) {
          const isRegistered = await isLocalCertRegistered(address, storedCert.pubkeyPem)
          if (isRegistered) {
            console.log('Local certificate IS registered on-chain')
            setCertificateMismatch(false)
          } else {
            console.log('Local certificate is NOT registered on-chain')
            console.log('This is OK - we can publish our local cert as additional cert')
            setCertificateMismatch(true)
          }
        }
      } else {
        console.log('No certificate found on-chain')
      }
    } catch (e) {
      console.error('Error checking certificate:', e)
    } finally {
      setIsCheckingCertificate(false)
    }
  }, [address])

  /**
   * Generate a new certificate and publish it to the blockchain
   *
   * Akash allows multiple certificates per address, so we:
   * 1. Check if we have a local certificate with private key
   * 2. If not, generate a new one
   * 3. Check if our local cert is already on-chain (by pubkey match)
   * 4. If not on-chain, publish it (even if other certs exist)
   */
  const createAndPublishCertificate = useCallback(async (): Promise<boolean> => {
    if (!address) {
      console.error('No address available for certificate creation')
      return false
    }

    setIsCertificateLoading(true)
    setError(null)

    try {
      // Check if we already have a local certificate
      let cert = loadCertificate(address)

      // Generate new certificate if needed
      if (!cert || !isCertificateValid(cert)) {
        console.log('Generating new certificate...')
        cert = await generateCertificate(address)
        storeCertificate(address, cert)
        setCertificate(cert)
        setHasLocalCertificate(true)
        console.log('Certificate generated and stored locally')
      }

      // Check if OUR certificate is already registered on-chain
      const isRegistered = await isLocalCertRegistered(address, cert.pubkeyPem)
      if (isRegistered) {
        console.log('Our local certificate is already registered on-chain')
        setHasCertificateOnChain(true)
        setCertificateMismatch(false)
        return true
      }

      // Our cert isn't registered, publish it
      console.log('Local certificate not found on-chain, publishing...')

      // Get signing client and publish
      const signingClient = await getSigningClient()
      if (!signingClient) {
        throw new Error('Failed to get signing client')
      }

      console.log('Publishing certificate to blockchain...')
      await publishCertificateToChain(signingClient, address, cert.certPem, cert.pubkeyPem)

      setHasCertificateOnChain(true)
      setCertificateMismatch(false)
      console.log('Certificate published successfully')
      return true
    } catch (e) {
      console.error('Error creating/publishing certificate:', e)
      setError(e instanceof Error ? e.message : 'Certificate creation failed')
      return false
    } finally {
      setIsCertificateLoading(false)
    }
  }, [address, getSigningClient])

  /**
   * Clear the locally stored certificate and generate a fresh one
   * Useful when the stored cert has wrong format or is corrupted
   */
  const clearAndRegenerateCertificate = useCallback(async (): Promise<boolean> => {
    if (!address) {
      console.error('No address available for certificate regeneration')
      return false
    }

    setIsCertificateLoading(true)
    setError(null)

    try {
      // Clear the old certificate from localStorage
      console.log('Clearing old certificate from localStorage...')
      removeCertificate(address)
      setCertificate(null)
      setHasLocalCertificate(false)

      // Generate fresh certificate
      console.log('Generating new certificate...')
      const cert = await generateCertificate(address)
      storeCertificate(address, cert)
      setCertificate(cert)
      setHasLocalCertificate(true)
      console.log('New certificate generated and stored locally')

      // Publish to blockchain
      const signingClient = await getSigningClient()
      if (!signingClient) {
        throw new Error('Failed to get signing client')
      }

      console.log('Publishing new certificate to blockchain...')
      await publishCertificateToChain(signingClient, address, cert.certPem, cert.pubkeyPem)

      setHasCertificateOnChain(true)
      setCertificateMismatch(false)
      console.log('New certificate published successfully')
      return true
    } catch (e) {
      console.error('Error regenerating certificate:', e)
      setError(e instanceof Error ? e.message : 'Certificate regeneration failed')
      return false
    } finally {
      setIsCertificateLoading(false)
    }
  }, [address, getSigningClient])

  return (
    <KeplrContext.Provider
      value={{
        isConnected,
        isConnecting,
        address,
        keplr,
        qrUri,
        connect,
        disconnect,
        getSigningClient,
        error,
        // Certificate state and functions
        certificate,
        hasCertificateOnChain,
        hasLocalCertificate,
        certificateMismatch,
        isCheckingCertificate,
        isCertificateLoading,
        checkAndLoadCertificate,
        createAndPublishCertificate,
        // revokeAndRecreateCertificate not needed - we can have multiple certs
        revokeAndRecreateCertificate: createAndPublishCertificate,
        clearAndRegenerateCertificate
      }}
    >
      {children}
    </KeplrContext.Provider>
  )
}

export function useKeplr() {
  const context = useContext(KeplrContext)
  if (!context) {
    throw new Error('useKeplr must be used within a KeplrProvider')
  }
  return context
}
