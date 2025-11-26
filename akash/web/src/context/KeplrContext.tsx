/**
 * Keplr WalletConnect context for sharing wallet state across components
 */

import { createContext, useContext, useState, useCallback, ReactNode } from 'react'
import { KeplrWalletConnectV2 } from '@keplr-wallet/wc-client'
import { SigningStargateClient } from '@cosmjs/stargate'
import { connectKeplrMobile, disconnectKeplr, getKeplrInstance, setQRUriCallback } from '../lib/keplrWalletConnect'
import { AKASH_RPC } from '../config/constants'

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
}

const KeplrContext = createContext<KeplrContextValue | null>(null)

export function KeplrProvider({ children }: { children: ReactNode }) {
  const [isConnected, setIsConnected] = useState(false)
  const [isConnecting, setIsConnecting] = useState(false)
  const [address, setAddress] = useState<string | null>(null)
  const [keplr, setKeplr] = useState<KeplrWalletConnectV2 | null>(null)
  const [qrUri, setQrUri] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

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
      const client = await SigningStargateClient.connectWithSigner(AKASH_RPC, offlineSigner)
      return client
    } catch (e) {
      console.error('Error getting signing client:', e)
      return null
    }
  }, [])

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
        error
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
