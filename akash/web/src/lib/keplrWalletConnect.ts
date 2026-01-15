/**
 * Keplr WalletConnect integration using official Keplr packages
 */

import SignClient from '@walletconnect/sign-client'
import { KeplrWalletConnectV2 } from '@keplr-wallet/wc-client'
import { WALLETCONNECT_PROJECT_ID } from '../config/constants'

let signClient: SignClient | null = null
let keplrInstance: KeplrWalletConnectV2 | null = null

const metadata = {
  name: 'Quip Protocol Mining',
  description: 'Deploy decentralized mining experiments on Akash Network',
  icons: ['https://quip.network/favicon.ico']
}

// Minimal required methods for Cosmos signing
const REQUIRED_METHODS = [
  'cosmos_getAccounts',
  'cosmos_signAmino',
  'cosmos_signDirect',
]

// Keplr-specific methods as optional
const OPTIONAL_METHODS = [
  'keplr_getKey',
  'keplr_signAmino',
  'keplr_signDirect',
  'keplr_signArbitrary',
  'keplr_enable',
]

const REQUIRED_EVENTS: string[] = []
const OPTIONAL_EVENTS = ['chainChanged', 'accountsChanged', 'keplr_accountsChanged']

// Callback for QR code URI
let qrUriCallback: ((uri: string) => void) | null = null

export function setQRUriCallback(callback: (uri: string) => void): void {
  qrUriCallback = callback
}

async function cleanCorruptedDatabases(): Promise<void> {
  try {
    const databases = await indexedDB.databases()
    for (const db of databases) {
      if (db.name && (db.name.includes('wc') || db.name.includes('walletconnect') || db.name.includes('WALLET_CONNECT'))) {
        const checkDb = indexedDB.open(db.name)
        const isCorrupted = await new Promise<boolean>((resolve) => {
          checkDb.onerror = () => resolve(true)
          checkDb.onsuccess = () => {
            const database = checkDb.result
            const hasKeyValueStorage = database.objectStoreNames.contains('keyvaluestorage')
            database.close()
            resolve(!hasKeyValueStorage)
          }
        })

        if (isCorrupted) {
          console.log('Deleting corrupted WC database:', db.name)
          await new Promise<void>((resolve) => {
            const deleteReq = indexedDB.deleteDatabase(db.name)
            deleteReq.onsuccess = () => resolve()
            deleteReq.onerror = () => resolve()
            deleteReq.onblocked = () => resolve()
          })
        }
      }
    }
  } catch (e) {
    console.error('Error checking databases:', e)
  }
}

export async function initSignClient(): Promise<SignClient> {
  if (signClient) return signClient

  await cleanCorruptedDatabases()
  await new Promise(resolve => setTimeout(resolve, 100))

  const timeoutMs = 15000
  const initPromise = SignClient.init({
    projectId: WALLETCONNECT_PROJECT_ID,
    metadata,
  })

  const timeoutPromise = new Promise<never>((_, reject) => {
    setTimeout(() => {
      reject(new Error(`SignClient.init() timed out after ${timeoutMs}ms. Try clearing browser storage.`))
    }, timeoutMs)
  })

  try {
    signClient = await Promise.race([initPromise, timeoutPromise])
  } catch (err) {
    console.error('SignClient.init() failed:', err)
    throw err
  }

  // Clean up keplrInstance when session is deleted
  signClient.on('session_delete', () => {
    keplrInstance = null
  })

  return signClient
}

export async function connectKeplrMobile(chainIds: string[] = ['akashnet-2']): Promise<{
  keplr: KeplrWalletConnectV2
  session: unknown
}> {
  const client = await initSignClient()

  // Check for existing session
  const existingSessions = client.session.getAll()
  if (existingSessions.length > 0) {
    const session = existingSessions[existingSessions.length - 1]
    console.log('Using existing session:', session.topic)

    if (!keplrInstance) {
      keplrInstance = new KeplrWalletConnectV2(client, {
        sessionProperties: session.sessionProperties
      })
    }

    return { keplr: keplrInstance, session }
  }

  // Build chain namespaces
  const chainNamespaces = chainIds.map(id => `cosmos:${id}`)

  const { uri, approval } = await client.connect({
    requiredNamespaces: {
      cosmos: {
        methods: REQUIRED_METHODS,
        chains: chainNamespaces,
        events: REQUIRED_EVENTS,
      },
    },
    optionalNamespaces: {
      cosmos: {
        methods: OPTIONAL_METHODS,
        chains: chainNamespaces,
        events: OPTIONAL_EVENTS,
      },
    },
  })

  if (!uri) {
    throw new Error('Failed to generate WalletConnect URI')
  }

  console.log('WalletConnect URI generated')

  // Notify the UI to display the QR code
  if (qrUriCallback) {
    qrUriCallback(uri)
  }

  // Wait for wallet approval
  const session = await approval()
  console.log('Session approved')

  keplrInstance = new KeplrWalletConnectV2(client, {
    sessionProperties: session.sessionProperties
  })

  return { keplr: keplrInstance, session }
}

export async function disconnectKeplr(): Promise<void> {
  if (!signClient) return

  const sessions = signClient.session.getAll()
  for (const session of sessions) {
    try {
      await signClient.disconnect({
        topic: session.topic,
        reason: { code: 6000, message: 'User disconnected' }
      })
    } catch (e) {
      console.error('Error disconnecting session:', e)
    }
  }

  keplrInstance = null
}

export function getKeplrInstance(): KeplrWalletConnectV2 | null {
  return keplrInstance
}
