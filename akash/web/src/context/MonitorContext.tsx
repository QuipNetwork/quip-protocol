/**
 * Monitor Context
 *
 * Provides deployment monitoring with automatic status polling
 * and optional auto-close when containers exit.
 */

import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  useRef,
  type ReactNode
} from 'react'
import { useKeplr } from './KeplrContext'
import {
  checkDeploymentStatus,
  shouldAutoClose,
  type DeploymentStatus
} from '../utils/deploymentMonitor'
import { closeDeployment } from '../utils/akashApi'

// Default polling interval in milliseconds
const DEFAULT_POLL_INTERVAL = 30000 // 30 seconds

// Auto-close countdown in seconds
const AUTO_CLOSE_COUNTDOWN = 60

/**
 * Monitored deployment with auto-close state
 */
interface MonitoredDeployment {
  dseq: string
  status: DeploymentStatus
  autoCloseCountdown: number | null // seconds remaining, null = not closing
  closePending: boolean
  closeError?: string
}

/**
 * Monitor context value
 */
interface MonitorContextValue {
  // Settings
  autoCloseEnabled: boolean
  setAutoCloseEnabled: (enabled: boolean) => void
  pollingInterval: number
  setPollingInterval: (ms: number) => void

  // State
  monitoredDeployments: Map<string, MonitoredDeployment>
  isPolling: boolean

  // Actions
  startMonitoring: (dseq: string) => void
  stopMonitoring: (dseq: string) => void
  checkNow: (dseq?: string) => Promise<void>
  cancelAutoClose: (dseq: string) => void
  closeNow: (dseq: string) => Promise<void>
}

const MonitorContext = createContext<MonitorContextValue | null>(null)

/**
 * Monitor provider component
 */
export function MonitorProvider({ children }: { children: ReactNode }) {
  const { address, getSigningClient } = useKeplr()

  // Settings
  const [autoCloseEnabled, setAutoCloseEnabled] = useState(() => {
    return localStorage.getItem('auto_close_enabled') === 'true'
  })
  const [pollingInterval, setPollingInterval] = useState(DEFAULT_POLL_INTERVAL)

  // State
  const [monitoredDeployments, setMonitoredDeployments] = useState<Map<string, MonitoredDeployment>>(new Map())
  const [isPolling, setIsPolling] = useState(false)

  // Refs for intervals
  const pollingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const countdownIntervalsRef = useRef<Map<string, ReturnType<typeof setInterval>>>(new Map())

  // Persist auto-close setting
  useEffect(() => {
    localStorage.setItem('auto_close_enabled', autoCloseEnabled ? 'true' : 'false')
  }, [autoCloseEnabled])

  // Check status for a single deployment
  const checkDeployment = useCallback(async (dseq: string): Promise<DeploymentStatus | null> => {
    if (!address) return null

    try {
      const status = await checkDeploymentStatus(address, dseq)
      return status
    } catch (error) {
      console.error(`Error checking deployment ${dseq}:`, error)
      return null
    }
  }, [address])

  // Start monitoring a deployment
  const startMonitoring = useCallback((dseq: string) => {
    setMonitoredDeployments(prev => {
      const newMap = new Map(prev)
      if (!newMap.has(dseq)) {
        newMap.set(dseq, {
          dseq,
          status: {
            dseq,
            owner: address || '',
            deploymentState: 'unknown',
            leaseState: 'unknown',
            containerState: 'unknown',
            isComplete: false,
            isError: false,
            lastChecked: new Date()
          },
          autoCloseCountdown: null,
          closePending: false
        })
      }
      return newMap
    })
  }, [address])

  // Stop monitoring a deployment
  const stopMonitoring = useCallback((dseq: string) => {
    setMonitoredDeployments(prev => {
      const newMap = new Map(prev)
      newMap.delete(dseq)
      return newMap
    })

    // Clear any countdown interval
    const countdownInterval = countdownIntervalsRef.current.get(dseq)
    if (countdownInterval) {
      clearInterval(countdownInterval)
      countdownIntervalsRef.current.delete(dseq)
    }
  }, [])

  // Cancel auto-close for a deployment
  const cancelAutoClose = useCallback((dseq: string) => {
    // Clear countdown interval
    const countdownInterval = countdownIntervalsRef.current.get(dseq)
    if (countdownInterval) {
      clearInterval(countdownInterval)
      countdownIntervalsRef.current.delete(dseq)
    }

    // Reset countdown state
    setMonitoredDeployments(prev => {
      const newMap = new Map(prev)
      const deployment = newMap.get(dseq)
      if (deployment) {
        newMap.set(dseq, { ...deployment, autoCloseCountdown: null })
      }
      return newMap
    })
  }, [])

  // Close a deployment immediately
  const closeNow = useCallback(async (dseq: string) => {
    if (!address) return

    // Cancel any pending auto-close
    cancelAutoClose(dseq)

    // Mark as pending
    setMonitoredDeployments(prev => {
      const newMap = new Map(prev)
      const deployment = newMap.get(dseq)
      if (deployment) {
        newMap.set(dseq, { ...deployment, closePending: true, closeError: undefined })
      }
      return newMap
    })

    try {
      const client = await getSigningClient()
      if (!client) {
        throw new Error('Failed to get signing client')
      }

      await closeDeployment(client, address, dseq)

      // Update status after close
      setMonitoredDeployments(prev => {
        const newMap = new Map(prev)
        const deployment = newMap.get(dseq)
        if (deployment) {
          newMap.set(dseq, {
            ...deployment,
            closePending: false,
            status: {
              ...deployment.status,
              deploymentState: 'closed',
              leaseState: 'closed',
              containerState: 'exited',
              isComplete: true
            }
          })
        }
        return newMap
      })
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error'
      setMonitoredDeployments(prev => {
        const newMap = new Map(prev)
        const deployment = newMap.get(dseq)
        if (deployment) {
          newMap.set(dseq, { ...deployment, closePending: false, closeError: message })
        }
        return newMap
      })
    }
  }, [address, getSigningClient, cancelAutoClose])

  // Start auto-close countdown for a deployment
  const startAutoCloseCountdown = useCallback((dseq: string) => {
    // Don't start if already closing
    const existing = monitoredDeployments.get(dseq)
    if (existing?.autoCloseCountdown !== null || existing?.closePending) {
      return
    }

    // Set initial countdown
    setMonitoredDeployments(prev => {
      const newMap = new Map(prev)
      const deployment = newMap.get(dseq)
      if (deployment) {
        newMap.set(dseq, { ...deployment, autoCloseCountdown: AUTO_CLOSE_COUNTDOWN })
      }
      return newMap
    })

    // Start countdown interval
    const interval = setInterval(() => {
      setMonitoredDeployments(prev => {
        const newMap = new Map(prev)
        const deployment = newMap.get(dseq)
        if (deployment && deployment.autoCloseCountdown !== null) {
          const newCountdown = deployment.autoCloseCountdown - 1
          if (newCountdown <= 0) {
            // Countdown complete - trigger close
            clearInterval(interval)
            countdownIntervalsRef.current.delete(dseq)
            // Close will be triggered by effect
            newMap.set(dseq, { ...deployment, autoCloseCountdown: 0 })
          } else {
            newMap.set(dseq, { ...deployment, autoCloseCountdown: newCountdown })
          }
        }
        return newMap
      })
    }, 1000)

    countdownIntervalsRef.current.set(dseq, interval)
  }, [monitoredDeployments])

  // Check all monitored deployments
  const checkNow = useCallback(async (specificDseq?: string) => {
    if (!address) return

    const dseqsToCheck = specificDseq
      ? [specificDseq]
      : Array.from(monitoredDeployments.keys())

    setIsPolling(true)

    try {
      const results = await Promise.all(
        dseqsToCheck.map(async dseq => {
          const status = await checkDeployment(dseq)
          return { dseq, status }
        })
      )

      setMonitoredDeployments(prev => {
        const newMap = new Map(prev)
        for (const { dseq, status } of results) {
          if (status) {
            const existing = newMap.get(dseq)
            if (existing) {
              newMap.set(dseq, { ...existing, status })

              // Check if should auto-close
              if (autoCloseEnabled && shouldAutoClose(status) && existing.autoCloseCountdown === null && !existing.closePending) {
                // Will trigger countdown via effect
                setTimeout(() => startAutoCloseCountdown(dseq), 0)
              }
            }
          }
        }
        return newMap
      })
    } finally {
      setIsPolling(false)
    }
  }, [address, monitoredDeployments, checkDeployment, autoCloseEnabled, startAutoCloseCountdown])

  // Effect to close deployment when countdown reaches 0
  useEffect(() => {
    for (const [dseq, deployment] of monitoredDeployments) {
      if (deployment.autoCloseCountdown === 0 && !deployment.closePending) {
        closeNow(dseq)
      }
    }
  }, [monitoredDeployments, closeNow])

  // Start/stop polling based on monitored deployments
  useEffect(() => {
    if (monitoredDeployments.size === 0) {
      // No deployments to monitor - stop polling
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current)
        pollingIntervalRef.current = null
      }
      return
    }

    // Start polling
    checkNow() // Initial check

    pollingIntervalRef.current = setInterval(() => {
      checkNow()
    }, pollingInterval)

    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current)
        pollingIntervalRef.current = null
      }
    }
  }, [monitoredDeployments.size, pollingInterval]) // Note: not including checkNow to avoid re-creating interval

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current)
      }
      for (const interval of countdownIntervalsRef.current.values()) {
        clearInterval(interval)
      }
    }
  }, [])

  const value: MonitorContextValue = {
    autoCloseEnabled,
    setAutoCloseEnabled,
    pollingInterval,
    setPollingInterval,
    monitoredDeployments,
    isPolling,
    startMonitoring,
    stopMonitoring,
    checkNow,
    cancelAutoClose,
    closeNow
  }

  return (
    <MonitorContext.Provider value={value}>
      {children}
    </MonitorContext.Provider>
  )
}

/**
 * Hook to use monitor context
 */
export function useMonitor(): MonitorContextValue {
  const context = useContext(MonitorContext)
  if (!context) {
    throw new Error('useMonitor must be used within a MonitorProvider')
  }
  return context
}
