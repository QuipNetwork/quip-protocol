import { useState, useEffect, useMemo } from 'react'
import yaml from 'yaml'
import { useKeplr } from '../context/KeplrContext'
import { useMonitor } from '../context/MonitorContext'
import {
  fetchDeployments,
  fetchLeases,
  fetchBlockHeight,
  fetchProvider,
  fetchDeploymentDetails,
  fetchLeaseStatus,
  extractServiceUrls,
  closeDeployment,
  type Deployment,
  type Lease,
  type ForwardedPort
} from '../utils/akashApi'
import {
  getStatusMessage,
  getStatusColor
} from '../utils/deploymentMonitor'
import { formatAKT, formatDate, calculateUptime } from '../utils/format'
import { generateSDL, type SDLConfig } from '../utils/sdl'
import { DEFAULTS } from '../config/constants'

interface DeploymentWithLease extends Deployment {
  leases: Lease[]
  providerName?: string
  providerHostUri?: string
  forwardedPorts?: ForwardedPort[]
  serviceUrls?: string[]
}

export function DeploymentList() {
  const { address, isConnected, getSigningClient } = useKeplr()
  const {
    autoCloseEnabled,
    setAutoCloseEnabled,
    monitoredDeployments,
    startMonitoring,
    stopMonitoring,
    cancelAutoClose,
    isPolling
  } = useMonitor()

  const [deployments, setDeployments] = useState<DeploymentWithLease[]>([])
  const [currentBlock, setCurrentBlock] = useState(0)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [expandedSdlDseq, setExpandedSdlDseq] = useState<string | null>(null)
  const [copiedDseq, setCopiedDseq] = useState<string | null>(null)
  const [closingDseq, setClosingDseq] = useState<string | null>(null)
  const [closeSuccess, setCloseSuccess] = useState<string | null>(null)
  const [closingAll, setClosingAll] = useState(false)
  const [closeAllProgress, setCloseAllProgress] = useState<{ current: number; total: number } | null>(null)

  // Generate default SDL config for existing deployments
  const defaultSdlConfig: SDLConfig = useMemo(() => ({
    minerType: DEFAULTS.minerType,
    fleetSize: DEFAULTS.fleetSize,
    miningDuration: DEFAULTS.duration,
    difficultyEnergy: DEFAULTS.difficulty,
    minDiversity: DEFAULTS.diversity,
    minSolutions: DEFAULTS.minSolutions,
  }), [])

  // Generate SDL YAML for display
  const getSdlYaml = (config: SDLConfig): string => {
    const sdl = generateSDL(config)
    return yaml.stringify(sdl)
  }

  // Copy SDL to clipboard
  const copySdlToClipboard = async (dseq: string) => {
    const sdlYaml = getSdlYaml(defaultSdlConfig)
    try {
      await navigator.clipboard.writeText(sdlYaml)
      setCopiedDseq(dseq)
      setTimeout(() => setCopiedDseq(null), 2000)
    } catch (err) {
      console.error('Failed to copy SDL:', err)
    }
  }

  // Close deployment handler
  const handleCloseDeployment = async (dseq: string) => {
    if (!address) return

    // Confirm before closing
    const confirmed = window.confirm(
      `Are you sure you want to close deployment #${dseq}?\n\nThis will stop all containers and release resources. Any remaining balance will be returned to your account.`
    )
    if (!confirmed) return

    setClosingDseq(dseq)
    setError(null)
    setCloseSuccess(null)

    try {
      const client = await getSigningClient()
      if (!client) {
        throw new Error('Failed to get signing client')
      }

      await closeDeployment(client, address, dseq)
      setCloseSuccess(`Deployment #${dseq} closed successfully!`)

      // Refresh the list after a short delay
      setTimeout(() => {
        refresh()
        setCloseSuccess(null)
      }, 2000)
    } catch (err) {
      console.error('Failed to close deployment:', err)
      const message = err instanceof Error ? err.message : 'Unknown error'
      if (message.includes('rejected') || message.includes('cancelled')) {
        setError('Transaction cancelled')
      } else {
        setError(`Failed to close deployment: ${message}`)
      }
    } finally {
      setClosingDseq(null)
    }
  }

  // Close all active deployments handler
  const handleCloseAllDeployments = async () => {
    if (!address) return

    const activeDeployments = deployments.filter(d => d.deployment.state === 'active')
    if (activeDeployments.length === 0) {
      setError('No active deployments to close')
      return
    }

    // Confirm before closing all
    const confirmed = window.confirm(
      `Are you sure you want to close ALL ${activeDeployments.length} active deployment(s)?\n\nThis will stop all containers and release all resources. Remaining balances will be returned to your account.\n\nYou will need to approve ${activeDeployments.length} transaction(s) in your wallet.`
    )
    if (!confirmed) return

    setClosingAll(true)
    setCloseAllProgress({ current: 0, total: activeDeployments.length })
    setError(null)
    setCloseSuccess(null)

    const client = await getSigningClient()
    if (!client) {
      setError('Failed to get signing client')
      setClosingAll(false)
      setCloseAllProgress(null)
      return
    }

    let successCount = 0
    const errors: string[] = []

    for (let i = 0; i < activeDeployments.length; i++) {
      const deployment = activeDeployments[i]
      setCloseAllProgress({ current: i + 1, total: activeDeployments.length })

      try {
        await closeDeployment(client, address, deployment.deployment.id.dseq)
        successCount++
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Unknown error'
        if (message.includes('rejected') || message.includes('cancelled')) {
          // User cancelled - stop the batch
          errors.push(`Cancelled at deployment #${deployment.deployment.id.dseq}`)
          break
        }
        errors.push(`#${deployment.deployment.id.dseq}: ${message}`)
      }
    }

    setClosingAll(false)
    setCloseAllProgress(null)

    if (successCount > 0) {
      setCloseSuccess(`Successfully closed ${successCount} deployment(s)`)
    }
    if (errors.length > 0) {
      setError(`Failed to close some deployments:\n${errors.join('\n')}`)
    }

    // Refresh the list after a short delay
    setTimeout(() => {
      refresh()
      if (errors.length === 0) {
        setCloseSuccess(null)
      }
    }, 2000)
  }

  const refresh = async () => {
    if (!address) return

    setLoading(true)
    setError(null)

    try {
      // Fetch deployments and current block height in parallel
      const [data, blockHeight] = await Promise.all([
        fetchDeployments(address),
        fetchBlockHeight()
      ])

      setCurrentBlock(blockHeight)

      // Fetch leases and console details for each deployment
      const deploymentsWithLeases: DeploymentWithLease[] = await Promise.all(
        data.map(async (d) => {
          const leases = await fetchLeases(address, d.deployment.id.dseq)

          // Get provider name for active leases
          let providerName: string | undefined
          let providerHostUri: string | undefined
          let forwardedPorts: ForwardedPort[] | undefined
          const activeLease = leases.find(l => l.lease.state === 'active')

          if (activeLease) {
            const provider = await fetchProvider(activeLease.lease.id.provider)
            providerName = provider?.attributes?.find(a => a.key === 'organization')?.value
              || activeLease.lease.id.provider.slice(0, 12) + '...'
            providerHostUri = provider?.host_uri
          }

          // Fetch detailed deployment info from Console API for forwarded ports
          const consoleDetails = await fetchDeploymentDetails(address, d.deployment.id.dseq)
          console.log(`Console details for ${d.deployment.id.dseq}:`, consoleDetails)
          if (consoleDetails?.forwardedPorts) {
            forwardedPorts = consoleDetails.forwardedPorts
          }
          // Also try to get provider URI from console API if not found
          if (!providerHostUri && consoleDetails?.leases?.[0]?.provider?.hostUri) {
            providerHostUri = consoleDetails.leases[0].provider.hostUri
          }
          // Check if provider name is available from console API
          if (!providerName && consoleDetails?.leases?.[0]?.provider?.organization) {
            providerName = consoleDetails.leases[0].provider.organization
          }

          // Fetch lease status for service URLs (for active leases)
          let serviceUrls: string[] | undefined
          if (activeLease) {
            const leaseStatus = await fetchLeaseStatus(
              address,
              d.deployment.id.dseq,
              activeLease.lease.id.gseq,
              activeLease.lease.id.oseq
            )
            serviceUrls = extractServiceUrls(leaseStatus)
            if (serviceUrls.length > 0) {
              console.log(`Service URLs for ${d.deployment.id.dseq}:`, serviceUrls)
            }
          }

          return { ...d, leases, providerName, providerHostUri, forwardedPorts, serviceUrls }
        })
      )

      setDeployments(deploymentsWithLeases)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch deployments')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (isConnected && address) {
      refresh()
    }
  }, [isConnected, address])

  const getStatusClass = (state: string) => {
    switch (state.toLowerCase()) {
      case 'active':
        return 'status-active'
      case 'closed':
        return 'status-closed'
      default:
        return 'status-pending'
    }
  }

  if (!isConnected) {
    return (
      <div className="tab-content">
        <div className="alert alert-info">
          <strong>Active Deployments:</strong> Connect wallet to view deployments
        </div>
        <p className="empty-state">Connect wallet to view deployments</p>
      </div>
    )
  }

  return (
    <div className="tab-content">
      <div className="alert alert-info">
        <strong>Active Deployments:</strong> View and manage your running deployments
      </div>

      <div style={{ display: 'flex', gap: '12px', marginBottom: '16px', flexWrap: 'wrap', alignItems: 'center' }}>
        <button className="btn btn-secondary" onClick={refresh} disabled={loading || closingAll}>
          {loading ? 'Refreshing...' : 'Refresh Deployments'}
          {isPolling && <span style={{ marginLeft: '8px', opacity: 0.7 }}>...</span>}
        </button>

        {deployments.filter(d => d.deployment.state === 'active').length > 0 && (
          <button
            className="btn btn-danger"
            onClick={handleCloseAllDeployments}
            disabled={loading || closingAll || !!closingDseq}
            style={{ minWidth: '180px' }}
          >
            {closingAll
              ? `Closing ${closeAllProgress?.current}/${closeAllProgress?.total}...`
              : `Close All (${deployments.filter(d => d.deployment.state === 'active').length})`
            }
          </button>
        )}

        {/* Auto-close toggle */}
        <label style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          cursor: 'pointer',
          padding: '8px 12px',
          background: autoCloseEnabled ? 'rgba(102, 126, 234, 0.15)' : 'rgba(113, 128, 150, 0.1)',
          borderRadius: '6px',
          border: autoCloseEnabled ? '1px solid rgba(102, 126, 234, 0.3)' : '1px solid transparent',
          transition: 'all 0.2s'
        }}>
          <input
            type="checkbox"
            checked={autoCloseEnabled}
            onChange={(e) => setAutoCloseEnabled(e.target.checked)}
          />
          <span style={{ fontWeight: autoCloseEnabled ? 600 : 400, fontSize: '14px' }}>
            Auto-Close Exited
          </span>
          {autoCloseEnabled && (
            <span style={{
              background: '#667eea',
              color: 'white',
              fontSize: '10px',
              padding: '2px 6px',
              borderRadius: '4px',
              fontWeight: 600
            }}>
              ON
            </span>
          )}
        </label>
      </div>

      {autoCloseEnabled && (
        <div className="alert alert-info" style={{ marginBottom: '16px' }}>
          <strong>Auto-Close Enabled:</strong> Deployments will automatically close 60 seconds after containers exit.
          {monitoredDeployments.size > 0 && (
            <span style={{ marginLeft: '8px' }}>
              Monitoring {monitoredDeployments.size} deployment(s).
            </span>
          )}
        </div>
      )}

      {error && (
        <div className="alert alert-error">
          {error}
        </div>
      )}

      {closeSuccess && (
        <div className="alert alert-success">
          {closeSuccess}
        </div>
      )}

      <div className="deployment-list">
        {deployments.length === 0 ? (
          <p className="empty-state">No active deployments found</p>
        ) : (
          deployments.map((d) => {
            // Get balance from v1beta4 escrow_account.state.funds array
            const funds = d.escrow_account?.state?.funds?.[0]
            const balance = funds?.amount || '0'

            // Find active lease
            const activeLease = d.leases.find(l => l.lease.state === 'active')
            const hasActiveLease = !!activeLease
            const leaseCreatedAt = activeLease?.lease.created_at

            // Get lease status
            const getLeaseStatus = () => {
              if (d.leases.length === 0) return { text: 'No bids accepted', class: 'status-pending' }
              if (activeLease) return { text: 'Running', class: 'status-active' }
              const closedLease = d.leases.find(l => l.lease.state === 'closed')
              if (closedLease) return { text: 'Lease closed', class: 'status-closed' }
              return { text: 'Pending', class: 'status-pending' }
            }
            const leaseStatus = getLeaseStatus()

            // Get monitoring info if available
            const monitorInfo = monitoredDeployments.get(d.deployment.id.dseq)
            const isMonitored = !!monitorInfo
            const hasAutoCloseCountdown = monitorInfo?.autoCloseCountdown !== null && monitorInfo?.autoCloseCountdown !== undefined

            return (
              <div key={d.deployment.id.dseq} className="deployment-item">
                {/* Auto-close countdown banner */}
                {hasAutoCloseCountdown && (
                  <div style={{
                    background: 'linear-gradient(135deg, #f6ad55 0%, #fc8181 100%)',
                    color: 'white',
                    padding: '12px',
                    borderRadius: '6px',
                    marginBottom: '12px',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center'
                  }}>
                    <div>
                      <strong>Auto-closing in {monitorInfo?.autoCloseCountdown}s</strong>
                      <span style={{ marginLeft: '8px', opacity: 0.9 }}>
                        Container exited - will close to recover escrow
                      </span>
                    </div>
                    <button
                      onClick={() => cancelAutoClose(d.deployment.id.dseq)}
                      style={{
                        background: 'rgba(255,255,255,0.2)',
                        color: 'white',
                        border: '1px solid rgba(255,255,255,0.4)',
                        padding: '4px 12px',
                        borderRadius: '4px',
                        cursor: 'pointer',
                        fontWeight: 500,
                        fontSize: '13px'
                      }}
                    >
                      Cancel
                    </button>
                  </div>
                )}

                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <h3>Deployment #{d.deployment.id.dseq}</h3>

                  {/* Monitor toggle for active deployments */}
                  {d.deployment.state === 'active' && autoCloseEnabled && (
                    <label style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px',
                      cursor: 'pointer',
                      fontSize: '12px',
                      padding: '4px 8px',
                      background: isMonitored ? 'rgba(72, 187, 120, 0.15)' : 'transparent',
                      borderRadius: '4px'
                    }}>
                      <input
                        type="checkbox"
                        checked={isMonitored}
                        onChange={(e) => {
                          if (e.target.checked) {
                            startMonitoring(d.deployment.id.dseq)
                          } else {
                            stopMonitoring(d.deployment.id.dseq)
                          }
                        }}
                      />
                      <span>Monitor</span>
                    </label>
                  )}
                </div>

                {/* Container status from monitor */}
                {isMonitored && monitorInfo?.status && (
                  <p style={{ marginBottom: '8px' }}>
                    <strong>Container Status:</strong>{' '}
                    <span style={{ color: getStatusColor(monitorInfo.status) }}>
                      {getStatusMessage(monitorInfo.status)}
                    </span>
                    {monitorInfo.status.lastChecked && (
                      <span style={{ marginLeft: '8px', fontSize: '11px', color: '#718096' }}>
                        (checked {Math.round((Date.now() - monitorInfo.status.lastChecked.getTime()) / 1000)}s ago)
                      </span>
                    )}
                  </p>
                )}

                <p>
                  <strong>Deployment State:</strong>{' '}
                  <span className={`deployment-status ${getStatusClass(d.deployment.state)}`}>
                    {d.deployment.state}
                  </span>
                </p>
                <p>
                  <strong>Lease Status:</strong>{' '}
                  <span className={`deployment-status ${leaseStatus.class}`}>
                    {leaseStatus.text}
                  </span>
                </p>
                {hasActiveLease && d.providerName && (
                  <p>
                    <strong>Provider:</strong> {d.providerName}
                  </p>
                )}
                {hasActiveLease && d.providerHostUri && (
                  <p>
                    <strong>Provider URI:</strong>{' '}
                    <span className="address" style={{ fontSize: '12px' }}>
                      {d.providerHostUri}
                    </span>
                  </p>
                )}
                {hasActiveLease && d.serviceUrls && d.serviceUrls.length > 0 && (
                  <div className="forwarded-ports">
                    <strong>Service URLs:</strong>
                    <ul style={{ margin: '8px 0 0 0', paddingLeft: '20px', listStyle: 'none' }}>
                      {d.serviceUrls.map((url, idx) => (
                        <li key={idx} style={{ marginBottom: '4px' }}>
                          <a
                            href={url.startsWith('http') ? url : `http://${url}`}
                            target="_blank"
                            rel="noopener noreferrer"
                            style={{ color: '#667eea', textDecoration: 'none' }}
                          >
                            {url}
                          </a>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {hasActiveLease && d.forwardedPorts && d.forwardedPorts.length > 0 && !d.serviceUrls?.length && (
                  <div className="forwarded-ports">
                    <strong>Forwarded Ports:</strong>
                    <ul style={{ margin: '8px 0 0 0', paddingLeft: '20px', listStyle: 'none' }}>
                      {d.forwardedPorts.map((port, idx) => {
                        const url = `http://${port.host}:${port.externalPort}`
                        return (
                          <li key={idx} style={{ marginBottom: '4px' }}>
                            <a
                              href={url}
                              target="_blank"
                              rel="noopener noreferrer"
                              style={{ color: '#667eea', textDecoration: 'none' }}
                            >
                              {url}
                            </a>
                            <span style={{ color: '#a0aec0', marginLeft: '8px', fontSize: '12px' }}>
                              ({port.name || 'service'} - port {port.port})
                            </span>
                          </li>
                        )
                      })}
                    </ul>
                  </div>
                )}
                {hasActiveLease && leaseCreatedAt && currentBlock > 0 && (
                  <p>
                    <strong>Uptime:</strong> {calculateUptime(leaseCreatedAt, currentBlock)}
                  </p>
                )}
                <p>
                  <strong>Created:</strong> {formatDate(d.deployment.created_at)}
                </p>
                <p>
                  <strong>Balance:</strong> {formatAKT(balance)}
                </p>
                {activeLease && (
                  <p>
                    <strong>Cost:</strong> {activeLease.lease.price.amount} uakt/block
                  </p>
                )}
                {d.groups.length > 0 && (
                  <p>
                    <strong>Groups:</strong> {d.groups.length}
                  </p>
                )}
                <div className="deployment-actions" style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                  <a
                    href={`https://stats.akash.network/addresses/${d.deployment.id.owner}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="btn btn-secondary"
                    style={{ fontSize: '14px', padding: '8px 16px', width: 'auto' }}
                  >
                    Akash Stats
                  </a>
                  <button
                    className="btn btn-primary"
                    style={{ fontSize: '14px', padding: '8px 16px', width: 'auto' }}
                    onClick={() => setExpandedSdlDseq(expandedSdlDseq === d.deployment.id.dseq ? null : d.deployment.id.dseq)}
                  >
                    {expandedSdlDseq === d.deployment.id.dseq ? 'Hide SDL' : 'Show SDL'}
                  </button>
                  <button
                    className="btn btn-secondary"
                    style={{ fontSize: '14px', padding: '8px 16px', width: 'auto', background: copiedDseq === d.deployment.id.dseq ? '#48bb78' : '#667eea' }}
                    onClick={() => copySdlToClipboard(d.deployment.id.dseq)}
                  >
                    {copiedDseq === d.deployment.id.dseq ? 'Copied!' : 'Copy SDL'}
                  </button>
                  <a
                    href="https://console.akash.network/deploy"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="btn"
                    style={{ fontSize: '14px', padding: '8px 16px', width: 'auto', background: '#764ba2' }}
                  >
                    Open Console
                  </a>
                  {d.deployment.state === 'active' && (
                    <button
                      className="btn btn-danger"
                      style={{ fontSize: '14px', padding: '8px 16px', width: 'auto' }}
                      onClick={(e) => {
                        e.stopPropagation()
                        handleCloseDeployment(d.deployment.id.dseq)
                      }}
                      disabled={closingDseq === d.deployment.id.dseq}
                    >
                      {closingDseq === d.deployment.id.dseq ? 'Closing...' : 'Close Deployment'}
                    </button>
                  )}
                </div>
                {expandedSdlDseq === d.deployment.id.dseq && (
                  <div style={{ marginTop: '12px', background: '#1a202c', borderRadius: '6px', padding: '12px', overflow: 'auto' }}>
                    <div style={{ marginBottom: '8px', fontSize: '12px', color: '#a0aec0' }}>
                      Default QUIP Miner SDL (CPU, {DEFAULTS.fleetSize} instances, {DEFAULTS.duration}):
                    </div>
                    <pre style={{ margin: 0, fontSize: '12px', color: '#e2e8f0', whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                      {getSdlYaml(defaultSdlConfig)}
                    </pre>
                  </div>
                )}
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}
