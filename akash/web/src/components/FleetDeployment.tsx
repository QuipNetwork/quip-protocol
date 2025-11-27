/**
 * Fleet Deployment Component
 *
 * Flow:
 * 1. Check network capacity (ensure providers exist)
 * 2. Show capacity overview and confirmation
 * 3. Create adaptive deployments - start with full request, reduce if no bids
 * 4. Show progress and results
 *
 * Adaptive strategy:
 * - For CPU: Start with count=N instances on one provider
 * - For GPU: Start with N GPUs on one provider
 * - If no bids, halve request and retry
 * - Continue until all resources allocated or minimum reached
 */

import { useState, useEffect, useCallback } from 'react'
import { useKeplr } from '../context/KeplrContext'
import {
  checkNetworkCapacity,
  deployFleetAutomatic,
  cancelFleet,
  getFleetSummary,
  type CapacityCheck,
  type FleetState,
  type FleetDeploymentStatus
} from '../utils/fleetManager'
import { estimateFleetCostAKT, type SDLConfig } from '../utils/sdl'

interface FleetDeploymentProps {
  config: SDLConfig
  onComplete?: (state: FleetState) => void
  onCancel?: () => void
}

type FleetPhase = 'loading' | 'capacity-check' | 'confirming' | 'deploying' | 'completed'

export function FleetDeployment({ config, onComplete, onCancel }: FleetDeploymentProps) {
  const { address, getSigningClient, hasCertificateOnChain, createAndPublishCertificate } = useKeplr()

  // Phase state
  const [phase, setPhase] = useState<FleetPhase>('loading')

  // Capacity check
  const [capacityCheck, setCapacityCheck] = useState<CapacityCheck | null>(null)
  const [capacityError, setCapacityError] = useState<string | null>(null)

  // Deployment state
  const [fleetState, setFleetState] = useState<FleetState | null>(null)

  // Error state
  const [error, setError] = useState<string | null>(null)

  // Determine resource type
  const isGpuFleet = config.minerType === 'cuda'
  const targetResources = config.fleetSize

  // Check capacity on mount
  useEffect(() => {
    checkCapacity()
  }, [])

  const checkCapacity = async () => {
    setPhase('loading')
    setCapacityError(null)

    try {
      const check = await checkNetworkCapacity(
        isGpuFleet ? 0 : targetResources,
        isGpuFleet ? targetResources : 0,
        {
          minProvidersRequired: 5,
          capacityMultiplier: 1.5
        }
      )
      setCapacityCheck(check)
      setPhase('capacity-check')
    } catch (err) {
      setCapacityError(err instanceof Error ? err.message : 'Failed to check network capacity')
      setPhase('capacity-check')
    }
  }

  // Start deployment
  const startDeployment = useCallback(async () => {
    if (!capacityCheck || !address) return

    setError(null)
    setPhase('deploying')

    try {
      // Ensure certificate exists
      if (!hasCertificateOnChain) {
        const certSuccess = await createAndPublishCertificate()
        if (!certSuccess) {
          throw new Error('Failed to create certificate')
        }
      }

      const client = await getSigningClient()
      if (!client) {
        throw new Error('Failed to get signing client')
      }

      const finalState = await deployFleetAutomatic(
        client,
        address,
        config,
        capacityCheck,
        (state) => setFleetState(state)
      )

      setFleetState(finalState)
      setPhase('completed')
      onComplete?.(finalState)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Deployment failed')
      setPhase('confirming')
    }
  }, [capacityCheck, address, config, hasCertificateOnChain, createAndPublishCertificate, getSigningClient, onComplete])

  // Cancel deployment
  const handleCancel = () => {
    if (fleetState && phase === 'deploying') {
      const cancelled = cancelFleet(fleetState)
      setFleetState(cancelled)
      setPhase('completed')
    } else {
      onCancel?.()
    }
  }

  // Parse duration for cost estimate
  const parseDuration = (duration: string): number => {
    const match = duration.match(/^(\d+)(m|h|d)$/)
    if (!match) return 90
    const value = parseInt(match[1], 10)
    const unit = match[2]
    switch (unit) {
      case 'm': return value
      case 'h': return value * 60
      case 'd': return value * 60 * 24
      default: return 90
    }
  }

  // Cost estimate
  const costEstimate = capacityCheck
    ? estimateFleetCostAKT(
        config.minerType,
        isGpuFleet ? capacityCheck.requestedGpu : capacityCheck.requestedCpu,
        parseDuration(config.miningDuration),
        capacityCheck.deploymentsNeeded
      )
    : null

  return (
    <div className="fleet-deployment">
      {/* Header */}
      <div className="fleet-header">
        <h2>
          Fleet Deployment: {targetResources} {isGpuFleet ? 'GPU' : 'CPU'}
          {targetResources > 1 ? 's' : ''}
        </h2>
        <p className="fleet-subtitle">
          Duration: {config.miningDuration} | Difficulty: {config.difficultyEnergy}
        </p>
      </div>

      {/* Error display */}
      {(error || capacityError) && (
        <div className="alert alert-error">
          {error || capacityError}
        </div>
      )}

      {/* Loading state */}
      {phase === 'loading' && (
        <div className="fleet-loading">
          <p>Checking network capacity...</p>
        </div>
      )}

      {/* Capacity check phase */}
      {phase === 'capacity-check' && capacityCheck && (
        <div className="fleet-planning">
          <div className="capacity-overview">
            <h3>Network Capacity</h3>

            {/* Capacity status */}
            {capacityCheck.canFill ? (
              <div className="alert alert-success">
                Network has sufficient capacity ({capacityCheck.capacityRatio.toFixed(1)}x your request)
              </div>
            ) : (
              <div className="alert alert-info">
                <strong>Warning:</strong> Network capacity is limited.
                {capacityCheck.providerCount < 5 && ` Only ${capacityCheck.providerCount} providers available.`}
                {capacityCheck.capacityRatio < 1.5 && ` Capacity ratio: ${capacityCheck.capacityRatio.toFixed(1)}x.`}
                <br />
                Some deployments may not receive bids.
              </div>
            )}

            <div className="capacity-stats">
              <div className="stat">
                <span className="stat-value">{capacityCheck.providerCount}</span>
                <span className="stat-label">Providers Online</span>
              </div>
              <div className="stat">
                <span className="stat-value">{capacityCheck.totalCpu}</span>
                <span className="stat-label">Total CPUs Available</span>
              </div>
              <div className="stat">
                <span className="stat-value">{capacityCheck.totalGpu}</span>
                <span className="stat-label">Total GPUs Available</span>
              </div>
            </div>

            {/* Top providers that could bid */}
            <div className="top-providers">
              <h4>Top Providers (may bid on your deployments)</h4>
              <div className="provider-list">
                {capacityCheck.topProviders.slice(0, 5).map((p) => (
                  <div key={p.address} className="provider-item">
                    <span className="provider-name">
                      {p.isReliable && <span className="reliable-badge">✓</span>}
                      {p.name}
                    </span>
                    <span className="provider-capacity">
                      {Math.floor(p.availableCpu / 1000)} CPU
                      {p.availableGpu > 0 && ` | ${p.availableGpu} GPU`}
                    </span>
                  </div>
                ))}
              </div>
              <small style={{ color: '#718096', marginTop: '8px', display: 'block' }}>
                Note: Any online provider can bid. Lowest price bids will be auto-selected.
              </small>
            </div>
          </div>

          <div className="fleet-actions">
            <button
              className="btn btn-primary"
              onClick={() => setPhase('confirming')}
              disabled={!capacityCheck.canFill && capacityCheck.providerCount === 0}
            >
              Continue to Deploy
            </button>
            <button className="btn btn-secondary" onClick={onCancel}>
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Confirming phase */}
      {phase === 'confirming' && capacityCheck && (
        <div className="fleet-confirming">
          <div className="plan-summary">
            <h3>Deployment Plan</h3>

            <div className="alert alert-info">
              {isGpuFleet ? (
                <>
                  Will request <strong>{capacityCheck.requestedGpu} GPUs</strong> from providers.
                  <br />
                  Starting with full request, adapting if providers can't fulfill.
                </>
              ) : (
                <>
                  Will deploy <strong>1 container with {capacityCheck.requestedCpu} CPUs</strong>.
                  <br />
                  Miner auto-parallelizes across all available cores.
                </>
              )}
            </div>

            <div className="plan-stats">
              <div className="stat">
                <span className="stat-value">
                  {isGpuFleet ? capacityCheck.requestedGpu : capacityCheck.requestedCpu}
                </span>
                <span className="stat-label">{isGpuFleet ? 'GPUs' : 'CPUs'} Total</span>
              </div>
              <div className="stat">
                <span className="stat-value">{capacityCheck.initialRequestSize}</span>
                <span className="stat-label">Initial Request</span>
              </div>
              {costEstimate && (
                <div className="stat">
                  <span className="stat-value">{costEstimate.totalCost.toFixed(2)}</span>
                  <span className="stat-label">Est. Cost (AKT)</span>
                </div>
              )}
            </div>

            <div style={{ marginTop: '16px', padding: '12px', background: '#2d3748', borderRadius: '6px' }}>
              <h4 style={{ margin: '0 0 8px 0', fontSize: '14px' }}>Adaptive Deployment Strategy</h4>
              <ol style={{ margin: 0, paddingLeft: '20px', fontSize: '13px', color: '#a0aec0' }}>
                <li>Request all {isGpuFleet ? 'GPUs' : 'CPUs'} from providers</li>
                <li>Wait for bids from providers who can fulfill</li>
                <li>If no bids, reduce request size and retry</li>
                <li>Auto-select lowest price bid</li>
                <li>Repeat until all resources allocated</li>
              </ol>
              <p style={{ margin: '8px 0 0 0', fontSize: '12px', color: '#718096' }}>
                {isGpuFleet
                  ? 'Example: 10 GPUs → Try 10, then 5, then 2, then 1 until fulfilled'
                  : 'Example: 10 CPUs → Try 10 CPUs, then 5, then 2, then 1 per deployment'}
              </p>
            </div>
          </div>

          <div className="fleet-actions">
            <button className="btn btn-primary" onClick={startDeployment}>
              Start Adaptive Deploy
            </button>
            <button className="btn btn-secondary" onClick={() => setPhase('capacity-check')}>
              Back
            </button>
            <button className="btn btn-secondary" onClick={onCancel}>
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Deploying phase */}
      {phase === 'deploying' && fleetState && (
        <div className="fleet-deploying">
          <div className="progress-header">
            <h3>Deploying Fleet...</h3>
            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{
                  width: `${(fleetState.stats.totalAllocated / fleetState.stats.totalRequested) * 100}%`
                }}
              />
            </div>
            <p className="progress-text">
              {fleetState.stats.totalAllocated} / {fleetState.stats.totalRequested} {isGpuFleet ? 'GPUs' : 'CPUs'} allocated
              {fleetState.remainingResources > 0 && ` (requesting ${fleetState.currentRequestSize})`}
            </p>
            {fleetState.stats.deploymentsFailed > 0 && (
              <p className="progress-text" style={{ color: '#fc8181' }}>
                {fleetState.stats.deploymentsFailed} deployment(s) failed - adapting...
              </p>
            )}
          </div>

          <div className="deployment-grid">
            {fleetState.deployments.map((d) => (
              <DeploymentCard key={d.index} deployment={d} isGpu={isGpuFleet} />
            ))}
          </div>

          <div className="fleet-actions">
            <button className="btn btn-danger" onClick={handleCancel}>
              Cancel Remaining
            </button>
          </div>
        </div>
      )}

      {/* Completed phase */}
      {phase === 'completed' && fleetState && (
        <div className="fleet-completed">
          <div className="summary-header">
            <h3>
              {fleetState.status === 'completed' ? '✓ Fleet Deployed' : '✗ Fleet Cancelled'}
            </h3>
          </div>

          {(() => {
            const summary = getFleetSummary(fleetState)
            return (
              <div className="summary-stats">
                <div className="stat stat-success">
                  <span className="stat-value">{summary.activeDeployments}</span>
                  <span className="stat-label">Active</span>
                </div>
                <div className="stat stat-error">
                  <span className="stat-value">{summary.failedDeployments}</span>
                  <span className="stat-label">Failed</span>
                </div>
                <div className="stat">
                  <span className="stat-value">
                    {isGpuFleet ? summary.totalGpuDeployed : summary.totalCpuDeployed}
                  </span>
                  <span className="stat-label">{isGpuFleet ? 'GPUs' : 'CPUs'} Deployed</span>
                </div>
              </div>
            )
          })()}

          <div className="deployment-grid">
            {fleetState.deployments.map((d) => (
              <DeploymentCard key={d.index} deployment={d} isGpu={isGpuFleet} />
            ))}
          </div>

          <div className="fleet-actions">
            <button className="btn btn-primary" onClick={onCancel}>
              Done
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

// ============================================================================
// Sub-components
// ============================================================================

interface DeploymentCardProps {
  deployment: {
    index: number
    status: FleetDeploymentStatus
    requestedCpu: number
    requestedGpu: number
    instanceCount: number
    dseq?: string
    selectedProviderName?: string
    error?: string
  }
  isGpu: boolean
}

function DeploymentCard({ deployment, isGpu }: DeploymentCardProps) {
  const statusConfig: Record<FleetDeploymentStatus, { icon: string; color: string }> = {
    'pending': { icon: '○', color: '#a0aec0' },
    'creating': { icon: '◐', color: '#667eea' },
    'waiting-bids': { icon: '◑', color: '#667eea' },
    'selecting-bid': { icon: '◑', color: '#667eea' },
    'accepting-bid': { icon: '◕', color: '#667eea' },
    'sending-manifest': { icon: '◕', color: '#667eea' },
    'active': { icon: '●', color: '#48bb78' },
    'failed': { icon: '✗', color: '#fc8181' }
  }

  const { icon, color } = statusConfig[deployment.status]

  // For CPU: show CPU count, for GPU: show GPU count
  const resourceDisplay = isGpu
    ? `${deployment.requestedGpu} GPU${deployment.requestedGpu > 1 ? 's' : ''}`
    : `${deployment.requestedCpu} CPU${deployment.requestedCpu > 1 ? 's' : ''}`

  return (
    <div className={`deployment-card status-${deployment.status}`}>
      <div className="card-header">
        <span className="card-index">#{deployment.index + 1}</span>
        <span className="card-status" style={{ color }}>
          {icon} {deployment.status.replace('-', ' ')}
        </span>
      </div>
      <div className="card-body">
        <div className="card-resources">
          {resourceDisplay}
        </div>
        {deployment.selectedProviderName && (
          <div className="card-provider">{deployment.selectedProviderName}</div>
        )}
        {deployment.dseq && (
          <div className="card-dseq">DSEQ: {deployment.dseq}</div>
        )}
        {deployment.error && (
          <div className="card-error">{deployment.error}</div>
        )}
      </div>
    </div>
  )
}
