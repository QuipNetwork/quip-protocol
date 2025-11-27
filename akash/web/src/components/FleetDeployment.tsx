/**
 * Fleet Deployment Component
 *
 * Shows:
 * 1. Provider capacity overview (before deploying)
 * 2. Allocation plan preview
 * 3. Deployment progress during execution
 * 4. Results summary after completion
 */

import { useState, useEffect, useCallback } from 'react'
import { useKeplr } from '../context/KeplrContext'
import {
  planFleetAllocation,
  deployFleet,
  cancelFleet,
  getFleetSummary,
  type AllocationPlan,
  type FleetState,
  type FleetDeploymentStatus
} from '../utils/fleetManager'
import {
  fetchAllProviders,
  filterProvidersByCapacity,
  getTotalCapacity,
  type ProviderCapacity
} from '../utils/akashApi'
import { estimateFleetCostAKT, type SDLConfig } from '../utils/sdl'

interface FleetDeploymentProps {
  config: Omit<SDLConfig, 'cpuUnits' | 'gpuUnits' | 'memoryGi'>
  onComplete?: (state: FleetState) => void
  onCancel?: () => void
}

type FleetPhase = 'loading' | 'planning' | 'confirming' | 'deploying' | 'completed'

export function FleetDeployment({ config, onComplete, onCancel }: FleetDeploymentProps) {
  const { address, getSigningClient, hasCertificateOnChain, createAndPublishCertificate } = useKeplr()

  // Phase state
  const [phase, setPhase] = useState<FleetPhase>('loading')

  // Provider data
  const [providers, setProviders] = useState<ProviderCapacity[]>([])
  const [loadingProviders, setLoadingProviders] = useState(true)
  const [providerError, setProviderError] = useState<string | null>(null)

  // Allocation plan
  const [plan, setPlan] = useState<AllocationPlan | null>(null)

  // Deployment state
  const [fleetState, setFleetState] = useState<FleetState | null>(null)

  // Error state
  const [error, setError] = useState<string | null>(null)

  // Determine if we're doing CPU or GPU fleet
  const isGpuFleet = config.minerType === 'cuda'
  const targetResources = config.fleetSize

  // Load providers on mount
  useEffect(() => {
    loadProviders()
  }, [])

  const loadProviders = async () => {
    setLoadingProviders(true)
    setProviderError(null)

    try {
      const allProviders = await fetchAllProviders()
      const filtered = filterProvidersByCapacity(allProviders, {
        minCpu: isGpuFleet ? 0 : 1,
        minGpu: isGpuFleet ? 1 : 0,
        onlyOnline: true
      })
      setProviders(filtered)
      setPhase('planning')
    } catch (err) {
      setProviderError(err instanceof Error ? err.message : 'Failed to load providers')
    } finally {
      setLoadingProviders(false)
    }
  }

  // Create allocation plan
  const createPlan = useCallback(async () => {
    setError(null)
    try {
      const newPlan = await planFleetAllocation(
        isGpuFleet ? 0 : targetResources,
        isGpuFleet ? targetResources : 0,
        {
          onlyReliable: false  // Include all online providers
        }
      )
      setPlan(newPlan)
      setPhase('confirming')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create plan')
    }
  }, [isGpuFleet, targetResources])

  // Execute deployment
  const executePlan = async () => {
    if (!plan || !address) return

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

      const finalState = await deployFleet(
        client,
        address,
        config,
        plan,
        (state) => setFleetState(state)
      )

      setFleetState(finalState)
      setPhase('completed')
      onComplete?.(finalState)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Deployment failed')
      setPhase('confirming')  // Go back to plan view
    }
  }

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

  // Get capacity summary
  const capacity = getTotalCapacity(providers)

  // Get cost estimate
  const costEstimate = plan
    ? estimateFleetCostAKT(
        config.minerType,
        isGpuFleet ? plan.totalGpu : plan.totalCpu,
        parseDuration(config.miningDuration),
        plan.deploymentsNeeded
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
      {(error || providerError) && (
        <div className="alert alert-error">
          {error || providerError}
        </div>
      )}

      {/* Loading state */}
      {phase === 'loading' && (
        <div className="fleet-loading">
          <p>Loading provider capacity...</p>
        </div>
      )}

      {/* Planning phase - show capacity */}
      {phase === 'planning' && (
        <div className="fleet-planning">
          <div className="capacity-overview">
            <h3>Available Capacity</h3>
            <div className="capacity-stats">
              <div className="stat">
                <span className="stat-value">{capacity.providerCount}</span>
                <span className="stat-label">Providers Online</span>
              </div>
              <div className="stat">
                <span className="stat-value">{capacity.totalCpu}</span>
                <span className="stat-label">Total CPUs</span>
              </div>
              <div className="stat">
                <span className="stat-value">{capacity.totalGpu}</span>
                <span className="stat-label">Total GPUs</span>
              </div>
            </div>

            {/* Top providers */}
            <div className="top-providers">
              <h4>Top Providers by Capacity</h4>
              <div className="provider-list">
                {providers.slice(0, 5).map((p) => (
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
            </div>
          </div>

          <div className="fleet-actions">
            <button className="btn btn-primary" onClick={createPlan}>
              Create Allocation Plan
            </button>
            <button className="btn btn-secondary" onClick={onCancel}>
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Confirming phase - show plan */}
      {phase === 'confirming' && plan && (
        <div className="fleet-confirming">
          <div className="plan-summary">
            <h3>Allocation Plan</h3>

            {plan.shortfall.cpu > 0 || plan.shortfall.gpu > 0 ? (
              <div className="alert alert-info">
                <strong>Note:</strong> Only {isGpuFleet ? plan.totalGpu : plan.totalCpu} of{' '}
                {targetResources} {isGpuFleet ? 'GPU' : 'CPU'}s available.
                {plan.shortfall.cpu > 0 && ` Short by ${plan.shortfall.cpu} CPUs.`}
                {plan.shortfall.gpu > 0 && ` Short by ${plan.shortfall.gpu} GPUs.`}
              </div>
            ) : (
              <div className="alert alert-success">
                ✓ Full allocation available: {isGpuFleet ? plan.totalGpu : plan.totalCpu}{' '}
                {isGpuFleet ? 'GPU' : 'CPU'}s across {plan.deploymentsNeeded} deployment
                {plan.deploymentsNeeded > 1 ? 's' : ''}
              </div>
            )}

            <div className="plan-stats">
              <div className="stat">
                <span className="stat-value">{plan.deploymentsNeeded}</span>
                <span className="stat-label">Deployments</span>
              </div>
              <div className="stat">
                <span className="stat-value">{isGpuFleet ? plan.totalGpu : plan.totalCpu}</span>
                <span className="stat-label">{isGpuFleet ? 'GPUs' : 'CPUs'}</span>
              </div>
              {costEstimate && (
                <div className="stat">
                  <span className="stat-value">{costEstimate.totalCost.toFixed(2)}</span>
                  <span className="stat-label">Est. Cost (AKT)</span>
                </div>
              )}
            </div>

            {/* Allocation details */}
            <div className="allocation-list">
              <h4>Allocation Details</h4>
              {plan.allocations.map((alloc, idx) => (
                <div key={idx} className="allocation-item">
                  <span className="alloc-index">#{idx + 1}</span>
                  <span className="alloc-provider">{alloc.provider.name}</span>
                  <span className="alloc-resources">
                    {alloc.requestedCpu} CPU
                    {alloc.requestedGpu > 0 && ` | ${alloc.requestedGpu} GPU`}
                    {` | ${alloc.requestedMemoryGi}Gi RAM`}
                  </span>
                </div>
              ))}
            </div>
          </div>

          <div className="fleet-actions">
            <button className="btn btn-primary" onClick={executePlan}>
              Deploy Fleet ({plan.deploymentsNeeded} transactions)
            </button>
            <button className="btn btn-secondary" onClick={() => setPhase('planning')}>
              Back
            </button>
            <button className="btn btn-secondary" onClick={onCancel}>
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Deploying phase - show progress */}
      {phase === 'deploying' && fleetState && (
        <div className="fleet-deploying">
          <div className="progress-header">
            <h3>Deploying Fleet...</h3>
            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{
                  width: `${((fleetState.stats.active + fleetState.stats.failed) / fleetState.stats.total) * 100}%`
                }}
              />
            </div>
            <p className="progress-text">
              {fleetState.stats.active + fleetState.stats.failed} / {fleetState.stats.total} complete
            </p>
          </div>

          <div className="deployment-grid">
            {fleetState.deployments.map((d) => (
              <DeploymentCard key={d.index} deployment={d} />
            ))}
          </div>

          <div className="fleet-actions">
            <button className="btn btn-danger" onClick={handleCancel}>
              Cancel Remaining
            </button>
          </div>
        </div>
      )}

      {/* Completed phase - show summary */}
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
              <DeploymentCard key={d.index} deployment={d} />
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
    allocation: {
      provider: { name: string }
      requestedCpu: number
      requestedGpu: number
    }
    dseq?: string
    error?: string
  }
}

function DeploymentCard({ deployment }: DeploymentCardProps) {
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

  return (
    <div className={`deployment-card status-${deployment.status}`}>
      <div className="card-header">
        <span className="card-index">#{deployment.index + 1}</span>
        <span className="card-status" style={{ color }}>
          {icon} {deployment.status.replace('-', ' ')}
        </span>
      </div>
      <div className="card-body">
        <div className="card-provider">{deployment.allocation.provider.name}</div>
        <div className="card-resources">
          {deployment.allocation.requestedCpu} CPU
          {deployment.allocation.requestedGpu > 0 && ` | ${deployment.allocation.requestedGpu} GPU`}
        </div>
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

// ============================================================================
// Helpers
// ============================================================================

function parseDuration(duration: string): number {
  const match = duration.match(/^(\d+)(m|h|d)$/)
  if (!match) return 90  // Default 90 minutes

  const value = parseInt(match[1], 10)
  const unit = match[2]

  switch (unit) {
    case 'm': return value
    case 'h': return value * 60
    case 'd': return value * 60 * 24
    default: return 90
  }
}
