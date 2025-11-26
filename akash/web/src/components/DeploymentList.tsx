import { useState, useEffect } from 'react'
import { useKeplr } from '../context/KeplrContext'
import { fetchDeployments, type Deployment } from '../utils/akashApi'
import { formatAKT, formatDate } from '../utils/format'

export function DeploymentList() {
  const { address, isConnected } = useKeplr()
  const [deployments, setDeployments] = useState<Deployment[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const refresh = async () => {
    if (!address) return

    setLoading(true)
    setError(null)

    try {
      const data = await fetchDeployments(address)
      setDeployments(data)
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

      <button className="btn btn-secondary" onClick={refresh} disabled={loading}>
        {loading ? 'Refreshing...' : 'Refresh Deployments'}
      </button>

      {error && (
        <div className="alert alert-error">
          {error}
        </div>
      )}

      <div className="deployment-list">
        {deployments.length === 0 ? (
          <p className="empty-state">No active deployments found</p>
        ) : (
          deployments.map((d) => (
            <div key={d.deployment.deployment_id.dseq} className="deployment-item">
              <h3>Deployment #{d.deployment.deployment_id.dseq}</h3>
              <p>
                <strong>State:</strong>{' '}
                <span className={`deployment-status ${getStatusClass(d.deployment.state)}`}>
                  {d.deployment.state}
                </span>
              </p>
              <p>
                <strong>Created:</strong> {formatDate(d.deployment.created_at)}
              </p>
              <p>
                <strong>Balance:</strong> {formatAKT(d.escrow_account.balance.amount)}
              </p>
              {d.groups.length > 0 && (
                <p>
                  <strong>Groups:</strong> {d.groups.length}
                </p>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  )
}
