import { useState } from 'react'
import { useKeplr } from '../context/KeplrContext'

export function ResultsRetrieval() {
  const { isConnected } = useKeplr()
  const [deploymentId, setDeploymentId] = useState('')
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleRetrieve = async () => {
    if (!deploymentId) {
      setError('Please enter a deployment ID')
      return
    }

    setLoading(true)
    setError(null)
    setResults(null)

    try {
      // TODO: Implement actual results retrieval via lease endpoints
      setResults('Results retrieval not yet implemented. This will fetch mining results from the deployment\'s HTTP endpoint.')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to retrieve results')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="tab-content">
      <div className="alert alert-info">
        <strong>Retrieve Results:</strong> Download mining results from completed deployments
      </div>

      <div className="form-group">
        <label htmlFor="deploymentId">Deployment ID (DSEQ)</label>
        <input
          type="number"
          id="deploymentId"
          value={deploymentId}
          onChange={(e) => setDeploymentId(e.target.value)}
          placeholder="e.g., 12345678"
        />
        <small>Enter the deployment sequence number</small>
      </div>

      <button
        className="btn btn-primary"
        onClick={handleRetrieve}
        disabled={!isConnected || loading || !deploymentId}
      >
        {loading ? 'Retrieving...' : 'Get Results'}
      </button>

      {error && (
        <div className="alert alert-error" style={{ marginTop: '20px' }}>
          {error}
        </div>
      )}

      {results && (
        <div className="results-output" style={{ marginTop: '20px' }}>
          <div className="alert alert-info">
            {results}
          </div>
        </div>
      )}
    </div>
  )
}
