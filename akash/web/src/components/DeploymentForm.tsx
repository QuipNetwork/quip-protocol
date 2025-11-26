import { useState, useEffect } from 'react'
import { useKeplr } from '../context/KeplrContext'
import { DEFAULTS } from '../config/constants'
import { generateSDL, estimateCostAKT } from '../utils/sdl'
import { parseDurationToMinutes } from '../utils/format'
import { createDeployment } from '../utils/akashApi'

export function DeploymentForm() {
  const { isConnected, address, getSigningClient } = useKeplr()

  const [minerType, setMinerType] = useState<'cpu' | 'cuda'>(DEFAULTS.minerType)
  const [fleetSize, setFleetSize] = useState(DEFAULTS.fleetSize)
  const [miningDuration, setMiningDuration] = useState(DEFAULTS.duration)
  const [difficultyEnergy, setDifficultyEnergy] = useState(DEFAULTS.difficulty)
  const [minDiversity, setMinDiversity] = useState(DEFAULTS.diversity)
  const [minSolutions, setMinSolutions] = useState(DEFAULTS.minSolutions)
  const [estimatedCost, setEstimatedCost] = useState('0')
  const [isDeploying, setIsDeploying] = useState(false)
  const [alert, setAlert] = useState<{ type: 'info' | 'success' | 'error'; message: string } | null>(null)

  // Update cost estimate when form values change
  useEffect(() => {
    const durationMinutes = parseDurationToMinutes(miningDuration)
    const cost = estimateCostAKT(minerType, fleetSize, durationMinutes)
    setEstimatedCost(cost.toString())
  }, [minerType, fleetSize, miningDuration])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!isConnected || !address) {
      setAlert({ type: 'error', message: 'Please connect your wallet first' })
      return
    }

    setIsDeploying(true)
    setAlert({ type: 'info', message: 'Preparing deployment...' })

    try {
      const sdl = generateSDL({
        minerType,
        fleetSize,
        miningDuration,
        difficultyEnergy,
        minDiversity,
        minSolutions
      })

      console.log('Generated SDL:', sdl)
      setAlert({ type: 'info', message: 'Requesting wallet approval...' })

      // Get signing client from Keplr
      const signingClient = await getSigningClient()
      if (!signingClient) {
        throw new Error('Failed to get signing client')
      }

      // Create the deployment transaction
      const result = await createDeployment(signingClient, address, sdl)

      console.log('Deployment result:', result)
      setAlert({
        type: 'success',
        message: `Deployment created! TX: ${result.transactionHash}`
      })
    } catch (error) {
      console.error('Deployment error:', error)
      const message = error instanceof Error ? error.message : 'Unknown error'
      // Check for user rejection
      if (message.includes('rejected') || message.includes('cancelled')) {
        setAlert({ type: 'info', message: 'Transaction cancelled by user' })
      } else {
        setAlert({
          type: 'error',
          message: `Deployment failed: ${message}`
        })
      }
    } finally {
      setIsDeploying(false)
    }
  }

  return (
    <div className="tab-content active">
      <div className="alert alert-info">
        <strong>Tip:</strong> Mining duration is set to 90 minutes by default. Results will be available via HTTP after completion.
      </div>

      {alert && (
        <div className={`alert alert-${alert.type}`}>
          {alert.message}
        </div>
      )}

      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="minerType">Miner Type</label>
          <select
            id="minerType"
            value={minerType}
            onChange={(e) => setMinerType(e.target.value as 'cpu' | 'cuda')}
            required
          >
            <option value="cpu">CPU Miner (Simulated Annealing)</option>
            <option value="cuda">CUDA GPU Miner (NVIDIA)</option>
          </select>
          <small>CPU miners are cheaper but slower. GPU miners are faster but cost more.</small>
        </div>

        <div className="form-group">
          <label htmlFor="fleetSize">Fleet Size</label>
          <input
            type="number"
            id="fleetSize"
            value={fleetSize}
            onChange={(e) => setFleetSize(parseInt(e.target.value, 10) || 1)}
            min={1}
            max={100}
            required
          />
          <small>Number of instances to deploy (1-100)</small>
        </div>

        <div className="form-group">
          <label htmlFor="miningDuration">Mining Duration</label>
          <input
            type="text"
            id="miningDuration"
            value={miningDuration}
            onChange={(e) => setMiningDuration(e.target.value)}
            required
          />
          <small>Format: 5m, 30m, 1h, 2h (90 minutes recommended for experiments)</small>
        </div>

        <div className="form-group">
          <label htmlFor="difficultyEnergy">Difficulty Energy</label>
          <input
            type="number"
            id="difficultyEnergy"
            value={difficultyEnergy}
            onChange={(e) => setDifficultyEnergy(parseFloat(e.target.value))}
            step={1}
            required
          />
          <small>Energy threshold for valid blocks (e.g., -14900, -4100)</small>
        </div>

        <div className="form-group">
          <label htmlFor="minDiversity">Minimum Diversity</label>
          <input
            type="number"
            id="minDiversity"
            value={minDiversity}
            onChange={(e) => setMinDiversity(parseFloat(e.target.value))}
            step={0.01}
            min={0}
            max={1}
            required
          />
          <small>Solution diversity threshold (0.0 - 1.0)</small>
        </div>

        <div className="form-group">
          <label htmlFor="minSolutions">Minimum Solutions</label>
          <input
            type="number"
            id="minSolutions"
            value={minSolutions}
            onChange={(e) => setMinSolutions(parseInt(e.target.value, 10) || 1)}
            min={1}
            required
          />
          <small>Minimum number of valid solutions required</small>
        </div>

        <div className="alert alert-info cost-estimate">
          <strong>Estimated Cost:</strong> {estimatedCost} AKT for {miningDuration}
        </div>

        <button
          type="submit"
          className="btn btn-primary"
          disabled={!isConnected || isDeploying}
        >
          {isDeploying ? 'Deploying...' : 'Deploy to Akash'}
        </button>
      </form>
    </div>
  )
}
