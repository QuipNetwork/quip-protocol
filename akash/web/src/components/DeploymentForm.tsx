import { useState, useEffect, useMemo, useRef, useCallback } from 'react'
import { useKeplr } from '../context/KeplrContext'
import { DEFAULTS } from '../config/constants'
import { generateSDL, estimateCostAKT, estimateFleetCostAKT, GPU_MODELS, type GpuModelKey } from '../utils/sdl'
import { parseDurationToMinutes } from '../utils/format'
import yaml from 'yaml'
import {
  createDeployment,
  waitForBids,
  acceptBid,
  fetchProvider,
  checkProviderHealth,
  sendManifestViaConsole,
  type Bid,
  type ManifestSubmissionResult
} from '../utils/akashApi'
import {
  selectBestBid,
  getProviderDisplayName,
  type BidWithProvider
} from '../utils/bidSelection'
import type { SigningStargateClient } from '@cosmjs/stargate'
import { FleetDeployment } from './FleetDeployment'
import type { FleetState } from '../utils/fleetManager'

// Auto-deploy countdown duration in seconds
const AUTO_DEPLOY_COUNTDOWN_SECONDS = 5

export function DeploymentForm() {
  const {
    isConnected,
    address,
    getSigningClient,
    hasCertificateOnChain,
    hasLocalCertificate,
    certificateMismatch,
    isCheckingCertificate,
    isCertificateLoading,
    checkAndLoadCertificate,
    createAndPublishCertificate,
    clearAndRegenerateCertificate,
    error: certError
  } = useKeplr()

  const [minerType, setMinerType] = useState<'cpu' | 'cuda'>(DEFAULTS.minerType)
  const [gpuModel, setGpuModel] = useState<GpuModelKey>('any')
  const [fleetSize, setFleetSize] = useState(DEFAULTS.fleetSize)
  const [miningDuration, setMiningDuration] = useState(DEFAULTS.duration)
  const [difficultyEnergy, setDifficultyEnergy] = useState(DEFAULTS.difficulty)
  const [minDiversity, setMinDiversity] = useState(DEFAULTS.diversity)
  const [minSolutions, setMinSolutions] = useState(DEFAULTS.minSolutions)
  const [estimatedCost, setEstimatedCost] = useState('0')

  // IPFS Configuration
  const [ipfsEnabled, setIpfsEnabled] = useState(false)
  const [ipfsNode, setIpfsNode] = useState('')
  const [ipfsApiKey, setIpfsApiKey] = useState('')
  const [isDeploying, setIsDeploying] = useState(false)
  const [deploymentStatus, setDeploymentStatus] = useState<string>('')
  const [alert, setAlert] = useState<{ type: 'info' | 'success' | 'error'; message: string } | null>(null)

  // Bid selection state
  const [availableBids, setAvailableBids] = useState<BidWithProvider[]>([])
  const [currentDseq, setCurrentDseq] = useState<string | null>(null)
  const [signingClient, setSigningClient] = useState<SigningStargateClient | null>(null)
  const [isAcceptingBid, setIsAcceptingBid] = useState(false)
  const [currentSdl, setCurrentSdl] = useState<object | null>(null)
  const [currentManifestJson, setCurrentManifestJson] = useState<string | null>(null)  // Pre-computed manifest
  const [showSdl, setShowSdl] = useState(false)
  const [manifestResult, setManifestResult] = useState<ManifestSubmissionResult | null>(null)

  // Fleet deployment mode
  const [isFleetMode, setIsFleetMode] = useState(false)
  const [fleetResult, setFleetResult] = useState<FleetState | null>(null)

  // Auto-deploy mode
  const [autoDeployEnabled, setAutoDeployEnabled] = useState(() => {
    return localStorage.getItem('auto_deploy_enabled') === 'true'
  })
  const [autoDeployCountdown, setAutoDeployCountdown] = useState<number | null>(null)
  const [autoDeploySelectedBid, setAutoDeploySelectedBid] = useState<BidWithProvider | null>(null)
  const countdownIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Persist auto-deploy setting
  const handleAutoDeployToggle = useCallback((enabled: boolean) => {
    setAutoDeployEnabled(enabled)
    localStorage.setItem('auto_deploy_enabled', enabled ? 'true' : 'false')
  }, [])

  // Cancel auto-deploy countdown
  const cancelAutoDeployCountdown = useCallback(() => {
    if (countdownIntervalRef.current) {
      clearInterval(countdownIntervalRef.current)
      countdownIntervalRef.current = null
    }
    setAutoDeployCountdown(null)
    setAutoDeploySelectedBid(null)
  }, [])

  // Start auto-deploy countdown when bids arrive and auto-deploy is enabled
  const startAutoDeployCountdown = useCallback((bids: BidWithProvider[]) => {
    // Select best bid using our algorithm
    const bestBid = selectBestBid(bids, { preferReliable: true })
    if (!bestBid) return

    setAutoDeploySelectedBid(bestBid)
    setAutoDeployCountdown(AUTO_DEPLOY_COUNTDOWN_SECONDS)

    // Start countdown
    countdownIntervalRef.current = setInterval(() => {
      setAutoDeployCountdown(prev => {
        if (prev === null || prev <= 1) {
          // Countdown complete - will trigger auto-accept via effect
          if (countdownIntervalRef.current) {
            clearInterval(countdownIntervalRef.current)
            countdownIntervalRef.current = null
          }
          return 0
        }
        return prev - 1
      })
    }, 1000)
  }, [])

  // Auto-accept bid when countdown reaches 0
  useEffect(() => {
    if (autoDeployCountdown === 0 && autoDeploySelectedBid && !isAcceptingBid) {
      // Clear the countdown state before accepting
      setAutoDeployCountdown(null)
      // Accept the bid
      handleSelectBid(autoDeploySelectedBid)
    }
  }, [autoDeployCountdown, autoDeploySelectedBid, isAcceptingBid])

  // Cleanup interval on unmount
  useEffect(() => {
    return () => {
      if (countdownIntervalRef.current) {
        clearInterval(countdownIntervalRef.current)
      }
    }
  }, [])

  // Convert SDL to YAML for display
  const sdlYaml = useMemo(() => {
    if (!currentSdl) return ''
    return yaml.stringify(currentSdl)
  }, [currentSdl])

  // Copy SDL to clipboard
  const copySdlToClipboard = async () => {
    if (sdlYaml) {
      await navigator.clipboard.writeText(sdlYaml)
      setAlert({ type: 'success', message: 'SDL copied to clipboard!' })
    }
  }

  // Check for certificate when connected
  useEffect(() => {
    if (isConnected && address) {
      checkAndLoadCertificate()
    }
  }, [isConnected, address, checkAndLoadCertificate])

  // Load saved IPFS settings from localStorage
  useEffect(() => {
    const savedNode = localStorage.getItem('ipfs_node')
    const savedApiKey = localStorage.getItem('ipfs_api_key')
    if (savedNode) {
      setIpfsNode(savedNode)
      setIpfsEnabled(true)
    }
    if (savedApiKey) {
      setIpfsApiKey(savedApiKey)
    }
  }, [])

  // Update cost estimate when form values change
  // For fleet mode, estimate based on total resources
  useEffect(() => {
    const durationMinutes = parseDurationToMinutes(miningDuration)
    if (fleetSize > 1) {
      // Fleet mode: estimate total cost across deployments
      // Assume worst case: 1 resource per deployment = fleetSize deployments
      const fleetCost = estimateFleetCostAKT(minerType, fleetSize, durationMinutes, fleetSize)
      setEstimatedCost(fleetCost.totalCost.toFixed(2))
    } else {
      // Single deployment
      const cost = estimateCostAKT(minerType, 1, durationMinutes)
      setEstimatedCost(cost.toFixed(2))
    }
  }, [minerType, miningDuration, fleetSize])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!isConnected || !address) {
      setAlert({ type: 'error', message: 'Please connect your wallet first' })
      return
    }

    setIsDeploying(true)
    setAvailableBids([])
    setCurrentDseq(null)
    setDeploymentStatus('Preparing deployment...')
    setAlert({ type: 'info', message: 'Preparing deployment...' })

    try {
      // Step 0: Ensure certificate exists on-chain and matches our local cert
      // If certificateMismatch is true, our local cert isn't registered yet
      if (!hasCertificateOnChain || certificateMismatch) {
        const statusMsg = certificateMismatch
          ? 'Registering local certificate on-chain...'
          : 'Creating certificate for provider authentication...'
        setDeploymentStatus(statusMsg)
        setAlert({ type: 'info', message: statusMsg + ' Please approve the transaction.' })

        const certSuccess = await createAndPublishCertificate()
        if (!certSuccess) {
          setAlert({ type: 'error', message: 'Failed to create certificate. Please try again.' })
          setIsDeploying(false)
          setDeploymentStatus('')
          return
        }

        setAlert({ type: 'success', message: 'Certificate registered! Proceeding with deployment...' })
      }

      const sdl = generateSDL({
        minerType,
        fleetSize,
        miningDuration,
        difficultyEnergy,
        minDiversity,
        minSolutions,
        // Include GPU model for CUDA deployments
        gpuModel: minerType === 'cuda' ? gpuModel : undefined,
        // Include IPFS config if enabled
        ipfsNode: ipfsEnabled ? ipfsNode : undefined,
        ipfsApiKey: ipfsEnabled ? ipfsApiKey : undefined,
        ipfsPin: ipfsEnabled,
      })

      console.log('Generated SDL:', sdl)
      setCurrentSdl(sdl) // Store SDL for manifest sending later
      setDeploymentStatus('Requesting wallet approval for deployment...')
      setAlert({ type: 'info', message: 'Requesting wallet approval for deployment...' })

      // Get signing client from Keplr
      const client = await getSigningClient()
      if (!client) {
        throw new Error('Failed to get signing client')
      }
      setSigningClient(client)

      // Step 1: Create the deployment transaction
      const deployResult = await createDeployment(client, address, sdl)
      console.log('Deployment result:', deployResult)
      setCurrentDseq(deployResult.dseq)
      setCurrentManifestJson(deployResult.manifestJson)  // Store manifest for later

      // Step 2: Wait for bids from providers
      setDeploymentStatus(`Deployment created (dseq: ${deployResult.dseq}). Waiting for bids...`)
      setAlert({ type: 'info', message: `Waiting for provider bids... (dseq: ${deployResult.dseq})` })

      // Wait up to 2 minutes for bids, polling every 5 seconds
      const bids = await waitForBids(address, deployResult.dseq, 120000, 5000)

      if (bids.length === 0) {
        setAlert({
          type: 'error',
          message: `No bids received for deployment ${deployResult.dseq}. Try adjusting pricing or resources.`
        })
        setIsDeploying(false)
        return
      }

      console.log(`Received ${bids.length} bids:`, bids)

      // Sort by price - check ALL bids, not just top N
      // We'll filter down after checking reliability
      const sortedBids = [...bids].sort((a, b) => {
        const priceA = parseFloat(a.bid.price.amount)
        const priceB = parseFloat(b.bid.price.amount)
        return priceA - priceB
      })

      // Enrich bids with provider info and health status
      // Process in batches to show progress
      setDeploymentStatus(`Checking ${sortedBids.length} providers for reliability...`)
      const enrichedBids: BidWithProvider[] = await Promise.all(
        sortedBids.map(async (bid) => {
          const [providerInfo, healthStatus] = await Promise.all([
            fetchProvider(bid.bid.id.provider),
            checkProviderHealth(bid.bid.id.provider)
          ])
          const providerOrg = providerInfo?.attributes?.find(a => a.key === 'organization')?.value
          const providerName = providerOrg || bid.bid.id.provider.slice(0, 16) + '...'
          return {
            bid,
            providerName,
            providerOrg,
            isOnline: healthStatus.isOnline,
            isReliable: healthStatus.isReliable,
            isAudited: healthStatus.isAudited,
            activeLeaseCount: healthStatus.activeLeaseCount,
            uptime7d: healthStatus.uptime7d
          }
        })
      )

      // Count by category
      const reliableCount = enrichedBids.filter(b => b.isReliable).length
      const onlineCount = enrichedBids.filter(b => b.isOnline).length
      console.log(`Provider breakdown: ${reliableCount} reliable, ${onlineCount} online, ${enrichedBids.length} total`)

      // Sort all providers by: price first, then reliability indicators
      const sortedProviders = [...enrichedBids]
        .sort((a, b) => {
          // Price first (lowest = best)
          const priceA = parseFloat(a.bid.bid.price.amount)
          const priceB = parseFloat(b.bid.bid.price.amount)
          if (priceA !== priceB) return priceA - priceB
          // Then reliable providers
          if (a.isReliable !== b.isReliable) return a.isReliable ? -1 : 1
          // Then online providers
          if (a.isOnline !== b.isOnline) return a.isOnline ? -1 : 1
          // Audited providers
          if (a.isAudited !== b.isAudited) return a.isAudited ? -1 : 1
          // Then by active lease count (more = better)
          return b.activeLeaseCount - a.activeLeaseCount
        })

      // Show top 30 providers (mix of reliable and online)
      const displayProviders = sortedProviders.slice(0, 30)

      if (displayProviders.length === 0) {
        setAlert({
          type: 'error',
          message: `${enrichedBids.length} bids received but no providers could be verified. Try again later.`
        })
        setIsDeploying(false)
        return
      }

      // If we have no reliable providers, warn the user
      const reliableInList = displayProviders.filter(b => b.isReliable).length
      const onlineInList = displayProviders.filter(b => b.isOnline).length

      setAvailableBids(displayProviders)
      setDeploymentStatus('')

      // Check if auto-deploy is enabled
      if (autoDeployEnabled && displayProviders.length > 0) {
        const bestBid = selectBestBid(displayProviders, { preferReliable: true })
        if (bestBid) {
          const providerName = getProviderDisplayName(bestBid)
          setAlert({
            type: 'info',
            message: `Auto-deploying to ${providerName} in ${AUTO_DEPLOY_COUNTDOWN_SECONDS}s... (Click Cancel to choose manually)`
          })
          startAutoDeployCountdown(displayProviders)
          return
        }
      }

      if (reliableInList === 0 && onlineInList > 0) {
        setAlert({
          type: 'info',
          message: `${onlineInList} providers online but none verified reliable. Proceed with caution.`
        })
      } else if (reliableInList > 0) {
        setAlert({
          type: 'info',
          message: `${reliableInList} verified reliable + ${onlineInList - reliableInList} online providers. Select one below.`
        })
      } else {
        setAlert({
          type: 'info',
          message: `${displayProviders.length} providers found. Check status before selecting.`
        })
      }

    } catch (error) {
      console.error('Deployment error:', error)
      const message = error instanceof Error ? error.message : 'Unknown error'
      if (message.includes('rejected') || message.includes('cancelled')) {
        setAlert({ type: 'info', message: 'Transaction cancelled by user' })
      } else {
        setAlert({
          type: 'error',
          message: `Deployment failed: ${message}`
        })
      }
      setDeploymentStatus('')
      setIsDeploying(false)
    }
  }

  const handleSelectBid = async (selectedBid: BidWithProvider) => {
    if (!signingClient || !currentDseq || !address) {
      setAlert({ type: 'error', message: 'No active deployment session' })
      return
    }

    setIsAcceptingBid(true)
    setDeploymentStatus(`Accepting bid from ${selectedBid.providerName}...`)
    setAlert({
      type: 'info',
      message: `Accepting bid from ${selectedBid.providerName}. Please approve the transaction...`
    })

    try {
      // Step 1: Accept bid (create lease)
      const leaseResult = await acceptBid(signingClient, selectedBid.bid)
      console.log('Lease created:', leaseResult)

      // Step 2: Send manifest to provider
      // This is critical - without it, the provider doesn't know what to deploy
      if (currentManifestJson) {
        setDeploymentStatus('Sending manifest to provider...')
        setAlert({ type: 'info', message: 'Lease created! Sending manifest to provider...' })

        try {
          const result = await sendManifestViaConsole(
            address,
            currentDseq,
            selectedBid.bid.bid.id.provider,
            currentManifestJson  // Use the exact manifest JSON from deployment creation
          )
          console.log('Manifest submission result:', result)
          setManifestResult(result)

          if (result.success) {
            console.log('Manifest sent successfully in', result.durationMs, 'ms')
          }
        } catch (manifestError) {
          console.error('Failed to send manifest:', manifestError)
          // Store the error for display
          setManifestResult({
            success: false,
            timestamp: new Date().toISOString(),
            durationMs: 0,
            request: {
              url: 'https://console-api.akash.network/v1/proxy/provider/manifest',
              owner: address,
              dseq: currentDseq,
              provider: selectedBid.bid.bid.id.provider,
              manifestSize: 0
            },
            response: {
              status: 0,
              statusText: 'Error',
              body: manifestError instanceof Error ? manifestError.message : String(manifestError)
            },
            error: manifestError instanceof Error ? manifestError.message : String(manifestError)
          })
          // Don't fail the whole deployment - the lease is created
          // User can try to send manifest again or use Console
          setAlert({
            type: 'info',
            message: `Lease created but manifest send failed. Visit Akash Console to complete setup. DSEQ: ${currentDseq}`
          })
          setDeploymentStatus('')
          setAvailableBids([])
          setCurrentDseq(null)
          setCurrentSdl(null)
          setCurrentManifestJson(null)
          setSigningClient(null)
          setIsDeploying(false)
          setIsAcceptingBid(false)
          return
        }
      }

      setAlert({
        type: 'success',
        message: `Deployment active! Provider: ${selectedBid.providerName}, DSEQ: ${currentDseq}. Container starting...`
      })
      setDeploymentStatus('')
      setAvailableBids([])
      setCurrentDseq(null)
      setCurrentSdl(null)
      setCurrentManifestJson(null)
      setSigningClient(null)
      setIsDeploying(false)

    } catch (error) {
      console.error('Accept bid error:', error)
      const message = error instanceof Error ? error.message : 'Unknown error'
      if (message.includes('rejected') || message.includes('cancelled')) {
        setAlert({ type: 'info', message: 'Transaction cancelled. You can still select a provider.' })
      } else {
        setAlert({ type: 'error', message: `Failed to accept bid: ${message}` })
      }
      setDeploymentStatus('')
    } finally {
      setIsAcceptingBid(false)
    }
  }

  const handleCancelBidSelection = () => {
    // Cancel any auto-deploy countdown
    cancelAutoDeployCountdown()
    setAvailableBids([])
    setCurrentDseq(null)
    setCurrentSdl(null)
    setCurrentManifestJson(null)
    setSigningClient(null)
    setIsDeploying(false)
    setAlert({ type: 'info', message: 'Bid selection cancelled. Deployment remains open - you can close it from Manage Deployments.' })
  }

  // Fleet mode handlers
  const handleStartFleet = () => {
    setIsFleetMode(true)
    setFleetResult(null)
    setAlert(null)
  }

  const handleFleetComplete = (state: FleetState) => {
    setFleetResult(state)
    setAlert({
      type: state.stats.deploymentsActive > 0 ? 'success' : 'error',
      message: `Fleet deployed: ${state.stats.deploymentsActive} active, ${state.stats.deploymentsFailed} failed`
    })
  }

  const handleFleetCancel = () => {
    setIsFleetMode(false)
    setFleetResult(null)
  }

  // Build config for fleet deployment
  const fleetConfig = {
    minerType,
    fleetSize,
    miningDuration,
    difficultyEnergy,
    minDiversity,
    minSolutions,
    // Include GPU model for CUDA deployments
    gpuModel: minerType === 'cuda' ? gpuModel : undefined,
    // Include IPFS config if enabled
    ipfsNode: ipfsEnabled ? ipfsNode : undefined,
    ipfsApiKey: ipfsEnabled ? ipfsApiKey : undefined,
    ipfsPin: ipfsEnabled,
  }

  // Show fleet deployment UI if in fleet mode
  if (isFleetMode && !fleetResult) {
    return (
      <div className="tab-content active">
        <FleetDeployment
          config={fleetConfig}
          onComplete={handleFleetComplete}
          onCancel={handleFleetCancel}
        />
      </div>
    )
  }

  return (
    <div className="tab-content active">
      {/* Certificate Status */}
      {isConnected && (
        <div className={`alert ${hasCertificateOnChain && !certificateMismatch ? 'alert-success' : certError ? 'alert-error' : 'alert-info'}`} style={{ marginBottom: '16px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <strong>Certificate Status:</strong>{' '}
              {isCheckingCertificate ? (
                'Checking...'
              ) : isCertificateLoading ? (
                'Creating certificate...'
              ) : hasCertificateOnChain && !certificateMismatch ? (
                <span style={{ color: '#48bb78' }}>✓ Valid certificate on-chain</span>
              ) : certificateMismatch ? (
                <span style={{ color: '#f6ad55' }}>⚠ Local certificate not registered</span>
              ) : hasLocalCertificate ? (
                <span style={{ color: '#f6ad55' }}>⚠ Certificate needs publishing</span>
              ) : (
                <span style={{ color: '#f6ad55' }}>⚠ No certificate found</span>
              )}
            </div>
            <div style={{ display: 'flex', gap: '8px' }}>
              {(!hasCertificateOnChain || certificateMismatch) && !isCheckingCertificate && !isCertificateLoading && (
                <button
                  type="button"
                  onClick={createAndPublishCertificate}
                  className="btn btn-secondary"
                  style={{ padding: '4px 12px', fontSize: '13px' }}
                >
                  {certificateMismatch ? 'Register Certificate' : hasLocalCertificate ? 'Publish Certificate' : 'Create Certificate'}
                </button>
              )}
              {hasLocalCertificate && !isCheckingCertificate && !isCertificateLoading && (
                <button
                  type="button"
                  onClick={clearAndRegenerateCertificate}
                  className="btn btn-secondary"
                  style={{ padding: '4px 12px', fontSize: '13px', background: '#718096' }}
                  title="Clear local certificate and create a fresh one"
                >
                  Regenerate
                </button>
              )}
            </div>
          </div>
          {certError && (
            <small style={{ display: 'block', marginTop: '8px', color: '#fc8181' }}>
              <strong>Error:</strong> {certError}
            </small>
          )}
          {certificateMismatch && !isCheckingCertificate && !certError && (
            <small style={{ display: 'block', marginTop: '8px', color: '#718096' }}>
              Your local certificate is not registered on-chain. Click "Register Certificate" to publish it.
              (Akash allows multiple certificates per address - your old certificate will remain valid.)
            </small>
          )}
          {!hasCertificateOnChain && !certificateMismatch && !isCheckingCertificate && !certError && (
            <small style={{ display: 'block', marginTop: '8px', color: '#718096' }}>
              A certificate is required for providers to authenticate your deployments.
              It will be created automatically when you deploy, or you can create it now.
            </small>
          )}
        </div>
      )}

      <div className="alert alert-info">
        <strong>Tip:</strong> Mining duration is set to 90 minutes by default. Results will be available via HTTP after completion.
      </div>

      {alert && (
        <div className={`alert alert-${alert.type}`}>
          {alert.message}
        </div>
      )}

      {deploymentStatus && (
        <div className="alert alert-info" style={{ fontStyle: 'italic' }}>
          <strong>Status:</strong> {deploymentStatus}
        </div>
      )}

      {/* Manifest Submission Result */}
      {manifestResult && (
        <div className={`alert ${manifestResult.success ? 'alert-success' : 'alert-error'}`} style={{ marginBottom: '16px' }}>
          <div style={{ marginBottom: '8px' }}>
            <strong>Manifest Submission {manifestResult.success ? 'Succeeded' : 'Failed'}</strong>
            <span style={{ marginLeft: '8px', fontSize: '12px', color: '#718096' }}>
              {manifestResult.timestamp}
            </span>
          </div>
          <div style={{ fontSize: '13px', fontFamily: 'monospace', background: 'rgba(0,0,0,0.1)', padding: '8px', borderRadius: '4px' }}>
            <div><strong>URL:</strong> {manifestResult.request.url}</div>
            <div><strong>Owner:</strong> {manifestResult.request.owner}</div>
            <div><strong>DSEQ:</strong> {manifestResult.request.dseq}</div>
            <div><strong>Provider:</strong> {manifestResult.request.provider}</div>
            <div><strong>Manifest Size:</strong> {manifestResult.request.manifestSize} bytes</div>
            <div><strong>Duration:</strong> {manifestResult.durationMs}ms</div>
            <div style={{ marginTop: '8px', borderTop: '1px solid rgba(255,255,255,0.2)', paddingTop: '8px' }}>
              <strong>Response:</strong> {manifestResult.response.status} {manifestResult.response.statusText}
            </div>
            {manifestResult.response.body && (
              <div style={{ marginTop: '4px', wordBreak: 'break-all' }}>
                <strong>Body:</strong> {manifestResult.response.body.substring(0, 500)}
                {manifestResult.response.body.length > 500 && '...'}
              </div>
            )}
            {manifestResult.error && (
              <div style={{ marginTop: '4px', color: '#fc8181' }}>
                <strong>Error:</strong> {manifestResult.error}
              </div>
            )}
          </div>
          <button
            type="button"
            onClick={() => setManifestResult(null)}
            className="btn btn-secondary"
            style={{ marginTop: '8px', padding: '4px 12px', fontSize: '12px' }}
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Bid Selection UI */}
      {availableBids.length > 0 && (
        <div className="bid-selection">
          {/* Auto-deploy countdown banner */}
          {autoDeployCountdown !== null && autoDeploySelectedBid && (
            <div style={{
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              color: 'white',
              padding: '16px',
              borderRadius: '8px',
              marginBottom: '16px',
              textAlign: 'center'
            }}>
              <div style={{ fontSize: '18px', fontWeight: 600, marginBottom: '8px' }}>
                Auto-deploying in {autoDeployCountdown}s...
              </div>
              <div style={{ marginBottom: '12px' }}>
                Selected: <strong>{getProviderDisplayName(autoDeploySelectedBid)}</strong>
                {' '}- {autoDeploySelectedBid.bid.bid.price.amount} uakt/block
                {autoDeploySelectedBid.isReliable && (
                  <span style={{ marginLeft: '8px', background: 'rgba(255,255,255,0.2)', padding: '2px 8px', borderRadius: '4px', fontSize: '12px' }}>
                    Reliable
                  </span>
                )}
              </div>
              <button
                type="button"
                onClick={cancelAutoDeployCountdown}
                style={{
                  background: 'rgba(255,255,255,0.2)',
                  color: 'white',
                  border: '1px solid rgba(255,255,255,0.4)',
                  padding: '8px 24px',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontWeight: 500
                }}
              >
                Cancel - Choose Manually
              </button>
            </div>
          )}

          <h3>Select a Provider ({availableBids.length} available)</h3>
          <p className="bid-info">
            <span style={{ color: '#48bb78' }}>●</span> Verified (active leases){' '}
            <span style={{ color: '#f6ad55' }}>●</span> Available
          </p>

          <div className="bid-list">
            {availableBids.map((bidInfo, index) => {
              // Color based on verified status - green if reliable, orange otherwise
              const statusColor = bidInfo.isReliable ? '#48bb78' : '#f6ad55'

              return (
                <div
                  key={bidInfo.bid.bid.id.provider}
                  className="bid-item"
                  onClick={() => !isAcceptingBid && handleSelectBid(bidInfo)}
                  style={{
                    cursor: isAcceptingBid ? 'not-allowed' : 'pointer'
                  }}
                >
                  <div className="bid-rank">#{index + 1}</div>
                  <div className="bid-details">
                    <div className="bid-provider">
                      <span style={{ color: statusColor, marginRight: '6px' }}>●</span>
                      <strong>{bidInfo.providerName}</strong>
                      {bidInfo.isAudited && (
                        <span style={{
                          marginLeft: '8px',
                          fontSize: '10px',
                          background: '#667eea',
                          color: 'white',
                          padding: '2px 6px',
                          borderRadius: '4px',
                          fontWeight: 600
                        }}>
                          AUDITED
                        </span>
                      )}
                    </div>
                    <div className="bid-address">
                      {bidInfo.bid.bid.id.provider.slice(0, 20)}...
                      <span style={{ marginLeft: '8px', color: '#718096' }}>
                        {bidInfo.activeLeaseCount > 0 && `${bidInfo.activeLeaseCount} active lease${bidInfo.activeLeaseCount !== 1 ? 's' : ''}`}
                        {bidInfo.uptime7d !== undefined && ` • ${Math.round(bidInfo.uptime7d * 100)}% uptime`}
                      </span>
                    </div>
                  </div>
                  <div className="bid-price">
                    <strong>{bidInfo.bid.bid.price.amount}</strong>
                    <span className="bid-price-unit">uakt/block</span>
                  </div>
                </div>
              )
            })}
          </div>

          <button
            type="button"
            className="btn btn-secondary"
            onClick={handleCancelBidSelection}
            disabled={isAcceptingBid}
            style={{ marginTop: '1rem' }}
          >
            Cancel
          </button>
        </div>
      )}

      {/* SDL Display - show when deployment is in progress */}
      {currentSdl && currentDseq && (
        <div style={{
          background: '#f7fafc',
          borderRadius: '8px',
          padding: '16px',
          marginBottom: '20px',
          border: '1px solid #e2e8f0'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
            <h3 style={{ margin: 0, color: '#2d3748' }}>Deployment SDL (DSEQ: {currentDseq})</h3>
            <div style={{ display: 'flex', gap: '8px' }}>
              <button
                type="button"
                onClick={copySdlToClipboard}
                style={{
                  background: '#667eea',
                  color: 'white',
                  border: 'none',
                  padding: '6px 12px',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '13px'
                }}
              >
                Copy SDL
              </button>
              <button
                type="button"
                onClick={() => setShowSdl(!showSdl)}
                style={{
                  background: '#718096',
                  color: 'white',
                  border: 'none',
                  padding: '6px 12px',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '13px'
                }}
              >
                {showSdl ? 'Hide SDL' : 'Show SDL'}
              </button>
            </div>
          </div>
          <p style={{ color: '#718096', fontSize: '13px', marginBottom: '12px' }}>
            Copy this SDL and paste it in Akash Console to complete the deployment setup.
          </p>
          {showSdl && (
            <pre style={{
              background: '#1a202c',
              color: '#e2e8f0',
              padding: '16px',
              borderRadius: '6px',
              overflow: 'auto',
              maxHeight: '400px',
              fontSize: '12px',
              fontFamily: 'monospace',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word'
            }}>
              {sdlYaml}
            </pre>
          )}
        </div>
      )}

      {/* Hide form when showing bids */}
      <form onSubmit={handleSubmit} style={{ display: availableBids.length > 0 ? 'none' : 'block' }}>
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

        {minerType === 'cuda' && (
          <div className="form-group">
            <label htmlFor="gpuModel">GPU Model</label>
            <select
              id="gpuModel"
              value={gpuModel}
              onChange={(e) => setGpuModel(e.target.value as GpuModelKey)}
            >
              {Object.entries(GPU_MODELS).map(([key, { label }]) => (
                <option key={key} value={key}>{label}</option>
              ))}
            </select>
            <small>Select a specific GPU model or "Any" to accept any NVIDIA GPU. Specific models may have fewer available providers.</small>
          </div>
        )}

        <div className="form-group">
          <label htmlFor="fleetSize">
            {minerType === 'cuda' ? 'Total GPUs' : 'Total CPUs'}
          </label>
          <input
            type="number"
            id="fleetSize"
            value={fleetSize}
            onChange={(e) => setFleetSize(parseInt(e.target.value, 10) || 1)}
            min={1}
            max={1000}
            required
          />
          <small>
            {fleetSize > 1
              ? `Will deploy across multiple providers to get ${fleetSize} ${minerType === 'cuda' ? 'GPUs' : 'CPUs'}`
              : 'Single deployment with 1 resource unit'}
          </small>
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

        {/* IPFS Configuration */}
        <div className="form-group">
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={ipfsEnabled}
              onChange={(e) => setIpfsEnabled(e.target.checked)}
            />
            Enable IPFS Upload
          </label>
          <small>Upload mining results to IPFS for permanent storage</small>
        </div>

        {ipfsEnabled && (
          <>
            <div className="form-group">
              <label htmlFor="ipfsNode">IPFS Node URL</label>
              <input
                type="url"
                id="ipfsNode"
                value={ipfsNode}
                onChange={(e) => {
                  setIpfsNode(e.target.value)
                  localStorage.setItem('ipfs_node', e.target.value)
                }}
                placeholder="https://your-ipfs-node.example.com"
                required={ipfsEnabled}
              />
              <small>Your IPFS node API endpoint (will be saved for future use)</small>
            </div>

            <div className="form-group">
              <label htmlFor="ipfsApiKey">IPFS API Key</label>
              <input
                type="password"
                id="ipfsApiKey"
                value={ipfsApiKey}
                onChange={(e) => {
                  setIpfsApiKey(e.target.value)
                  localStorage.setItem('ipfs_api_key', e.target.value)
                }}
                placeholder="Your API key"
                required={ipfsEnabled}
              />
              <small>Bearer token for IPFS API authentication (saved locally)</small>
            </div>
          </>
        )}

        {/* Auto-Deploy Toggle */}
        <div className="form-group" style={{
          background: autoDeployEnabled ? 'rgba(102, 126, 234, 0.1)' : 'transparent',
          padding: '12px',
          borderRadius: '8px',
          border: autoDeployEnabled ? '1px solid rgba(102, 126, 234, 0.3)' : '1px solid transparent',
          marginTop: '8px'
        }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={autoDeployEnabled}
              onChange={(e) => handleAutoDeployToggle(e.target.checked)}
            />
            <span style={{ fontWeight: autoDeployEnabled ? 600 : 400 }}>
              Auto-Accept Best Bid
            </span>
            {autoDeployEnabled && (
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
          <small style={{ display: 'block', marginTop: '4px' }}>
            {autoDeployEnabled
              ? 'Will automatically select the lowest-price reliable provider after a 5-second countdown. You can cancel to choose manually.'
              : 'When enabled, automatically accepts the best bid without manual selection.'}
          </small>
        </div>

        <div className="alert alert-info cost-estimate">
          <strong>Estimated Cost:</strong> {estimatedCost} AKT for {miningDuration}
          {fleetSize > 1 && ` (${fleetSize} ${minerType === 'cuda' ? 'GPUs' : 'CPUs'})`}
        </div>

        <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
          {fleetSize > 1 ? (
            <>
              <button
                type="button"
                className="btn btn-primary"
                onClick={handleStartFleet}
                disabled={!isConnected || isDeploying}
              >
                Smart Deploy Fleet
              </button>
              <button
                type="submit"
                className="btn btn-secondary"
                disabled={!isConnected || isDeploying}
              >
                {isDeploying ? 'Deploying...' : 'Single Deploy (1 CPU)'}
              </button>
            </>
          ) : (
            <button
              type="submit"
              className="btn btn-primary"
              disabled={!isConnected || isDeploying}
            >
              {isDeploying ? 'Deploying...' : 'Deploy to Akash'}
            </button>
          )}
        </div>
        {fleetSize > 1 && (
          <small style={{ display: 'block', marginTop: '8px', color: '#718096' }}>
            Smart Deploy will query provider capacity and create optimal allocations across providers.
          </small>
        )}
      </form>
    </div>
  )
}
