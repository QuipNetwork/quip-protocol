/**
 * Wallet connection component using Keplr WalletConnect
 * With session management for delegated signing
 */

import { useState } from 'react'
import { QRCodeSVG } from 'qrcode.react'
import { useKeplr } from '../context/KeplrContext'
import { useSession } from '../context/SessionContext'
import { truncateAddress } from '../utils/format'
import {
  SESSION_DURATIONS,
  type SessionDuration
} from '../utils/sessionKeys'
import {
  createAllMsgGrants,
  getMsgTypeDescription,
  REQUIRED_MSG_TYPES
} from '../utils/authz'

export function WalletConnection() {
  const { isConnected, isConnecting, address, qrUri, connect, disconnect, error } = useKeplr()
  const {
    sessionState,
    sessionAddress,
    sessionExpiration,
    timeRemaining,
    missingMsgTypes,
    pendingSession,
    requestSession,
    checkGrants,
    clearSession,
    error: sessionError
  } = useSession()

  // Session request state
  const [showSessionRequest, setShowSessionRequest] = useState(false)
  const [selectedDuration, setSelectedDuration] = useState<SessionDuration>('8h')
  const [isRequesting, setIsRequesting] = useState(false)

  return (
    <section className="card wallet-section">
      <h2>Wallet Connection</h2>

      {error && (
        <div style={{ color: '#ff6b6b', marginBottom: '1rem', fontSize: '0.9em' }}>
          Error: {error}
        </div>
      )}

      {!isConnected ? (
        <div className="wallet-state">
          <p>Connect your Keplr Mobile wallet to deploy on Akash Network.</p>

          {qrUri ? (
            <div style={{ textAlign: 'center', marginTop: '1rem' }}>
              <p style={{ marginBottom: '1rem', color: '#666' }}>
                Scan this QR code with Keplr Mobile
              </p>
              <div style={{
                display: 'inline-block',
                padding: '16px',
                backgroundColor: 'white',
                borderRadius: '8px'
              }}>
                <QRCodeSVG value={qrUri} size={280} level="M" />
              </div>
              <p style={{ marginTop: '1rem', fontSize: '0.85em', color: '#888' }}>
                Waiting for approval...
              </p>
            </div>
          ) : (
            <button
              onClick={connect}
              className="btn btn-primary"
              disabled={isConnecting}
            >
              {isConnecting ? 'Initializing...' : 'Connect Keplr Mobile'}
            </button>
          )}
        </div>
      ) : (
        <div className="wallet-state wallet-connected">
          <div className="wallet-info">
            <div className="info-row">
              <span className="label">Address:</span>
              <span className="value address">{truncateAddress(address)}</span>
            </div>
          </div>

          {/* Session Status */}
          <div style={{
            marginTop: '16px',
            padding: '12px',
            background: sessionState === 'active'
              ? 'rgba(72, 187, 120, 0.1)'
              : sessionState === 'pending_grant'
                ? 'rgba(246, 173, 85, 0.1)'
                : 'rgba(113, 128, 150, 0.1)',
            borderRadius: '8px',
            border: sessionState === 'active'
              ? '1px solid rgba(72, 187, 120, 0.3)'
              : sessionState === 'pending_grant'
                ? '1px solid rgba(246, 173, 85, 0.3)'
                : '1px solid transparent'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
              <span style={{ fontWeight: 600, fontSize: '14px' }}>
                Session Signing
              </span>
              {sessionState === 'active' && (
                <span style={{
                  background: '#48bb78',
                  color: 'white',
                  fontSize: '10px',
                  padding: '2px 8px',
                  borderRadius: '4px',
                  fontWeight: 600
                }}>
                  ACTIVE
                </span>
              )}
              {sessionState === 'pending_grant' && (
                <span style={{
                  background: '#f6ad55',
                  color: 'white',
                  fontSize: '10px',
                  padding: '2px 8px',
                  borderRadius: '4px',
                  fontWeight: 600
                }}>
                  PENDING
                </span>
              )}
            </div>

            {sessionError && (
              <div style={{ color: '#fc8181', fontSize: '12px', marginBottom: '8px' }}>
                {sessionError}
              </div>
            )}

            {sessionState === 'none' && (
              <>
                <p style={{ fontSize: '13px', color: '#718096', marginBottom: '12px' }}>
                  Enable session signing to approve transactions automatically without mobile confirmations.
                </p>
                <button
                  onClick={() => setShowSessionRequest(true)}
                  className="btn btn-secondary"
                  style={{ fontSize: '13px', padding: '6px 12px' }}
                >
                  Enable Session
                </button>
              </>
            )}

            {sessionState === 'active' && (
              <>
                <div style={{ fontSize: '13px', color: '#718096' }}>
                  <div style={{ marginBottom: '4px' }}>
                    <strong>Session Key:</strong> {truncateAddress(sessionAddress)}
                  </div>
                  <div style={{ marginBottom: '4px' }}>
                    <strong>Expires:</strong> {timeRemaining} ({sessionExpiration?.toLocaleString()})
                  </div>
                </div>
                <button
                  onClick={clearSession}
                  className="btn btn-secondary"
                  style={{ fontSize: '13px', padding: '6px 12px', marginTop: '8px' }}
                >
                  End Session
                </button>
              </>
            )}

            {sessionState === 'pending_grant' && pendingSession && (
              <>
                <p style={{ fontSize: '13px', color: '#718096', marginBottom: '12px' }}>
                  Waiting for mobile wallet to grant permissions. Please approve the grant transaction in Keplr Mobile.
                </p>
                <div style={{ marginBottom: '12px' }}>
                  <div style={{ fontSize: '12px', marginBottom: '8px' }}>
                    <strong>Session Address:</strong>
                    <code style={{
                      display: 'block',
                      padding: '8px',
                      background: 'rgba(0,0,0,0.1)',
                      borderRadius: '4px',
                      marginTop: '4px',
                      fontSize: '11px',
                      wordBreak: 'break-all'
                    }}>
                      {pendingSession.address}
                    </code>
                  </div>
                  {missingMsgTypes.length > 0 && (
                    <div style={{ fontSize: '12px' }}>
                      <strong>Required Permissions:</strong>
                      <ul style={{ margin: '4px 0 0 16px', padding: 0, fontSize: '11px' }}>
                        {missingMsgTypes.map(msgType => (
                          <li key={msgType} style={{ marginBottom: '2px' }}>
                            {getMsgTypeDescription(msgType)}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
                <div style={{ display: 'flex', gap: '8px' }}>
                  <button
                    onClick={async () => {
                      await checkGrants()
                    }}
                    className="btn btn-primary"
                    style={{ fontSize: '13px', padding: '6px 12px' }}
                  >
                    Check Grants
                  </button>
                  <button
                    onClick={clearSession}
                    className="btn btn-secondary"
                    style={{ fontSize: '13px', padding: '6px 12px' }}
                  >
                    Cancel
                  </button>
                </div>
              </>
            )}
          </div>

          {/* Session Request Modal */}
          {showSessionRequest && (
            <div style={{
              marginTop: '16px',
              padding: '16px',
              background: '#2d3748',
              borderRadius: '8px'
            }}>
              <h4 style={{ margin: '0 0 12px 0', fontSize: '16px' }}>Request Session</h4>
              <p style={{ fontSize: '13px', color: '#a0aec0', marginBottom: '12px' }}>
                This will generate a temporary signing key. You'll need to approve a one-time grant transaction in your mobile wallet.
              </p>

              <div style={{ marginBottom: '12px' }}>
                <label style={{ display: 'block', fontSize: '13px', marginBottom: '4px' }}>
                  Session Duration
                </label>
                <select
                  value={selectedDuration}
                  onChange={(e) => setSelectedDuration(e.target.value as SessionDuration)}
                  style={{
                    width: '100%',
                    padding: '8px',
                    background: '#1a202c',
                    color: 'white',
                    border: '1px solid #4a5568',
                    borderRadius: '4px',
                    fontSize: '14px'
                  }}
                >
                  <option value="1h">1 Hour</option>
                  <option value="8h">8 Hours (Recommended)</option>
                  <option value="24h">24 Hours</option>
                  <option value="7d">7 Days</option>
                </select>
              </div>

              <div style={{ fontSize: '12px', color: '#718096', marginBottom: '12px' }}>
                <strong>Permissions required:</strong>
                <ul style={{ margin: '4px 0 0 16px', padding: 0 }}>
                  {REQUIRED_MSG_TYPES.map(msgType => (
                    <li key={msgType} style={{ marginBottom: '2px' }}>
                      {getMsgTypeDescription(msgType)}
                    </li>
                  ))}
                </ul>
              </div>

              <div style={{ display: 'flex', gap: '8px' }}>
                <button
                  onClick={async () => {
                    setIsRequesting(true)
                    try {
                      await requestSession(selectedDuration)
                      setShowSessionRequest(false)
                    } catch {
                      // Error handled by context
                    } finally {
                      setIsRequesting(false)
                    }
                  }}
                  className="btn btn-primary"
                  disabled={isRequesting}
                  style={{ fontSize: '13px', padding: '8px 16px' }}
                >
                  {isRequesting ? 'Generating...' : 'Generate Session Key'}
                </button>
                <button
                  onClick={() => setShowSessionRequest(false)}
                  className="btn btn-secondary"
                  style={{ fontSize: '13px', padding: '8px 16px' }}
                >
                  Cancel
                </button>
              </div>
            </div>
          )}

          <button onClick={disconnect} className="btn btn-danger" style={{ marginTop: '16px' }}>
            Disconnect
          </button>
        </div>
      )}
    </section>
  )
}
