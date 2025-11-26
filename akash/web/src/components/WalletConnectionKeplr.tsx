/**
 * Wallet connection component using Keplr WalletConnect
 */

import { QRCodeSVG } from 'qrcode.react'
import { useKeplr } from '../context/KeplrContext'
import { truncateAddress } from '../utils/format'

export function WalletConnection() {
  const { isConnected, isConnecting, address, qrUri, connect, disconnect, error } = useKeplr()

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
          <button onClick={disconnect} className="btn btn-danger">
            Disconnect
          </button>
        </div>
      )}
    </section>
  )
}
