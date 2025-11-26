import { KeplrProvider } from './context/KeplrContext'
import { WalletConnection } from './components/WalletConnectionKeplr'
import { DeploymentForm } from './components/DeploymentForm'
import { DeploymentList } from './components/DeploymentList'
import { ResultsRetrieval } from './components/ResultsRetrieval'
import { TabNavigation } from './components/TabNavigation'
import { useState } from 'react'

export default function App() {
  const [activeTab, setActiveTab] = useState<'deploy' | 'manage' | 'results'>('deploy')

  return (
    <KeplrProvider>
      <div className="container">
        <header className="header">
          <h1>Quip Protocol Mining</h1>
          <p className="subtitle">Deploy decentralized mining experiments on Akash Network</p>
        </header>

        {/* Keplr Mobile connection using official @keplr-wallet/wc-qrcode-modal */}
        <WalletConnection />

        <TabNavigation activeTab={activeTab} onTabChange={setActiveTab} />

        {activeTab === 'deploy' && <DeploymentForm />}
        {activeTab === 'manage' && <DeploymentList />}
        {activeTab === 'results' && <ResultsRetrieval />}
      </div>
    </KeplrProvider>
  )
}
