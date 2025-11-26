interface TabNavigationProps {
  activeTab: 'deploy' | 'manage' | 'results'
  onTabChange: (tab: 'deploy' | 'manage' | 'results') => void
}

export function TabNavigation({ activeTab, onTabChange }: TabNavigationProps) {
  return (
    <div className="tabs">
      <button
        className={`tab ${activeTab === 'deploy' ? 'active' : ''}`}
        onClick={() => onTabChange('deploy')}
      >
        Deploy
      </button>
      <button
        className={`tab ${activeTab === 'manage' ? 'active' : ''}`}
        onClick={() => onTabChange('manage')}
      >
        Manage Deployments
      </button>
      <button
        className={`tab ${activeTab === 'results' ? 'active' : ''}`}
        onClick={() => onTabChange('results')}
      >
        Retrieve Results
      </button>
    </div>
  )
}
