import ReactDOM from 'react-dom/client'
import App from './App'
import './styles/index.css'

// Prevent WalletConnect Core from using global singleton
// This fixes "WalletConnect Core is already initialized" warning
// @ts-expect-error - setting env var for WalletConnect
globalThis.process = globalThis.process || {}
// @ts-expect-error - setting env var for WalletConnect
globalThis.process.env = globalThis.process.env || {}
// @ts-expect-error - setting env var for WalletConnect
globalThis.process.env.DISABLE_GLOBAL_CORE = 'true'

// Note: StrictMode is disabled because cosmos-kit/WalletConnect doesn't handle
// double-initialization from StrictMode's intentional double-rendering
ReactDOM.createRoot(document.getElementById('root')!).render(<App />)
