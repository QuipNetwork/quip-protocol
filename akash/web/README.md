# Akash Web Interface - Keplr Wallet Integration

A web-based deployment interface for Quip Protocol mining experiments on Akash Network, with support for both desktop and mobile Keplr wallets via WalletConnect.

## Quick Start

### 1. Start the Server

```bash
cd akash/web
python3 serve.py
```

The server will start at `http://localhost:8000` and automatically open in your browser.

### 2. Connect Your Wallet

**Desktop (Keplr Extension):**
1. Install [Keplr Extension](https://www.keplr.app/download)
2. Click "Connect Keplr Wallet" in the web interface
3. Approve the connection

**Mobile (Keplr App):**
1. Install Keplr mobile app
2. Click "Connect via WalletConnect" in the web interface
3. Scan the QR code with Keplr app
4. Approve the connection

### 3. Deploy Mining Experiments

1. **Configure deployment:**
   - Miner Type: CPU or CUDA GPU
   - Fleet Size: 1-100 instances
   - Mining Duration: e.g., "90m"
   - Difficulty Energy: e.g., -14900
   - Min Diversity: 0.1
   - Min Solutions: 5

2. **Review cost estimate** (shown automatically)

3. **Click "Deploy to Akash"**

4. **Sign transaction** in Keplr wallet

5. **Wait 90 minutes** for mining to complete

6. **Retrieve results** from the "Retrieve Results" tab

## Features

### Wallet Integration
- ✅ **Keplr Browser Extension** - Desktop support
- ✅ **WalletConnect** - Mobile Keplr app via QR code
- ✅ **Auto-connect** - Remembers previous connection
- ✅ **Balance Display** - Shows AKT balance
- ✅ **Secure Signing** - All transactions signed by Keplr

### Deployment Management
- ✅ **Visual Configuration** - Form-based parameter setting
- ✅ **Cost Estimates** - Real-time pricing calculations
- ✅ **SDL Generation** - Automatic deployment configuration
- ✅ **Multi-instance** - Deploy 1-100 instances at once

### Monitoring & Results
- ✅ **Deployment List** - View all your deployments
- ✅ **Status Tracking** - Monitor deployment state
- ✅ **Result Retrieval** - Download JSON + logs via HTTP
- ✅ **One-click Close** - Stop billing instantly

## Architecture

### Frontend
- **HTML/CSS/JavaScript** - No build step required
- **Keplr Integration** - Direct browser extension access
- **WalletConnect** - Mobile wallet connection
- **Responsive Design** - Works on desktop and mobile

### Wallet Connection Flow

**Desktop:**
```
User → Click Connect → Keplr Extension → Approve → Connected
```

**Mobile:**
```
User → Click WalletConnect → QR Code → Keplr App → Approve → Connected
```

### Deployment Flow

```
Configure → Generate SDL → Sign TX (Keplr) → Broadcast → Wait for Bids → Create Lease
```

## Configuration

### Update Docker Registry

Edit `app.js` line 7:

```javascript
const DOCKER_REGISTRY = 'ghcr.io/your-username';  // UPDATE THIS
```

Replace `your-username` with your actual GitHub username or registry URL.

### Customize RPC Endpoints

Edit `app.js` lines 4-5:

```javascript
const AKASH_RPC = 'https://rpc.akashnet.net:443';
const AKASH_REST = 'https://api.akashnet.net';
```

Use alternative endpoints if needed:
- `https://rpc.ny.akash.farm:443`
- `https://akash-rpc.polkachu.com:443`

## Production Deployment

### For Full Functionality

Integrate [@akashnetwork/akashjs](https://github.com/akash-network/akashjs) library:

```bash
npm install @akashnetwork/akashjs
```

Then modify `app.js` to use:
- `DeploymentClient` - Create deployments
- `LeaseClient` - Manage leases
- `ProviderClient` - Query provider status

### Example Integration

```javascript
import { SigningStargateClient } from "@cosmjs/stargate";
import { Registry } from "@cosmjs/proto-signing";
import { akashnetwork } from "@akashnetwork/akashjs";

async function createDeployment(sdl) {
    // Get Keplr signer
    const offlineSigner = window.keplr.getOfflineSigner(AKASH_CHAIN_ID);

    // Create registry with Akash types
    const registry = new Registry([
        ...defaultRegistryTypes,
        ...akashnetwork.deployment.v1beta3.registry
    ]);

    // Create client
    const client = await SigningStargateClient.connectWithSigner(
        AKASH_RPC,
        offlineSigner,
        { registry }
    );

    // Create deployment message
    const msg = akashnetwork.deployment.v1beta3.MessageComposer.fromPartial.createDeployment({
        // ... deployment params
    });

    // Sign and broadcast
    const result = await client.signAndBroadcast(
        wallet.address,
        [msg],
        "auto"
    );

    return result;
}
```

## Limitations (Current Demo)

1. **Simplified Deployment** - Generates SDL but doesn't actually deploy
   - For real deployment, integrate @akashnetwork/akashjs
   - Or use Akash CLI commands shown in console

2. **No Bid Management** - Doesn't handle bid acceptance automatically
   - Use Akash CLI to accept bids manually
   - Or integrate bid polling and acceptance logic

3. **Basic Result Retrieval** - Shows instructions but doesn't fetch automatically
   - Provider URI needs to be queried from lease
   - Then fetch via HTTP from provider

## Roadmap

### Phase 1: Core Functionality ✅
- [x] Keplr browser extension integration
- [x] WalletConnect mobile support
- [x] SDL generation
- [x] Deployment configuration UI
- [x] Cost estimation

### Phase 2: Full Integration 🚧
- [ ] @akashnetwork/akashjs integration
- [ ] Automatic deployment creation
- [ ] Bid polling and acceptance
- [ ] Real-time deployment status
- [ ] Automatic result fetching

### Phase 3: Advanced Features 📋
- [ ] Multiple wallet support (Cosmostation, Leap)
- [ ] Deployment templates
- [ ] Batch operations
- [ ] Analytics dashboard
- [ ] Cost tracking

## Troubleshooting

### Keplr Not Found
**Problem:** "Please install Keplr extension" message

**Solution:**
1. Install [Keplr Extension](https://www.keplr.app/download)
2. Refresh the page
3. Try connecting again

### WalletConnect QR Code Not Showing
**Problem:** Mobile connection fails

**Solution:**
1. Check browser console for errors
2. Try different browser (Chrome/Firefox recommended)
3. Ensure WalletConnect library loaded correctly

### Insufficient Balance
**Problem:** Cannot deploy due to low AKT balance

**Solution:**
1. Buy AKT on exchanges (Kraken, Osmosis, etc.)
2. Send to your Keplr wallet address
3. Wait for confirmation (~6 seconds)
4. Refresh balance in web interface

### Deployment Not Created
**Problem:** Click deploy but nothing happens

**Solution:**
1. Check browser console for errors
2. Ensure wallet is connected
3. Verify sufficient AKT balance
4. For production, integrate @akashnetwork/akashjs

## Security Notes

✅ **Private keys never exposed** - Keplr handles all key management
✅ **Transactions signed in wallet** - Never in browser
✅ **Read-only API access** - No sensitive data stored
✅ **CORS enabled** - Only for local development

⚠️ **Important:**
- Never paste private keys or mnemonics into web forms
- Always verify transactions in Keplr before signing
- Use testnet for initial testing

## Support

- **Akash Discord**: https://discord.akash.network/
- **Keplr Support**: https://help.keplr.app/
- **GitHub Issues**: Open an issue for bugs/features

## License

Same as parent project (Quip Protocol)
