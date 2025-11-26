# Akash Network Mining Deployment

Deploy large-scale mining rate comparison experiments on Akash Network - a decentralized cloud computing platform offering 2-5x cost savings over traditional cloud providers.

**Quick Start:** Use the web interface with your Keplr wallet (desktop or mobile) to deploy in minutes!

## Table of Contents

1. [Required Tools](#required-tools)
2. [Build and Push Docker Images](#build-and-push-docker-images)
3. [Deploy via Web Interface (Recommended)](#deploy-via-web-interface-recommended)
4. [Deploy via Command Line (Optional)](#deploy-via-command-line-optional)
5. [Monitor and Collect Results](#monitor-and-collect-results)
6. [Configuration Reference](#configuration-reference)
7. [Troubleshooting](#troubleshooting)

---

## Required Tools

### 1. Docker (for building images)

**macOS:**
```bash
brew install docker
# Or download Docker Desktop from docker.com
```

**Linux:**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install docker.io docker-buildx-plugin

# Enable BuildKit for multi-platform builds
export DOCKER_BUILDKIT=1
```

### 2. Keplr Wallet

**Desktop (Browser Extension):**
- Install from: https://www.keplr.app/download
- Supported browsers: Chrome, Firefox, Edge

**Mobile:**
- Download Keplr app from App Store or Google Play
- Set up wallet and backup seed phrase

### 3. AKT Tokens

You'll need AKT tokens for deployment:
- Buy on exchanges (Osmosis, Kraken, etc.)
- Send to your Keplr wallet address
- Minimum: ~5 AKT per deployment (~$0.50 worth)

### 4. Python 3 (for web interface)

**macOS:**
```bash
brew install python3
```

**Linux:**
```bash
sudo apt-get install python3
```

**Verify:**
```bash
python3 --version  # Should be 3.8 or higher
```

### 5. jq (for JSON processing in collection scripts)

**macOS:**
```bash
brew install jq
```

**Linux:**
```bash
sudo apt-get install jq
```

**Verify:**
```bash
jq --version
```

### 6. curl (usually pre-installed)

**macOS:** Pre-installed

**Linux:**
```bash
# If not installed
sudo apt-get install curl
```

**Verify:**
```bash
curl --version
```

---

## Build and Push Docker Images

Before deploying, you need to build multi-platform Docker images and push them to a container registry.

### 1. Set Up Docker Hub Account

```bash
# Create account at https://hub.docker.com (free)

# Login to Docker Hub
docker login
# Enter username and password
```

### 2. Configure Image Names

Set your image registry and version:

```bash
export REGISTRY="docker.io/yourusername" # For Docker Hub
export VERSION="v0.0.1"
```

Or use the GitHub Container Registry (ghcr.io):
```bash
# Login to GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# Keep registry as:
export REGISTRY="ghcr.io/your-org"
```

### 3. Build Multi-Platform Images (amd64)

**IMPORTANT: Akash providers run amd64 (x86_64) architecture.** If you're building on Apple Silicon (M1/M2/M3/M4), you must cross-compile for amd64. The build script handles this automatically using `docker buildx`.

**Run from the project root directory (not from `akash/`):**

```bash
# Make sure you're in the project root
cd /path/to/quip-protocol

# Build all images for amd64 (required for Akash)
./akash/build_akash_images.sh

# This builds:
# - CPU miner (linux/amd64)
# - CUDA miner (linux/amd64 with CUDA support)
```

**What this does:**
- Creates a `docker buildx` builder for cross-platform builds
- Builds Docker images targeting `linux/amd64` architecture
- Cross-compiles from arm64 to amd64 if you're on Apple Silicon
- Tags with version and `latest`
- Pushes to Docker Hub (docker.io/carback1 by default)

**Build time:** ~10-15 minutes (may be longer on Apple Silicon due to cross-compilation)

#### Manual Cross-Platform Build (if script fails)

If you need to build manually or customize the build:

```bash
# One-time setup: create buildx builder for multi-arch
docker buildx create --name quip-multiarch --use --bootstrap

# Build and push CPU image for amd64
docker buildx build \
  --platform linux/amd64 \
  -f akash/Dockerfile.akash-cpu \
  -t carback1/quip-protocol-cpu-miner:latest \
  --push \
  .

# Build and push CUDA image for amd64
docker buildx build \
  --platform linux/amd64 \
  -f akash/Dockerfile.akash-cuda \
  -t carback1/quip-protocol-cuda-miner:latest \
  --push \
  .
```

#### Verify Image Architecture

After building, verify the images are amd64:

```bash
# Check architecture of pushed image
docker buildx imagetools inspect carback1/quip-protocol-cpu-miner:latest

# Look for: "Platform: linux/amd64"
```

**Verify images are pushed:**
```bash
# Check Docker Hub
# Visit: https://hub.docker.com/u/yourusername

# Or check with docker
docker pull yourusername/quip-protocol-cpu-miner:latest
docker pull yourusername/quip-protocol-cuda-miner:latest
```

### 4. Update SDL Files

After building, update the image references in SDL files:

**Edit `akash/deploy-cpu-ipfs.yaml`:**
```yaml
services:
  cpu-miner:
    image: docker.io/yourusername/quip-protocol-cpu-miner:latest
    # Or if using GitHub Container Registry:
    # image: ghcr.io/your-org/quip-protocol-cpu-miner:latest
```

**Edit `akash/deploy-cuda-ipfs.yaml`:**
```yaml
services:
  cuda-miner:
    image: docker.io/yourusername/quip-protocol-cuda-miner:latest
```

---

## Deploy via Web Interface (Recommended)

The web interface provides the easiest way to deploy with your Keplr wallet.

### 1. Configure IPFS Settings

Before deploying, configure your IPFS node settings in the SDL files.

**Edit `akash/deploy-cpu-ipfs.yaml` and `akash/deploy-cuda-ipfs.yaml`:**

```yaml
env:
  # REQUIRED: Your IPFS node URL
  - IPFS_NODE=https://your-ipfs-node.ngrok.io  # UPDATE THIS

  # REQUIRED: API key for upload authentication
  - IPFS_API_KEY=your-api-key-here  # UPDATE THIS

  # OPTIONAL: Pin files permanently (recommended)
  - IPFS_PIN=true

  # Mining parameters (adjust as needed)
  - MINING_DURATION=90m
  - DIFFICULTY_ENERGY=-14900
  - MIN_DIVERSITY=0.1
  - MIN_SOLUTIONS=5
```

**Common IPFS node URLs:**
- **ngrok tunnel:** `https://your-subdomain.ngrok.io`
- **Custom domain:** `https://ipfs.yourdomain.com`
- **Direct IP:** `http://your.server.ip:5001`
- **Localhost (testing):** `http://localhost:5001`

### 2. Start Web Interface

```bash
cd akash/web
python3 serve.py
```

This will start an HTTPS server (required for WalletConnect) at `https://localhost:8000`.

**HTTPS Mode (Default - Required for WalletConnect):**
```bash
python3 serve.py
```

**Output:**
```
🔐 HTTPS Server running at: https://localhost:8000

⚠️  FIRST TIME SETUP:
   Your browser will show a security warning because this is a
   self-signed certificate. This is normal for local development.

   Click "Advanced" → "Proceed to localhost" to continue.
```

**HTTP Mode (Optional - Desktop Keplr Extension Only):**
```bash
python3 serve.py --http
```

**Note:** HTTP mode won't work with WalletConnect (mobile). Use HTTPS mode for full functionality.

The web interface will automatically open in your browser.

### 3. Connect Keplr Wallet

**Desktop (Browser Extension):**
1. Click "💻 Connect Keplr (Desktop)"
2. Approve connection in Keplr extension popup
3. Select Akash Network (akashnet-2)
4. Approve connection

**Mobile (WalletConnect v2):**
1. Open the web interface on your **laptop browser** at `https://localhost:8000`
2. Click "📱 Connect Keplr Mobile (WalletConnect)"
3. A QR code modal will appear on your laptop screen
4. Open Keplr app on your phone
5. Scan the QR code using Keplr Mobile
6. Approve connection in Keplr app
7. ✅ You can now sign transactions on your phone while using the laptop interface!

**Verify connection:**
- Your wallet address should appear in the interface
- Balance should show your AKT amount

### 4. Configure Deployment

**Select Miner Type:**
- **CPU:** Standard deployment, cheaper, slower mining
- **CUDA:** GPU-accelerated, more expensive, faster mining

**Set Fleet Size:**
- Enter number of instances (1-100)
- Cost estimate updates automatically

**Configure Mining Parameters:**
- **Duration:** How long to mine (default: 90m)
- **Difficulty:** Energy threshold (default: -14900)
- **Min Diversity:** Solution diversity requirement (default: 0.1)
- **Min Solutions:** Minimum solutions required (default: 5)

**Review Cost Estimate:**
- Shows total cost in AKT and USD
- Per-instance cost breakdown
- Total deployment cost

### 5. Deploy to Akash

1. Click "Deploy to Akash" button
2. Review deployment details in popup
3. Sign transaction in Keplr wallet
4. Wait for deployment confirmation (~1-2 minutes)

**Deployment creates:**
- SDL manifest with your configuration
- Deployment transaction on Akash
- Lease with provider
- Running containers

**Deployment IDs are saved to:** `deployments_cpu_TIMESTAMP.txt`

### 6. Manage Deployments

Switch to the "Manage Deployments" tab:

**View active deployments:**
- Deployment ID (DSEQ)
- Status (Active/Inactive)
- Provider URI
- Created date

**Actions:**
- **View Logs:** Stream live container logs
- **Close Deployment:** Stop containers and end lease
- **Get URI:** Copy deployment URI for HTTP access

### 7. Monitor Progress

**Real-time monitoring:**
1. Click "View Logs" on a deployment
2. Watch mining progress in log stream
3. Look for progress indicators:
   - "Mining started..."
   - "Attempt X/Y..."
   - "✅ All files successfully uploaded to IPFS!"

**Typical 90-minute run:**
- Mining: 0-90 minutes
- IPFS upload: 90-92 minutes
- Auto-shutdown: 92 minutes
- **Container automatically stops after successful upload!**

---

## Deploy via Command Line (Optional)

For automation or advanced users, you can deploy via CLI.

### 1. Install Akash CLI

**macOS:**
```bash
brew tap ovrclk/tap
brew install akash-provider-services
```

**Linux:**
```bash
# Download latest release
wget https://github.com/akash-network/provider/releases/download/v0.4.6/akash-provider-services_0.4.6_linux_amd64.tar.gz
tar -xzf akash-provider-services_0.4.6_linux_amd64.tar.gz
sudo mv akash-provider /usr/local/bin/
```

### 2. Configure Wallet

```bash
# Create wallet (or import existing)
akash keys add mykey

# Fund with AKT tokens
# Send AKT to the address shown above

# Set environment variables
export AKASH_KEYRING_BACKEND=os
export AKASH_FROM=mykey
export AKASH_NODE=https://rpc.akashnet.net:443
export AKASH_CHAIN_ID=akashnet-2
```

### 3. Deploy

```bash
cd akash

# Deploy CPU miners
export FLEET_SIZE=10
./deploy.sh deploy-cpu-ipfs.yaml

# Deployment IDs saved to: deployments_cpu_TIMESTAMP.txt
```

### 4. Close Deployments

```bash
# Close all deployments from a file
./close_deployments.sh deployments_cpu_20250119_120000.txt

# Or close single deployment
akash tx deployment close --dseq <deployment-id> --from mykey
```

---

## Monitor and Collect Results

### Automatic Collection (IPFS)

Results are automatically uploaded to your IPFS node when mining completes.

**1. Set up admin credentials:**

```bash
# Create credential file
cat > ~/.ipfs_admin_credentials <<EOF
export IPFS_NODE=https://your-ipfs-node.ngrok.io
export IPFS_ADMIN_USER=admin
export IPFS_ADMIN_PASS=your-secure-password
EOF

chmod 600 ~/.ipfs_admin_credentials
```

**2. Collect all results:**

```bash
# Load credentials
source ~/.ipfs_admin_credentials

# Run collection script
./admin_collect_ipfs.sh
```

**What this does:**
- Connects to your IPFS node with admin credentials
- Lists all pinned files
- Identifies mining result manifests
- Downloads results.json, mining.log, and manifest.json
- Organizes by miner type and deployment ID

**Output:**
```
ipfs_results/
├── cpu/
│   ├── deployment_abc123/
│   │   ├── results.json      # Mining metrics
│   │   ├── mining.log        # Complete logs
│   │   └── manifest.json     # IPFS CIDs
│   └── deployment_def456/
│       └── ...
├── cuda/
│   └── deployment_ghi789/
│       └── ...
└── summary.txt               # Aggregated statistics
```

**3. Analyze results:**

```bash
# View summary
cat ipfs_results/summary.txt

# Extract specific metrics
cat ipfs_results/cpu/*/results.json | jq '.results.blocks_found'

# Compare CPU vs CUDA
python ../tools/compare_mining_rates.py \
  --cpu-results ipfs_results/cpu \
  --cuda-results ipfs_results/cuda
```

### Manual Collection (HTTP)

If IPFS upload fails, retrieve results via HTTP while containers are running:

**1. Get deployment URI:**

```bash
# Via Akash CLI
akash provider lease-status \
  --dseq <deployment-id> \
  --from mykey \
  --node https://rpc.akashnet.net:443

# Look for: "uri": "https://..."
```

**2. Download results:**

```bash
# Get results JSON
curl https://deployment-uri/latest.json -o results.json

# Get mining log
curl https://deployment-uri/latest.log -o mining.log

# Get manifest (if IPFS partially succeeded)
curl https://deployment-uri/latest_manifest.json -o manifest.json

# Check status
curl https://deployment-uri/status.txt
```

### Monitoring Logs

**Via Web Interface:**
1. Go to "Manage Deployments" tab
2. Click "View Logs" on deployment
3. Watch real-time output

**Via Akash CLI:**
```bash
# Stream logs
akash provider lease-logs \
  --dseq <deployment-id> \
  --from mykey \
  --node https://rpc.akashnet.net:443 \
  --follow

# Get last 100 lines
akash provider lease-logs \
  --dseq <deployment-id> \
  --from mykey \
  --node https://rpc.akashnet.net:443 \
  --tail 100
```

**What to look for:**
- ✅ `"Mining started..."` - Container initialized
- ✅ `"Attempt X/Y..."` - Mining in progress
- ✅ `"📦 IPFS Manifest CID: QmXXX..."` - Upload succeeded
- ✅ `"🛑 Auto-shutdown enabled - stopping container..."` - Success!
- ⚠️ `"IPFS upload incomplete or failed"` - Need manual retrieval

---

## Configuration Reference

### Miner Types

| Type | Hardware | Cost/90min | Mining Speed | Best For |
|------|----------|------------|--------------|----------|
| **CPU** | 4 vCPU, 8GB RAM | ~$0.15 | Baseline | Cost-effective testing |
| **CUDA** | RTX 3090, 16GB RAM | ~$0.80 | 5-10x faster | Production mining |

### Mining Parameters

**MINING_DURATION**
- How long to run mining
- Format: `90m`, `2h`, `30m`
- Default: `90m`
- Recommendation: 90 minutes for consistent results

**DIFFICULTY_ENERGY**
- Energy threshold for valid solutions
- Must be below this value to accept
- Default: `-14900` (for Zephyr Z9,2 topology)
- Lower = harder, fewer solutions
- Higher = easier, more solutions

**MIN_DIVERSITY**
- Minimum Hamming distance between solutions
- Range: 0.0 to 1.0
- Default: `0.1` (10% different)
- Higher = more diverse solutions required

**MIN_SOLUTIONS**
- Minimum number of valid solutions
- Default: `5`
- Must find at least this many to succeed

**TOPOLOGY_FILE**
- Graph topology to mine on
- Default: `advantage2_system1_7.json.gz`
- Available topologies in `dwave_topologies/topologies/`

### IPFS Configuration

**IPFS_NODE** (Required)
- Your IPFS server URL
- Examples:
  - ngrok: `https://your-subdomain.ngrok.io`
  - Direct: `http://your.server.ip:5001`
  - Domain: `https://ipfs.yourdomain.com`

**IPFS_API_KEY** (Required)
- API key for upload authentication
- Get from your IPFS administrator

**IPFS_PIN**
- Pin files permanently to IPFS
- `true` = keep forever (recommended)
- `false` = may be garbage collected

### Cost Estimates

**CPU Mining:**
- Single instance: $0.15 per 90 minutes
- 10 instances: $1.50 per 90 minutes
- 100 instances: $15.00 per 90 minutes

**CUDA Mining:**
- Single RTX 3090: $0.80 per 90 minutes
- 10 RTX 3090s: $8.00 per 90 minutes
- 100 RTX 3090s: $80.00 per 90 minutes

**With IPFS auto-shutdown:**
- Containers stop 2 minutes after mining completes
- No additional costs for idle containers
- Total runtime: ~92 minutes instead of indefinite

---

## Troubleshooting

### Docker Build Fails

**Error:** "failed to solve: error getting credentials"

**Solution:**
```bash
# Login to Docker Hub
docker login

# Or login to GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin
```

**Error:** "multiple platforms feature is currently not supported for docker driver"

**Solution:**
```bash
# Create buildx builder
docker buildx create --name multiplatform --use
docker buildx inspect --bootstrap

# Try build again
./build_akash_images.sh
```

### Image Pull Fails on Akash (Architecture Mismatch)

**Error:** "exec format error" or "cannot execute binary"

**Cause:** You pushed an arm64 image but Akash providers run amd64.

**Solution:**
```bash
# Verify current image architecture
docker buildx imagetools inspect carback1/quip-protocol-cpu-miner:latest

# If it shows "linux/arm64", rebuild for amd64:
docker buildx build \
  --platform linux/amd64 \
  -f akash/Dockerfile.akash-cpu \
  -t carback1/quip-protocol-cpu-miner:latest \
  --push \
  .
```

**Error:** "pull access denied, repository does not exist"

**Cause:** Wrong image name or image not pushed to registry.

**Solution:**
1. Verify image exists on Docker Hub: https://hub.docker.com/r/carback1/quip-protocol-cpu-miner
2. Check the image name in `akash/web/src/config/constants.ts` matches Docker Hub
3. Make sure the repository is public (not private)

### Keplr Connection Failed

**Error:** "Keplr not found"

**Solutions:**
1. Install Keplr browser extension from https://www.keplr.app/download
2. Refresh page after installing
3. Try a different browser (Chrome recommended)

**Error:** "User rejected request"

**Solutions:**
1. Approve connection in Keplr popup
2. Check that Akash Network is in your Keplr networks
3. Try disconnecting and reconnecting

**Error:** "WebSocket connection to 'wss://bridge.walletconnect.org' failed"

**Cause:** WalletConnect v1 has been deprecated. The old bridge servers are no longer operational.

**Solution:** The code has been updated to use WalletConnect v2. Make sure you:
1. Run the server in HTTPS mode (default):
   ```bash
   python3 serve.py  # without --http flag
   ```
2. Accept the browser security warning
3. Click "📱 Connect Keplr Mobile (WalletConnect)" button
4. Scan QR code with Keplr Mobile app
5. The new WalletConnect v2 infrastructure will be used automatically

**Error:** "Failed to connect via WalletConnect" or QR code doesn't appear

**Solutions:**
1. **Check browser console** for specific errors (F12 → Console tab)
2. **Verify HTTPS is enabled** - WalletConnect v2 requires secure connections
3. **Try a different browser** - Chrome/Firefox recommended
4. **Clear browser cache** and refresh the page
5. **Use desktop Keplr extension** as fallback (doesn't require WalletConnect)

### IPFS Upload Failed

**Symptoms:** Container stays running after 90+ minutes

**Check deployment logs:**
```bash
# Via CLI
akash provider lease-logs --dseq <id> ... | tail -50

# Look for:
# ❌ "IPFS upload failed"
# ⚠️  "IPFS upload incomplete or failed"
```

**Solutions:**

1. **Verify IPFS node is accessible:**
   ```bash
   curl https://your-ipfs-node.ngrok.io/api/v0/version
   ```

2. **Test API key:**
   ```bash
   curl -X POST \
     -H "Authorization: Bearer $IPFS_API_KEY" \
     -F "file=@test.txt" \
     "https://your-ipfs-node.ngrok.io/api/v0/add"
   ```

3. **Retrieve results manually:**
   ```bash
   # Get deployment URI from Akash
   URI=$(akash provider lease-status --dseq <id> ... | grep uri)

   # Download results
   curl "$URI/latest.json" -o results.json
   curl "$URI/latest.log" -o mining.log
   ```

4. **Check IPFS_NODE URL matches:**
   - SDL file uses: `IPFS_NODE=https://...`
   - Collection uses: `export IPFS_NODE=https://...`
   - **Must be identical!**

### No Results Found

**Error:** "No pinned files found on IPFS node"

**Solutions:**

1. **Check if mining completed:**
   ```bash
   # View logs
   akash provider lease-logs --dseq <id> ...

   # Look for "COMPLETED" in logs
   ```

2. **Check if IPFS_PIN was enabled:**
   ```yaml
   # In SDL file, should have:
   - IPFS_PIN=true
   ```

3. **Try different IPFS gateway:**
   ```bash
   # Use public gateway
   export IPFS_GATEWAY=https://ipfs.io
   ./admin_collect_ipfs.sh
   ```

### Deployment Costs Too High

**Solutions:**

1. **Use CPU instead of CUDA** (10x cheaper)
2. **Reduce fleet size** (start with 5-10 instances)
3. **Enable auto-shutdown** (stops billing after 92 minutes)
4. **Close promptly** (use web interface to close)
5. **Monitor spending** (check deployment list regularly)

### Container Won't Start

**Check provider compatibility:**
- Some providers don't support GPUs
- Some providers have resource limits
- Try different provider or reduce resources

**Check SDL syntax:**
```bash
# Validate SDL file
akash deployment validate deploy-cpu-ipfs.yaml
```

**Check image accessibility:**
```bash
# Verify image is public
docker pull yourusername/quip-protocol-cpu-miner:latest

# If private, make repository public on Docker Hub
```

---

## Quick Reference

### Deploy Workflow

1. **Build images** (once):
   ```bash
   docker login
   cd akash
   # Update REGISTRY in build_akash_images.sh
   ./build_akash_images.sh
   ```

2. **Configure IPFS** (once):
   ```bash
   # Edit deploy-cpu-ipfs.yaml and deploy-cuda-ipfs.yaml
   # Set IPFS_NODE and IPFS_API_KEY
   ```

3. **Deploy** (each run):
   ```bash
   cd akash/web
   python3 serve.py
   # Use web interface to deploy
   ```

4. **Wait** (90 minutes):
   - Containers mine automatically
   - Results upload to IPFS
   - Containers auto-shutdown

5. **Collect** (after mining):
   ```bash
   source ~/.ipfs_admin_credentials
   ./admin_collect_ipfs.sh
   ```

### Important Files

- `deploy-cpu-ipfs.yaml` - CPU miner configuration
- `deploy-cuda-ipfs.yaml` - CUDA miner configuration
- `build_akash_images.sh` - Docker build script
- `admin_collect_ipfs.sh` - Result collection script
- `deployments_*.txt` - Deployment ID lists

### Environment Variables

**For deployment (in SDL files):**
- `IPFS_NODE` - Your IPFS server URL
- `IPFS_API_KEY` - Upload authentication key
- `IPFS_PIN` - Pin files permanently (true/false)
- `MINING_DURATION` - How long to mine
- `DIFFICULTY_ENERGY` - Energy threshold
- `MIN_DIVERSITY` - Solution diversity
- `MIN_SOLUTIONS` - Minimum solutions

**For collection (in shell):**
- `IPFS_NODE` - Your IPFS server URL (must match SDL)
- `IPFS_ADMIN_USER` - Admin username
- `IPFS_ADMIN_PASS` - Admin password

### Support

- **Akash Network:** https://akash.network/
- **Akash Docs:** https://docs.akash.network/
- **Akash Discord:** https://discord.akash.network/
- **Keplr Wallet:** https://www.keplr.app/
- **IPFS Docs:** https://docs.ipfs.tech/

---

## Summary

Akash Network provides a cost-effective platform for large-scale mining experiments:

✅ **Easy deployment** - Web interface with Keplr wallet
✅ **50-80% cost savings** - vs AWS/GCP
✅ **Auto-shutdown** - Stop billing when mining completes
✅ **Decentralized storage** - IPFS for permanent results
✅ **Scalable** - Deploy 1 to 100+ instances
✅ **Multi-platform** - CPU and CUDA support

**Get started in 3 steps:**
1. Build and push Docker images
2. Configure IPFS settings
3. Deploy via web interface

Happy mining! ⛏️
