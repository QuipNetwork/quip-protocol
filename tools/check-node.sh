#!/bin/bash
# Check health and telemetry of any QuIP network node via its REST API.
#
# Usage:
#   bash tools/check-node.sh <host:port> [--https] [--verbose]
#
# Examples:
#   bash tools/check-node.sh mynode.example.com:20050
#   bash tools/check-node.sh 10.0.1.5:8080 --verbose
#   bash tools/check-node.sh mynode.example.com:443 --https
set -euo pipefail

# ── Args ────────────────────────────────────────────────────────────────────
TARGET=""
SCHEME="http"
VERBOSE=false

for arg in "$@"; do
    case "$arg" in
        --https)   SCHEME="https" ;;
        --verbose) VERBOSE=true ;;
        -*)        echo "Unknown option: $arg"; exit 1 ;;
        *)         TARGET="$arg" ;;
    esac
done

if [ -z "$TARGET" ]; then
    echo "Usage: $0 <host:port> [--https] [--verbose]"
    exit 1
fi

BASE_URL="${SCHEME}://${TARGET}"

# ── Colors ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

PASS=0
FAIL=0
WARN=0
TMPFILE=$(mktemp)
trap "rm -f $TMPFILE" EXIT

# ── Helpers ──────────────────────────────────────────────────────────────────
check_endpoint() {
    local method="$1"
    local path="$2"
    local label="${3:-$path}"
    local expected="${4:-200}"

    HTTP_CODE=$(curl -sk -o "$TMPFILE" -w "%{http_code}" \
        -X "$method" "${BASE_URL}${path}" 2>/dev/null || echo "000")

    if [ "$HTTP_CODE" = "$expected" ]; then
        echo -e "  ${GREEN}PASS${NC}  ${label} (${HTTP_CODE})"
        PASS=$((PASS + 1))
        return 0
    elif [ "$HTTP_CODE" = "503" ] && [ "$path" = "/api/v1/stats" ]; then
        echo -e "  ${YELLOW}WARN${NC}  ${label} (503 — stats cache warming up)"
        WARN=$((WARN + 1))
        return 0
    else
        echo -e "  ${RED}FAIL${NC}  ${label} (HTTP ${HTTP_CODE})"
        FAIL=$((FAIL + 1))
        return 1
    fi
}

jq_or_python() {
    # Extract a dotted key path from JSON on stdin, using jq if available.
    local keypath="$1"  # e.g. ".data.running"
    local pypath="$2"   # e.g. "d.get('data',{}).get('running','')"
    if command -v jq &>/dev/null; then
        jq -r "$keypath // empty" 2>/dev/null || true
    else
        python3 -c "import sys,json; d=json.load(sys.stdin); print($pypath)" 2>/dev/null || true
    fi
}

show_field() {
    local label="$1"
    local value="$2"
    printf "    ${CYAN}%-20s${NC} %s\n" "$label" "${value:-(n/a)}"
}

# ── Header ───────────────────────────────────────────────────────────────────
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  QuIP Node Health Check${NC}"
echo -e "${BLUE}  ${BASE_URL}${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# ── Phase 1: Reachability ────────────────────────────────────────────────────
echo -e "${BLUE}[1] Reachability${NC}"
if ! check_endpoint GET /health "/health"; then
    echo ""
    echo -e "  ${RED}Node is unreachable or unhealthy. Aborting.${NC}"
    exit 1
fi

if $VERBOSE; then
    RESP=$(cat "$TMPFILE")
    VERSION=$(echo "$RESP" | jq_or_python '.data.version' "d.get('data',{}).get('version','')")
    show_field "version" "$VERSION"
fi

# ── Phase 2: Endpoint checks ─────────────────────────────────────────────────
echo ""
echo -e "${BLUE}[2] Endpoint checks${NC}"
check_endpoint GET /api/v1/status  "GET /api/v1/status"
check_endpoint GET /api/v1/stats   "GET /api/v1/stats"
check_endpoint GET /api/v1/peers   "GET /api/v1/peers"
check_endpoint GET /api/v1/block/latest "GET /api/v1/block/latest"

# ── Phase 3: Telemetry content ───────────────────────────────────────────────
echo ""
echo -e "${BLUE}[3] Telemetry${NC}"

STATUS_RESP=$(curl -sk "${BASE_URL}/api/v1/status" 2>/dev/null || echo "{}")
RUNNING=$(echo "$STATUS_RESP"  | jq_or_python '.data.running'      "d.get('data',{}).get('running','')")
PEERS=$(echo "$STATUS_RESP"    | jq_or_python '.data.total_peers'  "d.get('data',{}).get('total_peers',0)")
BLOCK=$(echo "$STATUS_RESP"    | jq_or_python '.data.latest_block' "d.get('data',{}).get('latest_block',-1)")
HOST=$(echo "$STATUS_RESP"     | jq_or_python '.data.host'         "d.get('data',{}).get('host','')")
NODE_NAME=$(echo "$STATUS_RESP"| jq_or_python '.data.info.node_name' "d.get('data',{}).get('info',{}).get('node_name','')")
MINER_TYPE=$(echo "$STATUS_RESP"| jq_or_python '.data.info.miner_type' "d.get('data',{}).get('info',{}).get('miner_type','')")

if [ "$RUNNING" = "true" ] || [ "$RUNNING" = "True" ]; then
    echo -e "  ${GREEN}PASS${NC}  node is running"
    PASS=$((PASS + 1))
else
    echo -e "  ${RED}FAIL${NC}  node.running != true (got: ${RUNNING})"
    FAIL=$((FAIL + 1))
fi

if [ "${BLOCK:-"-1"}" -ge 0 ] 2>/dev/null; then
    echo -e "  ${GREEN}PASS${NC}  latest block index = ${BLOCK}"
    PASS=$((PASS + 1))
else
    echo -e "  ${YELLOW}WARN${NC}  no blocks yet (mining may still be starting)"
    WARN=$((WARN + 1))
fi

if $VERBOSE || true; then
    show_field "public host"  "$HOST"
    show_field "node name"    "$NODE_NAME"
    show_field "miner type"   "$MINER_TYPE"
    show_field "peers"        "$PEERS"
    show_field "latest block" "$BLOCK"
fi

BLOCK_RESP=$(curl -sk "${BASE_URL}/api/v1/block/latest" 2>/dev/null || echo "{}")
BLOCK_HASH=$(echo "$BLOCK_RESP" | jq_or_python '.data.header.prev_hash' \
    "d.get('data',{}).get('header',{}).get('prev_hash','')")
if $VERBOSE; then
    show_field "prev_hash" "${BLOCK_HASH:0:16}…"
fi

# ── Phase 4: Peer count ───────────────────────────────────────────────────────
echo ""
echo -e "${BLUE}[4] Peers${NC}"
PEER_RESP=$(curl -sk "${BASE_URL}/api/v1/peers" 2>/dev/null || echo "{}")
PEER_COUNT=$(echo "$PEER_RESP" | jq_or_python '.data.count' "d.get('data',{}).get('count',0)")

if [ "${PEER_COUNT:-0}" -ge 1 ] 2>/dev/null; then
    echo -e "  ${GREEN}PASS${NC}  node sees ${PEER_COUNT} peer(s)"
    PASS=$((PASS + 1))
else
    echo -e "  ${YELLOW}WARN${NC}  node sees 0 peers (may be bootstrapping or standalone)"
    WARN=$((WARN + 1))
fi

if $VERBOSE; then
    echo "$PEER_RESP" | jq_or_python '.data.peers | keys[]' \
        "[ print(k) for k in d.get('data',{}).get('peers',{}).keys() ]" \
        | while read -r peer; do
            show_field "peer" "$peer"
        done
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${BLUE}========================================${NC}"
printf "  Results: ${GREEN}%d passed${NC}" "$PASS"
[ "$WARN" -gt 0 ] && printf ", ${YELLOW}%d warnings${NC}" "$WARN"
[ "$FAIL" -gt 0 ] && printf ", ${RED}%d failed${NC}" "$FAIL"
echo ""
echo -e "${BLUE}========================================${NC}"

[ "$FAIL" -eq 0 ] && exit 0 || exit 1

