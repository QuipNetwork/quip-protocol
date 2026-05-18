#!/bin/bash
# Test REST API telemetry endpoints on local 3-node cluster
#
# Prerequisites:
#   docker compose -f docker/docker-compose.local.yml up --build -d
#
# Usage:
#   bash docker/test-rest-telemetry.sh
set -euo pipefail

NODES=("127.0.0.1:20050" "127.0.0.1:20051" "127.0.0.1:20052")
NODE_NAMES=("bootstrap" "node-2" "node-3")
MAX_WAIT=120
POLL_INTERVAL=3

ENDPOINTS=(
    "/health"
    "/api/v1/status"
    "/api/v1/stats"
    "/api/v1/peers"
    "/api/v1/block/latest"
    "/api/v1/system"
)

GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS=0
FAIL=0
WARN=0

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Quip REST API Telemetry Test${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Phase 1: Wait for all nodes to become healthy
echo -e "${BLUE}Phase 1: Waiting for nodes...${NC}"
for i in "${!NODES[@]}"; do
    printf "  %-12s (%s) " "${NODE_NAMES[$i]}" "${NODES[$i]}"
    elapsed=0
    while [ $elapsed -lt $MAX_WAIT ]; do
        if curl -sf "http://${NODES[$i]}/health" > /dev/null 2>&1; then
            echo -e "${GREEN}ready${NC}"
            break
        fi
        sleep $POLL_INTERVAL
        elapsed=$((elapsed + POLL_INTERVAL))
        printf "."
    done
    if [ $elapsed -ge $MAX_WAIT ]; then
        echo -e " ${RED}TIMEOUT (${MAX_WAIT}s)${NC}"
        echo ""
        echo "Hint: start the cluster first:"
        echo "  docker compose -f docker/docker-compose.local.yml up --build -d"
        exit 1
    fi
done

# Phase 2: Test each endpoint on each node
echo ""
echo -e "${BLUE}Phase 2: Endpoint checks...${NC}"
TMPFILE=$(mktemp)
trap "rm -f $TMPFILE" EXIT

for i in "${!NODES[@]}"; do
    echo -e "  ${BLUE}--- ${NODE_NAMES[$i]} (${NODES[$i]}) ---${NC}"
    for endpoint in "${ENDPOINTS[@]}"; do
        HTTP_CODE=$(curl -s -o "$TMPFILE" -w "%{http_code}" \
            "http://${NODES[$i]}${endpoint}" 2>/dev/null || echo "000")
        if [ "$HTTP_CODE" = "200" ]; then
            echo -e "    ${GREEN}PASS${NC}  $endpoint"
            PASS=$((PASS + 1))
        elif [ "$HTTP_CODE" = "503" ] && [ "$endpoint" = "/api/v1/stats" ]; then
            echo -e "    ${YELLOW}WARN${NC}  $endpoint (503 — stats cache warming up)"
            WARN=$((WARN + 1))
        else
            echo -e "    ${RED}FAIL${NC}  $endpoint (HTTP $HTTP_CODE)"
            FAIL=$((FAIL + 1))
        fi
    done
done

# Phase 3: Telemetry content verification
echo ""
echo -e "${BLUE}Phase 3: Telemetry verification...${NC}"

for i in "${!NODES[@]}"; do
    RESP=$(curl -sf "http://${NODES[$i]}/api/v1/status" 2>/dev/null || echo "{}")
    RUNNING=$(echo "$RESP" | python3 -c \
        "import sys,json; d=json.load(sys.stdin); print(d.get('data',{}).get('running',''))" \
        2>/dev/null || echo "")
    if [ "$RUNNING" = "True" ] || [ "$RUNNING" = "true" ]; then
        echo -e "  ${GREEN}PASS${NC}  ${NODE_NAMES[$i]} status.running=true"
        PASS=$((PASS + 1))
    else
        echo -e "  ${RED}FAIL${NC}  ${NODE_NAMES[$i]} status.running != true (got: $RUNNING)"
        FAIL=$((FAIL + 1))
    fi
done

# Check peer discovery (give bootstrap a moment to learn about peers)
echo ""
echo -e "${BLUE}Phase 4: Peer discovery...${NC}"
echo -n "  Waiting 10s for peer discovery..."
sleep 10
echo " done"

PEER_RESP=$(curl -sf "http://${NODES[0]}/api/v1/peers" 2>/dev/null || echo "{}")
PEER_COUNT=$(echo "$PEER_RESP" | python3 -c \
    "import sys,json; print(json.load(sys.stdin).get('data',{}).get('count',0))" \
    2>/dev/null || echo "0")

if [ "$PEER_COUNT" -ge 2 ]; then
    echo -e "  ${GREEN}PASS${NC}  bootstrap sees $PEER_COUNT peers (expected >= 2)"
    PASS=$((PASS + 1))
else
    echo -e "  ${YELLOW}WARN${NC}  bootstrap sees $PEER_COUNT peers (expected >= 2, may need more time)"
    WARN=$((WARN + 1))
fi

# Phase 5: System descriptor + scrubbing check
echo ""
echo -e "${BLUE}Phase 5: System descriptor...${NC}"
for i in "${!NODES[@]}"; do
    SYS_RESP=$(curl -sf "http://${NODES[$i]}/api/v1/system" 2>/dev/null || echo "{}")
    # Extract a few required fields
    DESC_VERSION=$(echo "$SYS_RESP" | python3 -c \
        "import sys,json; d=json.load(sys.stdin); print(d.get('data',{}).get('descriptor_version',''))" \
        2>/dev/null || echo "")
    CPU_BRAND=$(echo "$SYS_RESP" | python3 -c \
        "import sys,json; d=json.load(sys.stdin); print(d.get('data',{}).get('system_info',{}).get('cpu',{}).get('brand',''))" \
        2>/dev/null || echo "")
    MINER_TYPE=$(curl -sf "http://${NODES[$i]}/api/v1/status" 2>/dev/null \
        | python3 -c "import sys,json; d=json.load(sys.stdin); info=d.get('data',{}).get('info',{}); print(info.get('miner_type','') if isinstance(info, dict) else json.loads(info).get('miner_type',''))" \
        2>/dev/null || echo "")

    if [ "$DESC_VERSION" = "1" ] && [ -n "$CPU_BRAND" ]; then
        echo -e "  ${GREEN}PASS${NC}  ${NODE_NAMES[$i]} descriptor v$DESC_VERSION, cpu='$CPU_BRAND', miner_type='$MINER_TYPE'"
        PASS=$((PASS + 1))
    else
        echo -e "  ${RED}FAIL${NC}  ${NODE_NAMES[$i]} descriptor_version='$DESC_VERSION', cpu_brand='$CPU_BRAND'"
        FAIL=$((FAIL + 1))
    fi

    # Scrubbing: the JSON dump of system/status/peers/stats must never
    # contain listen IPs, peer lists, or heartbeat internals.
    ALL_BLOBS=$(
        curl -sf "http://${NODES[$i]}/api/v1/system" 2>/dev/null;
        curl -sf "http://${NODES[$i]}/api/v1/status" 2>/dev/null;
        curl -sf "http://${NODES[$i]}/api/v1/stats"  2>/dev/null;
        curl -sf "http://${NODES[$i]}/api/v1/peers"  2>/dev/null
    )
    LEAK=""
    # We look for TOML-config keys that must never appear in outward JSON.
    # "listen" and "heartbeat_interval" are the canonical smoking guns.
    for needle in '"listen"' '"heartbeat_interval"' '"secret"' '"tls_cert_file"'; do
        if echo "$ALL_BLOBS" | grep -q "$needle"; then
            LEAK="$LEAK $needle"
        fi
    done
    if [ -z "$LEAK" ]; then
        echo -e "  ${GREEN}PASS${NC}  ${NODE_NAMES[$i]} no sensitive config keys in REST output"
        PASS=$((PASS + 1))
    else
        echo -e "  ${RED}FAIL${NC}  ${NODE_NAMES[$i]} leaked:$LEAK"
        FAIL=$((FAIL + 1))
    fi
done

# Check that mining has started (at least genesis block)
BLOCK_RESP=$(curl -sf "http://${NODES[0]}/api/v1/block/latest" 2>/dev/null || echo "{}")
BLOCK_INDEX=$(echo "$BLOCK_RESP" | python3 -c \
    "import sys,json; d=json.load(sys.stdin); print(d.get('data',{}).get('header',{}).get('index',-1))" \
    2>/dev/null || echo "-1")

if [ "$BLOCK_INDEX" -ge 0 ]; then
    echo -e "  ${GREEN}PASS${NC}  latest block index=$BLOCK_INDEX"
    PASS=$((PASS + 1))
else
    echo -e "  ${YELLOW}WARN${NC}  no blocks yet (mining may still be starting)"
    WARN=$((WARN + 1))
fi

# Summary
echo ""
echo -e "${BLUE}========================================${NC}"
printf "  Results: ${GREEN}%d passed${NC}" "$PASS"
[ "$WARN" -gt 0 ] && printf ", ${YELLOW}%d warnings${NC}" "$WARN"
[ "$FAIL" -gt 0 ] && printf ", ${RED}%d failed${NC}" "$FAIL"
echo ""
echo -e "${BLUE}========================================${NC}"

[ "$FAIL" -eq 0 ] && exit 0 || exit 1
