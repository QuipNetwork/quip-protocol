#!/bin/bash
# Entrypoint script for Quip Network Node (CPU and GPU)
#
# Config philosophy: /data/config.toml is the source of truth.
# - First run: seeds TOML from template + ENV vars
# - Subsequent runs: reads from TOML; ENV vars override only when set
# - Node launches with just --config (no CLI arg construction)
set -e

echo "========================================"
echo "Quip Protocol Network Node"
echo "========================================"
echo "Start time: $(date)"
echo "Mode: $QUIP_MODE"

CONFIG_FILE="/data/config.toml"
TEMPLATE_FILE="/app/quip-node.docker.toml"

# ── Helper: read a TOML value from [global] section ──────────────
# Usage: toml_get <key> → prints value (unquoted), empty string if not found
toml_get() {
    local key="$1"
    # Match key = "value" or key = value in [global] section
    # Stops at the next [section] header
    sed -n '/^\[global\]/,/^\[/{
        s/^'"$key"' *= *"\(.*\)"/\1/p
        s/^'"$key"' *= *\([^"]\S*\)/\1/p
    }' "$CONFIG_FILE" | head -1
}

# ── Helper: set a TOML value in [global] section ─────────────────
# Usage: toml_set <key> <value> [type]
# type: "string" (default, wraps in quotes), "int", "bool", "array"
toml_set() {
    local key="$1" value="$2" type="${3:-string}"
    case "$type" in
        string) value="\"$value\"" ;;
        int|bool|array) ;;  # use raw value
    esac
    if grep -q "^$key *= " "$CONFIG_FILE"; then
        sed -i "s|^$key *= .*|$key = $value|" "$CONFIG_FILE"
    elif grep -q "^# *$key *= " "$CONFIG_FILE"; then
        sed -i "s|^# *$key *= .*|$key = $value|" "$CONFIG_FILE"
    else
        # Insert after [global] line
        sed -i "/^\[global\]/a $key = $value" "$CONFIG_FILE"
    fi
}

# ── Helper: resolve a value (ENV overrides TOML) ─────────────────
# Usage: resolve <env_var_name> <toml_key> [toml_type]
# Sets the shell variable named by env_var_name. If ENV is set, writes to TOML.
resolve() {
    local env_name="$1" toml_key="$2" toml_type="${3:-string}"
    local env_val="${!env_name:-}"
    if [ -n "$env_val" ]; then
        # ENV is set — write to TOML so config stays authoritative
        toml_set "$toml_key" "$env_val" "$toml_type"
    else
        # ENV not set — read from TOML
        local toml_val
        toml_val=$(toml_get "$toml_key")
        if [ -n "$toml_val" ]; then
            export "$env_name=$toml_val"
        fi
    fi
}

# ── First run: generate config from template ──────────────────────
if [ ! -f "$CONFIG_FILE" ]; then
    echo "First run detected - generating config with new secret..."
    mkdir -p /data

    # Generate random 64-char hex secret
    SECRET=$(openssl rand -hex 32)

    # Copy template and set secret
    cp "$TEMPLATE_FILE" "$CONFIG_FILE"
    sed -i "s/secret = \"GENERATE_ON_FIRST_RUN\"/secret = \"$SECRET\"/" "$CONFIG_FILE"

    echo "Config generated at $CONFIG_FILE"
    echo "Secret has been randomly generated and saved."
else
    echo "Using existing config at $CONFIG_FILE"
fi

# ── Copy genesis block to data dir if not present ─────────────────
GENESIS_CONFIG=$(toml_get genesis_config)
GENESIS_CONFIG="${GENESIS_CONFIG:-genesis_block.json}"
if [ ! -f "/data/$GENESIS_CONFIG" ]; then
    if [ -f "/app/$GENESIS_CONFIG" ]; then
        cp "/app/$GENESIS_CONFIG" "/data/$GENESIS_CONFIG"
        echo "Copied default genesis block to /data/$GENESIS_CONFIG"
    else
        echo "WARNING: genesis block $GENESIS_CONFIG not found in /app/"
    fi
fi

# ── Resolve all settings (ENV overrides TOML) ─────────────────────
resolve QUIP_LISTEN        listen        string
resolve QUIP_PORT          port          int
resolve QUIP_PUBLIC_HOST   public_host   string
resolve QUIP_PUBLIC_PORT   public_port   int
resolve QUIP_NODE_NAME     node_name     string
resolve QUIP_AUTO_MINE     auto_mine     bool

# Peers: ENV is comma-separated, TOML is array
if [ -n "${QUIP_PEERS:-}" ]; then
    # Convert "a,b,c" → ["a", "b", "c"]
    PEER_ARRAY=""
    IFS=',' read -ra PEERS <<< "$QUIP_PEERS"
    for p in "${PEERS[@]}"; do
        p=$(echo "$p" | xargs)
        if [ -n "$p" ]; then
            [ -n "$PEER_ARRAY" ] && PEER_ARRAY="$PEER_ARRAY, "
            PEER_ARRAY="$PEER_ARRAY\"$p\""
        fi
    done
    toml_set peer "[$PEER_ARRAY]" array
fi

echo "Config: listen=$(toml_get listen) port=$(toml_get port)"
echo "Config: public_host=$(toml_get public_host) public_port=$(toml_get public_port)"
echo "Config: node_name=$(toml_get node_name) auto_mine=$(toml_get auto_mine)"

# ── Auto-detect hardware and write into TOML config ───────────────
if [ "$QUIP_MODE" = "gpu" ]; then
    echo "----------------------------------------"
    echo "GPU Mode - Detecting NVIDIA GPUs..."
    nvidia-smi 2>&1 || { echo "ERROR: nvidia-smi not available - no GPUs detected. Run with --gpus all"; exit 1; }
    echo "----------------------------------------"

    NUM_GPUS=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
    echo "Detected GPUs: $NUM_GPUS"

    if [ "$NUM_GPUS" -eq 0 ]; then
        echo "ERROR: No GPUs detected. Make sure to run with --gpus all"
        exit 1
    fi

    # Write [cuda.N] device sections into config if not already present
    if ! grep -q '^\[cuda\.' "$CONFIG_FILE"; then
        echo "" >> "$CONFIG_FILE"
        for ((i=0; i<NUM_GPUS; i++)); do
            echo "[cuda.$i]" >> "$CONFIG_FILE"
            echo "" >> "$CONFIG_FILE"
        done
        echo "Wrote $NUM_GPUS GPU device section(s) into $CONFIG_FILE"
    else
        echo "GPU device sections already present in $CONFIG_FILE"
    fi

else
    # CPU mode (default) — write num_cpus into [cpu] section if not set
    NUM_CPUS=$(nproc)
    echo "CPU Mode - Detected CPUs: $NUM_CPUS"

    if ! grep -q '^num_cpus' "$CONFIG_FILE"; then
        sed -i '/^\[cpu\]/a num_cpus = '"$NUM_CPUS" "$CONFIG_FILE"
        echo "Set num_cpus = $NUM_CPUS in $CONFIG_FILE"
    else
        echo "num_cpus already configured in $CONFIG_FILE"
    fi
fi

# ── TLS Certificate Management (certbot) ──────────────────────────
# Activates when public_host is a DNS name AND CERT_EMAIL is set.
# Certs are written to /data/certs/private/ and injected into config.toml.

CERTBOT_SCRIPT="/data/certs/certbot"
PUBLIC_HOST=$(toml_get public_host)

# Returns 0 (true) if the argument is a DNS name, 1 if it's an IP address
is_dns_name() {
    python3 -c "import ipaddress; ipaddress.ip_address('$1')" 2>/dev/null && return 1
    return 0
}

if [ -n "$PUBLIC_HOST" ]; then
    if is_dns_name "$PUBLIC_HOST"; then
        if [ -n "${CERT_EMAIL:-}" ]; then
            echo "TLS: domain=$PUBLIC_HOST, email=$CERT_EMAIL — certbot enabled"

            # Seed default certbot wrapper if user hasn't provided one
            if [ ! -f "$CERTBOT_SCRIPT" ]; then
                mkdir -p /data/certs
                cp /app/certbot-default.sh "$CERTBOT_SCRIPT"
                chmod +x "$CERTBOT_SCRIPT"
                echo "TLS: seeded default certbot script at $CERTBOT_SCRIPT"
            else
                echo "TLS: using existing certbot script at $CERTBOT_SCRIPT"
            fi

            # Run certbot on startup
            if "$CERTBOT_SCRIPT" "$PUBLIC_HOST"; then
                CERT_FILE="/data/certs/private/fullchain.pem"
                KEY_FILE="/data/certs/private/privkey.pem"

                if [ -f "$CERT_FILE" ] && [ -f "$KEY_FILE" ]; then
                    chmod 600 "$KEY_FILE"
                    echo "TLS: certificate obtained, injecting paths into config"

                    # Inject cert paths into [global] section of config.toml
                    # Shared by both QUIC and REST API
                    # Use exact key match (^tls_) to avoid matching rest_tls_cert_file from old configs
                    if ! grep -q '^tls_cert_file' "$CONFIG_FILE"; then
                        toml_set tls_cert_file "$CERT_FILE" string
                        toml_set tls_key_file "$KEY_FILE" string
                    fi
                else
                    echo "TLS: WARNING — certbot succeeded but cert files not found, falling back to self-signed"
                fi
            else
                echo "TLS: WARNING — certbot failed, falling back to self-signed certificates"
            fi

            # Install cron job for daily renewal at 3am
            # Cron doesn't inherit environment, so pass CERT_* vars explicitly
            CRON_ENV=""
            for var in CERT_EMAIL CERT_CHALLENGE CERT_DNS_PLUGIN CERT_DNS_CREDENTIALS \
                       CERT_ACME_SERVER CERT_EAB_KID CERT_EAB_HMAC_KEY CERT_STAGING; do
                val="${!var:-}"
                if [ -n "$val" ]; then
                    CRON_ENV="$CRON_ENV $var=$val"
                fi
            done

            CRON_LINE="0 3 * * * $CRON_ENV $CERTBOT_SCRIPT $PUBLIC_HOST >> /data/certs/renewal.log 2>&1"
            mkdir -p /var/spool/cron/crontabs
            echo "$CRON_LINE" | busybox crontab -

            # Start crond in background (tini reaps if it dies)
            busybox crond -L /data/certs/renewal.log
            echo "TLS: cron renewal installed (daily 3am), crond started"
        else
            echo "TLS: DNS domain detected ($PUBLIC_HOST) but CERT_EMAIL not set — skipping certbot"
        fi
    else
        echo "TLS: public_host is an IP address ($PUBLIC_HOST) — no certbot (use DNS name for Let's Encrypt)"
    fi
else
    echo "TLS: public_host not set — skipping certbot"
fi

# ── Launch ────────────────────────────────────────────────────────
echo "========================================"
echo "Starting Quip Network Node..."
echo "========================================"

CMD="quip-network-node --config $CONFIG_FILE"
echo "Command: $CMD"
echo "----------------------------------------"

# Execute the network node (replace shell process for proper signal handling)
exec $CMD
