#!/bin/bash
# Default certbot wrapper for Quip Network Node Docker images.
# Seeded to /data/certs/certbot on first run — replace with custom logic
# (acme.sh, lego, manual) if needed. Cron calls the same script.
#
# Usage: ./certbot <domain>
# Reads CERT_* environment variables. Writes certs to /data/certs/private/.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
set -euo pipefail

DOMAIN="${1:?Usage: $0 <domain>}"
CERT_DIR="/data/certs/private"
CERT_FILE="$CERT_DIR/fullchain.pem"
KEY_FILE="$CERT_DIR/privkey.pem"

# Skip if certificate exists and has >30 days remaining
if [ -f "$CERT_FILE" ]; then
    if openssl x509 -checkend 2592000 -noout -in "$CERT_FILE" 2>/dev/null; then
        echo "certbot: certificate for $DOMAIN still valid (>30 days remaining), skipping."
        exit 0
    fi
    echo "certbot: certificate expires within 30 days, renewing..."
fi

mkdir -p "$CERT_DIR"
chmod 700 "$CERT_DIR"

# Build certbot arguments
CERTBOT_ARGS=(
    certonly
    --non-interactive
    --agree-tos
    --domain "$DOMAIN"
    --cert-path "$CERT_FILE"
    --key-path "$KEY_FILE"
    --fullchain-path "$CERT_FILE"
    --config-dir /data/certs/certbot-config
    --work-dir /data/certs/certbot-work
    --logs-dir /data/certs/certbot-logs
)

# Email (required by plan gate, but handle --register-unsafely-without-email as fallback)
if [ -n "${CERT_EMAIL:-}" ]; then
    CERTBOT_ARGS+=(--email "$CERT_EMAIL")
else
    CERTBOT_ARGS+=(--register-unsafely-without-email)
fi

# ACME server (default: Let's Encrypt production)
if [ -n "${CERT_ACME_SERVER:-}" ]; then
    CERTBOT_ARGS+=(--server "$CERT_ACME_SERVER")
fi

# Staging mode (Let's Encrypt staging server for testing)
if [ "${CERT_STAGING:-}" = "true" ]; then
    CERTBOT_ARGS+=(--staging)
fi

# External Account Binding (ZeroSSL, Buypass, etc.)
if [ -n "${CERT_EAB_KID:-}" ] && [ -n "${CERT_EAB_HMAC_KEY:-}" ]; then
    CERTBOT_ARGS+=(--eab-kid "$CERT_EAB_KID" --eab-hmac-key "$CERT_EAB_HMAC_KEY")
fi

# Challenge type
CHALLENGE="${CERT_CHALLENGE:-http}"
case "$CHALLENGE" in
    http)
        CERTBOT_ARGS+=(--standalone --http-01-port 80)
        ;;
    dns)
        PLUGIN="${CERT_DNS_PLUGIN:?CERT_DNS_PLUGIN required when CERT_CHALLENGE=dns}"
        CERTBOT_ARGS+=("--dns-$PLUGIN")
        if [ -n "${CERT_DNS_CREDENTIALS:-}" ]; then
            CERTBOT_ARGS+=("--dns-${PLUGIN}-credentials" "$CERT_DNS_CREDENTIALS")
        fi
        ;;
    *)
        echo "certbot: unknown CERT_CHALLENGE='$CHALLENGE' (expected 'http' or 'dns')" >&2
        exit 1
        ;;
esac

echo "certbot: obtaining certificate for $DOMAIN ..."
certbot "${CERTBOT_ARGS[@]}"

# Lock down private key
chmod 600 "$KEY_FILE" 2>/dev/null || true
chmod 644 "$CERT_FILE" 2>/dev/null || true

echo "certbot: certificate for $DOMAIN saved to $CERT_DIR"
