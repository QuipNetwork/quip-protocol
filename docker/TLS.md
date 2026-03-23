# TLS Certificate Management

Automated Let's Encrypt (and custom ACME) certificate management for Quip Network Node Docker images.

## Overview

Certbot activates when **both** conditions are met:

1. `QUIP_PUBLIC_HOST` resolves to a DNS name (not an IP address)
2. `CERT_EMAIL` is set

Certificates are stored at `/data/certs/private/` and automatically injected into the node's TOML config. Renewal runs daily at 3am via cron.

When either condition is missing, the node falls back to self-signed certificates (default behavior).

## HTTP-01 Challenge (Default)

The simplest setup. Requires port 80 to be reachable from the internet.

```bash
docker run -d --name quip-cpu \
  -v ~/quip-data:/data \
  -e QUIP_PUBLIC_HOST=mynode.example.com \
  -e CERT_EMAIL=admin@example.com \
  -p 20049:20049/udp -p 20049:20049/tcp \
  -p 80:80/tcp \
  registry.gitlab.com/piqued/quip-protocol/quip-network-node-cpu:latest
```

**How it works:** The certbot script auto-detects whether port 80 is available:

- **First boot (port 80 free):** Uses `--standalone` — certbot runs its own temporary HTTP server on port 80 for the ACME challenge, then releases it before the node starts.
- **Renewal (port 80 in use):** Uses `--webroot` — certbot writes challenge files to `/data/webroot/.well-known/acme-challenge/` and the node's HTTP server on port 80 serves them. No restart needed.

Override the webroot path with `CERT_WEBROOT` (default: `/data/webroot`).

## DNS-01 Challenge

No port 80 needed. Uses a DNS plugin to prove domain ownership via TXT records.

```bash
docker run -d --name quip-cpu \
  -v ~/quip-data:/data \
  -v ~/cloudflare.ini:/data/certs/cf.ini:ro \
  -e QUIP_PUBLIC_HOST=mynode.example.com \
  -e CERT_EMAIL=admin@example.com \
  -e CERT_CHALLENGE=dns \
  -e CERT_DNS_PLUGIN=cloudflare \
  -e CERT_DNS_CREDENTIALS=/data/certs/cf.ini \
  -p 20049:20049/udp -p 20049:20049/tcp \
  registry.gitlab.com/piqued/quip-protocol/quip-network-node-cpu:latest
```

## DNS Plugin Reference

| Plugin | Credential File Format | Documentation |
|--------|----------------------|---------------|
| `cloudflare` | `dns_cloudflare_api_token = <token>` | [certbot-dns-cloudflare](https://certbot-dns-cloudflare.readthedocs.io/) |
| `route53` | Uses AWS IAM (env vars or instance role) | [certbot-dns-route53](https://certbot-dns-route53.readthedocs.io/) |
| `google` | Service account JSON key file | [certbot-dns-google](https://certbot-dns-google.readthedocs.io/) |
| `digitalocean` | `dns_digitalocean_token = <token>` | [certbot-dns-digitalocean](https://certbot-dns-digitalocean.readthedocs.io/) |
| `ovh` | Application key, secret, consumer key | [certbot-dns-ovh](https://certbot-dns-ovh.readthedocs.io/) |
| `rfc2136` | TSIG key for dynamic DNS updates | [certbot-dns-rfc2136](https://certbot-dns-rfc2136.readthedocs.io/) |

**Cloudflare example** (`~/cloudflare.ini`):
```ini
dns_cloudflare_api_token = your-api-token-here
```

Set permissions: `chmod 600 ~/cloudflare.ini`

## Custom ACME Provider (ZeroSSL)

For providers other than Let's Encrypt, use External Account Binding (EAB):

1. Create an account at [ZeroSSL](https://zerossl.com/) and get EAB credentials from the developer dashboard
2. Run with:

```bash
docker run -d --name quip-cpu \
  -v ~/quip-data:/data \
  -e QUIP_PUBLIC_HOST=mynode.example.com \
  -e CERT_EMAIL=admin@example.com \
  -e CERT_ACME_SERVER=https://acme.zerossl.com/v2/DV90 \
  -e CERT_EAB_KID=your-eab-key-id \
  -e CERT_EAB_HMAC_KEY=your-eab-hmac-key \
  -p 20049:20049/udp -p 20049:20049/tcp -p 80:80/tcp \
  registry.gitlab.com/piqued/quip-protocol/quip-network-node-cpu:latest
```

This also works with Buypass (`https://api.buypass.com/acme/directory`) and other ACME-compliant CAs.

## Custom Certbot Script

The file `/data/certs/certbot` is the hook point. On first run, the entrypoint seeds it from the built-in default. You can replace it with any script that:

- Accepts `$1` as the domain name
- Writes `fullchain.pem` and `privkey.pem` to `/data/certs/private/`
- Exits 0 on success

Examples of replacement scripts:

**acme.sh:**
```bash
#!/bin/bash
acme.sh --issue -d "$1" --webroot /data/certs/webroot \
  --cert-file /data/certs/private/fullchain.pem \
  --key-file /data/certs/private/privkey.pem
```

**lego:**
```bash
#!/bin/bash
lego --email "$CERT_EMAIL" --domains "$1" --accept-tos \
  --path /data/certs/lego run
cp /data/certs/lego/certificates/"$1".crt /data/certs/private/fullchain.pem
cp /data/certs/lego/certificates/"$1".key /data/certs/private/privkey.pem
```

The daily cron job calls the same `/data/certs/certbot` script, so your replacement handles both initial issuance and renewal.

## Renewal

- **Schedule:** Daily at 3am (cron)
- **Logs:** `/data/certs/renewal.log`
- **Skip logic:** The default script checks certificate expiry and skips renewal if >30 days remain
- **On success:** Certificate files are updated in place; the node picks them up on next TLS handshake

## Staging Mode

For testing without hitting Let's Encrypt rate limits:

```bash
-e CERT_STAGING=true
```

This uses Let's Encrypt's staging server, which issues untrusted certificates but has much higher rate limits. Remove `CERT_STAGING` for production.

## Troubleshooting

**Port 80 not reachable:**
```bash
# Verify port mapping
docker port <container> 80
# Test from outside (should return 404, not connection refused)
curl -v http://mynode.example.com/.well-known/acme-challenge/test
```

**Webroot not served during renewal:**
If the node's HTTP server doesn't serve `/.well-known/acme-challenge/`, ensure `rest_insecure_port = 80` in config.toml and the webroot path matches (`CERT_WEBROOT`, default `/data/webroot`).

**DNS propagation delay:**
DNS-01 challenges may take a few minutes for TXT records to propagate. The DNS plugins handle waiting automatically.

**Rate limits:**
Let's Encrypt allows 5 duplicate certificates per week. Use `CERT_STAGING=true` while testing.

**View current certificates:**
```bash
docker exec <container> certbot certificates --config-dir /data/certs/certbot-config
```

**Force renewal:**
```bash
docker exec <container> /data/certs/certbot <domain>
```
The script skips renewal when >30 days remain. To force: delete `/data/certs/private/fullchain.pem` and re-run.

**Check cron:**
```bash
docker exec <container> busybox crontab -l
```
