#!/bin/bash
# Entrypoint for the quip-miner Docker image (CPU / CUDA).
#
# Everything the miner does is declared in /data/config.toml (seeded
# from the image template on first run): validators, mempool, faucet,
# telemetry, identity, and backend inventory. `quip-miner --config`
# reads it and spawns/supervises every process itself — the entrypoint
# only prepares the volume and drops privileges. Container args are
# forwarded to the supervisor (e.g. `--mode gpu` keeps one configured
# miner type and reports the dropped ones).
#
# Host environment (not miner configuration):
#   PUID / PGID   uid/gid to run as (default 1000:1000; PUID=0 = root)
#
# Persistent state under /data:
#   config.toml      — the configuration; edit it and restart.
#   keystore.json    — hybrid sr25519 + ML-DSA-44, plaintext seed.
#                      BACK THIS UP — the chain key lives here.
set -euo pipefail

CONFIG_FILE="/data/config.toml"
TEMPLATE_FILE="/app/quip-miner.docker.toml"
# Fixed path: the template's `signer_key` value and this keygen target
# are the same by construction. Mount an existing keystore here to
# bring your own key.
KEYSTORE_FILE="/data/keystore.json"

echo "========================================"
echo "Quip Miner (substrate)  $(date)"
echo "========================================"

# ── First run: seed config + keystore ─────────────────────────────
mkdir -p /data
if [ ! -f "$CONFIG_FILE" ]; then
    echo "First run: seeding $CONFIG_FILE from $TEMPLATE_FILE"
    cp "$TEMPLATE_FILE" "$CONFIG_FILE"
else
    echo "Using existing config at $CONFIG_FILE"
fi

if [ ! -f "$KEYSTORE_FILE" ]; then
    echo "First run: generating hybrid keystore at $KEYSTORE_FILE"
    quip-miner keygen --out "$KEYSTORE_FILE"
    # quip-miner keygen sets mode 0600 internally; re-assert for clarity.
    chmod 600 "$KEYSTORE_FILE"
else
    echo "Using existing keystore at $KEYSTORE_FILE"
fi

# ── GPU diagnostic + CUDA forward-compat guard (cuda image only) ──
# The nvidia/cuda:12.9.1 base image ships /usr/local/cuda/compat/
# libcuda.so (~575). The NVIDIA Container Toolkit's enable-cuda-compat
# hook prepends it to the ldcache whenever it is newer than the host
# driver. Forward compat is datacenter-only; on a consumer GPU it fails
# CUDA init with error 804 (cudaErrorCompatNotSupportedOnDevice) and the
# miner crash-loops. We probe once as root; on 804 we drop the compat
# libs so the host-mounted libcuda.so wins — our kernels JIT against the
# host driver via CUDA minor-version compatibility, so no driver upgrade
# is needed. Datacenter GPUs never raise 804, so their forward-compat
# path is left untouched. See QUI-896.
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "----------------------------------------"
    nvidia-smi --query-gpu=gpu_name,compute_cap --format=csv,noheader || true
    echo "----------------------------------------"

    if command -v python3 >/dev/null 2>&1; then
        # Prints one of: OK | COMPAT_UNSUPPORTED | FAIL:<detail>
        cuda_probe() {
            python3 - <<'PY'
try:
    import cupy as cp

    cp.cuda.Device(0).use()
    _ = cp.zeros(1)
    print("OK")
except Exception as exc:  # probe reports status, never handles it
    msg = str(exc).lower()
    status = getattr(exc, "status", None)
    if status == 804 or "forward compatibility" in msg or "compatnotsupported" in msg:
        print("COMPAT_UNSUPPORTED")
    else:
        print("FAIL:" + type(exc).__name__ + ": " + str(exc))
PY
        }

        cuda_status="$(cuda_probe)"
        if [ "$cuda_status" = "COMPAT_UNSUPPORTED" ]; then
            echo "CUDA init hit forward-compat error 804 (consumer GPU) —" \
                 "neutralizing compat libs (QUI-896)"
            for conf in /etc/ld.so.conf.d/*.conf; do
                [ -e "$conf" ] || continue
                if grep -qE '/usr/local/cuda.*/compat' "$conf" 2>/dev/null; then
                    echo "  removing compat ldconfig entry: $conf"
                    rm -f "$conf"
                fi
            done
            if compgen -G "/usr/local/cuda/compat/libcuda.so*" >/dev/null; then
                echo "  masking /usr/local/cuda/compat/libcuda.so*"
                mkdir -p /usr/local/cuda/compat/.disabled
                mv /usr/local/cuda/compat/libcuda.so* \
                    /usr/local/cuda/compat/.disabled/ 2>/dev/null || true
            fi
            ldconfig
            cuda_status="$(cuda_probe)"
        fi

        if [ "$cuda_status" = "OK" ]; then
            echo "CUDA init OK."
        else
            driver="$(nvidia-smi --query-gpu=driver_version \
                --format=csv,noheader 2>/dev/null | head -1 || true)"
            {
                echo "WARNING: CUDA is not usable (probe: $cuda_status;" \
                     "host driver: ${driver:-unknown})."
                echo "  This image is built on CUDA 12.9 and needs an NVIDIA"
                echo "  driver >= 575 (open kernel module) for GPU mining."
                echo "  Upgrade the host driver, or run a non-GPU backend."
                echo "  Continuing startup; GPU backends will fail to init."
                echo "  See QUI-896."
            } >&2
        fi
    fi
fi

# ── Privilege drop ─────────────────────────────────────────────────
# Container starts as root so we can chown /data and generate the
# keystore with the right ownership; the miner runs as `quip`.
# PUID/PGID map the in-container user to a host uid/gid (default
# 1000:1000; PUID=0 keeps root).
PUID="${PUID:-1000}"
PGID="${PGID:-1000}"

if [ "$PUID" = "0" ]; then
    echo "Privilege drop: PUID=0 — running as root (legacy mode)"
    exec quip-miner --config "$CONFIG_FILE" "$@"
fi

[ "$(id -g quip)" != "$PGID" ] && groupmod -g "$PGID" quip
[ "$(id -u quip)" != "$PUID" ] && usermod -u "$PUID" -g "$PGID" quip
chown -R quip:quip /data
chmod 600 "$KEYSTORE_FILE"
echo "Privilege drop: exec as uid=$PUID gid=$PGID"
exec gosu quip:quip quip-miner --config "$CONFIG_FILE" "$@"
