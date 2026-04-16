# Telemetry Indexer Specification

Specification for building an indexer that consumes the QuIP node telemetry stream API.

## Architecture

```
QuIP Node  --(writes)-->  telemetry/  --(serves)-->  Telemetry API
                                                          |
                                                     (REST / QUIC)
                                                          |
                                                      Indexer  --(stores)-->  Database
                                                          |
                                                      Dashboard
```

The indexer is a polling client that maintains a cursor (epoch + block_index) and incrementally fetches new data.

## Authentication

If the node operator configures an access token, include it on all REST requests:

```
Authorization: Bearer <token>
```

QUIC connections are authenticated via TLS — no additional token is needed.

Handle `401 Unauthorized` responses by prompting the operator for a valid token.

## REST API Reference

Base URL: `http(s)://<node-host>:<rest-port>` (default port: 20050)

All responses use the envelope:

```json
{
  "success": true,
  "data": { ... },
  "timestamp": 1700000000
}
```

Error responses:

```json
{
  "success": false,
  "error": "Human-readable message",
  "code": "ERROR_CODE",
  "timestamp": 1700000000
}
```

### GET /api/v1/telemetry/status

Returns a summary of available telemetry data.

**Response:**

```json
{
  "epochs": ["1775166921", "1775167182"],
  "latest_epoch": "1775167182",
  "latest_block_index": 6,
  "total_blocks": 11,
  "node_count": 1,
  "active_node_count": 0,
  "nodes_updated_at": "2026-04-02T23:31:53.610902+00:00"
}
```

**ETag:** Composite of `latest_epoch:latest_block_index:total_blocks`. Use `If-None-Match` to avoid re-downloading unchanged data.

### GET /api/v1/telemetry/nodes

Returns the full node registry (peer addresses, status, miner info).

**Response:**

```json
{
  "updated_at": "2026-04-02T23:31:53.610902+00:00",
  "node_count": 2,
  "active_count": 1,
  "nodes": {
    "127.0.0.1:8085": {
      "address": "127.0.0.1:8085",
      "miner_id": "node-0",
      "miner_type": "CPU",
      "ecdsa_public_key_hex": "02ab...8c",
      "status": "active",
      "last_heartbeat": 1775172713.610861,
      "first_seen": 1775172713.610861,
      "last_seen": 1775172713.610861
    }
  }
}
```

**ETag:** Value of `updated_at`.

**Node status values:** `active`, `initial_peer`, `failed`, `lost`

### GET /api/v1/telemetry/epochs

Lists all epochs with block counts.

**Response:**

```json
{
  "epochs": [
    {
      "epoch": "1775166921",
      "block_count": 5,
      "first_block": 1,
      "last_block": 5
    },
    {
      "epoch": "1775167182",
      "block_count": 6,
      "first_block": 1,
      "last_block": 6
    }
  ]
}
```

### GET /api/v1/telemetry/epochs/{epoch}/blocks/{block_index}

Returns a single block's telemetry data. One block at a time (no bulk export).

**Response:**

```json
{
  "block_index": 1,
  "block_hash": "94d12dee...",
  "timestamp": 1775166921,
  "previous_hash": "45416d50...",
  "miner": {
    "miner_id": "mac.carback.int",
    "miner_type": "{...}",
    "ecdsa_public_key": "042ab049..."
  },
  "quantum_proof": {
    "energy": -14484.0,
    "diversity": 0.4119,
    "num_valid_solutions": 5,
    "mining_time": 5,
    "nonce": 3014965652,
    "num_nodes": 4579,
    "num_edges": 41549
  },
  "requirements": {
    "difficulty_energy": -2688.39,
    "min_diversity": 0.2,
    "min_solutions": 5
  }
}
```

**ETag:** Value of `block_hash` (immutable once written).

### GET /api/v1/telemetry/latest

Shortcut returning the most recent block telemetry with metadata.

**Response:**

```json
{
  "epoch": "1775167182",
  "block_index": 6,
  "block": { ... }
}
```

### GET /api/v1/telemetry/stream

Server-Sent Events (SSE) endpoint. Pushes events as new data appears.

**Event types:**

```
event: block
data: {"epoch":"1775167182","block_index":7,"block":{...}}

event: nodes
data: {"updated_at":"...","node_count":2,...}
```

Keepalive comments (`:keepalive`) are sent every 15 seconds.

## QUIC Protocol Reference

Telemetry data is also accessible via QUIC message types on the node's P2P port (default: 20049). Connections use TLS 1.3 with ALPN `quip-v1`.

| Request | Code | Response | Code | Payload |
|---------|------|----------|------|---------|
| `TELEMETRY_STATUS_REQUEST` | `0x20` | `TELEMETRY_STATUS_RESPONSE` | `0xA0` | JSON (same as REST) |
| `TELEMETRY_NODES_REQUEST` | `0x21` | `TELEMETRY_NODES_RESPONSE` | `0xA1` | JSON |
| `TELEMETRY_EPOCHS_REQUEST` | `0x22` | `TELEMETRY_EPOCHS_RESPONSE` | `0xA2` | JSON |
| `TELEMETRY_BLOCK_REQUEST` | `0x23` | `TELEMETRY_BLOCK_RESPONSE` | `0xA3` | Request: `{"epoch":"...","block_index":N}` |
| `TELEMETRY_LATEST_REQUEST` | `0x24` | `TELEMETRY_LATEST_RESPONSE` | `0xA4` | JSON |

Wire format: `[1B msg_type][1B protocol_version][4B request_id][4B payload_len][payload]`

## Recommended Polling Strategy

```
1. GET /api/v1/telemetry/status
   - Record latest_epoch and latest_block_index as cursor
   - Send If-None-Match with previous ETag
   - If 304 Not Modified: sleep 5-10s, retry

2. If latest_block_index > cursor.block_index (same epoch):
   For each block_index from cursor+1 to latest_block_index:
     GET /api/v1/telemetry/epochs/{epoch}/blocks/{i}
     Store in database
     Update cursor

3. If latest_epoch != cursor.epoch (epoch transition):
   GET /api/v1/telemetry/epochs
   Discover new epoch, reset block cursor to first_block
   Fetch all blocks in new epoch one at a time

4. Periodically (every 30-60s):
   GET /api/v1/telemetry/nodes
   Update node registry in database

5. Sleep 5-10 seconds, goto 1
```

## SSE Strategy (Alternative)

```
1. Connect to GET /api/v1/telemetry/stream
2. For each "block" event: store the block data, update cursor
3. For each "nodes" event: update node registry
4. On disconnect: fall back to polling to catch up, then reconnect
```

## Epoch Transitions

The chain resets periodically (epoch transitions). When this happens:

- A new epoch directory appears with a different timestamp
- Block indices restart from 1
- The epoch directory name is the Unix timestamp of block 1

Indexers must track **epoch + block_index** as a composite cursor. When the epoch changes, fetch the epochs listing to discover the new epoch and begin indexing from its first block.

## Rate Limits

- Default: 30 requests per minute per IP (configurable by node operator)
- Token-bucket algorithm with burst allowance
- `429 Too Many Requests` when exceeded
- SSE: max 5 connections per IP, 20 total

**Recommended approach:** Poll status every 5-10 seconds (6-12 RPM), fetch 1 block per new block event. This stays well within limits.

**Backoff on 429:** Use exponential backoff starting at 5 seconds, capped at 60 seconds.

## Error Handling

| Status | Meaning | Action |
|--------|---------|--------|
| 200 | Success | Process data |
| 304 | Not Modified | Data unchanged, skip |
| 401 | Unauthorized | Check access token |
| 404 | Not Found | Block/epoch doesn't exist or was pruned |
| 429 | Rate Limited | Exponential backoff |
| 500 | Server Error | Retry with backoff |
| 503 | Telemetry Disabled | Node has telemetry disabled |

## Data Retention

Telemetry files persist as long as the node operator keeps them. Old epochs may be pruned by the operator. Indexers should handle 404 responses for historical blocks gracefully.
