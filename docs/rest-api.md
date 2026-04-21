# QuIP REST API

The network node exposes an HTTP REST API alongside its QUIC protocol. Every QUIC
message type has an HTTP equivalent so browsers, tooling, and scripts can query
and drive a node without a QUIC client.

There is **no RPC envelope**. Paths are exact matches and methods are enforced
(`GET` for reads, `POST` for writes).

## Base URL and configuration

| Config key            | Default     | Meaning                                                        |
| --------------------- | ----------- | -------------------------------------------------------------- |
| `rest_host`           | `127.0.0.1` | Bind address. Remote hosts get connection refused by default.  |
| `rest_insecure_port`  | `20050`     | Plain HTTP port. Set to `-1` to disable.                       |
| `rest_port`           | `-1`        | HTTPS port. Requires a certificate via `CertificateManager`.   |

The API is enabled when either port is `> 0`.

```
http://<rest_host>:20050/...
https://<rest_host>:<rest_port>/...
```

CORS is enabled for all origins (`Access-Control-Allow-Origin: *`). `OPTIONS`
preflight succeeds on any path.

## Response envelope

Every endpoint returns JSON with a uniform envelope.

```json
{
  "success": true,
  "data":    { ... },
  "timestamp": 1700000000
}
```

On error:

```json
{
  "success": false,
  "error":   "Human readable message",
  "code":    "MACHINE_READABLE_CODE",
  "timestamp": 1700000000
}
```

Common error codes:

| Code              | HTTP | Meaning                                              |
| ----------------- | ---- | ---------------------------------------------------- |
| `NOT_FOUND`       | 404  | Unknown endpoint. `GET /` lists everything.         |
| `INVALID_JSON`    | 400  | POST body was not valid JSON.                        |
| `MISSING_FIELDS`  | 400  | Required field absent in POST body.                  |
| `BLOCK_NOT_FOUND` | 404  | Requested block index not in the store.              |
| `RATE_LIMITED`    | 429  | Per-IP rate limit tripped (solve, telemetry).        |
| `UNAUTHORIZED`    | 401  | Bearer token missing or wrong (telemetry only).      |
| `OVERLOADED`      | 503  | Internal queue full.                                 |
| `TIMEOUT`         | 504  | Block/gossip processor did not respond in time.      |
| `INTERNAL_ERROR`  | 500  | Unhandled exception; see server logs.                |

## Discoverability

`GET /` returns a self-describing index of every registered endpoint on the
running node. Prefer it over this document when you need to confirm what a
specific node exposes — telemetry routes only appear when the telemetry cache
is enabled.

```bash
curl -sS http://127.0.0.1:20050/ | python -m json.tool
```

## Public endpoints

### GET /

API index. Returns `name`, `version`, `response_envelope`, and an `endpoints`
array. Useful as a health probe and for clients that auto-discover routes.

### GET /health

Liveness check.

```bash
curl -sS http://127.0.0.1:20050/health
```

```json
{"success": true, "data": {"status": "healthy", "node_running": true, "version": "0.1.5"}, ...}
```

### GET /api/v1/status

Node status: public host, miner info, hardware descriptor, peer count, latest
block index, and uptime.

### GET /api/v1/system

Hardware survey plus whitelisted config (same `descriptor` block returned by
`/status`, without runtime counters).

### GET /api/v1/stats

Cached mining and network statistics. Returns `STATS_NOT_READY` (503) until the
stats cache has been populated for the first time.

### GET /api/v1/peers

Map of known peers keyed by `host:port`.

```json
{"success": true, "data": {"peers": {"1.2.3.4:20049": {...}}, "count": 1}, ...}
```

### GET /api/v1/block/latest

Latest accepted block.

### GET /api/v1/block/{block_number}

Block by index. Returns `BLOCK_NOT_FOUND` (404) if absent, `INVALID_BLOCK_NUMBER`
(400) if the path segment is not an integer.

### GET /api/v1/block/{block_number}/header

Header-only variant of the block lookup. Same error codes.

### POST /api/v1/join

Announce this node to a peer and receive its peer list back. Body:

```json
{
  "host":       "<public-host>:<port>",
  "info":       <MinerInfo JSON>,
  "descriptor": <NodeDescriptor JSON>
}
```

Response includes the peer map with the target node's own entry appended.

### POST /api/v1/block

Submit a new block for validation and propagation. Debug endpoint — mining nodes
normally emit blocks via gossip. Body:

```json
{"raw": "<hex-encoded block bytes>", "signature": "<hex>"}
```

Returns `{"status": "accepted"}` or `{"status": "rejected"}`.

### POST /api/v1/gossip

Forward a gossip message encoded as the raw QUIC wire format.

```bash
curl -sS -X POST --data-binary @gossip.bin \
  http://127.0.0.1:20050/api/v1/gossip
```

### POST /api/v1/solve

Submit an Ising `(h, J)` problem to the node's first available miner. Subject
to per-IP rate limiting (default 10 req/min). Body:

```json
{
  "h":           [0, -1, 1, ...],
  "J":           {"(0,1)": 1.0, "(1,2)": -1.0},
  "num_samples": 64
}
```

`J` may also be a list of `[i, j, coupling]` triples.

Response includes `samples`, `energies`, and a `transaction_id`.

### POST /api/v1/heartbeat

Peer liveness ping. Body:

```json
{"sender": "<host>:<port>", "timestamp": 1700000000.0}
```

## Telemetry endpoints

Only registered when the node is configured with a telemetry cache. If a bearer
token is configured (`telemetry_access_token`), requests must include
`Authorization: Bearer <token>`. All telemetry endpoints are rate-limited.

| Path                                                          | Description                                |
| ------------------------------------------------------------- | ------------------------------------------ |
| `GET /api/v1/telemetry/status`                                 | Telemetry status snapshot + ETag           |
| `GET /api/v1/telemetry/nodes`                                  | Node roster + ETag                         |
| `GET /api/v1/telemetry/epochs`                                 | Available epochs                           |
| `GET /api/v1/telemetry/epochs/{epoch}/blocks/{block_index}`   | Block detail within an epoch               |
| `GET /api/v1/telemetry/latest`                                 | Latest telemetry block                     |
| `GET /api/v1/telemetry/stream`                                 | Server-sent events stream                  |

Telemetry `GET` endpoints honor `If-None-Match` and return `304 Not Modified`
when the ETag matches.

## Common pitfalls

- **`/status` or `/nodes` returns 404.** These are not the real paths. Use
  `/api/v1/status` and `/api/v1/peers`. `GET /` lists everything.
- **Connection refused from another host.** `rest_host` defaults to
  `127.0.0.1`. Set it to `0.0.0.0` (or an external interface) to accept
  remote traffic.
- **HTTPS returns certificate error.** The HTTPS listener only starts when
  `CertificateManager` has a valid cert/key. Check node logs for
  `Failed to start HTTPS server`; until then only the HTTP port serves.
- **`/api/v1/solve` returns 429.** Per-IP rate limit, default 10/min.
- **Telemetry endpoint returns 401.** A `telemetry_access_token` is
  configured; include `Authorization: Bearer <token>`.
