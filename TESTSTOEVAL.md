# Tests to Evaluate

Tests removed from the suite during the Phase 1c triage because they
don't exercise any component documented in
[`ARCHITECTURE.md`](./ARCHITECTURE.md). Recovered from git history if
the underlying code is later judged to be load-bearing.

The categorization rule: a test is "orphan" if its test target sits
outside §§2–8 of `ARCHITECTURE.md` and the cross-cutting list (crypto,
signer, keystore, SCALE codec, version, work_context, ising
primitives). Auxiliary tools (`faucet_bot.py`), one-off scripts, and
out-of-band utilities count as outside the architecture.

## Removed

| File | What it tested | Why orphan |
|---|---|---|
| `tests/test_faucet_normalize.py` | `faucet_bot._normalize_dest` — SS58/hex canonicalization that keys the per-destination throttle in the standalone faucet bot. | `faucet_bot.py` is an auxiliary tool, not part of the runtime architecture documented in `ARCHITECTURE.md`. If the faucet is still operated, the tests should move into the faucet's own repo or be revived after re-scoping the project's tool layer. |
| `tests/test_faucet_unit.py` | `faucet_bot.py` helper functions (rate limiting, request parsing, mock-driven) that don't need a live chain. | Same as above — `faucet_bot.py` is auxiliary. |
| `tests/test_substrate_faucet.py` | Live-chain integration test for `faucet_bot.py`: brings the faucet up on an ephemeral port, requests funding from a fresh key, asserts the destination balance moves. Requires `ws://localhost:9944` (docker compose). | Same as above. The faucet is operationally separate from the miner runtime; integration coverage belongs alongside the faucet, not in the miner test suite. |

## Triage outcome

| Test bucket | Count | Disposition |
|---|---|---|
| Directly map to `ARCHITECTURE.md` §§2–8 | 56 | Kept |
| Cross-cutting (kept by exception list above) | 1 | Kept |
| Orphan (this file) | 3 | Removed; recoverable from git history |

If you want to revive any of the removed tests, look up the file in
`git log --diff-filter=D --name-only -- tests/` and restore from the
prior commit. The faucet bot itself (`faucet_bot.py`) was not touched
in this phase — only its tests.
