# AGENTS.md — QuIP Protocol Project Instructions

Cross-tool instructions for AI coding assistants (Claude Code, Codex, Cursor, Gemini CLI).

QuIP v0.3 is a Rust workspace. The coordinator owns chain access and work routing; it spawns and supervises the miner subprocesses that do the sampling. The miner binaries ship from their own repos. A small PyO3 extension exposes the consensus primitives to Python.

## Repository layout

- `crates/` — the Cargo workspace (root manifest: `Cargo.toml`):
  - `quip-proto` — protobuf/gRPC types generated from `proto/` (prost/tonic).
  - `quip-protocol` — consensus primitives: scoring, wire codec, ChaCha8 draw, session types. The source of truth for the math.
  - `quip-protocol-py` — PyO3 `cdylib` exposing `quip-protocol` as `quip_proto._core`. Built by maturin, not the default cargo build.
  - `quip-miner-core` — shared miner harness: the gRPC session loop and the `Sampler` trait each backend supplies.
  - `quip-coordinator` — the coordinator binary: chain access, routing, miner supervision.
  - `quip-mock-coordinator`, `quip-mock-miner` — test doubles.
- `python/` — the `quip_proto` SDK (maturin `python-source`): the `_core` re-export shim plus the generated `quip.v1` gRPC stubs.
- `proto/` — the normative `.proto` IDL.
- `conformance/` — pytest suite pinning the PyO3 SDK to golden vectors.
- `docker/` — image builds (`Dockerfile.quip-miner`, `Dockerfile.quip-miner-cuda`) and `config.toml`.
- `dwave_topologies/` — D-Wave embedding data.
- `docs/VERSIONING.md` — the release-tag standard.

The miner binaries live in their own repos: `quip.network/quip-miner-{cpu,cuda,metal,dwave}`.

## Build and test

The workspace builds from the repo root. The toolchain is pinned to 1.97.1 (`rust-toolchain.toml`).

```bash
# Build every crate except the PyO3 cdylib (maturin builds that one).
cargo build --workspace --exclude quip-protocol-py

# Test (mirrors CI; the CI test job also excludes quip-coordinator).
cargo test --workspace --exclude quip-coordinator --exclude quip-protocol-py

# Lint and format.
cargo clippy --workspace --exclude quip-coordinator --exclude quip-protocol-py --all-targets -- -D warnings
cargo fmt --all --check
```

The Python SDK and its conformance suite need maturin:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install maturin
maturin develop -E dev     # builds quip_proto._core, installs quip_proto + dev extras
pytest conformance/        # 19 tests
```

The PyO3 crate is a workspace member held out of `default-members`, so a bare `cargo build`/`cargo test` skips it. Build it with maturin, or target it directly with `cargo build -p quip-protocol-py`.

## Running the coordinator

`quip-coordinator` has two modes.

```bash
# Production: read the config, connect to the validators, spawn and supervise the
# configured miners, feed them work, submit proofs. Stops on SIGINT/SIGTERM.
quip-coordinator --config ./docker/config.toml

# Drive: spawn one miner and feed it synthetic problems. No chain, no submit —
# for benchmarking and matched-condition parity runs.
quip-coordinator drive --miner ./miners/quip-cpu-sa --source random \
  --topology-preset advantage2-system1 --count 100
```

The config registers miners and their launch plan. Each backend section (`[cpu]`, `[cuda.0]`, `[metal]`, `[dwave]`) becomes one supervised subprocess; `binary` selects the executable — `quip-cpu-sa`, or the chromatic `quip-cpu-gibbs`. See `crates/quip-coordinator/config.toml.example`. Miner binaries are fetched per host by `crates/quip-coordinator/tools/fetch-miners.sh` from the standalone repos and resolved on `PATH` or by absolute path.

`signer_key` is a dev account URI (`//Alice`) or a path to a hybrid sr25519 + ML-DSA-44 keystore. `.env` holds credentials such as `DWAVE_API_KEY` — **never read or display its contents.**

## The PyO3 SDK

`quip_proto` is the only Python surface. It re-exports the Rust consensus primitives (`scoring`, `wire`, `ExitCode`) from `quip_proto._core` and the generated gRPC stubs from `quip.v1`. The Rust in `quip-protocol` is the source of truth; the Python can't drift from it. The D-Wave miner repo depends on this wheel.

Regenerate the `quip.v1` stubs from `proto/` with the pinned `grpcio-tools==1.82.1`; CI diffs the result against the checked-in copies, so a version bump and a stub regen go together.

## Task tracking

This project tracks work in beads (`bd`), not to-do lists or markdown. Run `bd prime` for the workflow and `bd ready` for available work.

## Conventions

- Rust standards and lints follow the `lang-rust` guidance; `cargo clippy -- -D warnings` is the gate.
- Follow the existing crate layout — shared code goes in the crate that owns it, not a new top-level module.
- No `Co-Authored-By` or other assistant-attribution trailers in commits.
- Never read `.env`.
