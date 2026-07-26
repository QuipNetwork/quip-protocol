# quip-protocol

> **Experimental software.** Use at your own risk. No production warranties.

The v0.3 mining stack for the
[quip-protocol-rs](https://gitlab.com/quip.network/quip-protocol-rs) Substrate
chain. A coordinator follows the chain head and stages Ising problems for the
miner subprocesses; when a solution clears the difficulty gate, the coordinator
submits a `QuantumPow.submit_proof` extrinsic. The miners run simulated
annealing and Gibbs sampling on CPU, CUDA, and Metal, and quantum annealing on
D-Wave hardware.

This is a Rust workspace. The coordinator and the chain integration are Rust; a
small PyO3 extension exposes the consensus primitives to Python; the miner
binaries ship from their own repositories.

## Repository layout

- `crates/` — the Cargo workspace: `quip-coordinator` (the binary),
  `quip-miner-core` (the shared miner harness), `quip-protocol` (consensus
  primitives), `quip-protocol-py` (the PyO3 extension), `quip-proto` (generated
  protocol types), and two test doubles.
- `python/` — the `quip_proto` SDK (PyO3 core + generated gRPC stubs).
- `proto/` — the normative `.proto` IDL.
- `conformance/` — the PyO3 parity test suite.
- `docker/` — image builds and the example coordinator config.

The miner binaries live in `quip.network/quip-miner-{cpu,cuda,metal,dwave}`.

## Documentation

- `AGENTS.md` — build, test, and run commands; repository conventions.
- `COORDINATOR.md` — how the coordinator works (chain access, feeder, routing,
  supervision, session protocol).
- `MINER.md` — the `quip-miner-core` harness and its helper modules.
- `NEWMINER.md` — how to add a new miner.
- `docs/VERSIONING.md` — the release-tag standard.

## Build and run

The workspace builds from the repository root (toolchain pinned in
`rust-toolchain.toml`):

```bash
cargo build --workspace --exclude quip-protocol-py
```

Run the coordinator against a config that lists the miners and the validators:

```bash
quip-coordinator --config ./docker/config.toml
```

The Python SDK builds with maturin:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install maturin
maturin develop -E dev
pytest conformance/
```

See `AGENTS.md` for the full command set and `COORDINATOR.md` for the run model.

## What changed from v0.2

In v0.2 this repository was a single Python mining stack that attached to the
chain. v0.3 splits it apart:

- The coordinator and chain integration are Rust (`crates/`), talking to the
  node over subxt.
- Each miner is a standalone binary in its own repository, supervised by the
  coordinator over a local socket. The Python miner stack is gone.
- The only remaining Python is the `quip_proto` SDK, a PyO3 wrapper over the
  Rust consensus primitives plus the generated gRPC stubs.

## License

AGPL-3.0-or-later. See LICENSE.
