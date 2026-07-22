//! quip-coordinator binary: CLI, runtime wiring, graceful shutdown.

use clap::{Parser, Subcommand, ValueEnum};
use quip_coordinator::chain::{FakeChain, MiningSnapshot, RealChainClient};
use quip_coordinator::config::{parse_config, LaunchEntry};
use quip_coordinator::drive::{
    aggregate, drain_all, parse_topology_spec, print_table, run_drive, write_jsonl,
    DriveManyParams, ListSource, RandomSource,
};
use quip_coordinator::session::gen_session_token;
use quip_coordinator::topology::Topology;
use quip_proto::v1::{Configure, Job};
use quip_protocol::session::ExitCode;
use std::path::PathBuf;
use std::process::ExitCode as StdExitCode;
use std::sync::Arc;
use std::time::Duration;

#[derive(Parser, Debug)]
#[command(
    name = "quip-coordinator",
    version = concat!(env!("CARGO_PKG_VERSION"), " protocol 1"),
    about = "QuIP v0.3 coordinator: chain access, routing, miner supervision"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    /// Path to coordinator config.toml (ignored when a subcommand is given)
    #[arg(long)]
    config: Option<PathBuf>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Drive a spawned miner with synthetic work; no chain, no submit.
    Drive(DriveArgs),
}

#[derive(ValueEnum, Clone, Debug)]
enum DriveSourceKind {
    Random,
    List,
}

#[derive(clap::Args, Debug)]
struct DriveArgs {
    /// Miner binary to spawn.
    #[arg(long)]
    miner: PathBuf,
    /// Job source: golden-draw random problems, or a JSONL replay list.
    #[arg(long, value_enum)]
    source: DriveSourceKind,
    /// Topology spec JSON. Required for `--source random`; for
    /// `--source list`, required only if the list has a nonce-ref entry.
    #[arg(long)]
    topology: Option<PathBuf>,
    /// Number of problems to draw (`--source random`).
    #[arg(long, default_value_t = 10)]
    count: u32,
    /// Draw seed: same seed + topology draws the same jobs (`--source random`).
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// JSONL model list (`--source list`).
    #[arg(long)]
    list: Option<PathBuf>,
    /// Per-job deadline, milliseconds from now.
    #[arg(long, default_value_t = 3_600_000)]
    deadline_ms: u64,
    /// Optional path to write a per-job + aggregate JSONL report.
    #[arg(long)]
    report: Option<PathBuf>,
}

fn main() -> StdExitCode {
    let cli = Cli::parse();
    match cli.command {
        Some(Command::Drive(args)) => run_drive_cli(args),
        None => run_config_path(cli.config),
    }
}

fn run_config_path(config: Option<PathBuf>) -> StdExitCode {
    // --help is handled by clap (exit 0). Missing/invalid config → exit 64.
    let Some(config_path) = config else {
        eprintln!("error: --config <path> is required");
        return StdExitCode::from(ExitCode::ConfigInvalid as u8);
    };

    let text = match std::fs::read_to_string(&config_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: cannot read config {}: {e}", config_path.display());
            return StdExitCode::from(ExitCode::ConfigInvalid as u8);
        }
    };

    let cfg = match parse_config(&text) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: invalid config: {e}");
            return StdExitCode::from(ExitCode::ConfigInvalid as u8);
        }
    };

    // Real chain is CONFIRM-isolated; construct for wiring only.
    let _chain = RealChainClient::new(cfg.validators.clone(), cfg.signer_key.clone());
    let _tokens: Vec<String> = cfg.launch.iter().map(|_| gen_session_token()).collect();

    // Full runtime (producers + session server + supervisor) requires a live
    // node via RealChainClient. Until external crates are wired, exit cleanly
    // after a successful config load so operators can validate configs.
    // Integration is covered by tests/e2e.rs with FakeChain.
    eprintln!(
        "quip-coordinator: config ok ({} miners); real chain submit is CONFIRM-pending",
        cfg.launch.len()
    );

    // Block until SIGINT/SIGTERM for a realistic process shape when used under
    // a process supervisor; tests use --config /nonexistent and never reach here.
    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("error: runtime: {e}");
            return StdExitCode::from(ExitCode::InternalFatal as u8);
        }
    };
    rt.block_on(async {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
        }
    });
    StdExitCode::from(ExitCode::Clean as u8)
}

fn now_unix_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Placeholder snapshot for chain-less drive runs with no topology at all
/// (explicit-only lists): the `FakeChain` still needs one to construct.
fn empty_snapshot() -> MiningSnapshot {
    MiningSnapshot {
        last_proof_block_hash: [0u8; 32],
        topology_hash: vec![0u8; 32],
        nodes: vec![],
        edges: vec![],
        allowed_h_milli: vec![0],
        allowed_j_milli: vec![0],
        min_solutions: 0,
        max_energy_milli: 0,
        min_diversity_milli: 0,
        block_number: 0,
    }
}

type BuiltJobs = (Vec<Job>, Option<Topology>, Option<MiningSnapshot>);

/// Load the requested job source, returning its jobs plus the topology/
/// snapshot (if any) so drive setup can wire `FakeChain` and the wire
/// `Topology` message consistently.
fn build_jobs(args: &DriveArgs, deadline_ms: u64) -> Result<BuiltJobs, String> {
    match args.source {
        DriveSourceKind::Random => {
            let topo_path = args
                .topology
                .as_ref()
                .ok_or("--source random requires --topology <spec.json>")?;
            let text = std::fs::read_to_string(topo_path)
                .map_err(|e| format!("cannot read topology spec: {e}"))?;
            let spec = parse_topology_spec(&text).map_err(|e| e.to_string())?;
            let miner_account = [0u8; 32];
            let mut src =
                RandomSource::new(&spec, miner_account, args.seed, args.count, deadline_ms);
            let jobs = drain_all(&mut src);
            let snapshot = spec.to_snapshot();
            Ok((jobs, Some(spec.topology), Some(snapshot)))
        }
        DriveSourceKind::List => {
            let list_path = args
                .list
                .as_ref()
                .ok_or("--source list requires --list <models.jsonl>")?;
            let (topology, snapshot) = match &args.topology {
                Some(p) => {
                    let text = std::fs::read_to_string(p)
                        .map_err(|e| format!("cannot read topology spec: {e}"))?;
                    let spec = parse_topology_spec(&text).map_err(|e| e.to_string())?;
                    (Some(spec.topology.clone()), Some(spec.to_snapshot()))
                }
                None => (None, None),
            };
            let mut src = ListSource::load(list_path, snapshot.as_ref(), deadline_ms)
                .map_err(|e| e.to_string())?;
            let jobs = drain_all(&mut src);
            Ok((jobs, topology, snapshot))
        }
    }
}

fn run_drive_cli(args: DriveArgs) -> StdExitCode {
    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("error: runtime: {e}");
            return StdExitCode::from(ExitCode::InternalFatal as u8);
        }
    };
    rt.block_on(drive_main(args))
}

async fn drive_main(args: DriveArgs) -> StdExitCode {
    let deadline_ms = now_unix_ms() + args.deadline_ms;
    let (jobs, topology, snapshot) = match build_jobs(&args, deadline_ms) {
        Ok(v) => v,
        Err(msg) => {
            eprintln!("error: {msg}");
            return StdExitCode::from(ExitCode::ConfigInvalid as u8);
        }
    };

    let chain = Arc::new(FakeChain::new(
        snapshot.unwrap_or_else(empty_snapshot),
        None,
    ));
    let entry = LaunchEntry {
        miner_id: "drive-0".into(),
        binary: args.miner.to_string_lossy().into_owned(),
        configure: Configure {
            queue_depth: 3,
            idle_timeout_s: 30,
            heartbeat_s: 15,
            reconnect_window_s: 60,
            backend_toml: String::new(),
        },
    };
    let sock = format!("/tmp/quip-coordinator-drive-{}.sock", std::process::id());
    let token = gen_session_token();
    let overall_timeout = Duration::from_secs(30 + jobs.len() as u64 * 5);

    let report = run_drive(DriveManyParams {
        miner_bin: &entry.binary,
        sock_path: &sock,
        miner_id: &entry.miner_id,
        token: &token,
        entry: &entry,
        topology,
        jobs,
        chain,
        overall_timeout,
    })
    .await;

    if !report.handshake_ok {
        eprintln!(
            "error: miner handshake failed (exit code {})",
            report.miner_exit_code
        );
        return StdExitCode::from(ExitCode::InternalFatal as u8);
    }

    let agg = aggregate(&report.rows);
    print_table(&report.rows, &agg);
    if let Some(path) = &args.report {
        if let Err(e) = write_jsonl(path, &report.rows, &agg) {
            eprintln!("error: cannot write report {}: {e}", path.display());
            return StdExitCode::from(ExitCode::InternalFatal as u8);
        }
    }
    // A truncated run (miner crash / dropped Result / timeout) leaves fewer rows
    // than jobs. That is a run failure, not per-job data, so it must not exit 0.
    if report.rows.len() < report.total {
        eprintln!(
            "error: incomplete run — {} of {} jobs produced no result",
            report.total - report.rows.len(),
            report.total
        );
        return StdExitCode::from(ExitCode::InternalFatal as u8);
    }
    StdExitCode::from(ExitCode::Clean as u8)
}
