//! quip-coordinator binary: CLI, runtime wiring, graceful shutdown.

use clap::Parser;
use quip_coordinator::chain::RealChainClient;
use quip_coordinator::config::parse_config;
use quip_coordinator::session::gen_session_token;
use quip_protocol::session::ExitCode;
use std::path::PathBuf;
use std::process::ExitCode as StdExitCode;

#[derive(Parser, Debug)]
#[command(
    name = "quip-coordinator",
    version = concat!(env!("CARGO_PKG_VERSION"), " protocol 1"),
    about = "QuIP v0.3 coordinator: chain access, routing, miner supervision"
)]
struct Cli {
    /// Path to coordinator config.toml
    #[arg(long)]
    config: Option<PathBuf>,
}

fn main() -> StdExitCode {
    let cli = Cli::parse();

    // --help is handled by clap (exit 0). Missing/invalid config → exit 64.
    let Some(config_path) = cli.config else {
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

    // Real chain is wired (RPC + hybrid sign). Full producers/session loop
    // still needs a live node; config-load path validates wiring only.
    // Integration is covered by tests/e2e.rs with FakeChain.
    let _chain = RealChainClient::new(cfg.validators.clone(), cfg.signer_key.clone());
    let _tokens: Vec<String> = cfg.launch.iter().map(|_| gen_session_token()).collect();

    eprintln!(
        "quip-coordinator: config ok ({} miners); chain client ready (needs live node for RPC)",
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
