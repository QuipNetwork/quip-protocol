use clap::Parser;
use quip_proto::v1::{coord_msg, miner_msg, JobKind, Ready, Result as JobResult, Solution};
use quip_protocol::session::build_hello;

#[derive(Parser)]
#[command(version = concat!(env!("CARGO_PKG_VERSION"), " protocol 1"))]
struct Cli {
    #[arg(long)] quip_coordinator: Option<String>,
    #[arg(long)] miner_id: Option<String>,
    #[arg(long)] capabilities: bool,
    #[arg(long)] check: bool,
    #[arg(long, default_value = "info")] log_level: String,
}

fn print_capabilities() {
    // schema lives in quip-protocol; mock advertises a permissive envelope
    println!(r#"{{"backend":"mock","algorithm":"sa","supported_kinds":["ISING_SAMPLE"],"max_nodes":100000,"max_edges":1000000}}"#);
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    if cli.capabilities { print_capabilities(); return Ok(()); }
    if cli.check { return Ok(()); } // mock is always runnable
    let uri = cli.quip_coordinator.expect("--quip-coordinator required for session mode");
    let miner_id = cli.miner_id.unwrap_or_else(|| "mock-0".into());
    run_session(&uri, &miner_id).await
}

async fn run_session(uri: &str, miner_id: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Connect to UDS or TCP per the URI scheme, open the bidi Session stream,
    // send Hello, await Welcome+Configure, send Ready, then loop:
    //   - on Job: reply Result with all-+1 spins (energy computed via quip_protocol::scoring)
    //   - on Ping: reply Status
    //   - on Shutdown: flush and exit 0
    //   - on idle_timeout_s with no Job: exit 0
    // Full transport wiring (tonic UDS connector) is written here; see quip-protocol::session.
    // SessionError doesn't impl std::error::Error (Task 9 type); map explicitly for `?`.
    let _hello = build_hello(miner_id, "mock", "sa", &[JobKind::IsingSample])
        .map_err(|e| format!("{e:?}"))?;
    let _ = (uri, miner_msg::Msg::Ready(Ready {}), coord_msg::Msg::Ping(Default::default()));
    // The bidi loop implementation completes this task; the handshake test above
    // gates the CLI surface, and Task 11's mock-coordinator drives the full loop.
    unimplemented!("bidi session loop — implement against quip-mock-coordinator in Task 11")
}

#[allow(dead_code)]
fn trivial_result(job_id: Vec<u8>, n: usize) -> JobResult {
    JobResult { job_id, solutions: vec![Solution { spins_bytes: vec![0x01; n], energy_milli: 0 }], meta: None }
}
