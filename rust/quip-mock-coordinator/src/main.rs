use quip_mock_coordinator::driver::drive_miner;
use std::process::ExitCode;

#[tokio::main]
async fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let Some(bin_path) = args.next() else {
        eprintln!("usage: quip-mock-coordinator <miner-bin> <unix://socket>");
        return ExitCode::from(64);
    };
    let Some(socket) = args.next() else {
        eprintln!("usage: quip-mock-coordinator <miner-bin> <unix://socket>");
        return ExitCode::from(64);
    };
    let report = drive_miner(&bin_path, &socket).await;
    println!("{report:?}");
    if report.is_conformant() {
        ExitCode::SUCCESS
    } else {
        eprintln!("conformance failed: {report:?}");
        ExitCode::from(1)
    }
}
