use quip_mock_coordinator::driver::drive_miner;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let bin_path = args
        .next()
        .ok_or("usage: quip-mock-coordinator <miner-bin> <unix://socket>")?;
    let socket = args
        .next()
        .ok_or("usage: quip-mock-coordinator <miner-bin> <unix://socket>")?;
    let report = drive_miner(&bin_path, &socket).await;
    println!("{report:?}");
    if report.handshake_ok && report.exit_code == 0 {
        Ok(())
    } else {
        Err(format!("conformance failed: {report:?}").into())
    }
}
