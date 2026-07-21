use std::process::Command;

fn bin() -> &'static str { env!("CARGO_BIN_EXE_quip-mock-miner") }

#[test]
fn capabilities_prints_json_and_exits_zero() {
    let out = Command::new(bin()).arg("--capabilities").output().unwrap();
    assert!(out.status.success());
    let s = String::from_utf8(out.stdout).unwrap();
    assert!(s.contains("\"backend\""), "capabilities JSON must include backend, got: {s}");
}

#[test]
fn version_exits_zero() {
    let out = Command::new(bin()).arg("--version").output().unwrap();
    assert!(out.status.success());
    assert!(String::from_utf8(out.stdout).unwrap().contains("protocol"));
}

#[test]
fn check_exits_zero_on_this_host() {
    assert!(Command::new(bin()).arg("--check").status().unwrap().success());
}
