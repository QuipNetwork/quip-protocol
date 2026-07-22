use std::process::Command;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_quip-coordinator")
}

#[test]
fn help_exits_zero() {
    let out = Command::new(bin())
        .arg("--help")
        .output()
        .expect("run --help");
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn missing_config_exits_64() {
    let out = Command::new(bin())
        .arg("--config")
        .arg("/nonexistent/quip-coordinator-config.toml")
        .output()
        .expect("run --config");
    assert_eq!(out.status.code(), Some(64));
}

#[test]
fn bad_toml_exits_64() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.toml");
    std::fs::write(&path, "not = [valid").unwrap();
    let out = Command::new(bin())
        .arg("--config")
        .arg(&path)
        .output()
        .expect("run bad config");
    assert_eq!(out.status.code(), Some(64));
}
