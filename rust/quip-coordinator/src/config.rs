//! Parse `config.toml` into a per-miner launch plan.

use quip_proto::v1::Configure;

#[derive(Debug, PartialEq)]
pub enum ConfigError {
    MissingMiner,
    BadToml(String),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::MissingMiner => write!(f, "missing [miner] section"),
            ConfigError::BadToml(e) => write!(f, "bad toml: {e}"),
        }
    }
}

impl std::error::Error for ConfigError {}

pub struct CoordinatorConfig {
    pub validators: Vec<String>,
    pub signer_key: String,
    pub launch: Vec<LaunchEntry>,
}

pub struct LaunchEntry {
    pub miner_id: String,
    pub binary: String,
    pub configure: Configure,
}

fn make_configure(table: &toml::Table) -> Configure {
    let u32_of = |k: &str, d: u32| {
        table
            .get(k)
            .and_then(|v| v.as_integer())
            .map(|i| i as u32)
            .unwrap_or(d)
    };
    // Strip coordinator-owned keys; the rest passes through verbatim.
    let mut passthrough = table.clone();
    for k in [
        "binary",
        "queue_depth",
        "idle_timeout_s",
        "heartbeat_s",
        "reconnect_window_s",
    ] {
        passthrough.remove(k);
    }
    Configure {
        queue_depth: u32_of("queue_depth", 3),
        idle_timeout_s: u32_of("idle_timeout_s", 300),
        heartbeat_s: u32_of("heartbeat_s", 15),
        reconnect_window_s: u32_of("reconnect_window_s", 60),
        backend_toml: toml::to_string(&passthrough).unwrap_or_default(),
    }
}

fn default_binary(backend: &str) -> String {
    match backend {
        "dwave" => "quip-dwave-qa".into(),
        other => format!("quip-{other}-sa"),
    }
}

fn entry(miner_id: &str, backend: &str, table: &toml::Table) -> LaunchEntry {
    let binary = table
        .get("binary")
        .and_then(|v| v.as_str())
        .map(String::from)
        .unwrap_or_else(|| default_binary(backend));
    LaunchEntry {
        miner_id: miner_id.into(),
        binary,
        configure: make_configure(table),
    }
}

/// Parse a coordinator TOML config into validators, signer, and launch plan.
///
/// Section mapping: `[cpu]`→`cpu-0`, `[cuda.N]`→`cuda-N`, `[metal]`→`metal-0`,
/// `[dwave]`/`[qpu]`→`qpu-0`.
pub fn parse_config(toml_text: &str) -> Result<CoordinatorConfig, ConfigError> {
    let root: toml::Table =
        toml::from_str(toml_text).map_err(|e| ConfigError::BadToml(e.to_string()))?;
    let miner = root
        .get("miner")
        .and_then(|v| v.as_table())
        .ok_or(ConfigError::MissingMiner)?;
    let validators = miner
        .get("validators")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    let signer_key = miner
        .get("signer_key")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let mut launch = Vec::new();
    if let Some(t) = root.get("cpu").and_then(|v| v.as_table()) {
        launch.push(entry("cpu-0", "cpu", t));
    }
    if let Some(cuda) = root.get("cuda").and_then(|v| v.as_table()) {
        let mut idxs: Vec<&String> = cuda.keys().collect();
        idxs.sort();
        for k in idxs {
            if let Some(t) = cuda.get(k).and_then(|v| v.as_table()) {
                launch.push(entry(&format!("cuda-{k}"), "cuda", t));
            }
        }
    }
    if let Some(t) = root.get("metal").and_then(|v| v.as_table()) {
        launch.push(entry("metal-0", "metal", t));
    }
    // Prefer [dwave], fall back to [qpu].
    if let Some(t) = root
        .get("dwave")
        .or_else(|| root.get("qpu"))
        .and_then(|v| v.as_table())
    {
        launch.push(entry("qpu-0", "dwave", t));
    }
    Ok(CoordinatorConfig {
        validators,
        signer_key,
        launch,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
[miner]
validators = ["ws://127.0.0.1:9944"]
signer_key = "//Alice"

[cpu]
binary = "quip-cpu-gibbs"
num_cpus = 8
queue_depth = 4
idle_timeout_s = 120

[cuda.0]
device_index = 0

[cuda.1]
device_index = 1

[dwave]
daily_budget = "30s"
"#;

    #[test]
    fn maps_sections_to_launch_entries() {
        let c = parse_config(SAMPLE).unwrap();
        assert_eq!(c.validators, vec!["ws://127.0.0.1:9944"]);
        let ids: Vec<&str> = c.launch.iter().map(|e| e.miner_id.as_str()).collect();
        assert_eq!(ids, vec!["cpu-0", "cuda-0", "cuda-1", "qpu-0"]);
    }

    #[test]
    fn binary_override_and_default() {
        let c = parse_config(SAMPLE).unwrap();
        let cpu = c.launch.iter().find(|e| e.miner_id == "cpu-0").unwrap();
        assert_eq!(cpu.binary, "quip-cpu-gibbs");
        assert_eq!(cpu.configure.queue_depth, 4);
        assert_eq!(cpu.configure.idle_timeout_s, 120);
        let cuda = c.launch.iter().find(|e| e.miner_id == "cuda-0").unwrap();
        assert_eq!(cuda.binary, "quip-cuda-sa");
    }

    #[test]
    fn backend_toml_carries_unknown_keys() {
        let c = parse_config(SAMPLE).unwrap();
        let dwave = c.launch.iter().find(|e| e.miner_id == "qpu-0").unwrap();
        assert!(dwave.configure.backend_toml.contains("daily_budget"));
    }

    #[test]
    fn dwave_default_binary_is_qa() {
        let c = parse_config(SAMPLE).unwrap();
        let dwave = c.launch.iter().find(|e| e.miner_id == "qpu-0").unwrap();
        assert_eq!(dwave.binary, "quip-dwave-qa");
    }
}
