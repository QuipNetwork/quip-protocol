//! Parse `config.toml` into a per-miner launch plan.

use quip_proto::v1::Configure;

/// Errors raised while parsing a coordinator TOML config.
#[derive(Debug, PartialEq)]
pub enum ConfigError {
    /// No `[miner]` section present.
    MissingMiner,
    /// TOML parse failure (message from the parser).
    BadToml(String),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingMiner => write!(f, "missing [miner] section"),
            Self::BadToml(e) => write!(f, "bad toml: {e}"),
        }
    }
}

impl std::error::Error for ConfigError {}

/// Fully parsed coordinator config: chain endpoints, signer, and launch plan.
pub struct CoordinatorConfig {
    /// Validator WebSocket endpoints.
    pub validators: Vec<String>,
    /// Signer key URI or keystore path.
    pub signer_key: String,
    /// One entry per supervised miner subprocess.
    pub launch: Vec<LaunchEntry>,
    /// Optional mining-attempt dashboard (`[dashboard]` section). `None`
    /// disables recording + the REST endpoint.
    pub dashboard: Option<DashboardConfig>,
}

/// `[dashboard]` config: where to serve the attempt logs and where they live.
pub struct DashboardConfig {
    /// HTTP listen address, e.g. `127.0.0.1:9090`.
    pub listen: String,
    /// Root directory for `<qblock_id>/attempts.jsonl` files.
    pub data_dir: String,
}

/// One supervised miner: identity, binary path, and handshake `Configure`.
pub struct LaunchEntry {
    /// Miner id used on the session wire (`cpu-0`, `cuda-1`, …).
    pub miner_id: String,
    /// Executable name or path for the miner subprocess.
    pub binary: String,
    /// Handshake configure payload (queue depths, heartbeat, backend TOML).
    pub configure: Configure,
}

fn make_configure(table: &toml::Table) -> Configure {
    let u32_of = |k: &str, d: u32| {
        table
            .get(k)
            .and_then(toml::Value::as_integer)
            .map_or(d, |i| {
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "config u32 knobs are small non-negative integers"
                )]
                {
                    i as u32
                }
            })
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
        let _ = passthrough.remove(k);
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
        .map_or_else(|| default_binary(backend), String::from);
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
///
/// # Errors
/// Returns [`ConfigError::MissingMiner`] when the `[miner]` section is absent,
/// or [`ConfigError::BadToml`] when the input is not valid TOML.
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
    let dashboard = root
        .get("dashboard")
        .and_then(|v| v.as_table())
        .and_then(|t| {
            let listen = t.get("listen").and_then(|v| v.as_str())?;
            let data_dir = t.get("data_dir").and_then(|v| v.as_str())?;
            Some(DashboardConfig {
                listen: listen.to_string(),
                data_dir: data_dir.to_string(),
            })
        });

    Ok(CoordinatorConfig {
        validators,
        signer_key,
        launch,
        dashboard,
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
min_block_budget = "90s"
budget_cap = "5m"
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
        // The dwave miner consumes its budget knobs from `backend_toml`; the
        // coordinator forwards every non-coordinator key verbatim, so all three
        // budget dimensions reach the miner.
        let c = parse_config(SAMPLE).unwrap();
        let dwave = c.launch.iter().find(|e| e.miner_id == "qpu-0").unwrap();
        let backend_toml = &dwave.configure.backend_toml;
        assert!(backend_toml.contains("daily_budget"));
        assert!(backend_toml.contains("min_block_budget"));
        assert!(backend_toml.contains("budget_cap"));
    }

    #[test]
    fn dwave_default_binary_is_qa() {
        let c = parse_config(SAMPLE).unwrap();
        let dwave = c.launch.iter().find(|e| e.miner_id == "qpu-0").unwrap();
        assert_eq!(dwave.binary, "quip-dwave-qa");
    }

    #[test]
    fn dashboard_section_parsed_when_present() {
        let c = parse_config(SAMPLE).unwrap();
        assert!(c.dashboard.is_none());

        let with = format!(
            "{SAMPLE}\n[dashboard]\nlisten = \"127.0.0.1:9090\"\ndata_dir = \"/data/attempts\"\n"
        );
        let d = parse_config(&with).unwrap().dashboard.unwrap();
        assert_eq!(d.listen, "127.0.0.1:9090");
        assert_eq!(d.data_dir, "/data/attempts");
    }

    #[test]
    fn dashboard_ignored_when_missing_keys() {
        // A [dashboard] section missing a required key disables it rather than
        // erroring, so a partial config still boots.
        let partial = format!("{SAMPLE}\n[dashboard]\nlisten = \"127.0.0.1:9090\"\n");
        assert!(parse_config(&partial).unwrap().dashboard.is_none());
    }
}
