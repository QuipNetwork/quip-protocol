use quip_proto::v1::{Configure, Hello, JobKind};

#[derive(Debug, PartialEq)]
pub enum SessionError { MissingToken, BadWelcome(u32) }

impl std::fmt::Display for SessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SessionError::MissingToken => write!(f, "QUIP_SESSION_TOKEN environment variable not set"),
            SessionError::BadWelcome(v) => write!(f, "unexpected protocol version in Welcome: {v}"),
        }
    }
}

impl std::error::Error for SessionError {}

#[repr(i32)]
pub enum ExitCode { Clean = 0, ConfigInvalid = 64, EnvIncompatible = 69, InternalFatal = 70, TokenRejected = 77 }

pub struct SessionConfig {
    pub miner_id: String,
    pub queue_depth: u32,
    pub idle_timeout_s: u32,
    pub heartbeat_s: u32,
    pub reconnect_window_s: u32,
}

impl SessionConfig {
    pub fn from_configure(miner_id: String, c: &Configure) -> Self {
        let d = |v: u32, default: u32| if v == 0 { default } else { v };
        SessionConfig {
            miner_id,
            queue_depth: d(c.queue_depth, 3),
            idle_timeout_s: d(c.idle_timeout_s, 300),
            heartbeat_s: d(c.heartbeat_s, 15),
            reconnect_window_s: d(c.reconnect_window_s, 60),
        }
    }
}

pub fn build_hello(miner_id: &str, backend: &str, algorithm: &str, supported: &[JobKind]) -> Result<Hello, SessionError> {
    let token = std::env::var("QUIP_SESSION_TOKEN").map_err(|_| SessionError::MissingToken)?;
    Ok(Hello {
        miner_id: miner_id.into(),
        session_token: token,
        protocol_version: 1,
        backend: backend.into(),
        algorithm: algorithm.into(),
        supported_kinds: supported.iter().map(|k| *k as i32).collect(),
        max_nodes: 0,
        max_edges: 0,
        native_topology_hash: None,
        features: vec![],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use quip_proto::v1::{Configure, JobKind};

    #[test]
    fn hello_requires_token() {
        std::env::remove_var("QUIP_SESSION_TOKEN");
        assert!(matches!(
            build_hello("cpu-0", "cpu", "sa", &[JobKind::IsingSample]),
            Err(SessionError::MissingToken)
        ));
        std::env::set_var("QUIP_SESSION_TOKEN", "tok-123");
        let h = build_hello("cpu-0", "cpu", "sa", &[JobKind::IsingSample]).unwrap();
        assert_eq!(h.session_token, "tok-123");
        assert_eq!(h.protocol_version, 1);
        assert_eq!(h.miner_id, "cpu-0");
    }

    #[test]
    fn configure_applies_defaults_for_zero_fields() {
        let c = Configure { queue_depth: 0, idle_timeout_s: 0, heartbeat_s: 0, reconnect_window_s: 0, backend_toml: String::new() };
        let cfg = SessionConfig::from_configure("cpu-0".into(), &c);
        assert_eq!(cfg.queue_depth, 3);
        assert_eq!(cfg.idle_timeout_s, 300);
        assert_eq!(cfg.heartbeat_s, 15);
        assert_eq!(cfg.reconnect_window_s, 60);
    }
}
