use quip_proto::v1::{Configure, Hello, JobKind, Welcome};

/// Protocol version this SDK speaks. `Welcome.protocol_version` must equal this.
pub const PROTOCOL_VERSION: u32 = 1;

#[derive(Debug, PartialEq)]
pub enum SessionError {
    MissingToken,
    BadWelcome(u32),
}

impl std::fmt::Display for SessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SessionError::MissingToken => {
                write!(f, "QUIP_SESSION_TOKEN environment variable not set")
            }
            SessionError::BadWelcome(v) => write!(f, "unexpected protocol version in Welcome: {v}"),
        }
    }
}

impl std::error::Error for SessionError {}

/// Sysexits-style process exit codes (also carried on `Fatal.exit_code`).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExitCode {
    Clean = 0,
    /// Missing/invalid CLI or config (e.g. no `--quip-coordinator`, bad Welcome).
    ConfigInvalid = 64,
    /// Host/env cannot run this miner (`--check` failed).
    EnvIncompatible = 69,
    /// Unexpected internal failure.
    InternalFatal = 70,
    /// `QUIP_SESSION_TOKEN` missing/empty or rejected.
    TokenRejected = 77,
}

impl ExitCode {
    pub fn as_i32(self) -> i32 {
        self as i32
    }
}

impl From<SessionError> for ExitCode {
    fn from(e: SessionError) -> Self {
        match e {
            SessionError::MissingToken => ExitCode::TokenRejected,
            SessionError::BadWelcome(_) => ExitCode::ConfigInvalid,
        }
    }
}

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

pub fn build_hello(
    miner_id: &str,
    backend: &str,
    algorithm: &str,
    supported: &[JobKind],
) -> Result<Hello, SessionError> {
    let token = std::env::var("QUIP_SESSION_TOKEN").map_err(|_| SessionError::MissingToken)?;
    if token.is_empty() {
        return Err(SessionError::MissingToken);
    }
    Ok(Hello {
        miner_id: miner_id.into(),
        session_token: token,
        protocol_version: PROTOCOL_VERSION,
        backend: backend.into(),
        algorithm: algorithm.into(),
        supported_kinds: supported.iter().map(|k| *k as i32).collect(),
        max_nodes: 0,
        max_edges: 0,
        native_topology_hash: None,
        features: vec![],
    })
}

/// Reject a `Welcome` whose `protocol_version` is not what this SDK speaks.
pub fn check_welcome(w: &Welcome) -> Result<(), SessionError> {
    if w.protocol_version != PROTOCOL_VERSION {
        return Err(SessionError::BadWelcome(w.protocol_version));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use quip_proto::v1::{Configure, JobKind, Welcome};

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

        // An empty (but present) token is treated the same as a missing one.
        std::env::set_var("QUIP_SESSION_TOKEN", "");
        assert!(matches!(
            build_hello("cpu-0", "cpu", "sa", &[JobKind::IsingSample]),
            Err(SessionError::MissingToken)
        ));
        std::env::remove_var("QUIP_SESSION_TOKEN");
    }

    #[test]
    fn configure_applies_defaults_for_zero_fields() {
        let c = Configure {
            queue_depth: 0,
            idle_timeout_s: 0,
            heartbeat_s: 0,
            reconnect_window_s: 0,
            backend_toml: String::new(),
        };
        let cfg = SessionConfig::from_configure("cpu-0".into(), &c);
        assert_eq!(cfg.queue_depth, 3);
        assert_eq!(cfg.idle_timeout_s, 300);
        assert_eq!(cfg.heartbeat_s, 15);
        assert_eq!(cfg.reconnect_window_s, 60);
    }

    #[test]
    fn welcome_rejects_non_v1_protocol_version() {
        assert!(check_welcome(&Welcome {
            protocol_version: 1
        })
        .is_ok());
        assert_eq!(
            check_welcome(&Welcome {
                protocol_version: 2
            }),
            Err(SessionError::BadWelcome(2))
        );
        assert_eq!(
            check_welcome(&Welcome {
                protocol_version: 0
            }),
            Err(SessionError::BadWelcome(0))
        );
    }

    #[test]
    fn session_error_maps_to_documented_exit_codes() {
        assert_eq!(
            ExitCode::from(SessionError::MissingToken),
            ExitCode::TokenRejected
        );
        assert_eq!(
            ExitCode::from(SessionError::BadWelcome(2)),
            ExitCode::ConfigInvalid
        );
        assert_eq!(ExitCode::ConfigInvalid.as_i32(), 64);
        assert_eq!(ExitCode::EnvIncompatible.as_i32(), 69);
        assert_eq!(ExitCode::InternalFatal.as_i32(), 70);
        assert_eq!(ExitCode::TokenRejected.as_i32(), 77);
    }
}
