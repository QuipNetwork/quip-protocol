//! Spawn miner binaries, restart policy by exit code, shutdown fan-out.

use crate::config::LaunchEntry;
use crate::session::gen_session_token;
use std::collections::HashMap;
use tokio::process::{Child, Command};

/// Restart decision for a miner child exit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Restart {
    /// Exit 0: respawn when jobs exist.
    OnDemand,
    /// 64/69/77: operator/env/token error — do not respawn.
    Never,
    /// 70 or signal: exponential backoff + failure budget.
    Backoff,
}

/// Map a process exit code to the coordinator restart policy.
pub fn restart_policy(exit_code: i32) -> Restart {
    match exit_code {
        0 => Restart::OnDemand,
        64 | 69 | 77 => Restart::Never,
        70 => Restart::Backoff,
        // Negative codes (signals) and anything else → backoff.
        _ => Restart::Backoff,
    }
}

/// Tracks per-miner children and tokens.
pub struct Supervisor {
    pub children: HashMap<String, Child>,
    pub tokens: HashMap<String, String>,
    pub sock: String,
}

impl Supervisor {
    pub fn new(sock: impl Into<String>) -> Self {
        Self {
            children: HashMap::new(),
            tokens: HashMap::new(),
            sock: sock.into(),
        }
    }

    /// Spawn a miner binary with `QUIP_SESSION_TOKEN` (never argv).
    pub async fn spawn(
        &mut self,
        entry: &LaunchEntry,
        token: &str,
        sock_uri: &str,
    ) -> std::io::Result<()> {
        let child = Command::new(&entry.binary)
            .arg("--quip-coordinator")
            .arg(sock_uri)
            .arg("--miner-id")
            .arg(&entry.miner_id)
            .env("QUIP_SESSION_TOKEN", token)
            .kill_on_drop(true)
            .spawn()?;
        self.tokens
            .insert(entry.miner_id.clone(), token.to_string());
        self.children.insert(entry.miner_id.clone(), child);
        Ok(())
    }

    /// Spawn with a freshly generated token; returns the token.
    pub async fn spawn_with_new_token(
        &mut self,
        entry: &LaunchEntry,
        sock_uri: &str,
    ) -> std::io::Result<String> {
        let token = gen_session_token();
        self.spawn(entry, &token, sock_uri).await?;
        Ok(token)
    }

    /// Kill all children after a grace period (in-band Shutdown is preferred
    /// when a live session channel is available).
    pub async fn shutdown_all(&mut self, grace_ms: u32) {
        tokio::time::sleep(std::time::Duration::from_millis(grace_ms as u64)).await;
        for child in self.children.values_mut() {
            let _ = child.kill().await;
        }
        for child in self.children.values_mut() {
            let _ = child.wait().await;
        }
        self.children.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restart_policy_matches_exit_codes() {
        assert!(matches!(restart_policy(0), Restart::OnDemand));
        assert!(matches!(restart_policy(64), Restart::Never));
        assert!(matches!(restart_policy(69), Restart::Never));
        assert!(matches!(restart_policy(77), Restart::Never));
        assert!(matches!(restart_policy(70), Restart::Backoff));
        assert!(matches!(restart_policy(-9), Restart::Backoff)); // SIGKILL
    }
}
