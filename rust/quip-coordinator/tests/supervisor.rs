use quip_coordinator::supervisor::{restart_policy, Restart};

#[test]
fn restart_policy_matches_exit_codes() {
    assert!(matches!(restart_policy(0), Restart::OnDemand));
    assert!(matches!(restart_policy(64), Restart::Never));
    assert!(matches!(restart_policy(69), Restart::Never));
    assert!(matches!(restart_policy(77), Restart::Never));
    assert!(matches!(restart_policy(70), Restart::Backoff));
    assert!(matches!(restart_policy(-9), Restart::Backoff));
}
