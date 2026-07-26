//! quip-coordinator: chain access, job production, routing, validation,
//! miner session server, and process supervision for the v0.3 protocol.

pub mod attempt;
pub mod chain;
pub mod config;
pub mod dashboard;
pub mod decay;
pub mod drive;
pub mod producer;
pub mod router;
pub mod runtime;
pub mod session;
pub mod supervisor;
pub mod timing;
pub mod topology;
pub mod validate;
