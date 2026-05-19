//! AIVCS integration for oxidizedRAG.
//!
//! Bridges GraphRAG `ask()` runs to the AIVCS ledger for run tracking,
//! content-addressed config specs, and observability hooks.

pub mod adapter;
pub mod aivcs_adapter;
pub mod config_hasher;
pub mod persistence;
pub mod recorder;
pub mod run_recorder;
pub mod spec;

pub use adapter::RagAdapter;
pub use aivcs_adapter::RagToAivcsAdapter;
pub use config_hasher::RagConfigDigest;
pub use run_recorder::RagRunRecorder;
pub use spec::GraphRagSpec;
