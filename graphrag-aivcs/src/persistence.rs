//! SurrealDB Persistence Layer - bridges RAG runs to AIVCS for persistent storage
//!
//! This module integrates GraphRAG execution tracking with AIVCS for full version control:
//! - Records RAG runs to SurrealDB via AIVCS GraphRunRecorder
//! - Enables comparing RAG executions across experiments
//! - Supports evaluation and replay of RAG agent behavior

use crate::run_recorder::RagRunRecorder;
use aivcs_core::domain::run::{Event, EventKind};
use aivcs_core::GraphRunRecorder;
use oxidized_state::{ContentDigest, RunLedger, RunMetadata, RunSummary, StorageResult};
use serde_json::json;
use std::convert::TryFrom;
use std::sync::Arc;
use uuid::Uuid;

/// Persists a RAG run to SurrealDB via AIVCS GraphRunRecorder
///
/// This adapter captures the full RAG execution lifecycle:
/// 1. Each retrieval operation becomes a "tool_called" + "tool_returned" pair
/// 2. Each LLM interaction becomes a "tool_called" + "tool_returned" pair
/// 3. Config digest serves as the AgentSpec for reproducibility
/// 4. Run metadata captures query and execution context
///
/// # Example
///
/// ```ignore
/// let ledger = Arc::new(SurrealRunLedger::new(...).await?);
/// let mut recorder = RagRunRecorder::new("What is Rust?");
/// recorder.record_retrieval("chunk 1", 5, 0.95, 100);
/// recorder.record_llm_call("prompt", "response", 150, 500);
///
/// let config_digest = RagConfigDigest::from_config(config).as_hex().to_string();
/// let persisted = RagRunPersister::persist_run(
///     &recorder,
///     &config_digest,
///     "graphrag-agent",
///     ledger.clone(),
/// ).await?;
///
/// println!("Persisted run: {}", persisted.run_id);
/// ```
pub struct RagRunPersister;

/// Result of persisting a RAG run to SurrealDB
#[derive(Debug, Clone)]
pub struct PersistedRagRun {
    pub run_id: Uuid,
    pub query: String,
    pub event_count: u64,
    pub retrieval_count: u64,
    pub llm_calls: u64,
}

impl RagRunPersister {
    /// Persist a RAG run to SurrealDB
    ///
    /// Converts RAG events to AIVCS domain events and stores them in SurrealDB
    /// with full traceability for later comparison and evaluation.
    pub async fn persist_run(
        recorder: &RagRunRecorder,
        config_digest: &str,
        agent_name: &str,
        ledger: Arc<dyn RunLedger>,
    ) -> StorageResult<PersistedRagRun> {
        let summary = recorder.summary();
        let run_uuid = Uuid::parse_str(&summary.run_id).unwrap_or_else(|_| Uuid::new_v4());

        // Create run metadata
        let metadata = RunMetadata {
            git_sha: None,
            agent_name: agent_name.to_string(),
            tags: json!(["rag", "graphrag"]),
        };

        // Start the run with the config digest as the spec
        let content_digest = ContentDigest::try_from(config_digest.to_string())?;
        let graph_recorder =
            GraphRunRecorder::start(ledger.clone(), &content_digest, metadata).await?;

        // Convert each RAG event to AIVCS events
        let mut event_seq = 0u64;
        for rag_event in recorder.events() {
            event_seq += 1;

            // Tool called event
            let tool_name = format!("rag.{}", rag_event.event_type);
            let tool_called_event = Event::new(
                run_uuid,
                event_seq,
                EventKind::ToolCalled {
                    tool_name: tool_name.clone(),
                },
                rag_event.metadata.clone(),
            );
            graph_recorder.record(&tool_called_event).await?;

            event_seq += 1;

            // Tool returned event (success)
            let tool_returned_event = Event::new(
                run_uuid,
                event_seq,
                EventKind::ToolReturned {
                    tool_name: tool_name.clone(),
                },
                serde_json::json!({
                    "duration_ms": rag_event.duration_ms,
                    "status": "success",
                }),
            );
            graph_recorder.record(&tool_returned_event).await?;
        }

        // Complete the run
        let run_summary = RunSummary {
            duration_ms: summary.total_duration_ms as u64,
            total_events: event_seq,
            success: true,
            final_state_digest: Some(content_digest.clone()),
        };
        graph_recorder.finish_ok(run_summary).await?;

        Ok(PersistedRagRun {
            run_id: run_uuid,
            query: summary.query,
            event_count: summary.event_count as u64,
            retrieval_count: summary.retrieval_count as u64,
            llm_calls: summary.llm_calls as u64,
        })
    }

    /// Persist multiple RAG runs for batch experiment tracking
    pub async fn persist_runs(
        runs: &[&RagRunRecorder],
        config_digest: &str,
        agent_name: &str,
        ledger: Arc<dyn RunLedger>,
    ) -> StorageResult<Vec<PersistedRagRun>> {
        let mut results = Vec::new();
        for recorder in runs {
            let result =
                Self::persist_run(recorder, config_digest, agent_name, ledger.clone()).await?;
            results.push(result);
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_persisted_rag_run_creation() {
        let persisted = PersistedRagRun {
            run_id: Uuid::new_v4(),
            query: "test query".to_string(),
            event_count: 3,
            retrieval_count: 2,
            llm_calls: 1,
        };

        assert_eq!(persisted.query, "test query");
        assert_eq!(persisted.event_count, 3);
        assert_eq!(persisted.retrieval_count, 2);
        assert_eq!(persisted.llm_calls, 1);
    }

    #[test]
    fn test_persisted_rag_run_has_valid_uuid() {
        let run_id = Uuid::new_v4();
        let persisted = PersistedRagRun {
            run_id,
            query: "test".to_string(),
            event_count: 1,
            retrieval_count: 1,
            llm_calls: 0,
        };

        assert_eq!(persisted.run_id, run_id);
        assert!(!persisted.run_id.to_string().is_empty());
    }
}
