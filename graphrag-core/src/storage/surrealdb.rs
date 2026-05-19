//! SurrealDB-backed delta storage for incremental graph updates
//!
//! Provides ACID-guaranteed persistence for `GraphDelta`, `ChangeRecord`,
//! and transaction metadata using SurrealDB.
//!
//! Supports multiple connection modes:
//! - In-memory: `mem://` (for testing)
//! - File-based: `file://path/to/db`
//! - Remote: `ws://host:port` or `wss://host:port`
//!
//! Requires the `surrealdb-storage` feature:
//! ```toml
//! graphrag-core = { version = "0.1", features = ["surrealdb-storage"] }
//! ```

use crate::core::{GraphRAGError, Result};
use crate::graph::incremental::{ChangeRecord, DeltaStatus, GraphDelta, RollbackData, UpdateId};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use surrealdb::engine::any::{connect, Any};
use surrealdb::opt::auth::Root;
use surrealdb::sql::Thing;
use surrealdb::Surreal;

/// SurrealDB-backed storage for incremental graph deltas and transactions.
pub struct SurrealDeltaStorage {
    db: Surreal<Any>,
}

/// Builder for remote SurrealDB connections with authentication.
pub struct SurrealDeltaStorageBuilder {
    url: String,
    username: Option<String>,
    password: Option<String>,
    namespace: String,
    database: String,
}

impl SurrealDeltaStorageBuilder {
    /// Set authentication credentials.
    pub fn credentials(mut self, username: impl Into<String>, password: impl Into<String>) -> Self {
        self.username = Some(username.into());
        self.password = Some(password.into());
        self
    }

    /// Set the namespace.
    pub fn namespace(mut self, ns: impl Into<String>) -> Self {
        self.namespace = ns.into();
        self
    }

    /// Set the database name.
    pub fn database(mut self, db: impl Into<String>) -> Self {
        self.database = db.into();
        self
    }

    /// Build and connect to SurrealDB.
    pub async fn build(self) -> Result<SurrealDeltaStorage> {
        let db: Surreal<Any> = connect(&self.url)
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("SurrealDB connect failed: {e}"),
            })?;

        if let (Some(user), Some(pass)) = (self.username, self.password) {
            db.signin(Root {
                username: &user,
                password: &pass,
            })
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("SurrealDB auth failed: {e}"),
            })?;
        }

        db.use_ns(&self.namespace)
            .use_db(&self.database)
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("SurrealDB use ns/db failed: {e}"),
            })?;

        let storage = SurrealDeltaStorage { db };
        storage.setup_schema().await?;
        Ok(storage)
    }
}

// -- Internal record types for SurrealDB serialization --

#[derive(Debug, Serialize, Deserialize)]
struct DeltaRecord {
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<Thing>,
    delta_id: String,
    timestamp: String,
    status: String,
    dependencies: Vec<String>,
    changes: serde_json::Value,
    rollback_data: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChangeRecordRow {
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<Thing>,
    change_id: String,
    delta_id: String,
    timestamp: String,
    change_type: String,
    operation: String,
    entity_id: Option<String>,
    data: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct TransactionRecord {
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<Thing>,
    tx_id: String,
    status: String,
    created_at: String,
    committed_at: Option<String>,
}

impl SurrealDeltaStorage {
    /// Create an in-memory storage instance (for testing).
    pub async fn memory() -> Result<Self> {
        let db: Surreal<Any> = connect("mem://")
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("SurrealDB connect failed: {e}"),
            })?;

        db.use_ns("graphrag")
            .use_db("incremental")
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("SurrealDB use ns/db failed: {e}"),
            })?;

        let storage = Self { db };
        storage.setup_schema().await?;
        Ok(storage)
    }

    /// Create a file-based storage instance.
    pub async fn file(path: impl AsRef<str>) -> Result<Self> {
        let url = format!("file://{}", path.as_ref());
        let db: Surreal<Any> = connect(&url).await.map_err(|e| GraphRAGError::Storage {
            message: format!("SurrealDB connect failed: {e}"),
        })?;

        db.use_ns("graphrag")
            .use_db("incremental")
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("SurrealDB use ns/db failed: {e}"),
            })?;

        let storage = Self { db };
        storage.setup_schema().await?;
        Ok(storage)
    }

    /// Connect to a remote SurrealDB server (e.g. `wss://surrealdb.stevedores.org`).
    pub fn connect_remote(url: impl Into<String>) -> SurrealDeltaStorageBuilder {
        SurrealDeltaStorageBuilder {
            url: url.into(),
            username: None,
            password: None,
            namespace: "graphrag".to_string(),
            database: "incremental".to_string(),
        }
    }

    async fn setup_schema(&self) -> Result<()> {
        self.db
            .query(
                r#"
                DEFINE INDEX IF NOT EXISTS idx_delta_status ON TABLE deltas COLUMNS status;
                DEFINE INDEX IF NOT EXISTS idx_delta_timestamp ON TABLE deltas COLUMNS timestamp;
                DEFINE INDEX IF NOT EXISTS idx_change_delta ON TABLE change_records COLUMNS delta_id;
                DEFINE INDEX IF NOT EXISTS idx_change_entity ON TABLE change_records COLUMNS entity_id;
                DEFINE INDEX IF NOT EXISTS idx_tx_status ON TABLE transactions COLUMNS status;
                "#,
            )
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to setup schema: {e}"),
            })?;
        Ok(())
    }

    /// Persist a complete GraphDelta with all its change records.
    pub async fn persist_delta(&self, delta: &GraphDelta) -> Result<()> {
        let delta_id_str = delta.delta_id.as_str().to_string();

        let record = DeltaRecord {
            id: None,
            delta_id: delta_id_str.clone(),
            timestamp: delta.timestamp.to_rfc3339(),
            status: serde_json::to_string(&delta.status).unwrap_or_default(),
            dependencies: delta
                .dependencies
                .iter()
                .map(|d| d.as_str().to_string())
                .collect(),
            changes: serde_json::to_value(&delta.changes).unwrap_or_default(),
            rollback_data: delta
                .rollback_data
                .as_ref()
                .and_then(|rd| serde_json::to_value(rd).ok()),
        };

        let _: Option<DeltaRecord> = self
            .db
            .create(("deltas", delta_id_str.as_str()))
            .content(record)
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to persist delta: {e}"),
            })?;

        // Also persist individual change records for queryability
        for change in &delta.changes {
            self.persist_change(&delta_id_str, change).await?;
        }

        Ok(())
    }

    /// Persist an individual change record associated with a delta.
    async fn persist_change(&self, delta_id: &str, change: &ChangeRecord) -> Result<()> {
        let change_id_str = change.change_id.as_str().to_string();

        let row = ChangeRecordRow {
            id: None,
            change_id: change_id_str.clone(),
            delta_id: delta_id.to_string(),
            timestamp: change.timestamp.to_rfc3339(),
            change_type: serde_json::to_string(&change.change_type).unwrap_or_default(),
            operation: serde_json::to_string(&change.operation).unwrap_or_default(),
            entity_id: change.entity_id.as_ref().map(|eid| eid.to_string()),
            data: serde_json::to_value(&change.data).unwrap_or_default(),
        };

        let _: Option<ChangeRecordRow> = self
            .db
            .create(("change_records", change_id_str.as_str()))
            .content(row)
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to persist change record: {e}"),
            })?;

        Ok(())
    }

    /// Load a delta by ID.
    pub async fn load_delta(&self, delta_id: &UpdateId) -> Result<GraphDelta> {
        let record: Option<DeltaRecord> = self
            .db
            .select(("deltas", delta_id.as_str()))
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to load delta: {e}"),
            })?;

        let record = record.ok_or_else(|| GraphRAGError::Storage {
            message: format!("Delta {} not found", delta_id),
        })?;

        self.delta_record_to_graph_delta(record)
    }

    /// Delete a delta and its associated change records.
    pub async fn delete_delta(&self, delta_id: &UpdateId) -> Result<()> {
        // Delete change records first
        let id_str = delta_id.as_str().to_string();
        self.db
            .query("DELETE FROM change_records WHERE delta_id = $delta_id")
            .bind(("delta_id", id_str))
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to delete change records: {e}"),
            })?;

        // Delete the delta itself
        let _: Option<DeltaRecord> = self
            .db
            .delete(("deltas", delta_id.as_str()))
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to delete delta: {e}"),
            })?;

        Ok(())
    }

    /// List deltas, optionally filtered by timestamp.
    pub async fn list_deltas(&self, since: Option<DateTime<Utc>>) -> Result<Vec<GraphDelta>> {
        let records: Vec<DeltaRecord> = if let Some(since_time) = since {
            let since_str = since_time.to_rfc3339();
            let mut result = self
                .db
                .query("SELECT * FROM deltas WHERE timestamp >= $since ORDER BY timestamp ASC")
                .bind(("since", since_str))
                .await
                .map_err(|e| GraphRAGError::Storage {
                    message: format!("Failed to list deltas: {e}"),
                })?;

            result.take(0).map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to parse deltas: {e}"),
            })?
        } else {
            let mut result = self
                .db
                .query("SELECT * FROM deltas ORDER BY timestamp ASC")
                .await
                .map_err(|e| GraphRAGError::Storage {
                    message: format!("Failed to list deltas: {e}"),
                })?;

            result.take(0).map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to parse deltas: {e}"),
            })?
        };

        records
            .into_iter()
            .map(|r| self.delta_record_to_graph_delta(r))
            .collect()
    }

    /// Update the status of a delta.
    pub async fn update_delta_status(
        &self,
        delta_id: &UpdateId,
        status: DeltaStatus,
    ) -> Result<()> {
        let status_str = serde_json::to_string(&status).unwrap_or_default();
        self.db
            .query("UPDATE deltas SET status = $status WHERE delta_id = $delta_id")
            .bind(("delta_id", delta_id.as_str().to_string()))
            .bind(("status", status_str))
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to update delta status: {e}"),
            })?;
        Ok(())
    }

    /// Record a transaction.
    pub async fn record_transaction(&self, tx_id: &str, status: &str) -> Result<()> {
        let record = TransactionRecord {
            id: None,
            tx_id: tx_id.to_string(),
            status: status.to_string(),
            created_at: Utc::now().to_rfc3339(),
            committed_at: None,
        };

        let _: Option<TransactionRecord> = self
            .db
            .create(("transactions", tx_id))
            .content(record)
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to record transaction: {e}"),
            })?;

        Ok(())
    }

    /// Update transaction status.
    pub async fn update_transaction_status(&self, tx_id: &str, status: &str) -> Result<()> {
        let committed_at = if status == "committed" {
            Some(Utc::now().to_rfc3339())
        } else {
            None
        };

        self.db
            .query(
                "UPDATE transactions SET status = $status, committed_at = $committed_at WHERE tx_id = $tx_id",
            )
            .bind(("tx_id", tx_id.to_string()))
            .bind(("status", status.to_string()))
            .bind(("committed_at", committed_at))
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to update transaction status: {e}"),
            })?;

        Ok(())
    }

    /// List pending (uncommitted) transactions for crash recovery.
    pub async fn list_pending_transactions(&self) -> Result<Vec<String>> {
        let mut result = self
            .db
            .query("SELECT tx_id FROM transactions WHERE status = 'active'")
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to list pending transactions: {e}"),
            })?;

        #[derive(Deserialize)]
        struct TxIdRow {
            tx_id: String,
        }

        let rows: Vec<TxIdRow> = result.take(0).map_err(|e| GraphRAGError::Storage {
            message: format!("Failed to parse transactions: {e}"),
        })?;

        Ok(rows.into_iter().map(|r| r.tx_id).collect())
    }

    /// Get committed deltas for recovery replay.
    pub async fn get_committed_deltas(&self) -> Result<Vec<GraphDelta>> {
        let status_str = serde_json::to_string(&DeltaStatus::Committed).unwrap_or_default();
        let mut result = self
            .db
            .query("SELECT * FROM deltas WHERE status = $status ORDER BY timestamp ASC")
            .bind(("status", status_str))
            .await
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to get committed deltas: {e}"),
            })?;

        let records: Vec<DeltaRecord> = result.take(0).map_err(|e| GraphRAGError::Storage {
            message: format!("Failed to parse deltas: {e}"),
        })?;

        records
            .into_iter()
            .map(|r| self.delta_record_to_graph_delta(r))
            .collect()
    }

    fn delta_record_to_graph_delta(&self, record: DeltaRecord) -> Result<GraphDelta> {
        let timestamp = chrono::DateTime::parse_from_rfc3339(&record.timestamp)
            .map_err(|e| GraphRAGError::Storage {
                message: format!("Failed to parse timestamp: {e}"),
            })?
            .with_timezone(&Utc);

        let status: DeltaStatus =
            serde_json::from_str(&record.status).unwrap_or(DeltaStatus::Pending);

        let changes: Vec<ChangeRecord> = serde_json::from_value(record.changes).unwrap_or_default();

        let rollback_data: Option<RollbackData> = record
            .rollback_data
            .and_then(|v| serde_json::from_value(v).ok());

        let dependencies: Vec<UpdateId> = record
            .dependencies
            .into_iter()
            .map(UpdateId::from_string)
            .collect();

        Ok(GraphDelta {
            delta_id: UpdateId::from_string(record.delta_id),
            timestamp,
            changes,
            dependencies,
            status,
            rollback_data,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::incremental::{ChangeData, ChangeType, Operation};

    fn make_test_delta() -> GraphDelta {
        GraphDelta {
            delta_id: UpdateId::new(),
            timestamp: Utc::now(),
            changes: vec![ChangeRecord {
                change_id: UpdateId::new(),
                timestamp: Utc::now(),
                change_type: ChangeType::EntityAdded,
                entity_id: Some(crate::core::EntityId::new("test-entity-1".into())),
                document_id: None,
                operation: Operation::Insert,
                data: ChangeData::Empty,
                metadata: std::collections::HashMap::new(),
            }],
            dependencies: vec![],
            status: DeltaStatus::Pending,
            rollback_data: None,
        }
    }

    #[tokio::test]
    async fn test_memory_storage_crud() {
        let storage = SurrealDeltaStorage::memory().await.unwrap();

        let delta = make_test_delta();
        let delta_id = delta.delta_id.clone();

        // Persist
        storage.persist_delta(&delta).await.unwrap();

        // Load
        let loaded = storage.load_delta(&delta_id).await.unwrap();
        assert_eq!(loaded.delta_id, delta_id);
        assert_eq!(loaded.changes.len(), 1);

        // Update status
        storage
            .update_delta_status(&delta_id, DeltaStatus::Committed)
            .await
            .unwrap();

        // List committed
        let committed = storage.get_committed_deltas().await.unwrap();
        assert_eq!(committed.len(), 1);

        // Delete
        storage.delete_delta(&delta_id).await.unwrap();
        assert!(storage.load_delta(&delta_id).await.is_err());
    }

    #[tokio::test]
    async fn test_list_deltas_since() {
        let storage = SurrealDeltaStorage::memory().await.unwrap();

        let delta1 = make_test_delta();
        storage.persist_delta(&delta1).await.unwrap();

        let cutoff = Utc::now();
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        let delta2 = make_test_delta();
        storage.persist_delta(&delta2).await.unwrap();

        let all = storage.list_deltas(None).await.unwrap();
        assert_eq!(all.len(), 2);

        let recent = storage.list_deltas(Some(cutoff)).await.unwrap();
        assert_eq!(recent.len(), 1);
    }

    #[tokio::test]
    async fn test_transaction_tracking() {
        let storage = SurrealDeltaStorage::memory().await.unwrap();

        storage
            .record_transaction("tx-001", "active")
            .await
            .unwrap();
        storage
            .record_transaction("tx-002", "active")
            .await
            .unwrap();

        let pending = storage.list_pending_transactions().await.unwrap();
        assert_eq!(pending.len(), 2);

        storage
            .update_transaction_status("tx-001", "committed")
            .await
            .unwrap();

        let pending = storage.list_pending_transactions().await.unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0], "tx-002");
    }
}
