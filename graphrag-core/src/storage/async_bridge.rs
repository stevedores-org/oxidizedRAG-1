//! Sync/async bridge for incremental knowledge graph updates.
//!
//! [`AsyncKnowledgeGraph`] wraps a sync [`KnowledgeGraph`] and an async
//! [`ProductionGraphStore`], keeping both in sync through the incremental
//! update pipeline.

use crate::core::{Entity, EntityId, KnowledgeGraph, Relationship, Result};
use crate::graph::incremental::{
    ConflictResolver, ConflictStrategy, IncrementalConfig, IncrementalGraphStore,
    ProductionGraphStore, UpdateId,
};
use std::sync::Arc;
use tokio::sync::Mutex;

/// Bridges the synchronous [`KnowledgeGraph`] with the async
/// [`ProductionGraphStore`] for incremental updates.
///
/// All mutations go through the `ProductionGraphStore` (which maintains
/// change logs, conflict resolution, cache invalidation, and optional
/// SurrealDB persistence), and the underlying `KnowledgeGraph` is updated
/// as a side-effect.
pub struct AsyncKnowledgeGraph {
    store: Arc<Mutex<ProductionGraphStore>>,
}

impl AsyncKnowledgeGraph {
    /// Create a new async bridge with default configuration.
    pub fn new(graph: KnowledgeGraph) -> Self {
        let config = IncrementalConfig::default();
        let resolver = ConflictResolver::new(ConflictStrategy::KeepNew);
        let store = ProductionGraphStore::new(graph, config, resolver);
        Self {
            store: Arc::new(Mutex::new(store)),
        }
    }

    /// Create from an existing `ProductionGraphStore`.
    pub fn from_store(store: ProductionGraphStore) -> Self {
        Self {
            store: Arc::new(Mutex::new(store)),
        }
    }

    /// Add an entity through the incremental pipeline.
    pub async fn add_entity(&self, entity: Entity) -> Result<UpdateId> {
        let mut store = self.store.lock().await;
        store.upsert_entity(entity).await
    }

    /// Add a relationship through the incremental pipeline.
    pub async fn add_relationship(&self, rel: Relationship) -> Result<UpdateId> {
        let mut store = self.store.lock().await;
        store.upsert_relationship(rel).await
    }

    /// Delete an entity through the incremental pipeline.
    pub async fn delete_entity(&self, entity_id: &EntityId) -> Result<UpdateId> {
        let mut store = self.store.lock().await;
        store.delete_entity(entity_id).await
    }

    /// Delete a relationship through the incremental pipeline.
    pub async fn delete_relationship(
        &self,
        source: &EntityId,
        target: &EntityId,
        relation_type: &str,
    ) -> Result<UpdateId> {
        let mut store = self.store.lock().await;
        store
            .delete_relationship(source, target, relation_type)
            .await
    }

    /// Batch upsert entities with conflict resolution.
    pub async fn batch_add_entities(
        &self,
        entities: Vec<Entity>,
        strategy: ConflictStrategy,
    ) -> Result<Vec<UpdateId>> {
        let mut store = self.store.lock().await;
        store.batch_upsert_entities(entities, strategy).await
    }

    /// Batch upsert relationships with conflict resolution.
    pub async fn batch_add_relationships(
        &self,
        relationships: Vec<Relationship>,
        strategy: ConflictStrategy,
    ) -> Result<Vec<UpdateId>> {
        let mut store = self.store.lock().await;
        store
            .batch_upsert_relationships(relationships, strategy)
            .await
    }

    /// Get a reference to the underlying store for advanced operations.
    pub fn store(&self) -> &Arc<Mutex<ProductionGraphStore>> {
        &self.store
    }
}
