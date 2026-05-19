//! Performance gate tests for incremental indexing.
//!
//! Validates that incremental updates cost <5% of a full rebuild.

use graphrag_core::core::{Entity, EntityId, KnowledgeGraph, Relationship};
use graphrag_core::graph::incremental::{
    ConflictResolver, ConflictStrategy, IncrementalConfig, IncrementalGraphStore,
    ProductionGraphStore,
};
use std::time::Instant;

fn create_test_entity(i: usize) -> Entity {
    Entity {
        id: EntityId::new(format!("entity-{i}")),
        name: format!("Entity {i}"),
        entity_type: if i % 3 == 0 {
            "person"
        } else if i % 3 == 1 {
            "organization"
        } else {
            "location"
        }
        .to_string(),
        confidence: 0.85 + (i % 10) as f32 * 0.01,
        mentions: vec![],
        embedding: Some(vec![i as f32 * 0.01; 16]),
    }
}

fn create_test_relationship(src: usize, tgt: usize) -> Relationship {
    Relationship {
        source: EntityId::new(format!("entity-{src}")),
        target: EntityId::new(format!("entity-{tgt}")),
        relation_type: "related_to".to_string(),
        confidence: 0.75,
        context: vec![],
    }
}

/// Measure time for a full graph rebuild with N entities.
fn measure_full_rebuild(entity_count: usize, edge_count: usize) -> std::time::Duration {
    let start = Instant::now();
    let mut graph = KnowledgeGraph::new();

    for i in 0..entity_count {
        graph.add_entity(create_test_entity(i)).unwrap();
    }

    for i in 0..edge_count.min(entity_count.saturating_sub(1)) {
        graph
            .add_relationship(create_test_relationship(i, i + 1))
            .unwrap();
    }

    start.elapsed()
}

#[tokio::test]
async fn test_incremental_overhead_1k_entities() {
    let base_count = 1_000;
    let edge_count = 500;
    let delta_count = 10; // 1% delta

    // Measure full rebuild
    let full_time = measure_full_rebuild(base_count, edge_count);

    // Build base graph incrementally, then measure delta
    let config = IncrementalConfig::default();
    let resolver = ConflictResolver::new(ConflictStrategy::KeepNew);
    let mut graph = KnowledgeGraph::new();
    for i in 0..base_count {
        graph.add_entity(create_test_entity(i)).unwrap();
    }
    for i in 0..edge_count.min(base_count.saturating_sub(1)) {
        graph
            .add_relationship(create_test_relationship(i, i + 1))
            .unwrap();
    }

    let mut store = ProductionGraphStore::new(graph, config, resolver);

    // Measure incremental delta
    let start = Instant::now();
    for i in base_count..(base_count + delta_count) {
        store.upsert_entity(create_test_entity(i)).await.unwrap();
    }
    let incremental_time = start.elapsed();

    let overhead_pct = (incremental_time.as_secs_f64() / full_time.as_secs_f64()) * 100.0;

    println!(
        "[1K] Full rebuild: {:?}, Incremental ({} entities): {:?}, Overhead: {:.2}%",
        full_time, delta_count, incremental_time, overhead_pct
    );

    assert!(
        overhead_pct < 5.0,
        "1K graph: incremental overhead {:.2}% exceeds 5% gate",
        overhead_pct
    );
}

#[tokio::test]
async fn test_incremental_overhead_10k_entities() {
    let base_count = 10_000;
    let edge_count = 5_000;
    let delta_count = 100; // 1% delta

    // Measure full rebuild
    let full_time = measure_full_rebuild(base_count, edge_count);

    // Build base graph
    let config = IncrementalConfig::default();
    let resolver = ConflictResolver::new(ConflictStrategy::KeepNew);
    let mut graph = KnowledgeGraph::new();
    for i in 0..base_count {
        graph.add_entity(create_test_entity(i)).unwrap();
    }
    for i in 0..edge_count.min(base_count.saturating_sub(1)) {
        graph
            .add_relationship(create_test_relationship(i, i + 1))
            .unwrap();
    }

    let mut store = ProductionGraphStore::new(graph, config, resolver);

    // Measure incremental delta
    let start = Instant::now();
    for i in base_count..(base_count + delta_count) {
        store.upsert_entity(create_test_entity(i)).await.unwrap();
    }
    let incremental_time = start.elapsed();

    let overhead_pct = (incremental_time.as_secs_f64() / full_time.as_secs_f64()) * 100.0;

    println!(
        "[10K] Full rebuild: {:?}, Incremental ({} entities): {:?}, Overhead: {:.2}%",
        full_time, delta_count, incremental_time, overhead_pct
    );

    assert!(
        overhead_pct < 5.0,
        "10K graph: incremental overhead {:.2}% exceeds 5% gate",
        overhead_pct
    );
}

#[tokio::test]
async fn test_incremental_delete_performance() {
    let base_count = 5_000;
    let delete_count = 50; // 1% deletion

    // Measure full rebuild for baseline
    let full_time = measure_full_rebuild(base_count, 0);

    // Build base graph
    let config = IncrementalConfig::default();
    let resolver = ConflictResolver::new(ConflictStrategy::KeepNew);
    let mut graph = KnowledgeGraph::new();
    for i in 0..base_count {
        graph.add_entity(create_test_entity(i)).unwrap();
    }
    let mut store = ProductionGraphStore::new(graph, config, resolver);

    // Measure incremental delete
    let start = Instant::now();
    for i in 0..delete_count {
        let entity_id = EntityId::new(format!("entity-{i}"));
        store.delete_entity(&entity_id).await.unwrap();
    }
    let delete_time = start.elapsed();

    let overhead_pct = (delete_time.as_secs_f64() / full_time.as_secs_f64()) * 100.0;

    println!(
        "[5K] Full rebuild: {:?}, Incremental delete ({} entities): {:?}, Overhead: {:.2}%",
        full_time, delete_count, delete_time, overhead_pct
    );

    assert!(
        overhead_pct < 5.0,
        "5K graph: delete overhead {:.2}% exceeds 5% gate",
        overhead_pct
    );
}

#[tokio::test]
async fn test_batch_upsert_performance() {
    let base_count = 10_000;
    let batch_size = 100;

    // Measure full rebuild for baseline
    let full_time = measure_full_rebuild(base_count, 0);

    // Build base graph
    let config = IncrementalConfig::default();
    let resolver = ConflictResolver::new(ConflictStrategy::KeepNew);
    let mut graph = KnowledgeGraph::new();
    for i in 0..base_count {
        graph.add_entity(create_test_entity(i)).unwrap();
    }
    let mut store = ProductionGraphStore::new(graph, config, resolver);

    // Measure batch upsert
    let batch: Vec<Entity> = (base_count..(base_count + batch_size))
        .map(|i| create_test_entity(i))
        .collect();

    let start = Instant::now();
    store
        .batch_upsert_entities(batch, ConflictStrategy::KeepNew)
        .await
        .unwrap();
    let batch_time = start.elapsed();

    let overhead_pct = (batch_time.as_secs_f64() / full_time.as_secs_f64()) * 100.0;

    println!(
        "[10K] Full rebuild: {:?}, Batch upsert ({} entities): {:?}, Overhead: {:.2}%",
        full_time, batch_size, batch_time, overhead_pct
    );

    assert!(
        overhead_pct < 5.0,
        "10K graph: batch upsert overhead {:.2}% exceeds 5% gate",
        overhead_pct
    );
}
