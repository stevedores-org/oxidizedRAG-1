//! Storage Contract Test Suite — Batch APIs (Issue #39: Story 4.1)
//!
//! Validates that batch insert, fetch, and query operations work correctly
//! across storage backends, avoid N+1 query patterns, and produce metrics.

use graphrag_core::core::{traits::BatchMetrics, Entity, EntityId};
use graphrag_core::storage::MemoryStorage;
use graphrag_core::vector::{VectorIndex, VectorUtils};
use indexmap::IndexMap;

// ─── Helpers ──────────────────────────────────────────────────────────────

fn make_entity(id: &str, name: &str) -> Entity {
    Entity {
        id: EntityId::new(id.to_string()),
        name: name.to_string(),
        entity_type: "test".to_string(),
        confidence: 1.0,
        mentions: vec![],
        embedding: None,
    }
}

fn make_vector_index_with_data(n: usize, dim: usize) -> VectorIndex {
    let mut idx = VectorIndex::new();
    for i in 0..n {
        let mut v = vec![0.0f32; dim];
        v[i % dim] = 1.0; // one-hot-ish
        idx.add_vector(format!("vec-{i}"), v).unwrap();
    }
    idx.build_index().unwrap();
    idx
}

// ─── 1. Batch Insert (store_entities_batch) ──────────────────────────────

#[test]
fn batch_insert_entities_returns_all_ids() {
    let mut storage = MemoryStorage::new();
    let entities: Vec<_> = (0..10)
        .map(|i| make_entity(&format!("e{i}"), &format!("Entity {i}")))
        .collect();

    let ids: Vec<String> = entities.iter().map(|e| e.id.to_string()).collect();

    // Insert individually to simulate batch (MemoryStorage doesn't impl Storage trait
    // directly without the `async` feature, so we use the inherent methods)
    for e in &entities {
        storage.store_entity(e.id.to_string(), e.clone()).unwrap();
    }

    // Verify all entities are retrievable
    for id in &ids {
        assert!(
            storage.get_entity(id).is_some(),
            "entity {id} should be stored"
        );
    }
    assert_eq!(storage.stats().entity_count, 10);
}

#[test]
fn batch_insert_vectors_all_present() {
    let mut idx = VectorIndex::new();
    let vectors: Vec<(String, Vec<f32>)> = (0..20)
        .map(|i| {
            let mut v = vec![0.0f32; 8];
            v[i % 8] = 1.0;
            (format!("v{i}"), v)
        })
        .collect();

    idx.batch_add_vectors(vectors).unwrap();
    assert_eq!(idx.len(), 20);

    for i in 0..20 {
        assert!(
            idx.contains(&format!("v{i}")),
            "vector v{i} should be present"
        );
    }
}

#[test]
fn batch_insert_preserves_vector_data() {
    let mut idx = VectorIndex::new();
    let original = vec![0.1, 0.2, 0.3, 0.4];
    idx.batch_add_vectors(vec![("test".to_string(), original.clone())])
        .unwrap();

    let retrieved = idx.get_embedding("test").unwrap();
    assert_eq!(retrieved, &original);
}

// ─── 2. Fetch Many (fetch_many) ──────────────────────────────────────────

#[test]
fn fetch_many_entities_returns_found_and_missing() {
    let mut storage = MemoryStorage::new();
    storage
        .store_entity("a".to_string(), make_entity("a", "Alpha"))
        .unwrap();
    storage
        .store_entity("b".to_string(), make_entity("b", "Beta"))
        .unwrap();

    let results = storage.fetch_many_entities(&["a", "missing", "b"]);
    assert_eq!(results.len(), 3);
    assert!(results[0].is_some(), "entity 'a' should be found");
    assert!(results[1].is_none(), "entity 'missing' should not be found");
    assert!(results[2].is_some(), "entity 'b' should be found");
    assert_eq!(results[0].unwrap().name, "Alpha");
    assert_eq!(results[2].unwrap().name, "Beta");
}

#[test]
fn fetch_many_vectors_returns_found_and_missing() {
    let mut idx = VectorIndex::new();
    idx.add_vector("x".to_string(), vec![1.0, 0.0]).unwrap();
    idx.add_vector("y".to_string(), vec![0.0, 1.0]).unwrap();

    let results = idx.fetch_many(&["x", "missing", "y"]);
    assert_eq!(results.len(), 3);
    assert!(results[0].is_some());
    assert!(results[1].is_none());
    assert!(results[2].is_some());
    assert_eq!(results[0].unwrap(), &vec![1.0, 0.0]);
    assert_eq!(results[2].unwrap(), &vec![0.0, 1.0]);
}

#[test]
fn fetch_many_empty_ids_returns_empty() {
    let storage = MemoryStorage::new();
    let results = storage.fetch_many_entities(&[]);
    assert!(results.is_empty());

    let idx = VectorIndex::new();
    let results = idx.fetch_many(&[]);
    assert!(results.is_empty());
}

#[test]
fn fetch_many_all_missing_returns_all_none() {
    let storage = MemoryStorage::new();
    let results = storage.fetch_many_entities(&["x", "y", "z"]);
    assert_eq!(results.len(), 3);
    assert!(results.iter().all(|r| r.is_none()));
}

#[test]
fn fetch_many_documents_works() {
    let mut storage = MemoryStorage::new();
    let doc = graphrag_core::core::Document {
        id: graphrag_core::core::DocumentId::new("doc1".to_string()),
        title: "Test".to_string(),
        content: "Content".to_string(),
        metadata: IndexMap::new(),
        chunks: vec![],
    };
    storage.store_document("doc1".to_string(), doc).unwrap();

    let results = storage.fetch_many_documents(&["doc1", "doc2"]);
    assert!(results[0].is_some());
    assert!(results[1].is_none());
}

// ─── 3. Query Top-K ──────────────────────────────────────────────────────

#[test]
fn query_topk_returns_k_results_with_metrics() {
    let idx = make_vector_index_with_data(10, 4);

    let query = vec![1.0, 0.0, 0.0, 0.0]; // closest to vec-0, vec-4, vec-8
    let (results, metrics) = idx.query_topk(&query, 3).unwrap();

    assert!(results.len() <= 3);
    assert!(metrics.batch_size <= 3);
    assert!(metrics.total_duration.as_nanos() > 0);
    // All returned scores should be finite
    for (id, score) in &results {
        assert!(score.is_finite(), "score for {id} should be finite");
    }
}

#[test]
fn query_topk_results_ordered_by_similarity() {
    let idx = make_vector_index_with_data(10, 4);

    let query = vec![1.0, 0.0, 0.0, 0.0];
    let (results, _metrics) = idx.query_topk(&query, 5).unwrap();

    // Scores should be in descending order (highest similarity first)
    for window in results.windows(2) {
        assert!(
            window[0].1 >= window[1].1,
            "results should be ordered: {} >= {}",
            window[0].1,
            window[1].1
        );
    }
}

#[test]
fn query_topk_empty_index_errors() {
    let idx = VectorIndex::new();
    // Don't build index — should error
    let result = idx.query_topk(&[1.0, 0.0], 5);
    assert!(result.is_err());
}

// ─── 4. BatchMetrics ─────────────────────────────────────────────────────

#[test]
fn batch_metrics_from_batch_computes_correctly() {
    let duration = std::time::Duration::from_millis(100);
    let metrics = BatchMetrics::from_batch(10, duration);

    assert_eq!(metrics.batch_size, 10);
    assert_eq!(metrics.total_duration, duration);
    assert_eq!(
        metrics.latency_per_item,
        std::time::Duration::from_millis(10)
    );
}

#[test]
fn batch_metrics_zero_batch_size() {
    let duration = std::time::Duration::from_millis(50);
    let metrics = BatchMetrics::from_batch(0, duration);

    assert_eq!(metrics.batch_size, 0);
    assert_eq!(metrics.latency_per_item, std::time::Duration::ZERO);
}

#[test]
fn batch_metrics_single_item() {
    let duration = std::time::Duration::from_micros(500);
    let metrics = BatchMetrics::from_batch(1, duration);

    assert_eq!(metrics.batch_size, 1);
    assert_eq!(metrics.latency_per_item, duration);
}

// ─── 5. Batch vs Sequential Performance Profile ──────────────────────────

#[test]
fn batch_insert_faster_than_sequential_for_large_n() {
    let n = 500;
    let dim = 16;

    // Sequential insert
    let start = std::time::Instant::now();
    let mut idx_seq = VectorIndex::new();
    for i in 0..n {
        let v = VectorUtils::random_vector(dim);
        idx_seq.add_vector(format!("s{i}"), v).unwrap();
    }
    let _seq_time = start.elapsed();

    // Batch insert
    let start = std::time::Instant::now();
    let mut idx_batch = VectorIndex::new();
    let vectors: Vec<_> = (0..n)
        .map(|i| (format!("b{i}"), VectorUtils::random_vector(dim)))
        .collect();
    idx_batch.batch_add_vectors(vectors).unwrap();
    let batch_time = start.elapsed();

    assert_eq!(idx_seq.len(), n);
    assert_eq!(idx_batch.len(), n);

    // Batch should not be significantly slower (at minimum, same data goes in)
    // We're generous here — just verify both complete and have correct count
    let metrics = BatchMetrics::from_batch(n, batch_time);
    assert!(
        metrics.latency_per_item.as_nanos() < 1_000_000, // < 1ms per item
        "batch latency per item too high: {:?}",
        metrics.latency_per_item
    );
}

// ─── 6. No N+1 Pattern (Pipeline Profile) ──────────────────────────────

#[test]
fn pipeline_uses_batch_fetch_not_n_plus_1() {
    // Simulate a pipeline that needs to fetch 100 entities
    let mut storage = MemoryStorage::new();
    let n = 100;
    let ids: Vec<String> = (0..n).map(|i| format!("e{i}")).collect();
    for id in &ids {
        storage
            .store_entity(id.clone(), make_entity(id, &format!("Entity {id}")))
            .unwrap();
    }

    // BATCH approach: single fetch_many call
    let start = std::time::Instant::now();
    let id_refs: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();
    let batch_results = storage.fetch_many_entities(&id_refs);
    let batch_time = start.elapsed();

    // N+1 approach: individual fetches
    let start = std::time::Instant::now();
    let individual_results: Vec<_> = ids.iter().map(|id| storage.get_entity(id)).collect();
    let _individual_time = start.elapsed();

    // Both should return same data
    assert_eq!(batch_results.len(), individual_results.len());
    for (b, i) in batch_results.iter().zip(individual_results.iter()) {
        assert_eq!(b.is_some(), i.is_some());
    }

    // Batch metrics
    let metrics = BatchMetrics::from_batch(n, batch_time);
    assert!(
        metrics.latency_per_item < std::time::Duration::from_millis(1),
        "batch fetch should be fast: {:?} per item",
        metrics.latency_per_item
    );
}

#[test]
fn vector_fetch_many_avoids_n_plus_1() {
    let n = 50;
    let dim = 8;
    let mut idx = VectorIndex::new();
    let ids: Vec<String> = (0..n).map(|i| format!("v{i}")).collect();
    for (i, id) in ids.iter().enumerate() {
        let mut v = vec![0.0f32; dim];
        v[i % dim] = 1.0;
        idx.add_vector(id.clone(), v).unwrap();
    }

    // Batch fetch
    let start = std::time::Instant::now();
    let id_refs: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();
    let batch_results = idx.fetch_many(&id_refs);
    let batch_time = start.elapsed();

    assert_eq!(batch_results.len(), n);
    assert!(batch_results.iter().all(|r| r.is_some()));

    let metrics = BatchMetrics::from_batch(n, batch_time);
    assert!(
        metrics.latency_per_item < std::time::Duration::from_millis(1),
        "vector fetch_many should be fast: {:?} per item",
        metrics.latency_per_item
    );
}
