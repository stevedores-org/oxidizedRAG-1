//! Mock embedder for CI determinism
//!
//! Provides a deterministic, hash-based embedder implementing `AsyncEmbedder`.
//! Uses the same hash-based pattern as `EmbeddingGenerator` in `vector/mod.rs`
//! and tracks stats via `AtomicU64` (same pattern as `AsyncMockLLM`).

use crate::core::traits::AsyncEmbedder;
use crate::core::{GraphRAGError, Result};
use async_trait::async_trait;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};

/// Statistics tracking for the mock embedder
#[derive(Debug, Default)]
pub struct MockEmbedderStats {
    /// Total number of embed calls
    pub total_requests: AtomicU64,
    /// Total number of individual texts embedded
    pub total_texts_embedded: AtomicU64,
}

/// Deterministic mock embedder implementing `AsyncEmbedder`.
///
/// Uses hash-based vector generation for reproducible outputs in tests and CI.
#[derive(Debug)]
pub struct MockEmbedder {
    dimension: usize,
    stats: MockEmbedderStats,
}

impl MockEmbedder {
    /// Create a new mock embedder with the given dimension.
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            stats: MockEmbedderStats::default(),
        }
    }

    /// Get the total number of embed requests made.
    pub fn total_requests(&self) -> u64 {
        self.stats.total_requests.load(Ordering::Relaxed)
    }

    /// Get the total number of individual texts embedded.
    pub fn total_texts_embedded(&self) -> u64 {
        self.stats.total_texts_embedded.load(Ordering::Relaxed)
    }

    /// Generate a deterministic embedding for the given text.
    fn hash_embed(&self, text: &str) -> Vec<f32> {
        let mut embedding = Vec::with_capacity(self.dimension);
        for i in 0..self.dimension {
            let mut hasher = DefaultHasher::new();
            text.hash(&mut hasher);
            i.hash(&mut hasher);
            let hash = hasher.finish();
            // Map hash to [-1.0, 1.0]
            embedding.push((hash as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32);
        }
        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut embedding {
                *v /= norm;
            }
        }
        embedding
    }
}

#[async_trait]
impl AsyncEmbedder for MockEmbedder {
    type Error = GraphRAGError;

    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.stats.total_requests.fetch_add(1, Ordering::Relaxed);
        self.stats
            .total_texts_embedded
            .fetch_add(1, Ordering::Relaxed);
        Ok(self.hash_embed(text))
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.stats.total_requests.fetch_add(1, Ordering::Relaxed);
        self.stats
            .total_texts_embedded
            .fetch_add(texts.len() as u64, Ordering::Relaxed);
        Ok(texts.iter().map(|t| self.hash_embed(t)).collect())
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    async fn is_ready(&self) -> bool {
        true
    }

    async fn health_check(&self) -> Result<bool> {
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_embed_single() {
        let embedder = MockEmbedder::new(128);
        let result = embedder.embed("hello world").await.unwrap();
        assert_eq!(result.len(), 128);
    }

    #[tokio::test]
    async fn test_embed_batch() {
        let embedder = MockEmbedder::new(64);
        let result = embedder.embed_batch(&["foo", "bar", "baz"]).await.unwrap();
        assert_eq!(result.len(), 3);
        assert!(result.iter().all(|v| v.len() == 64));
    }

    #[tokio::test]
    async fn test_deterministic_output() {
        let embedder = MockEmbedder::new(32);
        let a = embedder.embed("test").await.unwrap();
        let b = embedder.embed("test").await.unwrap();
        assert_eq!(a, b, "Same input must produce same output");
    }

    #[tokio::test]
    async fn test_dimension() {
        let embedder = MockEmbedder::new(256);
        assert_eq!(embedder.dimension(), 256);
        let v = embedder.embed("x").await.unwrap();
        assert_eq!(v.len(), 256);
    }

    #[tokio::test]
    async fn test_health_check() {
        let embedder = MockEmbedder::new(64);
        assert!(embedder.health_check().await.unwrap());
        assert!(embedder.is_ready().await);
    }

    #[tokio::test]
    async fn test_stats_tracking() {
        let embedder = MockEmbedder::new(64);
        embedder.embed("a").await.unwrap();
        embedder.embed_batch(&["b", "c"]).await.unwrap();
        assert_eq!(embedder.total_requests(), 2);
        assert_eq!(embedder.total_texts_embedded(), 3);
    }
}
