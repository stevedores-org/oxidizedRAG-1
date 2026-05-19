//! Deterministic mock providers for offline CI testing
//!
//! This module provides mock implementations of core traits that produce
//! deterministic, reproducible results without requiring external API calls.
//! Enables CI pipelines to run fully offline.
//!
//! # Providers
//!
//! - [`DeterministicEmbedder`] — Hash-based embedding that maps text to
//!   consistent vectors using SHA-256. Same input always yields the same output.
//! - [`MockLanguageModel`] — Returns canned responses keyed by prompt prefix,
//!   with a configurable fallback.

use crate::core::traits::{
    AsyncEmbedder, AsyncLanguageModel, Embedder, GenerationParams, LanguageModel, ModelInfo,
};
use crate::core::Result;
use async_trait::async_trait;
use sha2::{Digest, Sha256};
use std::collections::HashMap;

/// A deterministic embedder that produces consistent vectors from text via SHA-256.
///
/// The hash is expanded to fill the requested dimension by repeatedly hashing
/// with an incrementing counter, then each byte is mapped to `[-1.0, 1.0]`
/// and L2-normalized. This guarantees:
///
/// - **Determinism**: identical text always yields the identical vector.
/// - **Offline**: no network calls, no GPU, no model weights.
/// - **Similarity**: different texts produce different vectors (collision-resistant).
#[derive(Debug, Clone)]
pub struct DeterministicEmbedder {
    dimension: usize,
}

impl DeterministicEmbedder {
    /// Create a new deterministic embedder with the given vector dimension.
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    fn hash_to_vector(&self, text: &str) -> Vec<f32> {
        let mut vector = Vec::with_capacity(self.dimension);
        let mut counter: u32 = 0;

        while vector.len() < self.dimension {
            let mut hasher = Sha256::new();
            hasher.update(text.as_bytes());
            hasher.update(counter.to_le_bytes());
            let hash = hasher.finalize();

            for byte in hash.iter() {
                if vector.len() >= self.dimension {
                    break;
                }
                // Map byte [0,255] to [-1.0, 1.0]
                vector.push((*byte as f32 / 127.5) - 1.0);
            }
            counter += 1;
        }

        // L2-normalize
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut vector {
                *v /= norm;
            }
        }

        vector
    }
}

impl Embedder for DeterministicEmbedder {
    type Error = crate::core::GraphRAGError;

    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        Ok(self.hash_to_vector(text))
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.hash_to_vector(t)).collect())
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn is_ready(&self) -> bool {
        true
    }
}

#[async_trait]
impl AsyncEmbedder for DeterministicEmbedder {
    type Error = crate::core::GraphRAGError;

    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        Ok(self.hash_to_vector(text))
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.hash_to_vector(t)).collect())
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    async fn is_ready(&self) -> bool {
        true
    }
}

/// A mock language model that returns canned responses for testing.
///
/// Responses are matched by prompt prefix: the first registered prefix that
/// matches the beginning of the prompt wins. If no prefix matches, the
/// configurable `default_response` is returned.
#[derive(Debug, Clone)]
pub struct MockLanguageModel {
    responses: HashMap<String, String>,
    default_response: String,
    model_name: String,
}

impl MockLanguageModel {
    /// Create a new mock language model with a default response.
    pub fn new(default_response: impl Into<String>) -> Self {
        Self {
            responses: HashMap::new(),
            default_response: default_response.into(),
            model_name: "mock-llm-v1".to_string(),
        }
    }

    /// Register a canned response for prompts starting with `prefix`.
    pub fn with_response(mut self, prefix: impl Into<String>, response: impl Into<String>) -> Self {
        self.responses.insert(prefix.into(), response.into());
        self
    }

    fn lookup(&self, prompt: &str) -> String {
        for (prefix, response) in &self.responses {
            if prompt.starts_with(prefix.as_str()) {
                return response.clone();
            }
        }
        self.default_response.clone()
    }
}

impl LanguageModel for MockLanguageModel {
    type Error = crate::core::GraphRAGError;

    fn complete(&self, prompt: &str) -> Result<String> {
        Ok(self.lookup(prompt))
    }

    fn complete_with_params(&self, prompt: &str, _params: GenerationParams) -> Result<String> {
        Ok(self.lookup(prompt))
    }

    fn is_available(&self) -> bool {
        true
    }

    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            name: self.model_name.clone(),
            version: Some("1.0.0".to_string()),
            max_context_length: Some(4096),
            supports_streaming: false,
        }
    }
}

#[async_trait]
impl AsyncLanguageModel for MockLanguageModel {
    type Error = crate::core::GraphRAGError;

    async fn complete(&self, prompt: &str) -> Result<String> {
        Ok(self.lookup(prompt))
    }

    async fn complete_with_params(
        &self,
        prompt: &str,
        _params: GenerationParams,
    ) -> Result<String> {
        Ok(self.lookup(prompt))
    }

    async fn is_available(&self) -> bool {
        true
    }

    async fn model_info(&self) -> ModelInfo {
        ModelInfo {
            name: self.model_name.clone(),
            version: Some("1.0.0".to_string()),
            max_context_length: Some(4096),
            supports_streaming: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── DeterministicEmbedder ──

    #[test]
    fn embed_is_deterministic() {
        let embedder = DeterministicEmbedder::new(128);
        let v1 = Embedder::embed(&embedder, "hello world").unwrap();
        let v2 = Embedder::embed(&embedder, "hello world").unwrap();
        assert_eq!(v1, v2);
    }

    #[test]
    fn embed_dimension_matches() {
        for dim in [64, 128, 384, 768, 1536] {
            let embedder = DeterministicEmbedder::new(dim);
            let v = Embedder::embed(&embedder, "test").unwrap();
            assert_eq!(v.len(), dim);
        }
    }

    #[test]
    fn embed_different_texts_differ() {
        let embedder = DeterministicEmbedder::new(128);
        let v1 = Embedder::embed(&embedder, "hello").unwrap();
        let v2 = Embedder::embed(&embedder, "world").unwrap();
        assert_ne!(v1, v2);
    }

    #[test]
    fn embed_is_unit_normalized() {
        let embedder = DeterministicEmbedder::new(384);
        let v = Embedder::embed(&embedder, "normalize me").unwrap();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm = {norm}");
    }

    #[test]
    fn embed_batch_matches_individual() {
        let embedder = DeterministicEmbedder::new(128);
        let texts = ["alpha", "beta", "gamma"];
        let batch = Embedder::embed_batch(&embedder, &texts).unwrap();
        for (text, batch_vec) in texts.iter().zip(batch.iter()) {
            let single = Embedder::embed(&embedder, text).unwrap();
            assert_eq!(&single, batch_vec);
        }
    }

    #[test]
    fn embedder_is_ready() {
        let embedder = DeterministicEmbedder::new(64);
        assert!(Embedder::is_ready(&embedder));
    }

    // ── MockLanguageModel ──

    #[test]
    fn mock_llm_default_response() {
        let llm = MockLanguageModel::new("I don't know");
        assert_eq!(
            LanguageModel::complete(&llm, "anything").unwrap(),
            "I don't know"
        );
    }

    #[test]
    fn mock_llm_prefix_match() {
        let llm = MockLanguageModel::new("default")
            .with_response("summarize", "This is a summary.")
            .with_response("extract entities", "[\"Alice\", \"Bob\"]");

        assert_eq!(
            LanguageModel::complete(&llm, "summarize the following text").unwrap(),
            "This is a summary."
        );
        assert_eq!(
            LanguageModel::complete(&llm, "extract entities from: ...").unwrap(),
            "[\"Alice\", \"Bob\"]"
        );
        assert_eq!(
            LanguageModel::complete(&llm, "unknown prompt").unwrap(),
            "default"
        );
    }

    #[test]
    fn mock_llm_params_ignored() {
        let llm = MockLanguageModel::new("ok");
        let params = GenerationParams {
            max_tokens: Some(10),
            temperature: Some(0.0),
            ..Default::default()
        };
        assert_eq!(
            LanguageModel::complete_with_params(&llm, "test", params).unwrap(),
            "ok"
        );
    }

    #[test]
    fn mock_llm_model_info() {
        let llm = MockLanguageModel::new("x");
        let info = LanguageModel::model_info(&llm);
        assert_eq!(info.name, "mock-llm-v1");
        assert!(!info.supports_streaming);
    }

    #[test]
    fn mock_llm_is_available() {
        let llm = MockLanguageModel::new("x");
        assert!(LanguageModel::is_available(&llm));
    }

    // ── Async variants ──

    #[tokio::test]
    async fn async_embed_is_deterministic() {
        let embedder = DeterministicEmbedder::new(128);
        let v1 = AsyncEmbedder::embed(&embedder, "hello world")
            .await
            .unwrap();
        let v2 = AsyncEmbedder::embed(&embedder, "hello world")
            .await
            .unwrap();
        assert_eq!(v1, v2);
    }

    #[tokio::test]
    async fn async_mock_llm_prefix_match() {
        let llm = MockLanguageModel::new("default").with_response("summarize", "summary");
        let result = AsyncLanguageModel::complete(&llm, "summarize this")
            .await
            .unwrap();
        assert_eq!(result, "summary");
    }
}
