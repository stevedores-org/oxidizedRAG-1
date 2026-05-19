//! RAG Configuration Hasher - creates deterministic digests for RAG configs
//!
//! Uses canonical JSON to ensure identical RAG configurations produce identical digests,
//! enabling reproducible experiment tracking and version control.

use serde_json::Value;
use sha2::{Digest, Sha256};

/// Represents a hashed RAG configuration for versioning
#[derive(Debug, Clone)]
pub struct RagConfigDigest {
    /// SHA256 hex digest of canonical config JSON
    pub digest: String,
    /// Original config (for reference)
    pub config: Value,
}

impl RagConfigDigest {
    /// Create a digest from a RAG configuration
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = serde_json::json!({
    ///     "retrieval_strategy": "hybrid",
    ///     "chunk_size": 512,
    ///     "model": "gpt-4"
    /// });
    /// let digest = RagConfigDigest::from_config(config);
    /// // digest.digest will be stable across multiple calls with same config
    /// ```
    pub fn from_config(config: Value) -> Self {
        // Sort keys recursively to ensure canonical form
        let canonical = Self::canonicalize(&config);
        let json_str = serde_json::to_string(&canonical).expect("config must be JSON-serializable");

        // Compute SHA256
        let mut hasher = Sha256::new();
        hasher.update(json_str.as_bytes());
        let digest = format!("{:x}", hasher.finalize());

        Self { digest, config }
    }

    /// Get the digest as a hex string
    pub fn as_hex(&self) -> &str {
        &self.digest
    }

    /// Recursively canonicalize JSON by sorting object keys
    fn canonicalize(value: &Value) -> Value {
        match value {
            Value::Object(map) => {
                let mut sorted_map = serde_json::Map::new();
                let mut keys: Vec<_> = map.keys().collect();
                keys.sort();

                for key in keys {
                    if let Some(val) = map.get(key) {
                        sorted_map.insert(key.to_string(), Self::canonicalize(val));
                    }
                }
                Value::Object(sorted_map)
            },
            Value::Array(arr) => Value::Array(arr.iter().map(|v| Self::canonicalize(v)).collect()),
            other => other.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_digest_stable() {
        let config1 = serde_json::json!({
            "model": "gpt-4",
            "retrieval": "hybrid",
        });

        let config2 = serde_json::json!({
            "retrieval": "hybrid",
            "model": "gpt-4",
        });

        let digest1 = RagConfigDigest::from_config(config1);
        let digest2 = RagConfigDigest::from_config(config2);

        assert_eq!(
            digest1.digest, digest2.digest,
            "same config with different key order should produce same digest"
        );
    }

    #[test]
    fn test_config_digest_changes_on_difference() {
        let config1 = serde_json::json!({
            "model": "gpt-4",
            "retrieval": "hybrid",
        });

        let config2 = serde_json::json!({
            "model": "gpt-3.5",
            "retrieval": "hybrid",
        });

        let digest1 = RagConfigDigest::from_config(config1);
        let digest2 = RagConfigDigest::from_config(config2);

        assert_ne!(
            digest1.digest, digest2.digest,
            "different configs should produce different digests"
        );
    }

    #[test]
    fn test_nested_config_canonical() {
        let config = serde_json::json!({
            "retrieval": {
                "strategy": "hybrid",
                "vector_weight": 0.7,
                "bm25_weight": 0.3,
            },
            "model": "gpt-4",
        });

        let digest = RagConfigDigest::from_config(config);
        assert!(!digest.digest.is_empty());
        assert_eq!(digest.digest.len(), 64, "SHA256 hex should be 64 chars");
    }
}
