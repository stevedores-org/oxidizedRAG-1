//! Stage-level caching/memoization for pipeline stages.
//!
//! `CachedStage<I,O>` wraps any `Stage<I,O>` with content-hash caching.
//! Supports two backends:
//! - **moka** (feature `caching`): in-memory TTL-based cache
//! - **DualModeCache** (feature `persistent-cache`): in-memory or RocksDB-backed persistent cache
//!
//! When both features are enabled, moka acts as a fast L1 cache and DualModeCache
//! as a persistent L2 fallback.

use crate::pipeline::stage::{Stage, StageError};
use async_trait::async_trait;
use serde::{de::DeserializeOwned, Serialize};
use sha2::{Digest, Sha256};
use std::hash::Hash;
use std::sync::Arc;
use std::time::Duration;

#[cfg(feature = "caching")]
use moka::future::Cache;

use super::dual_mode_cache::DualModeCache;

/// A cached wrapper around any `Stage<I,O>`.
///
/// Caches stage outputs based on a content hash of the input.
/// Requires `I: Serialize + Hash` and `O: Serialize + DeserializeOwned + Clone`.
pub struct CachedStage<I, O>
where
    I: Send + 'static,
    O: Send + 'static,
{
    inner: Arc<dyn Stage<I, O>>,
    #[cfg(feature = "caching")]
    l1_cache: Cache<String, Vec<u8>>,
    /// Optional persistent/dual-mode L2 cache.
    l2_cache: Option<Arc<DualModeCache>>,
    ttl: Duration,
    #[cfg(not(feature = "caching"))]
    _phantom: std::marker::PhantomData<(I, O)>,
}

impl<I, O> CachedStage<I, O>
where
    I: Serialize + Hash + Send + 'static,
    O: Serialize + DeserializeOwned + Clone + Send + 'static,
{
    /// Create a new cached stage wrapping the given inner stage (moka-only).
    ///
    /// `max_capacity` controls maximum number of cached entries.
    /// `ttl` is the time-to-live for each cache entry.
    pub fn new(inner: Arc<dyn Stage<I, O>>, max_capacity: u64, ttl: Duration) -> Self {
        Self {
            inner,
            #[cfg(feature = "caching")]
            l1_cache: Cache::builder()
                .max_capacity(max_capacity)
                .time_to_live(ttl)
                .build(),
            l2_cache: None,
            ttl,
            #[cfg(not(feature = "caching"))]
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create a cached stage with a DualModeCache as persistent L2.
    ///
    /// When moka (L1) misses, the L2 cache is checked before running the inner stage.
    /// Results are written to both L1 and L2.
    pub fn with_dual_mode_cache(
        inner: Arc<dyn Stage<I, O>>,
        max_capacity: u64,
        ttl: Duration,
        l2: Arc<DualModeCache>,
    ) -> Self {
        Self {
            inner,
            #[cfg(feature = "caching")]
            l1_cache: Cache::builder()
                .max_capacity(max_capacity)
                .time_to_live(ttl)
                .build(),
            l2_cache: Some(l2),
            ttl,
            #[cfg(not(feature = "caching"))]
            _phantom: std::marker::PhantomData,
        }
    }

    /// Compute a cache key from the input using SHA-256.
    fn cache_key(input: &I) -> String
    where
        I: Serialize,
    {
        let serialized = serde_json::to_vec(input).unwrap_or_default();
        let hash = Sha256::digest(&serialized);
        hex::encode(hash)
    }

    /// Get the configured TTL.
    pub fn ttl(&self) -> Duration {
        self.ttl
    }

    /// Check if an L2 (DualModeCache) backend is configured.
    pub fn has_l2_cache(&self) -> bool {
        self.l2_cache.is_some()
    }
}

// hex encoding helper (avoiding a dependency)
mod hex {
    pub fn encode(bytes: impl AsRef<[u8]>) -> String {
        bytes
            .as_ref()
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect()
    }
}

#[async_trait]
impl<I, O> Stage<I, O> for CachedStage<I, O>
where
    I: Serialize + Hash + Send + Sync + 'static,
    O: Serialize + DeserializeOwned + Clone + Send + Sync + 'static,
{
    async fn execute(&self, input: I) -> Result<O, StageError> {
        let key = Self::cache_key(&input);

        // L1: check moka
        #[cfg(feature = "caching")]
        {
            if let Some(cached_bytes) = self.l1_cache.get(&key).await {
                if let Ok(output) = serde_json::from_slice::<O>(&cached_bytes) {
                    return Ok(output);
                }
            }
        }

        // L2: check DualModeCache
        if let Some(ref l2) = self.l2_cache {
            if let Ok(Some(bytes)) = l2.get(&key) {
                if let Ok(output) = serde_json::from_slice::<O>(&bytes) {
                    // Promote back to L1
                    #[cfg(feature = "caching")]
                    {
                        self.l1_cache.insert(key, bytes).await;
                    }
                    return Ok(output);
                }
            }
        }

        // Cache miss — run inner stage
        let output = self.inner.execute(input).await?;

        // Store in L1
        #[cfg(feature = "caching")]
        {
            if let Ok(bytes) = serde_json::to_vec(&output) {
                self.l1_cache.insert(key.clone(), bytes).await;
            }
        }

        // Store in L2
        if let Some(ref l2) = self.l2_cache {
            if let Ok(bytes) = serde_json::to_vec(&output) {
                let _ = l2.set(key, bytes);
            }
        }

        Ok(output)
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn version(&self) -> &str {
        self.inner.version()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::stage::StageError;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// A counting stage that tracks how many times execute() is called.
    struct CountingStage {
        call_count: AtomicU64,
    }

    impl CountingStage {
        fn new() -> Self {
            Self {
                call_count: AtomicU64::new(0),
            }
        }

        fn calls(&self) -> u64 {
            self.call_count.load(Ordering::Relaxed)
        }
    }

    #[async_trait]
    impl Stage<String, String> for CountingStage {
        async fn execute(&self, input: String) -> Result<String, StageError> {
            self.call_count.fetch_add(1, Ordering::Relaxed);
            Ok(format!("processed:{}", input))
        }
        fn name(&self) -> &str {
            "counting"
        }
        fn version(&self) -> &str {
            "1.0.0"
        }
    }

    #[tokio::test]
    async fn test_cache_miss() {
        let inner = Arc::new(CountingStage::new());
        let cached = CachedStage::new(inner.clone(), 100, Duration::from_secs(60));

        let result = cached.execute("hello".to_string()).await.unwrap();
        assert_eq!(result, "processed:hello");
        assert_eq!(inner.calls(), 1);
    }

    #[cfg(feature = "caching")]
    #[tokio::test]
    async fn test_cache_hit() {
        let inner = Arc::new(CountingStage::new());
        let cached = CachedStage::new(inner.clone(), 100, Duration::from_secs(60));

        // First call — miss
        let r1 = cached.execute("hello".to_string()).await.unwrap();
        assert_eq!(r1, "processed:hello");
        assert_eq!(inner.calls(), 1);

        // Second call — hit
        let r2 = cached.execute("hello".to_string()).await.unwrap();
        assert_eq!(r2, "processed:hello");
        assert_eq!(inner.calls(), 1); // inner not called again
    }

    #[cfg(feature = "caching")]
    #[tokio::test]
    async fn test_cache_ttl_expiry() {
        let inner = Arc::new(CountingStage::new());
        let cached = CachedStage::new(
            inner.clone(),
            100,
            Duration::from_millis(50), // Very short TTL
        );

        cached.execute("hello".to_string()).await.unwrap();
        assert_eq!(inner.calls(), 1);

        // Wait for TTL to expire
        tokio::time::sleep(Duration::from_millis(100)).await;

        // moka eviction may require a get after expiry
        cached.execute("hello".to_string()).await.unwrap();
        // After TTL, inner should be called again
        assert!(inner.calls() >= 2);
    }

    #[tokio::test]
    async fn test_l2_cache_fallback() {
        let inner = Arc::new(CountingStage::new());
        let l2 = Arc::new(DualModeCache::new_memory());

        let cached = CachedStage::with_dual_mode_cache(
            inner.clone(),
            100,
            Duration::from_secs(60),
            l2.clone(),
        );

        assert!(cached.has_l2_cache());

        // First call — miss both L1 and L2
        let r1 = cached.execute("hello".to_string()).await.unwrap();
        assert_eq!(r1, "processed:hello");
        assert_eq!(inner.calls(), 1);

        // Verify L2 was populated
        let key = CachedStage::<String, String>::cache_key(&"hello".to_string());
        assert!(l2.contains(&key).unwrap());
    }

    #[tokio::test]
    async fn test_l2_cache_hit_on_l1_miss() {
        let inner = Arc::new(CountingStage::new());
        let l2 = Arc::new(DualModeCache::new_memory());

        // Pre-populate L2 with a cached value
        let key = CachedStage::<String, String>::cache_key(&"hello".to_string());
        let value = serde_json::to_vec(&"processed:hello".to_string()).unwrap();
        l2.set(key, value).unwrap();

        // Create CachedStage without L1 cache populated
        let cached =
            CachedStage::with_dual_mode_cache(inner.clone(), 100, Duration::from_secs(60), l2);

        // Should hit L2 without calling inner
        let result = cached.execute("hello".to_string()).await.unwrap();
        assert_eq!(result, "processed:hello");
        assert_eq!(inner.calls(), 0); // inner never called
    }

    #[tokio::test]
    async fn test_version_delegation() {
        let inner = Arc::new(CountingStage::new());
        let cached = CachedStage::new(inner.clone(), 100, Duration::from_secs(60));

        assert_eq!(cached.name(), "counting");
        assert_eq!(cached.version(), "1.0.0");
    }
}
