//! Dual-mode cache supporting both in-memory and persistent backends.
//!
//! Enables seamless switching between in-memory (fast) and disk-based (persistent) caching.

use super::persistent_cache::{CacheStats, PersistentCacheBackend};
use std::collections::HashMap;
use std::sync::Mutex;

/// Dual-mode cache that can operate in-memory or persistent mode.
pub struct DualModeCache {
    mode: CacheMode,
    stats: CacheStatistics,
}

/// Cache mode selection.
#[derive(Clone)]
pub enum CacheMode {
    /// In-memory HashMap (fast, lost on restart)
    InMemory(std::sync::Arc<Mutex<HashMap<String, Vec<u8>>>>),
    /// Persistent RocksDB backend
    #[cfg(feature = "persistent-cache")]
    Persistent(std::sync::Arc<dyn PersistentCacheBackend>),
}

/// Thread-safe statistics tracking.
#[derive(Debug, Default)]
struct CacheStatistics {
    hits: std::sync::atomic::AtomicU64,
    misses: std::sync::atomic::AtomicU64,
    evictions: std::sync::atomic::AtomicU64,
}

impl DualModeCache {
    /// Create a new in-memory cache.
    pub fn new_memory() -> Self {
        Self {
            mode: CacheMode::InMemory(std::sync::Arc::new(Mutex::new(HashMap::new()))),
            stats: CacheStatistics::default(),
        }
    }

    /// Create a new persistent cache backed by RocksDB.
    #[cfg(feature = "persistent-cache")]
    pub fn new_persistent<P: AsRef<std::path::Path>>(path: P) -> Result<Self, String> {
        use crate::pipeline::RocksDBCache;
        let db = RocksDBCache::new(path)?;
        Ok(Self {
            mode: CacheMode::Persistent(std::sync::Arc::new(db)),
            stats: CacheStatistics::default(),
        })
    }

    /// Get a value from the cache.
    pub fn get(&self, key: &str) -> Result<Option<Vec<u8>>, String> {
        match &self.mode {
            CacheMode::InMemory(map) => {
                let map = map.lock().unwrap();
                if let Some(value) = map.get(key) {
                    self.stats
                        .hits
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    Ok(Some(value.clone()))
                } else {
                    self.stats
                        .misses
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    Ok(None)
                }
            },
            #[cfg(feature = "persistent-cache")]
            CacheMode::Persistent(backend) => match backend.get(key) {
                Ok(Some(value)) => {
                    self.stats
                        .hits
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    Ok(Some(value))
                },
                Ok(None) => {
                    self.stats
                        .misses
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    Ok(None)
                },
                Err(e) => Err(e),
            },
        }
    }

    /// Store a value in the cache.
    pub fn set(&self, key: String, value: Vec<u8>) -> Result<(), String> {
        match &self.mode {
            CacheMode::InMemory(map) => {
                let mut map = map.lock().unwrap();
                map.insert(key, value);
                Ok(())
            },
            #[cfg(feature = "persistent-cache")]
            CacheMode::Persistent(backend) => backend.set(key, value),
        }
    }

    /// Delete a value from the cache.
    pub fn delete(&self, key: &str) -> Result<(), String> {
        match &self.mode {
            CacheMode::InMemory(map) => {
                let mut map = map.lock().unwrap();
                map.remove(key);
                Ok(())
            },
            #[cfg(feature = "persistent-cache")]
            CacheMode::Persistent(backend) => backend.delete(key),
        }
    }

    /// Check if a key exists in the cache.
    pub fn contains(&self, key: &str) -> Result<bool, String> {
        match &self.mode {
            CacheMode::InMemory(map) => {
                let map = map.lock().unwrap();
                Ok(map.contains_key(key))
            },
            #[cfg(feature = "persistent-cache")]
            CacheMode::Persistent(backend) => backend.contains(key),
        }
    }

    /// Get the number of entries in the cache.
    pub fn len(&self) -> Result<usize, String> {
        match &self.mode {
            CacheMode::InMemory(map) => {
                let map = map.lock().unwrap();
                Ok(map.len())
            },
            #[cfg(feature = "persistent-cache")]
            CacheMode::Persistent(backend) => {
                // RocksDB doesn't efficiently support len(), so estimate or return error
                backend.len()
            },
        }
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> Result<bool, String> {
        self.len().map(|len| len == 0)
    }

    /// Get cache statistics.
    pub fn stats(&self) -> Result<CacheStats, String> {
        let base_stats = match &self.mode {
            CacheMode::InMemory(_) => CacheStats {
                total_entries: self.len().unwrap_or(0),
                size_bytes: 0,
                hits: self.stats.hits.load(std::sync::atomic::Ordering::Relaxed),
                misses: self.stats.misses.load(std::sync::atomic::Ordering::Relaxed),
                evictions: self
                    .stats
                    .evictions
                    .load(std::sync::atomic::Ordering::Relaxed),
            },
            #[cfg(feature = "persistent-cache")]
            CacheMode::Persistent(backend) => {
                let mut stats = backend.stats()?;
                // Merge our tracked stats
                stats.hits += self.stats.hits.load(std::sync::atomic::Ordering::Relaxed);
                stats.misses += self.stats.misses.load(std::sync::atomic::Ordering::Relaxed);
                stats.evictions += self
                    .stats
                    .evictions
                    .load(std::sync::atomic::Ordering::Relaxed);
                stats
            },
        };
        Ok(base_stats)
    }

    /// Clear all entries (only works for in-memory).
    pub fn clear(&self) -> Result<(), String> {
        match &self.mode {
            CacheMode::InMemory(map) => {
                let mut map = map.lock().unwrap();
                map.clear();
                Ok(())
            },
            #[cfg(feature = "persistent-cache")]
            CacheMode::Persistent(backend) => backend.clear(),
        }
    }

    /// Return mode description for logging.
    pub fn mode_description(&self) -> &'static str {
        match &self.mode {
            CacheMode::InMemory(_) => "in-memory",
            #[cfg(feature = "persistent-cache")]
            CacheMode::Persistent(_) => "persistent (RocksDB)",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dual_mode_new_memory() {
        let cache = DualModeCache::new_memory();
        assert_eq!(cache.mode_description(), "in-memory");
        assert!(cache.is_empty().unwrap());
    }

    #[test]
    fn test_dual_mode_memory_get_set() {
        let cache = DualModeCache::new_memory();

        let key = "test".to_string();
        let value = vec![1, 2, 3];

        cache.set(key.clone(), value.clone()).unwrap();
        let retrieved = cache.get(&key).unwrap();
        assert_eq!(retrieved, Some(value));
    }

    #[test]
    fn test_dual_mode_memory_delete() {
        let cache = DualModeCache::new_memory();

        let key = "test".to_string();
        cache.set(key.clone(), vec![1, 2, 3]).unwrap();
        assert!(cache.contains(&key).unwrap());

        cache.delete(&key).unwrap();
        assert!(!cache.contains(&key).unwrap());
    }

    #[test]
    fn test_dual_mode_memory_stats() {
        let cache = DualModeCache::new_memory();

        // Miss
        let _ = cache.get("nonexistent");
        // Hit
        cache.set("key".to_string(), vec![1, 2, 3]).unwrap();
        let _ = cache.get("key");

        let stats = cache.stats().unwrap();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_dual_mode_memory_clear() {
        let cache = DualModeCache::new_memory();

        cache.set("key1".to_string(), vec![1]).unwrap();
        cache.set("key2".to_string(), vec![2]).unwrap();
        assert_eq!(cache.len().unwrap(), 2);

        cache.clear().unwrap();
        assert!(cache.is_empty().unwrap());
    }

    #[cfg(feature = "persistent-cache")]
    #[test]
    fn test_dual_mode_persistent() {
        use tempfile::TempDir;

        let dir = TempDir::new().unwrap();
        let cache = DualModeCache::new_persistent(dir.path()).unwrap();
        assert_eq!(cache.mode_description(), "persistent (RocksDB)");

        let key = "test".to_string();
        let value = vec![1, 2, 3];

        cache.set(key.clone(), value.clone()).unwrap();
        let retrieved = cache.get(&key).unwrap();
        assert_eq!(retrieved, Some(value));
    }

    #[cfg(feature = "persistent-cache")]
    #[test]
    fn test_dual_mode_persistent_stats() {
        use tempfile::TempDir;

        let dir = TempDir::new().unwrap();
        let cache = DualModeCache::new_persistent(dir.path()).unwrap();

        let _ = cache.get("nonexistent");
        cache.set("key".to_string(), vec![1, 2, 3]).unwrap();
        let _ = cache.get("key");

        let stats = cache.stats().unwrap();
        assert!(stats.hits > 0);
        assert!(stats.misses > 0);
    }
}
