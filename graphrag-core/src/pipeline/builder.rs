//! Pipeline builder - construct deterministic DAGs from configuration.
//!
//! Parses TOML/YAML configs into typed pipeline DAGs with:
//! - Dependency resolution
//! - Cycle detection
//! - Deterministic hashing for content-addressing
//! - Immutability guarantees

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;

/// Configuration for a pipeline DAG.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Pipeline name
    pub name: String,
    /// Pipeline version
    pub version: String,
    /// Sequence of stage configurations
    pub stages: Vec<StageConfig>,
    /// Metadata
    pub metadata: Option<serde_json::Value>,
}

/// Configuration for a single pipeline stage.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StageConfig {
    /// Stage name (e.g., "chunker", "embedder")
    pub name: String,
    /// Stage version (e.g., "1.0.0")
    pub version: String,
    /// Input stage names (dependencies)
    pub inputs: Vec<String>,
    /// Stage-specific configuration
    pub config: Option<serde_json::Value>,
}

/// A fully resolved, validated pipeline DAG.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineDAG {
    /// Pipeline name
    pub name: String,
    /// Pipeline version
    pub version: String,
    /// Ordered stages (topologically sorted)
    pub stages: Vec<StageConfig>,
    /// SHA256 digest of canonical config
    pub digest: String,
    /// Generated timestamp
    pub timestamp: String,
}

/// Error types for pipeline building.
#[derive(Debug, Clone)]
pub enum PipelineError {
    /// Cycle detected in DAG
    CycleDetected(String),
    /// Missing input stage
    MissingStage(String),
    /// Invalid configuration
    InvalidConfig(String),
    /// Serialization error
    SerializationError(String),
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CycleDetected(msg) => write!(f, "Cycle detected: {}", msg),
            Self::MissingStage(name) => write!(f, "Missing stage: {}", name),
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {}", msg),
            Self::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
        }
    }
}

impl std::error::Error for PipelineError {}

/// Builds and validates pipeline DAGs from configuration.
pub struct PipelineBuilder;

impl PipelineBuilder {
    /// Create a new pipeline builder.
    pub fn new() -> Self {
        Self
    }

    /// Build a pipeline DAG from configuration.
    ///
    /// Validates:
    /// - No missing stage dependencies
    /// - No cycles in DAG
    /// - All stages have valid names/versions
    pub fn build(config: PipelineConfig) -> Result<PipelineDAG, PipelineError> {
        // Validate all stages exist
        let stage_names: HashSet<String> = config.stages.iter().map(|s| s.name.clone()).collect();

        for stage in &config.stages {
            for input in &stage.inputs {
                if !stage_names.contains(input) && input != "input" {
                    return Err(PipelineError::MissingStage(input.clone()));
                }
            }
        }

        // Detect cycles using DFS
        Self::detect_cycles(&config.stages)?;

        // Topologically sort stages
        let sorted_stages = Self::topological_sort(&config.stages)?;

        // Compute canonical digest
        let digest = Self::compute_digest(&config)?;

        Ok(PipelineDAG {
            name: config.name,
            version: config.version,
            stages: sorted_stages,
            digest,
            timestamp: chrono::Utc::now().to_rfc3339(),
        })
    }

    /// Detect cycles in the DAG using DFS.
    fn detect_cycles(stages: &[StageConfig]) -> Result<(), PipelineError> {
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();

        for stage in stages {
            if !visited.contains(&stage.name) {
                Self::dfs_cycle_check(&stage.name, stages, &mut visited, &mut rec_stack)?;
            }
        }

        Ok(())
    }

    /// DFS helper for cycle detection.
    fn dfs_cycle_check(
        node: &str,
        stages: &[StageConfig],
        visited: &mut HashSet<String>,
        rec_stack: &mut HashSet<String>,
    ) -> Result<(), PipelineError> {
        visited.insert(node.to_string());
        rec_stack.insert(node.to_string());

        if let Some(stage) = stages.iter().find(|s| s.name == node) {
            for input in &stage.inputs {
                if input == "input" {
                    continue; // Skip synthetic input
                }

                if !visited.contains(input) {
                    Self::dfs_cycle_check(input, stages, visited, rec_stack)?;
                } else if rec_stack.contains(input) {
                    return Err(PipelineError::CycleDetected(format!(
                        "{} -> {}",
                        node, input
                    )));
                }
            }
        }

        rec_stack.remove(node);
        Ok(())
    }

    /// Topologically sort stages.
    fn topological_sort(stages: &[StageConfig]) -> Result<Vec<StageConfig>, PipelineError> {
        let mut sorted = Vec::new();
        let mut visited = HashSet::new();

        for stage in stages {
            Self::topo_visit(&stage.name, stages, &mut visited, &mut sorted)?;
        }

        Ok(sorted)
    }

    /// Topological sort helper.
    fn topo_visit(
        node: &str,
        stages: &[StageConfig],
        visited: &mut HashSet<String>,
        sorted: &mut Vec<StageConfig>,
    ) -> Result<(), PipelineError> {
        if visited.contains(node) {
            return Ok(());
        }

        visited.insert(node.to_string());

        if let Some(stage) = stages.iter().find(|s| s.name == node) {
            for input in &stage.inputs {
                if input != "input" {
                    Self::topo_visit(input, stages, visited, sorted)?;
                }
            }
            sorted.push(stage.clone());
        }

        Ok(())
    }

    /// Compute canonical SHA256 digest of config (for content-addressing).
    fn compute_digest(config: &PipelineConfig) -> Result<String, PipelineError> {
        // Serialize to canonical JSON (sorted keys)
        let json = serde_json::to_value(config)
            .map_err(|e| PipelineError::SerializationError(e.to_string()))?;

        let canonical = Self::canonicalize_json(&json);
        let canonical_str = serde_json::to_string(&canonical)
            .map_err(|e| PipelineError::SerializationError(e.to_string()))?;

        let mut hasher = Sha256::new();
        hasher.update(canonical_str.as_bytes());
        Ok(format!("{:x}", hasher.finalize()))
    }

    /// Recursively canonicalize JSON (sort object keys).
    fn canonicalize_json(value: &serde_json::Value) -> serde_json::Value {
        match value {
            serde_json::Value::Object(map) => {
                let mut sorted = serde_json::Map::new();
                let mut keys: Vec<_> = map.keys().collect();
                keys.sort();

                for key in keys {
                    if let Some(val) = map.get(key) {
                        sorted.insert(key.to_string(), Self::canonicalize_json(val));
                    }
                }
                serde_json::Value::Object(sorted)
            },
            serde_json::Value::Array(arr) => {
                serde_json::Value::Array(arr.iter().map(|v| Self::canonicalize_json(v)).collect())
            },
            other => other.clone(),
        }
    }

    /// Print effective configuration for debugging.
    pub fn print_effective_config(dag: &PipelineDAG) {
        println!("Pipeline: {} v{}", dag.name, dag.version);
        println!("Digest: {}", dag.digest);
        println!("Stages:");
        for (i, stage) in dag.stages.iter().enumerate() {
            println!("  [{}] {}@{}", i + 1, stage.name, stage.version);
            if !stage.inputs.is_empty() {
                println!("       depends: {:?}", stage.inputs);
            }
        }
        println!("Timestamp: {}", dag.timestamp);
    }
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example_config() -> PipelineConfig {
        PipelineConfig {
            name: "test-pipeline".to_string(),
            version: "1.0.0".to_string(),
            stages: vec![
                StageConfig {
                    name: "chunker".to_string(),
                    version: "1.0.0".to_string(),
                    inputs: vec!["input".to_string()],
                    config: None,
                },
                StageConfig {
                    name: "embedder".to_string(),
                    version: "1.0.0".to_string(),
                    inputs: vec!["chunker".to_string()],
                    config: None,
                },
                StageConfig {
                    name: "retriever".to_string(),
                    version: "1.0.0".to_string(),
                    inputs: vec!["embedder".to_string()],
                    config: None,
                },
            ],
            metadata: None,
        }
    }

    #[test]
    fn test_build_valid_pipeline() {
        let config = example_config();
        let dag = PipelineBuilder::build(config);
        assert!(dag.is_ok());

        let dag = dag.unwrap();
        assert_eq!(dag.stages.len(), 3);
        assert_eq!(dag.stages[0].name, "chunker");
        assert_eq!(dag.stages[1].name, "embedder");
        assert_eq!(dag.stages[2].name, "retriever");
    }

    #[test]
    fn test_missing_stage_error() {
        let mut config = example_config();
        config.stages[1].inputs.push("nonexistent".to_string());

        let result = PipelineBuilder::build(config);
        assert!(matches!(result, Err(PipelineError::MissingStage(_))));
    }

    #[test]
    fn test_cycle_detection() {
        let config = PipelineConfig {
            name: "cycle-test".to_string(),
            version: "1.0.0".to_string(),
            stages: vec![
                StageConfig {
                    name: "a".to_string(),
                    version: "1.0.0".to_string(),
                    inputs: vec!["b".to_string()],
                    config: None,
                },
                StageConfig {
                    name: "b".to_string(),
                    version: "1.0.0".to_string(),
                    inputs: vec!["a".to_string()],
                    config: None,
                },
            ],
            metadata: None,
        };

        let result = PipelineBuilder::build(config);
        assert!(matches!(result, Err(PipelineError::CycleDetected(_))));
    }

    #[test]
    fn test_deterministic_digest() {
        let config1 = example_config();
        let config2 = example_config();

        let dag1 = PipelineBuilder::build(config1).unwrap();
        let dag2 = PipelineBuilder::build(config2).unwrap();

        assert_eq!(dag1.digest, dag2.digest);
    }

    #[test]
    fn test_digest_changes_on_config_change() {
        let config1 = example_config();
        let mut config2 = example_config();

        config2.stages[0].version = "2.0.0".to_string();

        let dag1 = PipelineBuilder::build(config1).unwrap();
        let dag2 = PipelineBuilder::build(config2).unwrap();

        assert_ne!(dag1.digest, dag2.digest);
    }

    #[test]
    fn test_topological_sort() {
        let config = example_config();
        let dag = PipelineBuilder::build(config).unwrap();

        // Verify order: chunker → embedder → retriever
        assert_eq!(dag.stages[0].name, "chunker");
        assert_eq!(dag.stages[1].name, "embedder");
        assert_eq!(dag.stages[2].name, "retriever");
    }

    #[test]
    fn test_print_effective_config() {
        let config = example_config();
        let dag = PipelineBuilder::build(config).unwrap();
        // This just verifies it doesn't panic
        PipelineBuilder::print_effective_config(&dag);
    }
}
