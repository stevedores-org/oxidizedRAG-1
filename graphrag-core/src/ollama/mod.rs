//! Ollama LLM integration
//!
//! This module provides integration with Ollama for local LLM inference.

use crate::core::traits::{AsyncLanguageModel, GenerationParams, ModelInfo};
use crate::core::{GraphRAGError, Result};
use async_trait::async_trait;

/// Ollama configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OllamaConfig {
    /// Enable Ollama integration
    pub enabled: bool,
    /// Ollama host URL
    pub host: String,
    /// Ollama port
    pub port: u16,
    /// Model for embeddings
    pub embedding_model: String,
    /// Model for chat/generation
    pub chat_model: String,
    /// Timeout in seconds
    pub timeout_seconds: u64,
    /// Maximum retry attempts
    pub max_retries: u32,
    /// Fallback to hash-based IDs on error
    pub fallback_to_hash: bool,
    /// Maximum tokens to generate
    pub max_tokens: Option<u32>,
    /// Temperature for generation (0.0 - 1.0)
    pub temperature: Option<f32>,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            host: "http://localhost".to_string(),
            port: 11434,
            embedding_model: "nomic-embed-text".to_string(),
            chat_model: "llama3.2:3b".to_string(),
            timeout_seconds: 30,
            max_retries: 3,
            fallback_to_hash: true,
            max_tokens: Some(2000),
            temperature: Some(0.7),
        }
    }
}

/// Ollama client for LLM inference
#[derive(Debug, Clone)]
pub struct OllamaClient {
    config: OllamaConfig,
    #[cfg(feature = "ureq")]
    client: ureq::Agent,
}

impl OllamaClient {
    /// Create a new Ollama client
    pub fn new(config: OllamaConfig) -> Self {
        Self {
            config: config.clone(),
            #[cfg(feature = "ureq")]
            client: ureq::AgentBuilder::new()
                .timeout(std::time::Duration::from_secs(config.timeout_seconds))
                .build(),
        }
    }

    /// Generate text completion using Ollama API
    #[cfg(feature = "ureq")]
    pub async fn generate(&self, prompt: &str) -> Result<String> {
        let endpoint = format!("{}:{}/api/generate", self.config.host, self.config.port);

        let mut request_body = serde_json::json!({
            "model": self.config.chat_model,
            "prompt": prompt,
            "stream": false,
        });

        // Add optional parameters
        if let Some(max_tokens) = self.config.max_tokens {
            request_body["options"] = serde_json::json!({
                "num_predict": max_tokens,
            });
        }

        if let Some(temperature) = self.config.temperature {
            if request_body.get("options").is_none() {
                request_body["options"] = serde_json::json!({});
            }
            request_body["options"]["temperature"] = serde_json::json!(temperature);
        }

        // Make HTTP request with retry logic
        let mut last_error = None;
        for attempt in 1..=self.config.max_retries {
            match self
                .client
                .post(&endpoint)
                .set("Content-Type", "application/json")
                .send_json(&request_body)
            {
                Ok(response) => {
                    let json_response: serde_json::Value =
                        response
                            .into_json()
                            .map_err(|e| GraphRAGError::Generation {
                                message: format!("Failed to parse JSON response: {}", e),
                            })?;

                    // Extract response text
                    if let Some(response_text) = json_response["response"].as_str() {
                        return Ok(response_text.to_string());
                    } else {
                        return Err(GraphRAGError::Generation {
                            message: format!("Invalid response format: {:?}", json_response),
                        });
                    }
                },
                Err(e) => {
                    tracing::warn!("Ollama API request failed (attempt {}): {}", attempt, e);
                    last_error = Some(e);

                    if attempt < self.config.max_retries {
                        // Wait before retry (exponential backoff)
                        tokio::time::sleep(std::time::Duration::from_millis(100 * attempt as u64))
                            .await;
                    }
                },
            }
        }

        Err(GraphRAGError::Generation {
            message: format!(
                "Ollama API failed after {} retries: {:?}",
                self.config.max_retries, last_error
            ),
        })
    }

    /// Generate text completion (sync fallback when ureq feature is disabled)
    #[cfg(not(feature = "ureq"))]
    pub async fn generate(&self, _prompt: &str) -> Result<String> {
        Err(GraphRAGError::Generation {
            message: "ureq feature required for Ollama integration".to_string(),
        })
    }
}

/// Async Ollama generator implementing AsyncLanguageModel
pub struct AsyncOllamaGenerator {
    client: OllamaClient,
}

impl AsyncOllamaGenerator {
    /// Create a new AsyncOllamaGenerator
    pub async fn new(config: OllamaConfig) -> Result<Self> {
        Ok(Self {
            client: OllamaClient::new(config),
        })
    }
}

#[async_trait]
impl AsyncLanguageModel for AsyncOllamaGenerator {
    type Error = GraphRAGError;

    async fn complete(&self, prompt: &str) -> Result<String> {
        self.client.generate(prompt).await
    }

    async fn complete_with_params(&self, prompt: &str, params: GenerationParams) -> Result<String> {
        // Clone config and update with params
        let mut config = self.client.config.clone();

        if let Some(max_tokens) = params.max_tokens {
            config.max_tokens = Some(max_tokens as u32);
        }

        if let Some(temperature) = params.temperature {
            config.temperature = Some(temperature);
        }

        // Create temporary client with updated config
        let client = OllamaClient::new(config);
        client.generate(prompt).await
    }

    async fn is_available(&self) -> bool {
        // Simple connectivity check
        // We could try to hit /api/tags or version endpoint
        #[cfg(feature = "ureq")]
        {
            let endpoint = format!(
                "{}:{}/api/version",
                self.client.config.host, self.client.config.port
            );
            match self.client.client.get(&endpoint).call() {
                Ok(_) => true,
                Err(_) => false,
            }
        }
        #[cfg(not(feature = "ureq"))]
        {
            false
        }
    }

    async fn model_info(&self) -> ModelInfo {
        ModelInfo {
            name: self.client.config.chat_model.clone(),
            version: None,
            max_context_length: Some(4096), // Assuming default Llama/Mistral context
            supports_streaming: false,
        }
    }
}
