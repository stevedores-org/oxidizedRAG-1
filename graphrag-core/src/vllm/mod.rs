//! vLLM / llm-d LLM integration
//!
//! This module provides integration with vLLM and llm-d inference servers
//! via their OpenAI-compatible API.

use crate::core::{GraphRAGError, Result};

/// Role of a chat message author (OpenAI-compatible).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// System prompt setting assistant behavior.
    System,
    /// User input message.
    User,
    /// Assistant (model) response.
    Assistant,
    /// Tool/function call result.
    Tool,
}

/// A single chat message with role and content.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChatMessage {
    /// The role of the message author.
    pub role: Role,
    /// The text content of the message.
    pub content: String,
}

/// vLLM configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VllmConfig {
    /// Enable vLLM integration
    pub enabled: bool,
    /// Base URL of the vLLM server (e.g. "http://localhost:8000")
    pub base_url: String,
    /// Model identifier
    pub model: String,
    /// Optional API key for authenticated deployments
    pub api_key: Option<String>,
    /// Timeout in seconds
    pub timeout_seconds: u64,
    /// Maximum tokens to generate
    pub max_tokens: Option<u32>,
    /// Temperature for generation (0.0 - 2.0)
    pub temperature: Option<f32>,
    /// Total number of attempts before giving up (1 = no retries)
    pub max_attempts: u32,
}

impl Default for VllmConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            base_url: "http://localhost:8000".to_string(),
            model: "meta-llama/Llama-3.1-8B-Instruct".to_string(),
            api_key: None,
            timeout_seconds: 30,
            max_tokens: Some(2000),
            temperature: Some(0.7),
            max_attempts: 3,
        }
    }
}

/// vLLM client for LLM inference via the OpenAI-compatible API
#[derive(Debug, Clone)]
pub struct VllmClient {
    config: VllmConfig,
    #[cfg(feature = "ureq")]
    client: ureq::Agent,
}

impl VllmClient {
    /// Create a new vLLM client
    pub fn new(config: VllmConfig) -> Self {
        Self {
            config: config.clone(),
            #[cfg(feature = "ureq")]
            client: ureq::AgentBuilder::new()
                .timeout(std::time::Duration::from_secs(config.timeout_seconds))
                .build(),
        }
    }

    /// Access the config
    pub fn config(&self) -> &VllmConfig {
        &self.config
    }

    /// Execute an HTTP POST with retry and exponential backoff.
    ///
    /// Returns the parsed JSON response on success. Retries up to
    /// `config.max_attempts` times with `100ms * attempt` backoff between
    /// failures. Uses `std::thread::sleep` — callers in async contexts should
    /// wrap this in `tokio::task::spawn_blocking`.
    #[cfg(feature = "ureq")]
    fn post_with_retry(
        &self,
        endpoint: &str,
        request_body: &serde_json::Value,
    ) -> Result<serde_json::Value> {
        let mut last_error = None;
        for attempt in 1..=self.config.max_attempts {
            let mut req = self
                .client
                .post(endpoint)
                .set("Content-Type", "application/json");

            if let Some(ref api_key) = self.config.api_key {
                req = req.set("Authorization", &format!("Bearer {api_key}"));
            }

            match req.send_json(request_body) {
                Ok(response) => {
                    return response.into_json().map_err(|e| GraphRAGError::Generation {
                        message: format!("Failed to parse vLLM response: {e}"),
                    });
                },
                Err(e) => {
                    log::warn!(
                        "vLLM request failed (attempt {attempt}/{max}): {e}",
                        max = self.config.max_attempts
                    );
                    last_error = Some(e);

                    if attempt < self.config.max_attempts {
                        std::thread::sleep(std::time::Duration::from_millis(
                            100 * u64::from(attempt),
                        ));
                    }
                },
            }
        }

        Err(GraphRAGError::Generation {
            message: format!(
                "vLLM API failed after {} attempts: {:?}",
                self.config.max_attempts, last_error
            ),
        })
    }

    /// Generate a chat completion from a single user prompt.
    #[cfg(feature = "ureq")]
    pub fn chat_completion(&self, prompt: &str) -> Result<String> {
        let messages = vec![ChatMessage {
            role: Role::User,
            content: prompt.to_string(),
        }];
        self.chat_completion_with_messages(&messages, None, None, None, None)
    }

    /// Generate a chat completion from structured multi-turn messages.
    ///
    /// Override parameters take precedence over config values when `Some`.
    #[cfg(feature = "ureq")]
    pub fn chat_completion_with_messages(
        &self,
        messages: &[ChatMessage],
        max_tokens_override: Option<u32>,
        temperature_override: Option<f32>,
        top_p_override: Option<f32>,
        stop_sequences: Option<&[String]>,
    ) -> Result<String> {
        let endpoint = format!("{}/v1/chat/completions", self.config.base_url);

        let messages_json: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| serde_json::json!({"role": m.role, "content": m.content}))
            .collect();

        let mut request_body = serde_json::json!({
            "model": self.config.model,
            "messages": messages_json,
            "stream": false,
        });

        let max_tokens = max_tokens_override.or(self.config.max_tokens);
        let temperature = temperature_override.or(self.config.temperature);

        if let Some(mt) = max_tokens {
            request_body["max_tokens"] = serde_json::json!(mt);
        }
        if let Some(t) = temperature {
            request_body["temperature"] = serde_json::json!(t);
        }
        if let Some(tp) = top_p_override {
            request_body["top_p"] = serde_json::json!(tp);
        }
        if let Some(stop) = stop_sequences {
            if !stop.is_empty() {
                request_body["stop"] = serde_json::json!(stop);
            }
        }

        let json_response = self.post_with_retry(&endpoint, &request_body)?;

        json_response["choices"]
            .as_array()
            .and_then(|choices| choices.first())
            .and_then(|choice| choice["message"]["content"].as_str())
            .map(Self::strip_think_tags)
            .ok_or_else(|| GraphRAGError::Generation {
                message: format!("Invalid vLLM response format: {json_response:?}"),
            })
    }

    /// Generate embeddings using the OpenAI-compatible endpoint.
    #[cfg(feature = "ureq")]
    pub fn embeddings(&self, inputs: &[&str]) -> Result<Vec<Vec<f32>>> {
        let endpoint = format!("{}/v1/embeddings", self.config.base_url);

        let input_value = if inputs.len() == 1 {
            serde_json::json!(inputs[0])
        } else {
            serde_json::json!(inputs)
        };

        let request_body = serde_json::json!({
            "model": self.config.model,
            "input": input_value,
        });

        let json_response = self.post_with_retry(&endpoint, &request_body)?;

        json_response["data"]
            .as_array()
            .map(|entries| {
                entries
                    .iter()
                    .filter_map(|item| {
                        item["embedding"].as_array().map(|emb| {
                            emb.iter()
                                .filter_map(|v| v.as_f64().map(|f| f as f32))
                                .collect()
                        })
                    })
                    .collect()
            })
            .ok_or_else(|| GraphRAGError::Generation {
                message: format!("Invalid embeddings response: {json_response:?}"),
            })
    }

    /// Generate chat completion (fallback when ureq feature is disabled)
    #[cfg(not(feature = "ureq"))]
    pub fn chat_completion(&self, _prompt: &str) -> Result<String> {
        Err(GraphRAGError::Generation {
            message: "ureq feature required for vLLM integration".to_string(),
        })
    }

    /// Generate chat completion with messages (fallback when ureq feature is disabled)
    #[cfg(not(feature = "ureq"))]
    pub fn chat_completion_with_messages(
        &self,
        _messages: &[ChatMessage],
        _max_tokens_override: Option<u32>,
        _temperature_override: Option<f32>,
        _top_p_override: Option<f32>,
        _stop_sequences: Option<&[String]>,
    ) -> Result<String> {
        Err(GraphRAGError::Generation {
            message: "ureq feature required for vLLM integration".to_string(),
        })
    }

    /// Generate embeddings (fallback when ureq feature is disabled)
    #[cfg(not(feature = "ureq"))]
    pub fn embeddings(&self, _inputs: &[&str]) -> Result<Vec<Vec<f32>>> {
        Err(GraphRAGError::Generation {
            message: "ureq feature required for vLLM integration".to_string(),
        })
    }

    /// Remove `<think>...</think>` tags from LLM output (Qwen3 and similar models).
    fn strip_think_tags(text: &str) -> String {
        let mut result = text.to_string();
        while let Some(start) = result.find("<think>") {
            if let Some(end) = result[start..].find("</think>") {
                let end_pos = start + end + "</think>".len();
                result.replace_range(start..end_pos, "");
            } else {
                result.replace_range(start..start + "<think>".len(), "");
                break;
            }
        }
        result.trim().to_string()
    }
}

/// Async vLLM generator implementing `AsyncLanguageModel`.
///
/// Wraps sync `VllmClient` calls in `spawn_blocking` to avoid blocking the
/// tokio runtime during retry backoff sleeps.
#[cfg(feature = "async-traits")]
pub struct AsyncVllmGenerator {
    client: VllmClient,
}

#[cfg(feature = "async-traits")]
impl AsyncVllmGenerator {
    /// Create a new async vLLM generator
    pub fn new(config: VllmConfig) -> Self {
        Self {
            client: VllmClient::new(config),
        }
    }
}

#[cfg(feature = "async-traits")]
#[async_trait::async_trait]
impl crate::core::traits::AsyncLanguageModel for AsyncVllmGenerator {
    type Error = GraphRAGError;

    async fn complete(&self, prompt: &str) -> Result<String> {
        let client = self.client.clone();
        let prompt = prompt.to_string();
        tokio::task::spawn_blocking(move || client.chat_completion(&prompt))
            .await
            .map_err(|e| GraphRAGError::Generation {
                message: format!("spawn_blocking join error: {e}"),
            })?
    }

    async fn complete_with_params(
        &self,
        prompt: &str,
        params: crate::core::traits::GenerationParams,
    ) -> Result<String> {
        let client = self.client.clone();
        let messages = vec![ChatMessage {
            role: Role::User,
            content: prompt.to_string(),
        }];
        let stop = params.stop_sequences.clone();
        tokio::task::spawn_blocking(move || {
            client.chat_completion_with_messages(
                &messages,
                params.max_tokens.map(|n| n as u32),
                params.temperature,
                params.top_p,
                stop.as_deref(),
            )
        })
        .await
        .map_err(|e| GraphRAGError::Generation {
            message: format!("spawn_blocking join error: {e}"),
        })?
    }

    async fn is_available(&self) -> bool {
        self.client.config.enabled
    }

    async fn model_info(&self) -> crate::core::traits::ModelInfo {
        crate::core::traits::ModelInfo {
            name: self.client.config.model.clone(),
            version: None,
            max_context_length: Some(8192),
            supports_streaming: false,
        }
    }
}

/// vLLM embedding provider implementing `EmbeddingProvider`.
#[cfg(feature = "async-traits")]
pub struct VllmEmbeddingProvider {
    client: VllmClient,
    dimensions: usize,
    initialized: bool,
}

#[cfg(feature = "async-traits")]
impl VllmEmbeddingProvider {
    /// Create a new vLLM embedding provider
    pub fn new(config: VllmConfig, dimensions: usize) -> Self {
        Self {
            client: VllmClient::new(config),
            dimensions,
            initialized: false,
        }
    }
}

#[cfg(feature = "async-traits")]
#[async_trait::async_trait]
impl crate::embeddings::EmbeddingProvider for VllmEmbeddingProvider {
    async fn initialize(&mut self) -> Result<()> {
        self.initialized = true;
        Ok(())
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        if !self.initialized {
            return Err(GraphRAGError::Generation {
                message: "VllmEmbeddingProvider not initialized — call initialize() first"
                    .to_string(),
            });
        }
        let client = self.client.clone();
        let text = text.to_string();
        let results = tokio::task::spawn_blocking(move || client.embeddings(&[text.as_str()]))
            .await
            .map_err(|e| GraphRAGError::Generation {
                message: format!("spawn_blocking join error: {e}"),
            })??;
        results
            .into_iter()
            .next()
            .ok_or_else(|| GraphRAGError::Generation {
                message: "No embedding returned".to_string(),
            })
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if !self.initialized {
            return Err(GraphRAGError::Generation {
                message: "VllmEmbeddingProvider not initialized — call initialize() first"
                    .to_string(),
            });
        }
        let client = self.client.clone();
        let owned: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        tokio::task::spawn_blocking(move || {
            let refs: Vec<&str> = owned.iter().map(|s| s.as_str()).collect();
            client.embeddings(&refs)
        })
        .await
        .map_err(|e| GraphRAGError::Generation {
            message: format!("spawn_blocking join error: {e}"),
        })?
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn is_available(&self) -> bool {
        self.client.config.enabled
    }

    fn provider_name(&self) -> &str {
        "vllm"
    }
}

/// Sync adapter for using `VllmClient` as an `LLMInterface` (e.g. for `AnswerGenerator`).
#[cfg(feature = "async")]
pub struct VllmLLMAdapter {
    client: VllmClient,
}

#[cfg(feature = "async")]
impl VllmLLMAdapter {
    /// Create a new sync adapter wrapping a VllmClient
    pub fn new(config: VllmConfig) -> Self {
        Self {
            client: VllmClient::new(config),
        }
    }
}

#[cfg(feature = "async")]
impl crate::generation::LLMInterface for VllmLLMAdapter {
    fn generate_response(&self, prompt: &str) -> Result<String> {
        self.client.chat_completion(prompt)
    }

    fn generate_summary(&self, content: &str, max_length: usize) -> Result<String> {
        let prompt =
            format!("Summarize the following in at most {max_length} characters:\n\n{content}");
        self.client.chat_completion(&prompt)
    }

    fn extract_key_points(&self, content: &str, num_points: usize) -> Result<Vec<String>> {
        let prompt = format!(
            "Extract {num_points} key points from:\n\n{content}\n\nReturn one point per line."
        );
        let response = self.client.chat_completion(&prompt)?;
        Ok(response
            .lines()
            .filter(|l| !l.trim().is_empty())
            .take(num_points)
            .map(String::from)
            .collect())
    }
}
