//! Advanced NLP Module
//!
//! This module provides advanced natural language processing capabilities:
//! - Multilingual support with automatic language detection
//! - Semantic chunking algorithms
//! - Custom NER training pipeline
//!
//! ## Features
//!
//! ### Multilingual Support
//! - Automatic language detection using n-gram analysis
//! - Support for 10+ languages (English, Spanish, French, German, Chinese, Japanese, Korean, Arabic, Russian, Portuguese)
//! - Language-specific text normalization and tokenization
//!
//! ### Semantic Chunking
//! - Multiple chunking strategies (sentence, paragraph, topic, semantic, hybrid)
//! - Intelligent boundary detection
//! - Coherence scoring
//! - Configurable chunk sizes and overlap
//!
//! ### Custom NER
//! - Pattern-based entity extraction
//! - Dictionary/gazetteer matching
//! - Rule-based extraction with priorities
//! - Training dataset management
//! - Active learning support

pub mod custom_ner;
pub mod multilingual;
pub mod semantic_chunking;
pub mod syntax_analyzer;

// Re-export main types
pub use multilingual::{
    DetectionResult, Language, LanguageDetector, MultilingualProcessor, ProcessedText,
};

pub use semantic_chunking::{
    ChunkingConfig, ChunkingStats, ChunkingStrategy, SemanticChunk, SemanticChunker,
};

pub use custom_ner::{
    AnnotatedExample, CustomNER, DatasetStatistics, EntityType, ExtractedEntity, ExtractionRule,
    RuleType, TrainingDataset,
};

pub use syntax_analyzer::{
    Dependency, DependencyRelation, NounPhrase, POSTag, SyntaxAnalyzer, SyntaxAnalyzerConfig, Token,
};
