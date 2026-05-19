//! Shared test fixtures and helpers for RAG agent tests

pub use graphrag_core::core::{Document, DocumentId, KnowledgeGraph};
use graphrag_core::graph::GraphBuilder;
pub use graphrag_core::text::TextProcessor;
pub use graphrag_core::Result;
use std::fs;

pub const FIXTURE_DIR: &str = "tests/fixtures/code_samples";

/// Load a fixture file by name
pub fn load_fixture(name: &str) -> String {
    let path = format!("{}/{}", FIXTURE_DIR, name);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("Failed to load fixture '{}': {}", path, e))
}

/// Create a Document from a fixture file
pub fn fixture_document(filename: &str) -> Document {
    let content = load_fixture(filename);
    let doc_id = DocumentId::new(filename.replace('.', "_"));
    Document::new(doc_id, filename.to_string(), content)
}

/// Index fixture files into a GraphRAG knowledge graph
pub fn index_fixtures(filenames: &[&str], chunk_size: usize) -> Result<KnowledgeGraph> {
    let mut graph = KnowledgeGraph::new();
    let processor = TextProcessor::new(chunk_size, chunk_size / 5)?;

    for filename in filenames {
        let doc = fixture_document(filename);
        let chunks = processor.chunk_text(&doc)?;
        let doc_with_chunks = Document { chunks, ..doc };
        graph.add_document(doc_with_chunks)?;
    }

    Ok(graph)
}

/// Build a knowledge graph from fixture files with entity extraction
pub fn build_graph_from_fixtures(filenames: &[&str]) -> Result<KnowledgeGraph> {
    let documents: Vec<Document> = filenames.iter().map(|f| fixture_document(f)).collect();
    let mut builder = GraphBuilder::new(500, 100, 0.5, 0.7, 10)?;
    builder.build_graph(documents)
}

/// Parse Rust code with tree-sitter and validate syntax
#[cfg(feature = "code-chunking")]
pub fn validate_rust_syntax(code: &str) -> std::result::Result<(), String> {
    use tree_sitter::Parser;

    let mut parser = Parser::new();
    let language = tree_sitter_rust::language();
    parser
        .set_language(&language)
        .map_err(|e| format!("Failed to load Rust grammar: {}", e))?;

    let tree = parser
        .parse(code, None)
        .ok_or_else(|| "Failed to parse code".to_string())?;

    let root = tree.root_node();
    if root.has_error() {
        // Walk the tree to find the first ERROR/MISSING node for a useful location
        let mut cursor = root.walk();
        let mut error_pos = None;
        loop {
            let node = cursor.node();
            if node.is_error() || node.is_missing() {
                let pos = node.start_position();
                error_pos = Some((node.start_byte(), pos.row + 1, pos.column + 1));
                break;
            }
            if !cursor.goto_first_child() {
                while !cursor.goto_next_sibling() {
                    if !cursor.goto_parent() {
                        break;
                    }
                }
                if cursor.node() == root {
                    break;
                }
            }
        }
        match error_pos {
            Some((byte, line, col)) => Err(format!(
                "Syntax error in generated code at byte {}, line {}:{}",
                byte, line, col
            )),
            None => Err("Syntax error in generated code (location unknown)".to_string()),
        }
    } else {
        Ok(())
    }
}

/// Represents a single turn in a multi-turn conversation
#[derive(Clone, Debug)]
pub struct ConversationTurn {
    pub user_query: String,
    pub retrieved_context: Vec<String>,
    pub generated_response: String,
    pub turn_number: usize,
}

/// Tracks multi-turn conversation context with knowledge graph integration
///
/// Preserves conversation history and validates response quality progression
/// across multiple turns using the knowledge graph for context retrieval.
#[derive(Debug)]
pub struct ConversationContext {
    turns: Vec<ConversationTurn>,
    knowledge_graph: KnowledgeGraph,
}

impl ConversationContext {
    /// Create a new conversation with initial knowledge graph
    pub fn new(graph: KnowledgeGraph) -> Self {
        ConversationContext {
            turns: Vec::new(),
            knowledge_graph: graph,
        }
    }

    /// Add a turn to the conversation, updating context history
    pub fn add_turn(&mut self, query: String, retrieved: Vec<String>, response: String) {
        self.turns.push(ConversationTurn {
            user_query: query,
            retrieved_context: retrieved,
            generated_response: response,
            turn_number: self.turns.len() + 1,
        });
    }

    /// Get the full conversation history formatted for review
    pub fn context_history(&self) -> String {
        self.turns
            .iter()
            .map(|t| {
                let context_summary = t
                    .retrieved_context
                    .iter()
                    .take(1)
                    .map(|c| format!("'{}'", c))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!(
                    "Turn {}: Query='{}', Context=[{}], Response='{}'",
                    t.turn_number, t.user_query, context_summary, t.generated_response
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Get turn count to validate conversation progression
    pub fn turn_count(&self) -> usize {
        self.turns.len()
    }

    /// Get reference to knowledge graph for retrieval during conversations
    pub fn knowledge_graph(&self) -> &KnowledgeGraph {
        &self.knowledge_graph
    }

    /// Get the last turn added to the conversation
    pub fn last_turn(&self) -> Option<&ConversationTurn> {
        self.turns.last()
    }

    /// Iterate over all turns in conversation
    pub fn turns(&self) -> impl Iterator<Item = &ConversationTurn> {
        self.turns.iter()
    }
}
