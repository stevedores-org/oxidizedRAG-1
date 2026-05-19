//! GraphRAG AIVCS Integration Demo
//!
//! This example demonstrates:
//! 1. Tracking RAG query execution with RagRunRecorder
//! 2. Recording RAG config versions with RagConfigDigest
//! 3. Converting RAG runs to AIVCS format for persistent storage

use graphrag_aivcs::{RagConfigDigest, RagRunRecorder, RagToAivcsAdapter};
use serde_json::json;

fn main() {
    println!("=== GraphRAG AIVCS Integration Demo ===\n");

    // Step 1: Create and record a RAG execution
    println!("Step 1: Recording RAG query execution...");
    let mut recorder = RagRunRecorder::new("What are the key features of Rust?");

    // Simulate retrieval operations
    recorder.record_retrieval("Rust chapter 1: ownership", 3, 0.92, 150);
    recorder.record_retrieval("Rust chapter 2: borrowing", 5, 0.88, 200);

    // Simulate LLM interactions
    recorder.record_llm_call(
        "Summarize these retrieval results...",
        "Rust features include ownership system, type safety, and memory safety...",
        425,
        800,
    );

    let run_summary = recorder.summary();
    println!("  Run ID: {}", run_summary.run_id);
    println!("  Query: {}", run_summary.query);
    println!("  Total events: {}", run_summary.event_count);
    println!("  Duration: {}ms", run_summary.total_duration_ms);
    println!(
        "  Retrievals: {}, LLM calls: {}\n",
        run_summary.retrieval_count, run_summary.llm_calls
    );

    // Step 2: Version the RAG configuration
    println!("Step 2: Creating versioned config digest...");
    let rag_config = json!({
        "retrieval_strategy": "hybrid",
        "chunk_size": 512,
        "model": "gpt-4",
        "vector_db": "qdrant",
        "temperature": 0.7,
    });

    let config_digest = RagConfigDigest::from_config(rag_config.clone());
    println!("  Config digest: {}", config_digest.as_hex());
    println!("  (stable across multiple runs with same config)\n");

    // Step 3: Convert RAG run to AIVCS format
    println!("Step 3: Converting RAG run to AIVCS format...");
    let aivcs_events = RagToAivcsAdapter::convert_run(&recorder, "graphrag-agent");
    let aivcs_summary = RagToAivcsAdapter::summarize_run(&recorder);

    println!(
        "  Converted {} RAG events to AIVCS tool calls",
        aivcs_events.len()
    );
    for (i, event) in aivcs_events.iter().enumerate() {
        println!(
            "    [{}] {} - seq={}",
            i + 1,
            event["tool_name"],
            event["seq"]
        );
    }
    println!();

    // Step 4: Show AIVCS summary
    println!("Step 4: AIVCS Run Summary");
    println!(
        "  {}",
        serde_json::to_string_pretty(&aivcs_summary).unwrap()
    );
    println!();

    // Step 5: Demonstrate reproducibility
    println!("Step 5: Demonstrating reproducibility...");
    let config_digest_2 = RagConfigDigest::from_config(json!({
        "temperature": 0.7,  // Different order
        "vector_db": "qdrant",
        "chunk_size": 512,
        "model": "gpt-4",
        "retrieval_strategy": "hybrid",
    }));

    println!("  Digest 1: {}", config_digest.as_hex());
    println!("  Digest 2: {}", config_digest_2.as_hex());
    println!(
        "  Digests match: {}",
        config_digest.as_hex() == config_digest_2.as_hex()
    );
    println!("  (canonical JSON ensures stable hashing)\n");

    println!("=== Integration Demo Complete ===");
    println!("Next steps:");
    println!("  1. Integrate with AIVCS GraphRunRecorder for persistent storage");
    println!("  2. Compare multiple runs using tool-call sequence diffing");
    println!("  3. Evaluate retrieval quality and LLM response coherence");
}
