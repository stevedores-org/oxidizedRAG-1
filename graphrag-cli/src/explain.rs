//! Formatting utilities for retrieval explainability traces.
//!
//! Provides human-readable table output for `QueryTrace` from
//! `graphrag_core::retrieval::explain`.

use graphrag_core::retrieval::explain::{QueryTrace, ScoreBreakdown};

/// Format a `QueryTrace` as a readable table suitable for terminal output.
///
/// The output includes:
/// - A header with the query string and total duration
/// - A table of stages with timing and candidate counts
/// - Score breakdowns for stages that include them
pub fn format_query_trace(trace: &QueryTrace) -> String {
    let mut out = String::new();

    // Header
    out.push_str(&format!("Query Trace: \"{}\"\n", trace.query));
    out.push_str(&format!(
        "Total duration: {}ms | Results: {}\n",
        trace.total_duration.as_millis(),
        trace.result_count,
    ));
    out.push('\n');

    // Stage table header
    out.push_str(&format!(
        "{:<12} {:>10} {:>12}\n",
        "Stage", "Duration", "Candidates"
    ));
    out.push_str(&"-".repeat(36));
    out.push('\n');

    for stage in &trace.stages {
        out.push_str(&format!(
            "{:<12} {:>8}ms {:>12}\n",
            stage.stage_name,
            stage.duration.as_millis(),
            stage.candidates_produced,
        ));
    }

    // Score breakdowns (if any stage has one)
    let breakdowns: Vec<(&str, &ScoreBreakdown)> = trace
        .stages
        .iter()
        .filter_map(|s| {
            s.score_breakdown
                .as_ref()
                .map(|b| (s.stage_name.as_str(), b))
        })
        .collect();

    if !breakdowns.is_empty() {
        out.push('\n');
        out.push_str(&format!(
            "{:<12} {:>8} {:>8} {:>8} {:>8}\n",
            "Stage", "Vector", "Graph", "Keyword", "Final"
        ));
        out.push_str(&"-".repeat(48));
        out.push('\n');

        for (name, bd) in &breakdowns {
            out.push_str(&format!(
                "{:<12} {:>8.3} {:>8.3} {:>8.3} {:>8.3}\n",
                name, bd.vector_score, bd.graph_score, bd.keyword_score, bd.final_score,
            ));
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use graphrag_core::retrieval::explain::{QueryTrace, ScoreBreakdown, StageTrace};
    use std::time::Duration;

    #[test]
    fn test_format_basic_trace() {
        let trace = QueryTrace {
            query: "what is rust?".to_string(),
            stages: vec![
                StageTrace {
                    stage_name: "semantic".to_string(),
                    duration: Duration::from_millis(50),
                    candidates_produced: 20,
                    score_breakdown: None,
                },
                StageTrace {
                    stage_name: "keyword".to_string(),
                    duration: Duration::from_millis(10),
                    candidates_produced: 15,
                    score_breakdown: None,
                },
            ],
            total_duration: Duration::from_millis(60),
            result_count: 10,
        };

        let output = format_query_trace(&trace);
        assert!(output.contains("what is rust?"));
        assert!(output.contains("60ms"));
        assert!(output.contains("semantic"));
        assert!(output.contains("keyword"));
        assert!(output.contains("20"));
        assert!(output.contains("15"));
        // No score breakdown section expected
        assert!(!output.contains("Vector"));
    }

    #[test]
    fn test_format_trace_with_scores() {
        let trace = QueryTrace {
            query: "test".to_string(),
            stages: vec![StageTrace {
                stage_name: "fusion".to_string(),
                duration: Duration::from_millis(5),
                candidates_produced: 3,
                score_breakdown: Some(ScoreBreakdown {
                    vector_score: 0.85,
                    graph_score: 0.1,
                    keyword_score: 0.62,
                    final_score: 0.74,
                }),
            }],
            total_duration: Duration::from_millis(5),
            result_count: 3,
        };

        let output = format_query_trace(&trace);
        assert!(output.contains("Vector"));
        assert!(output.contains("Graph"));
        assert!(output.contains("Keyword"));
        assert!(output.contains("Final"));
        assert!(output.contains("0.850"));
        assert!(output.contains("0.740"));
    }
}
