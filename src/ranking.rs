use std::collections::HashMap;

use crate::index::{DocId, InvertedIndex};

/// Hyperparameters for the Okapi BM25 ranking function.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BM25Params {
    /// Controls term frequency saturation.
    /// Higher values mean term frequency has a stronger, less saturated effect.
    /// Standard baseline: 1.2 (typical range 1.2 - 2.0).
    pub k1: f64,

    /// Controls the degree of document length normalization.
    /// 1.0 means full length normalization; 0.0 means no length normalization.
    /// Standard baseline: 0.75 (typical range 0.5 - 0.8).
    pub b: f64,
}

impl Default for BM25Params {
    fn default() -> Self {
        Self { k1: 1.2, b: 0.75 }
    }
}

/// Represents a document paired with its computed relevance score.
#[derive(Debug, PartialEq, Clone)]
pub struct ScoredDocument {
    /// The document identifier
    pub doc_id: DocId,
    /// The computed BM25 relevance score (higher is more relevant)
    pub score: f64,
}

/// Computes the Robertson-Spärck Jones Inverse Document Frequency (IDF).
///
/// Uses the smoothed formulation:
/// IDF(t) = ln( ((N - n + 0.5) / (n + 0.5)) + 1.0 )
///
/// Where:
/// - N is the total number of documents in the index.
/// - n is the number of documents containing the term (document frequency).
pub fn idf(total_docs: usize, doc_freq: usize) -> f64 {
    if total_docs == 0 || doc_freq == 0 {
        return 0.0;
    }

    let n = doc_freq as f64;
    let big_n = total_docs as f64;

    let numerator = big_n - n + 0.5;
    let denominator = n + 0.5;

    ((numerator / denominator) + 1.0).ln()
}

/// Scores all matching documents in the index for a given query string using BM25.
///
/// Results are returned sorted in descending order of relevance.
pub fn rank_bm25(index: &InvertedIndex, query: &str, params: &BM25Params) -> Vec<ScoredDocument> {
    let total_docs = index.total_documents();
    if total_docs == 0 {
        return Vec::new();
    }

    let avgdl = index.average_doc_length();
    if avgdl <= 0.0 {
        return Vec::new();
    }

    // Step 1: Analyze query using the engine's analyzer
    let query_terms = index.analyzer().analyze(query);
    if query_terms.is_empty() {
        return Vec::new();
    }

    // Step 2: Accumulate scores across all query terms.
    // Map: DocId -> accumulated BM25 score
    let mut scores: HashMap<DocId, f64> = HashMap::new();

    // Deduplicate query terms to avoid scoring the same term twice
    let mut seen_terms = std::collections::HashSet::new();

    for term in query_terms {
        if !seen_terms.insert(term.text.clone()) {
            continue;
        }

        if let Some(postings) = index.get_postings(&term.text) {
            let doc_freq = postings.len();
            let term_idf = idf(total_docs, doc_freq);

            for posting in postings {
                let doc_id = posting.doc_id;
                let tf = posting.term_frequency as f64;
                let doc_len = index.doc_length(doc_id).unwrap_or(avgdl as u32) as f64;

                // BM25 term weight formula
                let len_norm = 1.0 - params.b + (params.b * (doc_len / avgdl));
                let tf_component = (tf * (params.k1 + 1.0)) / (tf + (params.k1 * len_norm));

                let term_score = term_idf * tf_component;
                *scores.entry(doc_id).or_insert(0.0) += term_score;
            }
        }
    }

    // Step 3: Convert to ScoredDocument list and sort descending by score
    let mut ranked_docs: Vec<ScoredDocument> = scores
        .into_iter()
        .map(|(doc_id, score)| ScoredDocument { doc_id, score })
        .collect();

    // Sort descending by score.
    // If scores are equal, sort ascending by doc_id for determinism.
    ranked_docs.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.doc_id.cmp(&b.doc_id))
    });

    ranked_docs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_idf_smoothness() {
        // Term appearing in 1 out of 100 documents should have higher IDF
        // than term appearing in 50 out of 100 documents.
        let rare_idf = idf(100, 1);
        let common_idf = idf(100, 50);
        let universal_idf = idf(100, 100);

        assert!(rare_idf > common_idf);
        assert!(common_idf > universal_idf);
        assert!(universal_idf >= 0.0);
    }

    #[test]
    fn test_tf_saturation() {
        // Document A mentions "search" 2 times
        // Document B mentions "search" 20 times (10x more!)
        // In BM25, Doc B's score should be higher, but NOT 10x higher due to k1 saturation.
        let mut index = InvertedIndex::new();
        index.add_document(0, "search search"); // Doc A (length 2)
        index.add_document(
            1,
            "search search search search search search search search search search search search search search search search search search search search",
        ); // Doc B (length 20, tf 20)

        // Set b = 0.0 to isolate the pure Term Frequency effect without length penalty
        let params = BM25Params { k1: 1.2, b: 0.0 };
        let results = rank_bm25(&index, "search", &params);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].doc_id, 1); // Doc B ranks higher
        assert_eq!(results[1].doc_id, 0);

        let ratio = results[0].score / results[1].score;
        // Even though TF is 10x larger, the score ratio must be strictly less than 2.0!
        assert!(ratio < 2.0);
        assert!(ratio > 1.0);
    }

    #[test]
    fn test_document_length_normalization() {
        // Doc A is concise (length 2, mentions "rust" 1 time)
        // Doc B is verbose (length 100, mentions "rust" 1 time + 99 other words)
        // Doc A must rank higher than Doc B because Doc B matches purely due to being long!
        let mut index = InvertedIndex::new();
        index.add_document(0, "rust fast");

        let long_text = format!("rust {}", "word ".repeat(99));
        index.add_document(1, &long_text);

        let params = BM25Params::default(); // k1 = 1.2, b = 0.75
        let results = rank_bm25(&index, "rust", &params);

        assert_eq!(results[0].doc_id, 0); // Short, focused document wins!
        assert_eq!(results[1].doc_id, 1);
        assert!(results[0].score > results[1].score);
    }

    #[test]
    fn test_multi_term_query_ranking() {
        let mut index = InvertedIndex::new();
        index.add_document(0, "rust programming language");
        index.add_document(1, "rust memory safety guarantees");
        index.add_document(2, "python programming tutorial");

        let params = BM25Params::default();
        let results = rank_bm25(&index, "rust programming", &params);

        // Doc 0 matches BOTH query terms ("rust" AND "programming"), so it should rank #1
        assert_eq!(results[0].doc_id, 0);
        assert!(results.len() == 3);
    }
}
