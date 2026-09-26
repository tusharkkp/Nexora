use std::collections::{HashMap, HashSet};

use crate::index::{DocId, Field, InvertedIndex, MultiFieldIndex};

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

/// Hyperparameters for hybrid ranking combining BM25 relevance and PageRank link authority.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HybridRankingParams {
    /// Base BM25 textual scoring parameters
    pub bm25: BM25Params,
    /// Weighting coefficient applied to PageRank authority (default: 1.0)
    pub alpha: f64,
    /// Internal scaling factor for the logarithmic PageRank boost (default: 100.0)
    pub beta: f64,
}

impl Default for HybridRankingParams {
    fn default() -> Self {
        Self {
            bm25: BM25Params::default(),
            alpha: 1.0,
            beta: 100.0,
        }
    }
}

/// Computes combined ranking combining BM25 textual matching and PageRank link authority.
///
/// Formula:
/// CombinedScore(d, q) = BM25(d, q) + alpha * ln(1.0 + beta * PR(d))
pub fn rank_bm25_with_pagerank(
    index: &InvertedIndex,
    query: &str,
    pagerank_scores: &HashMap<DocId, f64>,
    params: &HybridRankingParams,
) -> Vec<ScoredDocument> {
    let mut results = rank_bm25(index, query, &params.bm25);

    for doc in &mut results {
        let pr = pagerank_scores.get(&doc.doc_id).copied().unwrap_or(0.0);
        let authority_boost = params.alpha * (1.0 + params.beta * pr).ln();
        doc.score += authority_boost;
    }

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.doc_id.cmp(&b.doc_id))
    });

    results
}

/// Configuration parameters for an individual field in BM25F ranking.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FieldConfig {
    /// Field importance weight (w_f).
    /// Typically: Title = 3.0 to 5.0, Body = 1.0, Anchor = 2.0 to 3.0.
    pub weight: f64,
    /// Field length normalization parameter (b_f) in range [0.0, 1.0].
    /// 1.0 applies full length penalization; 0.0 disables length penalization.
    pub b: f64,
}

impl FieldConfig {
    /// Creates a new field configuration.
    pub fn new(weight: f64, b: f64) -> Self {
        Self { weight, b }
    }
}

/// Hyperparameters for multi-field BM25F ranking.
#[derive(Debug, Clone, PartialEq)]
pub struct BM25FParams {
    /// Controls global term frequency saturation.
    /// Standard baseline: 1.2 (typical range 1.2 - 2.0).
    pub k1: f64,
    /// Per-field weights and length normalization factors.
    pub field_configs: HashMap<Field, FieldConfig>,
}

impl Default for BM25FParams {
    fn default() -> Self {
        let mut field_configs = HashMap::new();
        // Title matches are concise and high-signal -> high weight (4.0), moderate b (0.5)
        field_configs.insert(Field::Title, FieldConfig::new(4.0, 0.5));
        // Body matches are verbose -> standard weight (1.0), standard b (0.75)
        field_configs.insert(Field::Body, FieldConfig::new(1.0, 0.75));
        // Anchor text represents external citations -> strong weight (2.5), b (0.6)
        field_configs.insert(Field::Anchor, FieldConfig::new(2.5, 0.6));

        Self {
            k1: 1.2,
            field_configs,
        }
    }
}

impl BM25FParams {
    /// Sets or overrides the configuration for a specific field.
    pub fn set_field(&mut self, field: Field, weight: f64, b: f64) -> &mut Self {
        self.field_configs.insert(field, FieldConfig::new(weight, b));
        self
    }
}

/// Scores all matching documents across multiple fields (Title, Body, Anchor) using BM25F.
///
/// In BM25F, field term frequencies are length-normalized and linearly combined
/// *before* applying the non-linear saturation curve:
///
/// tf_tilde(t, d) = sum_{f in fields} ( w_f * tf(t, d, f) / (1 - b_f + b_f * (len(d, f) / avg_len_f)) )
/// Score(d, Q) = sum_{t in Q} ( IDF(t) * ( tf_tilde(t, d) / (k1 + tf_tilde(t, d)) ) )
pub fn rank_bm25f(
    multi_index: &MultiFieldIndex,
    query: &str,
    params: &BM25FParams,
) -> Vec<ScoredDocument> {
    let total_docs = multi_index.total_documents();
    if total_docs == 0 {
        return Vec::new();
    }

    let query_terms = multi_index.analyzer().analyze(query);
    if query_terms.is_empty() {
        return Vec::new();
    }

    // Precompute average field lengths
    let mut avg_lens: HashMap<Field, f64> = HashMap::new();
    for field in MultiFieldIndex::standard_fields() {
        avg_lens.insert(field, multi_index.average_field_length(field));
    }

    let mut scores: HashMap<DocId, f64> = HashMap::new();
    let mut seen_terms = HashSet::new();

    for term in query_terms {
        if !seen_terms.insert(term.text.clone()) {
            continue;
        }

        // Map: DocId -> Map<Field, term_frequency>
        let mut doc_field_tfs: HashMap<DocId, HashMap<Field, u32>> = HashMap::new();

        for &field in &MultiFieldIndex::standard_fields() {
            if let Some(index) = multi_index.get_field_index(field) {
                if let Some(postings) = index.get_postings(&term.text) {
                    for posting in postings {
                        doc_field_tfs
                            .entry(posting.doc_id)
                            .or_default()
                            .insert(field, posting.term_frequency);
                    }
                }
            }
        }

        if doc_field_tfs.is_empty() {
            continue;
        }

        // Collection-level document frequency: number of documents containing term in ANY field
        let doc_freq = doc_field_tfs.len();
        let term_idf = idf(total_docs, doc_freq);

        for (doc_id, field_tfs) in doc_field_tfs {
            let mut tf_tilde = 0.0;

            for &field in &MultiFieldIndex::standard_fields() {
                let config = params
                    .field_configs
                    .get(&field)
                    .copied()
                    .unwrap_or_else(|| FieldConfig::new(1.0, 0.75));

                if let Some(&tf) = field_tfs.get(&field) {
                    let doc_len = multi_index.field_doc_length(field, doc_id) as f64;
                    let avg_len = avg_lens.get(&field).copied().unwrap_or(0.0);

                    let len_norm = if avg_len > 0.0 {
                        1.0 - config.b + (config.b * (doc_len / avg_len))
                    } else {
                        1.0
                    };
                    let len_norm = if len_norm > 0.0 { len_norm } else { 1.0 };

                    let norm_tf = (tf as f64) / len_norm;
                    tf_tilde += config.weight * norm_tf;
                }
            }

            if tf_tilde > 0.0 {
                let term_score = term_idf * ((tf_tilde * (params.k1 + 1.0)) / (params.k1 + tf_tilde));
                *scores.entry(doc_id).or_insert(0.0) += term_score;
            }
        }
    }

    let mut ranked_docs: Vec<ScoredDocument> = scores
        .into_iter()
        .map(|(doc_id, score)| ScoredDocument { doc_id, score })
        .collect();

    ranked_docs.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.doc_id.cmp(&b.doc_id))
    });

    ranked_docs
}

/// Hyperparameters for hybrid BM25F ranking combining multi-field relevance and PageRank authority.
#[derive(Debug, Clone, PartialEq)]
pub struct HybridBM25FParams {
    /// Multi-field BM25F parameters
    pub bm25f: BM25FParams,
    /// Weighting coefficient applied to PageRank authority (default: 1.0)
    pub alpha: f64,
    /// Internal scaling factor for the logarithmic PageRank boost (default: 100.0)
    pub beta: f64,
}

impl Default for HybridBM25FParams {
    fn default() -> Self {
        Self {
            bm25f: BM25FParams::default(),
            alpha: 1.0,
            beta: 100.0,
        }
    }
}

/// Computes combined ranking combining BM25F multi-field textual matching and PageRank link authority.
///
/// Formula:
/// CombinedScore(d, q) = BM25F(d, q) + alpha * ln(1.0 + beta * PR(d))
pub fn rank_bm25f_with_pagerank(
    multi_index: &MultiFieldIndex,
    query: &str,
    pagerank_scores: &HashMap<DocId, f64>,
    params: &HybridBM25FParams,
) -> Vec<ScoredDocument> {
    let mut results = rank_bm25f(multi_index, query, &params.bm25f);

    for doc in &mut results {
        let pr = pagerank_scores.get(&doc.doc_id).copied().unwrap_or(0.0);
        let authority_boost = params.alpha * (1.0 + params.beta * pr).ln();
        doc.score += authority_boost;
    }

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.doc_id.cmp(&b.doc_id))
    });

    results
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

    #[test]
    fn test_hybrid_bm25_with_pagerank() {
        let mut index = InvertedIndex::new();
        // Both documents have identical text and length
        index.add_document(0, "search engine architecture");
        index.add_document(1, "search engine architecture");

        let mut pr_scores = HashMap::new();
        pr_scores.insert(0, 0.1);
        pr_scores.insert(1, 0.9); // Doc 1 has much higher link authority

        let hybrid_params = HybridRankingParams::default();
        let results = rank_bm25_with_pagerank(&index, "search engine", &pr_scores, &hybrid_params);

        assert_eq!(results.len(), 2);
        // Doc 1 wins due to PageRank authority boost!
        assert_eq!(results[0].doc_id, 1);
        assert_eq!(results[1].doc_id, 0);
        assert!(results[0].score > results[1].score);
    }

    #[test]
    fn test_bm25f_title_boost() {
        let mut multi = MultiFieldIndex::new();
        // Doc 0: Has "rust" in title, none in body
        multi.add_document(0, "Rust Programming", "Learn systems programming and safety", "");
        // Doc 1: Has "rust" 3 times in body, but title is unrelated
        multi.add_document(
            1,
            "Gardening Tools",
            "Metal tools get rust quickly. Prevent rust by cleaning iron rust.",
            "",
        );

        let params = BM25FParams::default(); // Title weight 4.0, Body weight 1.0
        let results = rank_bm25f(&multi, "rust", &params);

        assert_eq!(results.len(), 2);
        // Doc 0 must win due to title field weight!
        assert_eq!(results[0].doc_id, 0);
        assert_eq!(results[1].doc_id, 1);
        assert!(results[0].score > results[1].score);
    }

    #[test]
    fn test_bm25f_anchor_boost() {
        let mut multi = MultiFieldIndex::new();
        // Doc 0: Has "search" in body
        multi.add_document(0, "Index Page", "General web search tools", "");
        // Doc 1: Has "search engine" only in anchor text from incoming links
        multi.add_document(1, "Project Nexora", "High performance fast software", "");
        multi.append_anchor_text(1, "best rust search engine");

        let params = BM25FParams::default();
        let results = rank_bm25f(&multi, "search engine", &params);

        // Doc 1 matches both terms in anchor text, doc 0 only matches "search" in body
        assert_eq!(results[0].doc_id, 1);
    }

    #[test]
    fn test_bm25f_with_pagerank() {
        let mut multi = MultiFieldIndex::new();
        multi.add_document(0, "Search Engines", "Information retrieval system", "");
        multi.add_document(1, "Search Engines", "Information retrieval system", "");

        let mut pr_scores = HashMap::new();
        pr_scores.insert(0, 0.05);
        pr_scores.insert(1, 0.85);

        let params = HybridBM25FParams::default();
        let results = rank_bm25f_with_pagerank(&multi, "search engines", &pr_scores, &params);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].doc_id, 1); // Doc 1 boosted by PageRank
        assert!(results[0].score > results[1].score);
    }
}

