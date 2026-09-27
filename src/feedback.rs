use std::collections::{HashMap, HashSet};

use crate::index::{DocId, Field, InvertedIndex, MultiFieldIndex};
use crate::ranking::{
    BM25FParams, BM25Params, FieldConfig, ScoredDocument, idf, rank_bm25, rank_bm25_weighted,
    rank_bm25f, rank_bm25f_weighted,
};
use crate::stemmer::stem;

/// Generates the standard English stop-word set used during pseudo-relevance feedback.
///
/// Every word is normalized via Porter stemming to match the indexed vocabulary.
pub fn default_stop_words() -> HashSet<String> {
    let raw_words = [
        "a",
        "about",
        "above",
        "across",
        "after",
        "afterwards",
        "again",
        "against",
        "all",
        "almost",
        "alone",
        "along",
        "already",
        "also",
        "although",
        "always",
        "am",
        "among",
        "amongst",
        "amoungst",
        "amount",
        "an",
        "and",
        "another",
        "any",
        "anyhow",
        "anyone",
        "anything",
        "anyway",
        "anywhere",
        "are",
        "around",
        "as",
        "at",
        "back",
        "be",
        "became",
        "because",
        "become",
        "becomes",
        "becoming",
        "been",
        "before",
        "beforehand",
        "behind",
        "being",
        "below",
        "beside",
        "besides",
        "between",
        "beyond",
        "both",
        "bottom",
        "but",
        "by",
        "call",
        "can",
        "cannot",
        "cant",
        "co",
        "con",
        "could",
        "couldnt",
        "de",
        "describe",
        "detail",
        "do",
        "done",
        "down",
        "due",
        "during",
        "each",
        "eg",
        "eight",
        "either",
        "eleven",
        "else",
        "elsewhere",
        "empty",
        "enough",
        "etc",
        "even",
        "ever",
        "every",
        "everyone",
        "everything",
        "everywhere",
        "except",
        "few",
        "fifteen",
        "fifty",
        "fill",
        "find",
        "fire",
        "first",
        "five",
        "for",
        "former",
        "formerly",
        "forty",
        "found",
        "four",
        "from",
        "front",
        "full",
        "further",
        "get",
        "give",
        "go",
        "had",
        "has",
        "hasnt",
        "have",
        "he",
        "hence",
        "her",
        "here",
        "hereafter",
        "hereby",
        "herein",
        "hereupon",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "however",
        "hundred",
        "i",
        "ie",
        "if",
        "in",
        "inc",
        "indeed",
        "interest",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "keep",
        "last",
        "latter",
        "latterly",
        "least",
        "less",
        "ltd",
        "made",
        "many",
        "may",
        "me",
        "meanwhile",
        "might",
        "mill",
        "mine",
        "more",
        "moreover",
        "most",
        "mostly",
        "move",
        "much",
        "must",
        "my",
        "myself",
        "name",
        "namely",
        "neither",
        "never",
        "nevertheless",
        "next",
        "nine",
        "no",
        "nobody",
        "none",
        "noone",
        "nor",
        "not",
        "nothing",
        "now",
        "nowhere",
        "of",
        "off",
        "often",
        "on",
        "once",
        "one",
        "only",
        "onto",
        "or",
        "other",
        "others",
        "otherwise",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "part",
        "per",
        "perhaps",
        "please",
        "put",
        "rather",
        "re",
        "same",
        "see",
        "seem",
        "seemed",
        "seeming",
        "seems",
        "serious",
        "several",
        "she",
        "should",
        "show",
        "side",
        "since",
        "sincere",
        "six",
        "sixty",
        "so",
        "some",
        "somehow",
        "someone",
        "something",
        "sometime",
        "sometimes",
        "somewhere",
        "still",
        "such",
        "system",
        "take",
        "ten",
        "than",
        "that",
        "the",
        "their",
        "them",
        "themselves",
        "then",
        "thence",
        "there",
        "thereafter",
        "thereby",
        "therefore",
        "therein",
        "thereupon",
        "these",
        "they",
        "thick",
        "thin",
        "third",
        "this",
        "those",
        "though",
        "three",
        "through",
        "throughout",
        "thru",
        "thus",
        "to",
        "together",
        "too",
        "top",
        "toward",
        "towards",
        "twelve",
        "twenty",
        "two",
        "un",
        "under",
        "until",
        "up",
        "upon",
        "us",
        "very",
        "via",
        "was",
        "we",
        "well",
        "were",
        "what",
        "whatever",
        "when",
        "whence",
        "whenever",
        "where",
        "whereafter",
        "whereas",
        "whereby",
        "wherein",
        "whereupon",
        "wherever",
        "whether",
        "which",
        "while",
        "whither",
        "who",
        "whoever",
        "whole",
        "whom",
        "whose",
        "why",
        "will",
        "with",
        "within",
        "without",
        "would",
        "yet",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
    ];

    raw_words.iter().map(|w| stem(w)).collect()
}

/// Hyperparameters for Pseudo-Relevance Feedback (PRF) and Rocchio Query Expansion.
#[derive(Debug, Clone, PartialEq)]
pub struct PrfParams {
    /// Number of top documents from initial retrieval to treat as pseudo-relevant.
    /// Baseline default: 5 (typical range: 3 - 10).
    pub feedback_docs: usize,

    /// Number of expansion terms to extract and add to the reformulated query vector.
    /// Baseline default: 5 (typical range: 3 - 15).
    pub expansion_terms: usize,

    /// Rocchio alpha: weight assigned to original query terms (q_0).
    /// Baseline default: 1.0 (typical range: 0.75 - 1.25).
    pub alpha: f64,

    /// Rocchio beta: weight assigned to positive feedback terms extracted from D_R.
    /// Baseline default: 0.5 (typical range: 0.25 - 0.75).
    pub beta: f64,

    /// Maximum document frequency ratio (doc_freq / total_docs).
    /// Terms appearing in more than this fraction of documents are filtered out as non-discriminative.
    /// Baseline default: 0.40 (40% of the corpus).
    pub max_doc_freq_ratio: f64,

    /// Minimum character length of an expansion candidate term.
    /// Baseline default: 3.
    pub min_term_length: usize,

    /// Optional custom stop-word set. If None, `default_stop_words()` is used.
    pub stop_words: Option<HashSet<String>>,
}

impl Default for PrfParams {
    fn default() -> Self {
        Self {
            feedback_docs: 5,
            expansion_terms: 5,
            alpha: 1.0,
            beta: 0.5,
            max_doc_freq_ratio: 0.40,
            min_term_length: 3,
            stop_words: None,
        }
    }
}

impl PrfParams {
    /// Sets the number of pseudo-relevant feedback documents (|D_R|).
    pub fn with_feedback_docs(mut self, feedback_docs: usize) -> Self {
        self.feedback_docs = feedback_docs;
        self
    }

    /// Sets the number of expansion terms (m).
    pub fn with_expansion_terms(mut self, expansion_terms: usize) -> Self {
        self.expansion_terms = expansion_terms;
        self
    }

    /// Sets the Rocchio alpha weight for original query terms.
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Sets the Rocchio beta weight for feedback terms.
    pub fn with_beta(mut self, beta: f64) -> Self {
        self.beta = beta;
        self
    }

    /// Sets the maximum document frequency ratio filter.
    pub fn with_max_doc_freq_ratio(mut self, ratio: f64) -> Self {
        self.max_doc_freq_ratio = ratio;
        self
    }

    /// Sets the minimum term length in characters.
    pub fn with_min_term_length(mut self, min_term_length: usize) -> Self {
        self.min_term_length = min_term_length;
        self
    }

    /// Sets a custom stop-word set.
    pub fn with_stop_words(mut self, stop_words: HashSet<String>) -> Self {
        self.stop_words = Some(stop_words);
        self
    }

    /// Returns the effective stop-word set (custom or default).
    pub fn stop_words_set(&self) -> HashSet<String> {
        self.stop_words.clone().unwrap_or_else(default_stop_words)
    }
}

/// Represents an expanded query produced by Rocchio / PRF reformulation.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpandedQuery {
    /// The original raw query string
    pub original_query: String,
    /// Complete weighted query vector for 2nd-pass ranking: (normalized_term, rocchio_weight)
    pub term_weights: Vec<(String, f64)>,
    /// The subset of terms that were newly added by expansion: (normalized_term, feedback_score)
    pub expansion_terms: Vec<(String, f64)>,
}

impl ExpandedQuery {
    /// Returns only the text strings of the newly expanded terms.
    pub fn expansion_term_texts(&self) -> Vec<String> {
        self.expansion_terms
            .iter()
            .map(|(t, _)| t.clone())
            .collect()
    }

    /// Generates a reformulated query string suitable for text-only search interfaces.
    ///
    /// Original query terms and top expansion terms are included.
    pub fn to_query_string(&self) -> String {
        let mut parts = vec![self.original_query.clone()];
        for (term, _) in &self.expansion_terms {
            parts.push(term.clone());
        }
        parts.join(" ")
    }
}

/// Extracts candidate expansion terms and computes their Rocchio feedback scores
/// from the top-|D_R| feedback documents in a flat `InvertedIndex`.
fn extract_feedback_terms_bm25(
    index: &InvertedIndex,
    feedback_docs: &[ScoredDocument],
    bm25_params: &BM25Params,
    prf_params: &PrfParams,
) -> HashMap<String, f64> {
    let k = feedback_docs.len().min(prf_params.feedback_docs);
    if k == 0 {
        return HashMap::new();
    }

    let target_doc_ids: HashSet<DocId> = feedback_docs.iter().take(k).map(|d| d.doc_id).collect();
    let stop_words = prf_params.stop_words_set();
    let total_docs = index.total_documents();
    let avgdl = index.average_doc_length();

    if total_docs == 0 || avgdl <= 0.0 {
        return HashMap::new();
    }

    let mut term_scores: HashMap<String, f64> = HashMap::new();

    for (term_text, postings) in index.dictionary() {
        // Length filter
        if term_text.len() < prf_params.min_term_length {
            continue;
        }

        // Stop word filter
        if stop_words.contains(term_text) {
            continue;
        }

        // Collection document frequency filter
        let doc_freq = postings.len();
        if (doc_freq as f64) / (total_docs as f64) > prf_params.max_doc_freq_ratio {
            continue;
        }

        // Scan postings for occurrences within the feedback set D_R
        let mut accumulated_tf_comp = 0.0;
        let mut match_count = 0;

        for posting in postings {
            if target_doc_ids.contains(&posting.doc_id) {
                let tf = posting.term_frequency as f64;
                let doc_len = index.doc_length(posting.doc_id).unwrap_or(avgdl as u32) as f64;
                let len_norm = 1.0 - bm25_params.b + (bm25_params.b * (doc_len / avgdl));
                let len_norm = if len_norm > 0.0 { len_norm } else { 1.0 };
                let tf_comp = (tf * (bm25_params.k1 + 1.0)) / (tf + (bm25_params.k1 * len_norm));

                accumulated_tf_comp += tf_comp;
                match_count += 1;
                if match_count == k {
                    break;
                }
            }
        }

        if accumulated_tf_comp > 0.0 {
            let term_idf = idf(total_docs, doc_freq);
            // Rocchio feedback component: (beta / |D_R|) * IDF(t) * sum_{d in D_R} TF_norm(t, d)
            let feedback_score = (prf_params.beta / (k as f64)) * term_idf * accumulated_tf_comp;
            term_scores.insert(term_text.clone(), feedback_score);
        }
    }

    term_scores
}

/// Expands a query from a predetermined set of feedback documents on an `InvertedIndex`.
pub fn expand_from_feedback_bm25(
    index: &InvertedIndex,
    original_query: &str,
    feedback_docs: &[ScoredDocument],
    bm25_params: &BM25Params,
    prf_params: &PrfParams,
) -> ExpandedQuery {
    let query_terms = index.analyzer().analyze(original_query);
    let mut original_term_counts: HashMap<String, usize> = HashMap::new();
    for term in &query_terms {
        *original_term_counts.entry(term.text.clone()).or_insert(0) += 1;
    }

    let feedback_scores =
        extract_feedback_terms_bm25(index, feedback_docs, bm25_params, prf_params);

    // Filter candidate terms that are NOT in the original query
    let mut candidate_expansion: Vec<(String, f64)> = feedback_scores
        .iter()
        .filter(|(term, _)| !original_term_counts.contains_key(*term))
        .map(|(term, &score)| (term.clone(), score))
        .collect();

    // Sort candidate terms descending by Rocchio feedback score (tie-break alphabetically)
    candidate_expansion.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    let top_expansion_terms: Vec<(String, f64)> = candidate_expansion
        .into_iter()
        .take(prf_params.expansion_terms)
        .collect();

    // Assemble final Rocchio query vector q_m:
    // For original terms: alpha * q_0(t) + feedback_score(t)
    let mut term_weights: Vec<(String, f64)> = Vec::new();
    for (orig_term, count) in &original_term_counts {
        let fb_score = feedback_scores.get(orig_term).copied().unwrap_or(0.0);
        let weight = prf_params.alpha * (*count as f64) + fb_score;
        term_weights.push((orig_term.clone(), weight));
    }

    // For expanded terms: feedback_score(t)
    for (exp_term, score) in &top_expansion_terms {
        term_weights.push((exp_term.clone(), *score));
    }

    ExpandedQuery {
        original_query: original_query.to_string(),
        term_weights,
        expansion_terms: top_expansion_terms,
    }
}

/// Performs Pseudo-Relevance Feedback (PRF) query expansion on an `InvertedIndex`.
///
/// 1. Runs an initial BM25 search pass for `query`.
/// 2. Selects the top `prf_params.feedback_docs` documents.
/// 3. Extracts high-IDF, discriminative terms using the Rocchio algorithm.
/// 4. Returns an `ExpandedQuery` containing original and expanded weighted terms.
pub fn expand_query_bm25(
    index: &InvertedIndex,
    query: &str,
    bm25_params: &BM25Params,
    prf_params: &PrfParams,
) -> ExpandedQuery {
    let initial_results = rank_bm25(index, query, bm25_params);
    expand_from_feedback_bm25(index, query, &initial_results, bm25_params, prf_params)
}

/// Executes full end-to-end Pseudo-Relevance Feedback ranking using BM25.
///
/// 1. Initial pass: BM25 retrieval on the raw query.
/// 2. PRF pass: Rocchio expansion extracting the top terms from top-ranked documents.
/// 3. Final pass: Weighted BM25 scoring across both original and expansion terms.
pub fn rank_bm25_with_prf(
    index: &InvertedIndex,
    query: &str,
    bm25_params: &BM25Params,
    prf_params: &PrfParams,
) -> Vec<ScoredDocument> {
    let expanded = expand_query_bm25(index, query, bm25_params, prf_params);
    if expanded.term_weights.is_empty() {
        return Vec::new();
    }
    rank_bm25_weighted(index, &expanded.term_weights, bm25_params)
}

/// Extracts candidate expansion terms from multi-field documents across standard fields.
fn extract_feedback_terms_bm25f(
    multi_index: &MultiFieldIndex,
    feedback_docs: &[ScoredDocument],
    bm25f_params: &BM25FParams,
    prf_params: &PrfParams,
) -> HashMap<String, f64> {
    let k = feedback_docs.len().min(prf_params.feedback_docs);
    if k == 0 {
        return HashMap::new();
    }

    let target_doc_ids: HashSet<DocId> = feedback_docs.iter().take(k).map(|d| d.doc_id).collect();
    let stop_words = prf_params.stop_words_set();
    let total_docs = multi_index.total_documents();

    if total_docs == 0 {
        return HashMap::new();
    }

    let mut avg_lens: HashMap<Field, f64> = HashMap::new();
    for field in MultiFieldIndex::standard_fields() {
        avg_lens.insert(field, multi_index.average_field_length(field));
    }

    // Collect all candidate terms occurring in any field for docs in target_doc_ids
    // Map: term_text -> Map<doc_id, Map<Field, tf>>
    let mut candidate_term_field_tfs: HashMap<String, HashMap<DocId, HashMap<Field, u32>>> =
        HashMap::new();

    for &field in &MultiFieldIndex::standard_fields() {
        if let Some(field_idx) = multi_index.get_field_index(field) {
            for (term_text, postings) in field_idx.dictionary() {
                if term_text.len() < prf_params.min_term_length || stop_words.contains(term_text) {
                    continue;
                }

                for posting in postings {
                    if target_doc_ids.contains(&posting.doc_id) {
                        candidate_term_field_tfs
                            .entry(term_text.clone())
                            .or_default()
                            .entry(posting.doc_id)
                            .or_default()
                            .insert(field, posting.term_frequency);
                    }
                }
            }
        }
    }

    let mut term_scores: HashMap<String, f64> = HashMap::new();

    for (term_text, doc_map) in candidate_term_field_tfs {
        // Collection-level document frequency: across all docs in the entire multi_index
        let doc_freq = multi_index.document_frequency(&term_text);
        if (doc_freq as f64) / (total_docs as f64) > prf_params.max_doc_freq_ratio {
            continue;
        }

        let term_idf = idf(total_docs, doc_freq);
        let mut accumulated_doc_score = 0.0;

        for (doc_id, field_tfs) in doc_map {
            let mut tf_tilde = 0.0;

            for &field in &MultiFieldIndex::standard_fields() {
                let config = bm25f_params
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
                let doc_score = term_idf
                    * ((tf_tilde * (bm25f_params.k1 + 1.0)) / (bm25f_params.k1 + tf_tilde));
                accumulated_doc_score += doc_score;
            }
        }

        if accumulated_doc_score > 0.0 {
            let feedback_score = (prf_params.beta / (k as f64)) * accumulated_doc_score;
            term_scores.insert(term_text, feedback_score);
        }
    }

    term_scores
}

/// Expands a query from a predetermined set of feedback documents on a `MultiFieldIndex`.
pub fn expand_from_feedback_bm25f(
    multi_index: &MultiFieldIndex,
    original_query: &str,
    feedback_docs: &[ScoredDocument],
    bm25f_params: &BM25FParams,
    prf_params: &PrfParams,
) -> ExpandedQuery {
    let query_terms = multi_index.analyzer().analyze(original_query);
    let mut original_term_counts: HashMap<String, usize> = HashMap::new();
    for term in &query_terms {
        *original_term_counts.entry(term.text.clone()).or_insert(0) += 1;
    }

    let feedback_scores =
        extract_feedback_terms_bm25f(multi_index, feedback_docs, bm25f_params, prf_params);

    let mut candidate_expansion: Vec<(String, f64)> = feedback_scores
        .iter()
        .filter(|(term, _)| !original_term_counts.contains_key(*term))
        .map(|(term, &score)| (term.clone(), score))
        .collect();

    candidate_expansion.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    let top_expansion_terms: Vec<(String, f64)> = candidate_expansion
        .into_iter()
        .take(prf_params.expansion_terms)
        .collect();

    let mut term_weights: Vec<(String, f64)> = Vec::new();
    for (orig_term, count) in &original_term_counts {
        let fb_score = feedback_scores.get(orig_term).copied().unwrap_or(0.0);
        let weight = prf_params.alpha * (*count as f64) + fb_score;
        term_weights.push((orig_term.clone(), weight));
    }

    for (exp_term, score) in &top_expansion_terms {
        term_weights.push((exp_term.clone(), *score));
    }

    ExpandedQuery {
        original_query: original_query.to_string(),
        term_weights,
        expansion_terms: top_expansion_terms,
    }
}

/// Performs Pseudo-Relevance Feedback (PRF) query expansion on a `MultiFieldIndex`.
pub fn expand_query_bm25f(
    multi_index: &MultiFieldIndex,
    query: &str,
    bm25f_params: &BM25FParams,
    prf_params: &PrfParams,
) -> ExpandedQuery {
    let initial_results = rank_bm25f(multi_index, query, bm25f_params);
    expand_from_feedback_bm25f(
        multi_index,
        query,
        &initial_results,
        bm25f_params,
        prf_params,
    )
}

/// Executes full end-to-end Pseudo-Relevance Feedback ranking using BM25F.
pub fn rank_bm25f_with_prf(
    multi_index: &MultiFieldIndex,
    query: &str,
    bm25f_params: &BM25FParams,
    prf_params: &PrfParams,
) -> Vec<ScoredDocument> {
    let expanded = expand_query_bm25f(multi_index, query, bm25f_params, prf_params);
    if expanded.term_weights.is_empty() {
        return Vec::new();
    }
    rank_bm25f_weighted(multi_index, &expanded.term_weights, bm25f_params)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_stop_words() {
        let stops = default_stop_words();
        assert!(stops.contains("the"));
        assert!(stops.contains("and"));
        assert!(stops.contains("with"));
        // Stemmed check: 'running' -> 'run'
        assert!(stops.len() > 100);
    }

    #[test]
    fn test_prf_query_expansion_basic() {
        let mut index = InvertedIndex::new();
        // Doc 1: supersonic aerodynamics and wing flutter instability
        index.add_document(
            1,
            "supersonic aerodynamics and delta wing flutter aeroelastic instability",
        );
        // Doc 2: supersonic wind tunnel testing with wing flutter
        index.add_document(2, "supersonic wind tunnel testing with wing flutter");
        // Doc 3: supersonic flow over blunt bodies at high Mach number
        index.add_document(3, "supersonic flow over blunt bodies at high Mach number");
        // Doc 4: unrelated document on database index structures
        index.add_document(
            4,
            "database index structures btree inverted index transactions",
        );

        let bm25_params = BM25Params::default();
        let prf_params = PrfParams::default()
            .with_feedback_docs(2)
            .with_expansion_terms(3)
            .with_max_doc_freq_ratio(0.8)
            .with_beta(0.5);

        // Query: "flutter"
        let expanded = expand_query_bm25(&index, "flutter", &bm25_params, &prf_params);

        assert_eq!(expanded.original_query, "flutter");
        assert!(!expanded.expansion_terms.is_empty());
        assert!(expanded.expansion_terms.len() <= 3);

        // The expansion terms should be drawn from doc 1 and doc 2 (e.g. wing, supersonic, aerodynam)
        let expansion_words: HashSet<String> = expanded
            .expansion_terms
            .iter()
            .map(|(t, _)| t.clone())
            .collect();
        assert!(
            expansion_words.contains("wing") || expansion_words.contains("superson"),
            "Expected 'wing' or 'superson' in expansion, got: {:?}",
            expansion_words
        );

        // Original term 'flutter' must have weight >= 1.0 (alpha + feedback)
        let flutter_weight = expanded
            .term_weights
            .iter()
            .find(|(t, _)| t == "flutter")
            .map(|(_, w)| *w)
            .unwrap_or(0.0);
        assert!(flutter_weight >= 1.0, "flutter weight: {}", flutter_weight);

        // Run full PRF ranking
        let results = rank_bm25_with_prf(&index, "flutter", &bm25_params, &prf_params);
        assert!(!results.is_empty());
        // Doc 1 and 2 should be at the top
        assert!(results[0].doc_id == 1 || results[0].doc_id == 2);
    }

    #[test]
    fn test_prf_bm25f_expansion() {
        let mut multi = MultiFieldIndex::new();
        multi.add_document(
            1,
            "Supersonic Flutter in Aircraft",
            "Aeroelastic instability analysis for thin wings at high Mach numbers",
            "",
        );
        multi.add_document(
            2,
            "Aeroelasticity Experiments",
            "Wind tunnel flutter measurement of delta wings in supersonic regime",
            "",
        );
        multi.add_document(
            3,
            "Relational Database Systems",
            "SQL query optimization and disk storage indexing",
            "",
        );

        let bm25f_params = BM25FParams::default();
        let prf_params = PrfParams::default()
            .with_feedback_docs(2)
            .with_expansion_terms(3);

        let results = rank_bm25f_with_prf(&multi, "supersonic flutter", &bm25f_params, &prf_params);
        assert!(!results.is_empty());
        assert_eq!(results[0].doc_id, 1);
        assert_eq!(results[1].doc_id, 2);
    }
}
