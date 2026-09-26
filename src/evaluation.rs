use std::collections::{HashMap, HashSet};

use crate::index::{DocId, InvertedIndex};
use crate::ranking::BM25Params;

/// A single benchmark evaluation query paired with ground-truth relevance assessments.
#[derive(Debug, Clone)]
pub struct QueryJudgment {
    /// The search query string
    pub query: String,
    /// Maps each document ID to its graded relevance:
    /// - 0: Irrelevant / Not applicable
    /// - 1: Marginally relevant
    /// - 2: Highly relevant
    /// - 3: Perfect answer
    pub relevance: HashMap<DocId, u32>,
}

impl QueryJudgment {
    /// Helper to create a new judgment with binary relevance (grade 1 for all relevant docs).
    pub fn binary(query: &str, relevant_docs: &[DocId]) -> Self {
        let mut relevance = HashMap::new();
        for &doc_id in relevant_docs {
            relevance.insert(doc_id, 1);
        }
        Self {
            query: query.to_string(),
            relevance,
        }
    }

    /// Helper to create a new judgment with graded relevance.
    pub fn graded(query: &str, graded_pairs: &[(DocId, u32)]) -> Self {
        let mut relevance = HashMap::new();
        for &(doc_id, grade) in graded_pairs {
            relevance.insert(doc_id, grade);
        }
        Self {
            query: query.to_string(),
            relevance,
        }
    }

    /// Returns the set of all document IDs with grade > 0.
    pub fn relevant_doc_ids(&self) -> HashSet<DocId> {
        self.relevance
            .iter()
            .filter(|(_, grade)| **grade > 0)
            .map(|(id, _)| *id)
            .collect()
    }
}

/// Comprehensive search quality metrics computed across a benchmark suite.
#[derive(Debug, Clone, PartialEq)]
pub struct BenchmarkMetrics {
    /// The cutoff rank evaluated (e.g. top 5, top 10)
    pub k: usize,
    /// Mean Precision@K: Fraction of top K results that are relevant
    pub mean_precision: f64,
    /// Mean Recall@K: Fraction of total known relevant documents retrieved in top K
    pub mean_recall: f64,
    /// Mean Reciprocal Rank (MRR): Average of 1 / (rank of first relevant document)
    pub mean_reciprocal_rank: f64,
    /// Mean NDCG@K: Normalized Discounted Cumulative Gain accounting for graded relevance & position
    pub mean_ndcg: f64,
}

// -----------------------------------------------------------------------------
// Metric Computation Functions
// -----------------------------------------------------------------------------

/// Computes Precision@K: (number of relevant documents in top K) / K.
pub fn precision_at_k(retrieved: &[DocId], relevant: &HashSet<DocId>, k: usize) -> f64 {
    if k == 0 {
        return 0.0;
    }
    let top_k = retrieved.iter().take(k);
    let relevant_count = top_k.filter(|doc_id| relevant.contains(doc_id)).count();
    relevant_count as f64 / k as f64
}

/// Computes Recall@K: (number of relevant documents in top K) / (total relevant documents).
pub fn recall_at_k(retrieved: &[DocId], relevant: &HashSet<DocId>, k: usize) -> f64 {
    if relevant.is_empty() {
        return 0.0;
    }
    let top_k = retrieved.iter().take(k);
    let retrieved_relevant = top_k.filter(|doc_id| relevant.contains(doc_id)).count();
    retrieved_relevant as f64 / relevant.len() as f64
}

/// Computes Reciprocal Rank (RR): 1 / (1-based rank of the FIRST relevant document).
pub fn reciprocal_rank(retrieved: &[DocId], relevant: &HashSet<DocId>) -> f64 {
    for (index, doc_id) in retrieved.iter().enumerate() {
        if relevant.contains(doc_id) {
            return 1.0 / (index as f64 + 1.0);
        }
    }
    0.0
}

/// Computes Discounted Cumulative Gain (DCG@K) using standard logarithmic discounting:
/// DCG@K = sum_{i=0}^{K-1} (2^{grade} - 1) / log2(i + 2)
pub fn dcg_at_k(retrieved: &[DocId], relevance: &HashMap<DocId, u32>, k: usize) -> f64 {
    let mut dcg = 0.0;
    for (i, doc_id) in retrieved.iter().take(k).enumerate() {
        let grade = relevance.get(doc_id).copied().unwrap_or(0);
        if grade > 0 {
            let numerator = ((1 << grade) - 1) as f64; // 2^grade - 1
            let denominator = ((i + 2) as f64).log2(); // log2(rank + 1)
            dcg += numerator / denominator;
        }
    }
    dcg
}

/// Computes Ideal Discounted Cumulative Gain (IDCG@K) by sorting all true grades descending.
pub fn idcg_at_k(relevance: &HashMap<DocId, u32>, k: usize) -> f64 {
    let mut grades: Vec<u32> = relevance.values().copied().filter(|&g| g > 0).collect();
    grades.sort_by(|a, b| b.cmp(a)); // Sort descending

    let mut idcg = 0.0;
    for (i, &grade) in grades.iter().take(k).enumerate() {
        let numerator = ((1 << grade) - 1) as f64;
        let denominator = ((i + 2) as f64).log2();
        idcg += numerator / denominator;
    }
    idcg
}

/// Computes Normalized Discounted Cumulative Gain (NDCG@K): DCG@K / IDCG@K.
/// Always falls between 0.0 and 1.0 (where 1.0 is theoretically optimal).
pub fn ndcg_at_k(retrieved: &[DocId], relevance: &HashMap<DocId, u32>, k: usize) -> f64 {
    let idcg = idcg_at_k(relevance, k);
    if idcg <= 0.0 {
        return 0.0;
    }
    let dcg = dcg_at_k(retrieved, relevance, k);
    dcg / idcg
}

// -----------------------------------------------------------------------------
// Benchmark Runner
// -----------------------------------------------------------------------------

/// Runs a full search evaluation benchmark against an InvertedIndex using Okapi BM25.
pub fn evaluate_bm25(
    index: &InvertedIndex,
    benchmark: &[QueryJudgment],
    k: usize,
    params: &BM25Params,
) -> BenchmarkMetrics {
    if benchmark.is_empty() {
        return BenchmarkMetrics {
            k,
            mean_precision: 0.0,
            mean_recall: 0.0,
            mean_reciprocal_rank: 0.0,
            mean_ndcg: 0.0,
        };
    }

    let mut total_p = 0.0;
    let mut total_r = 0.0;
    let mut total_rr = 0.0;
    let mut total_ndcg = 0.0;

    for qj in benchmark {
        // Execute BM25 search
        let scored_docs = index.search_bm25(&qj.query, params);
        let retrieved_ids: Vec<DocId> = scored_docs.iter().map(|s| s.doc_id).collect();
        let relevant_set = qj.relevant_doc_ids();

        total_p += precision_at_k(&retrieved_ids, &relevant_set, k);
        total_r += recall_at_k(&retrieved_ids, &relevant_set, k);
        total_rr += reciprocal_rank(&retrieved_ids, &relevant_set);
        total_ndcg += ndcg_at_k(&retrieved_ids, &qj.relevance, k);
    }

    let n = benchmark.len() as f64;
    BenchmarkMetrics {
        k,
        mean_precision: total_p / n,
        mean_recall: total_r / n,
        mean_reciprocal_rank: total_rr / n,
        mean_ndcg: total_ndcg / n,
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_recall_mrr_metrics() {
        let retrieved = vec![10, 20, 30, 40, 50];
        let relevant: HashSet<DocId> = vec![20, 40, 99].into_iter().collect();

        // Top 5 contains 2 relevant docs (20, 40) out of 5 -> P@5 = 2/5 = 0.4
        assert_eq!(precision_at_k(&retrieved, &relevant, 5), 0.4);

        // Top 5 found 2 out of 3 total relevant docs in existence -> R@5 = 2/3 ≈ 0.6667
        let recall = recall_at_k(&retrieved, &relevant, 5);
        assert!((recall - (2.0 / 3.0)).abs() < 1e-4);

        // First relevant doc is at index 1 (rank 2) -> RR = 1/2 = 0.5
        assert_eq!(reciprocal_rank(&retrieved, &relevant), 0.5);
    }

    #[test]
    fn test_ndcg_perfect_and_suboptimal() {
        let mut relevance = HashMap::new();
        relevance.insert(1, 3); // Doc 1: grade 3
        relevance.insert(2, 2); // Doc 2: grade 2
        relevance.insert(3, 0); // Doc 3: grade 0

        // Perfect ranking [1, 2, 3] -> NDCG must be 1.0
        let perfect_order = vec![1, 2, 3];
        let ndcg_perfect = ndcg_at_k(&perfect_order, &relevance, 3);
        assert!((ndcg_perfect - 1.0).abs() < 1e-4);

        // Suboptimal ranking [2, 3, 1] -> NDCG must be strictly lower than 1.0
        let suboptimal_order = vec![2, 3, 1];
        let ndcg_suboptimal = ndcg_at_k(&suboptimal_order, &relevance, 3);
        assert!(ndcg_suboptimal < 1.0);
        assert!(ndcg_suboptimal > 0.5);
    }

    #[test]
    fn test_end_to_end_bm25_benchmark() {
        let mut index = InvertedIndex::new();
        index.add_document(0, "Rust systems programming with memory safety and performance");
        index.add_document(1, "Python machine learning, data science, and deep neural networks");
        index.add_document(2, "Search engines use inverted indexes and BM25 ranking algorithms");
        index.add_document(3, "Web crawlers fetch HTML pages over HTTP networks");

        let benchmark = vec![
            QueryJudgment::graded(
                "rust memory safety",
                &[(0, 3), (2, 0), (1, 0), (3, 0)],
            ),
            QueryJudgment::graded(
                "search engine inverted index",
                &[(2, 3), (3, 1), (0, 0), (1, 0)],
            ),
        ];

        let metrics = evaluate_bm25(&index, &benchmark, 3, &BM25Params::default());

        println!("\nBenchmark Quality Metrics (K = 3):");
        println!("  Mean Precision@3:  {:.2}%", metrics.mean_precision * 100.0);
        println!("  Mean Recall@3:     {:.2}%", metrics.mean_recall * 100.0);
        println!("  MRR:               {:.4}", metrics.mean_reciprocal_rank);
        println!("  Mean NDCG@3:       {:.2}%", metrics.mean_ndcg * 100.0);

        // High relevance expectations for our targeted benchmark
        assert_eq!(metrics.mean_reciprocal_rank, 1.0); // Rank 1 in both queries
        assert!(metrics.mean_ndcg > 0.90);
    }
}
