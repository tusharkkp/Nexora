use std::collections::{HashMap, HashSet};

use crate::index::{DocId, InvertedIndex, MultiFieldIndex};
use crate::ranking::{
    BM25FParams, BM25Params, HybridBM25FParams, HybridRankingParams, rank_bm25_with_pagerank,
    rank_bm25f, rank_bm25f_with_pagerank,
};

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

/// Computes Average Precision (AP) across all retrieved results.
/// AP = (1 / total_relevant) * sum_{k=1..N} (Precision@k * is_relevant(k))
pub fn average_precision(retrieved: &[DocId], relevant: &HashSet<DocId>) -> f64 {
    if relevant.is_empty() {
        return 0.0;
    }
    let mut num_relevant_seen = 0;
    let mut sum_precision = 0.0;
    for (idx, doc_id) in retrieved.iter().enumerate() {
        if relevant.contains(doc_id) {
            num_relevant_seen += 1;
            let precision_at_rank = num_relevant_seen as f64 / (idx + 1) as f64;
            sum_precision += precision_at_rank;
        }
    }
    sum_precision / relevant.len() as f64
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

/// Runs a search evaluation benchmark using Hybrid BM25 + PageRank ranking.
pub fn evaluate_hybrid_pagerank(
    index: &InvertedIndex,
    pagerank_scores: &HashMap<DocId, f64>,
    benchmark: &[QueryJudgment],
    k: usize,
    params: &HybridRankingParams,
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
        let scored_docs = rank_bm25_with_pagerank(index, &qj.query, pagerank_scores, params);
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

/// Evaluates multi-field BM25F ranking on a benchmark suite of queries and ground-truth judgments.
pub fn evaluate_bm25f(
    multi_index: &MultiFieldIndex,
    benchmark: &[QueryJudgment],
    params: &BM25FParams,
    k: usize,
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
        let scored_docs = rank_bm25f(multi_index, &qj.query, params);
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

/// Evaluates multi-field BM25F + PageRank hybrid ranking on a benchmark suite.
pub fn evaluate_bm25f_with_pagerank(
    multi_index: &MultiFieldIndex,
    pagerank_scores: &HashMap<DocId, f64>,
    benchmark: &[QueryJudgment],
    params: &HybridBM25FParams,
    k: usize,
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
        let scored_docs = rank_bm25f_with_pagerank(multi_index, &qj.query, pagerank_scores, params);
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

/// Side-by-side comparison of baseline BM25 vs Hybrid BM25 + PageRank ranking models.
#[derive(Debug, Clone, PartialEq)]
pub struct BenchmarkComparison {
    pub k: usize,
    pub bm25: BenchmarkMetrics,
    pub hybrid: BenchmarkMetrics,
}

impl BenchmarkComparison {
    /// Relative percentage lift in Precision@K: ((Hybrid - BM25) / BM25) * 100%
    pub fn precision_lift(&self) -> f64 {
        Self::calc_lift(self.bm25.mean_precision, self.hybrid.mean_precision)
    }

    /// Relative percentage lift in Recall@K
    pub fn recall_lift(&self) -> f64 {
        Self::calc_lift(self.bm25.mean_recall, self.hybrid.mean_recall)
    }

    /// Relative percentage lift in Mean Reciprocal Rank (MRR)
    pub fn mrr_lift(&self) -> f64 {
        Self::calc_lift(
            self.bm25.mean_reciprocal_rank,
            self.hybrid.mean_reciprocal_rank,
        )
    }

    /// Relative percentage lift in NDCG@K
    pub fn ndcg_lift(&self) -> f64 {
        Self::calc_lift(self.bm25.mean_ndcg, self.hybrid.mean_ndcg)
    }

    fn calc_lift(baseline: f64, candidate: f64) -> f64 {
        if baseline <= 1e-9 {
            if candidate > 1e-9 { 100.0 } else { 0.0 }
        } else {
            ((candidate - baseline) / baseline) * 100.0
        }
    }

    /// Formats the side-by-side comparison as an ASCII / Markdown table.
    pub fn format_table(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "| Metric (@{}) | BM25 Baseline | BM25 + PageRank | Relative Lift |\n",
            self.k
        ));
        out.push_str("|:-------------|:--------------|:----------------|:--------------|\n");
        out.push_str(&format!(
            "| Precision@{}  | {:.4}         | {:.4}           | {:+.2}%        |\n",
            self.k,
            self.bm25.mean_precision,
            self.hybrid.mean_precision,
            self.precision_lift()
        ));
        out.push_str(&format!(
            "| Recall@{}     | {:.4}         | {:.4}           | {:+.2}%        |\n",
            self.k,
            self.bm25.mean_recall,
            self.hybrid.mean_recall,
            self.recall_lift()
        ));
        out.push_str(&format!(
            "| MRR          | {:.4}         | {:.4}           | {:+.2}%        |\n",
            self.bm25.mean_reciprocal_rank,
            self.hybrid.mean_reciprocal_rank,
            self.mrr_lift()
        ));
        out.push_str(&format!(
            "| NDCG@{}       | {:.4}         | {:.4}           | {:+.2}%        |\n",
            self.k,
            self.bm25.mean_ndcg,
            self.hybrid.mean_ndcg,
            self.ndcg_lift()
        ));
        out
    }
}

/// Evaluates both baseline BM25 and Hybrid BM25 + PageRank, producing a side-by-side comparison.
pub fn compare_rankers(
    index: &InvertedIndex,
    pagerank_scores: &HashMap<DocId, f64>,
    benchmark: &[QueryJudgment],
    k: usize,
    hybrid_params: &HybridRankingParams,
) -> BenchmarkComparison {
    let bm25 = evaluate_bm25(index, benchmark, k, &hybrid_params.bm25);
    let hybrid = evaluate_hybrid_pagerank(index, pagerank_scores, benchmark, k, hybrid_params);
    BenchmarkComparison { k, bm25, hybrid }
}

// -----------------------------------------------------------------------------
// Cranfield Dataset Integration
// -----------------------------------------------------------------------------

/// Represents a scientific document from the standard Cranfield IR test collection.
#[derive(Debug, Clone, PartialEq)]
pub struct CranfieldDocument {
    pub id: DocId,
    pub title: String,
    pub body: String,
}

fn extract_xml_tag<'a>(source: &'a str, open_tag: &str, close_tag: &str) -> Option<&'a str> {
    let start = source.find(open_tag)? + open_tag.len();
    let end = source[start..].find(close_tag)? + start;
    Some(&source[start..end])
}

/// Parses documents from Cranfield collection XML (`cran.all.1400.xml`).
pub fn parse_cranfield_docs(content: &str) -> Vec<CranfieldDocument> {
    let mut docs = Vec::new();
    let mut remaining = content;

    while let Some(start_doc) = remaining.find("<doc>") {
        let after_start = &remaining[start_doc + 5..];
        let end_doc = match after_start.find("</doc>") {
            Some(pos) => pos,
            None => break,
        };
        let doc_block = &after_start[..end_doc];
        remaining = &after_start[end_doc + 6..];

        let id = extract_xml_tag(doc_block, "<docno>", "</docno>")
            .and_then(|s| s.trim().parse::<u32>().ok())
            .unwrap_or(0);
        let title = extract_xml_tag(doc_block, "<title>", "</title>")
            .map(|s| s.trim().replace('\n', " "))
            .unwrap_or_default();
        let body = extract_xml_tag(doc_block, "<text>", "</text>")
            .map(|s| s.trim().replace('\n', " "))
            .unwrap_or_default();

        docs.push(CranfieldDocument { id, title, body });
    }
    docs
}

/// Parses queries from Cranfield collection XML (`cran.qry.xml`).
pub fn parse_cranfield_queries(content: &str) -> Vec<(u32, String)> {
    let mut queries = Vec::new();
    let mut remaining = content;

    while let Some(start_top) = remaining.find("<top>") {
        let after_start = &remaining[start_top + 5..];
        let end_top = match after_start.find("</top>") {
            Some(pos) => pos,
            None => break,
        };
        let top_block = &after_start[..end_top];
        remaining = &after_start[end_top + 6..];

        let id = extract_xml_tag(top_block, "<num>", "</num>")
            .and_then(|s| s.trim().parse::<u32>().ok())
            .unwrap_or(0);
        let title = extract_xml_tag(top_block, "<title>", "</title>")
            .map(|s| s.trim().replace('\n', " "))
            .unwrap_or_default();

        queries.push((id, title));
    }
    queries
}

/// Parses relevance judgments from TREC format (`cranqrel.trec.txt`).
pub fn parse_cranfield_qrels(content: &str) -> HashMap<u32, HashMap<DocId, u32>> {
    let mut qrels: HashMap<u32, HashMap<DocId, u32>> = HashMap::new();
    for line in content.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 4 {
            if let (Ok(qid), Ok(docno), Ok(rel)) = (
                parts[0].parse::<u32>(),
                parts[2].parse::<DocId>(),
                parts[3].parse::<u32>(),
            ) {
                if rel > 0 {
                    qrels.entry(qid).or_default().insert(docno, rel);
                }
            }
        }
    }
    qrels
}

/// Parses the full Cranfield collection into documents and query judgments.
pub fn parse_cranfield_dataset(
    docs_xml: &str,
    queries_xml: &str,
    qrels_txt: &str,
) -> (Vec<CranfieldDocument>, Vec<QueryJudgment>) {
    let docs = parse_cranfield_docs(docs_xml);
    let queries = parse_cranfield_queries(queries_xml);
    let qrels = parse_cranfield_qrels(qrels_txt);

    let mut judgments = Vec::new();
    for (seq_idx, (raw_id, query_text)) in queries.into_iter().enumerate() {
        let qid_seq = (seq_idx + 1) as u32;
        // In the standard Cranfield collection, qrels are keyed by the 1-based sequential
        // index of the query in cran.qry (1..=225), not the historical sparse .I identifier.
        if let Some(rel_map) = qrels.get(&qid_seq).or_else(|| qrels.get(&raw_id)) {
            judgments.push(QueryJudgment {
                query: query_text,
                relevance: rel_map.clone(),
            });
        }
    }

    (docs, judgments)
}

/// Loads the Cranfield collection from a directory containing the three standard files.
pub fn load_cranfield_dataset<P: AsRef<std::path::Path>>(
    dir_path: P,
) -> std::io::Result<(Vec<CranfieldDocument>, Vec<QueryJudgment>)> {
    let dir = dir_path.as_ref();
    let docs_xml = std::fs::read_to_string(dir.join("cran.all.1400.xml"))?;
    let queries_xml = std::fs::read_to_string(dir.join("cran.qry.xml"))?;
    let qrels_txt = std::fs::read_to_string(dir.join("cranqrel.trec.txt"))?;
    Ok(parse_cranfield_dataset(&docs_xml, &queries_xml, &qrels_txt))
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
        index.add_document(
            0,
            "Rust systems programming with memory safety and performance",
        );
        index.add_document(
            1,
            "Python machine learning, data science, and deep neural networks",
        );
        index.add_document(
            2,
            "Search engines use inverted indexes and BM25 ranking algorithms",
        );
        index.add_document(3, "Web crawlers fetch HTML pages over HTTP networks");

        let benchmark = vec![
            QueryJudgment::graded("rust memory safety", &[(0, 3), (2, 0), (1, 0), (3, 0)]),
            QueryJudgment::graded(
                "search engine inverted index",
                &[(2, 3), (3, 1), (0, 0), (1, 0)],
            ),
        ];

        let metrics = evaluate_bm25(&index, &benchmark, 3, &BM25Params::default());

        println!("\nBenchmark Quality Metrics (K = 3):");
        println!(
            "  Mean Precision@3:  {:.2}%",
            metrics.mean_precision * 100.0
        );
        println!("  Mean Recall@3:     {:.2}%", metrics.mean_recall * 100.0);
        println!("  MRR:               {:.4}", metrics.mean_reciprocal_rank);
        println!("  Mean NDCG@3:       {:.2}%", metrics.mean_ndcg * 100.0);

        // High relevance expectations for our targeted benchmark
        assert_eq!(metrics.mean_reciprocal_rank, 1.0); // Rank 1 in both queries
        assert!(metrics.mean_ndcg > 0.90);
    }

    #[test]
    fn test_comparative_benchmark_pagerank_lift() {
        use crate::graph::{PageRankParams, WebGraph, compute_pagerank};

        let mut index = InvertedIndex::new();
        // Doc 0: Keyword-stuffed page, but unlinked
        index.add_document(0, "rust systems programming rust systems programming");
        // Doc 1: Authoritative official guide
        index.add_document(1, "rust systems programming language overview");
        // Docs 2, 3, 4: Peer sites
        index.add_document(2, "community guide");
        index.add_document(3, "developer resources");
        index.add_document(4, "curated crates");

        // Web graph: Docs 2, 3, 4 all link to Doc 1 (Authority)
        let mut graph = WebGraph::new();
        graph.add_edge(2, 1);
        graph.add_edge(3, 1);
        graph.add_edge(4, 1);
        graph.add_edge(1, 2);

        let pr_scores = compute_pagerank(&graph, &PageRankParams::default());

        // Ground truth: Doc 1 is the primary authoritative document (grade 3), Doc 0 is grade 1
        let benchmark = vec![QueryJudgment::graded(
            "rust systems programming",
            &[(1, 3), (0, 0)],
        )];

        let comparison = compare_rankers(
            &index,
            &pr_scores,
            &benchmark,
            2,
            &HybridRankingParams::default(),
        );

        println!("\n{}", comparison.format_table());

        // Under BM25 alone, Doc 0 ranks #1 due to term frequency (MRR = 0.5)
        assert_eq!(comparison.bm25.mean_reciprocal_rank, 0.5);

        // Under Hybrid BM25 + PageRank, Doc 1 is promoted to #1 (MRR = 1.0)
        assert_eq!(comparison.hybrid.mean_reciprocal_rank, 1.0);

        // PageRank delivers 100% lift in MRR and significant lift in NDCG@2!
        assert_eq!(comparison.mrr_lift(), 100.0);
        assert!(comparison.ndcg_lift() > 0.0);
    }

    #[test]
    fn test_bm25f_benchmark_evaluation() {
        let mut multi = MultiFieldIndex::new();
        multi.add_document(
            0,
            "Rust Programming",
            "Language syntax and memory safety",
            "",
        );
        multi.add_document(
            1,
            "Gardening Guide",
            "Tools get rust on them when wet. Iron rust can damage tools rust.",
            "",
        );

        let benchmark = vec![QueryJudgment::graded("rust", &[(0, 3), (1, 0)])];
        let params = BM25FParams::default();
        let metrics = evaluate_bm25f(&multi, &benchmark, &params, 1);

        assert_eq!(metrics.mean_precision, 1.0);
        assert_eq!(metrics.mean_reciprocal_rank, 1.0);
        assert_eq!(metrics.mean_ndcg, 1.0);
    }

    #[test]
    fn test_average_precision() {
        let relevant: HashSet<DocId> = vec![2, 4].into_iter().collect();

        // Perfect ranking [2, 4, 1, 3]: seen 1 at 1 (P=1.0), seen 2 at 2 (P=1.0) -> AP = 1.0
        let perfect = vec![2, 4, 1, 3];
        assert_eq!(average_precision(&perfect, &relevant), 1.0);

        // Suboptimal [1, 2, 3, 4]: seen 1 at 2 (P=0.5), seen 2 at 4 (P=0.5) -> AP = 0.5
        let sub = vec![1, 2, 3, 4];
        assert_eq!(average_precision(&sub, &relevant), 0.5);

        // Empty relevant set
        assert_eq!(average_precision(&sub, &HashSet::new()), 0.0);
    }

    #[test]
    fn test_cranfield_xml_parsing() {
        let doc_xml = "<doc><docno>42</docno><title>Aerodynamics of Wings</title><text>Study of lift.</text></doc>";
        let qry_xml = "<top><num>1</num><title>wing aerodynamics</title></top>";
        let qrel_txt = "1 0 42 1\n1 0 99 0\n";

        let (docs, judgments) = parse_cranfield_dataset(doc_xml, qry_xml, qrel_txt);
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].id, 42);
        assert_eq!(docs[0].title, "Aerodynamics of Wings");
        assert_eq!(docs[0].body, "Study of lift.");

        assert_eq!(judgments.len(), 1);
        assert_eq!(judgments[0].query, "wing aerodynamics");
        assert_eq!(judgments[0].relevance.get(&42), Some(&1));
        assert_eq!(judgments[0].relevance.get(&99), None); // 0 is filtered out
    }

    #[test]
    fn test_cranfield_file_inspection() {
        let path = std::path::Path::new("data/cranfield");
        if !path.exists() {
            return;
        }
        let (docs, judgments) = load_cranfield_dataset(path).unwrap();
        assert_eq!(docs.len(), 1400);
        assert_eq!(judgments.len(), 225);
    }
}
