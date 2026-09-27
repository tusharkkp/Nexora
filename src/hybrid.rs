use std::collections::{HashMap, HashSet};

use crate::index::{DocId, MultiFieldIndex};
use crate::ranking::{BM25FParams, rank_bm25f};
use crate::vector::{TextEmbedder, VectorIndex};

/// Operational retrieval mode for the hybrid search engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchMode {
    /// Fuses BM25F lexical matching with dense semantic vector retrieval (Best overall)
    Hybrid,
    /// Pure lexical retrieval using multi-field BM25F
    Lexical,
    /// Pure dense semantic vector similarity retrieval
    Semantic,
}

impl SearchMode {
    pub fn parse(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "lexical" | "bm25" => Self::Lexical,
            "semantic" | "vector" => Self::Semantic,
            _ => Self::Hybrid,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Hybrid => "hybrid",
            Self::Lexical => "lexical",
            Self::Semantic => "semantic",
        }
    }
}

/// Algorithm used to combine lexical and dense retrieval signals into a single score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HybridFusionStrategy {
    /// Reciprocal Rank Fusion (RRF):
    /// RRF(d) = w_lex / (k + rank_lex(d)) + w_sem / (k + rank_sem(d))
    /// Parameter-free, scale-invariant, and highly robust.
    Rrf { k: usize },

    /// Convex Normalized Linear Score Fusion:
    /// Score(d) = alpha * Norm(BM25F(d)) + (1 - alpha) * Norm(Cosine(d))
    LinearScore { alpha: f32 },
}

impl Default for HybridFusionStrategy {
    fn default() -> Self {
        Self::Rrf { k: 60 }
    }
}

/// Parameters controlling hybrid retrieval, channel weighting, and fusion.
#[derive(Debug, Clone, PartialEq)]
pub struct HybridSearchParams {
    pub mode: SearchMode,
    pub fusion_strategy: HybridFusionStrategy,
    /// Maximum candidate documents to pull from each channel before fusion
    pub candidate_k: usize,
    /// Relative weight assigned to lexical BM25F channel in RRF
    pub lexical_weight: f64,
    /// Relative weight assigned to semantic vector channel in RRF
    pub semantic_weight: f64,
    /// Whether to integrate PageRank link authority mass into the final score
    pub include_pagerank: bool,
    /// Multiplier scaling PageRank authority influence
    pub pagerank_weight: f64,
    /// BM25F parameters
    pub bm25f_params: BM25FParams,
}

impl Default for HybridSearchParams {
    fn default() -> Self {
        Self {
            mode: SearchMode::Hybrid,
            fusion_strategy: HybridFusionStrategy::default(),
            candidate_k: 60,
            lexical_weight: 0.55,
            semantic_weight: 0.45,
            include_pagerank: true,
            pagerank_weight: 15.0,
            bm25f_params: BM25FParams::default(),
        }
    }
}

/// A search result produced by the hybrid retrieval engine.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredHybridDocument {
    pub doc_id: DocId,
    /// Final fused relevance score
    pub score: f64,
    /// Raw lexical BM25F score (0.0 if not matched in lexical channel)
    pub bm25f_score: f64,
    /// Dense vector cosine similarity in `[-1.0, 1.0]` (0.0 if not matched in vector channel)
    pub semantic_similarity: f32,
    /// PageRank score of the document
    pub pagerank: f64,
    /// 1-based rank in the lexical BM25F candidate list
    pub lexical_rank: Option<usize>,
    /// 1-based rank in the semantic vector candidate list
    pub semantic_rank: Option<usize>,
}

/// Executes hybrid search combining multi-field BM25F, dense vector similarity, and PageRank.
pub fn rank_hybrid(
    multi_index: &MultiFieldIndex,
    vector_index: &VectorIndex,
    query: &str,
    embedder: &dyn TextEmbedder,
    pagerank_scores: &HashMap<DocId, f64>,
    params: &HybridSearchParams,
) -> Vec<ScoredHybridDocument> {
    let clean = query.trim();
    if clean.is_empty() {
        return Vec::new();
    }

    // 1. Channel A: Lexical BM25F retrieval
    let mut lexical_map: HashMap<DocId, (usize, f64)> = HashMap::new(); // doc_id -> (rank_1_indexed, bm25f_score)
    if params.mode != SearchMode::Semantic {
        let lexical_results = rank_bm25f(multi_index, clean, &params.bm25f_params);
        let top_lexical = lexical_results.iter().take(params.candidate_k);
        for (i, scored) in top_lexical.enumerate() {
            lexical_map.insert(scored.doc_id, (i + 1, scored.score));
        }
    }

    // 2. Channel B: Dense Vector Semantic retrieval
    let mut vector_map: HashMap<DocId, (usize, f32)> = HashMap::new(); // doc_id -> (rank_1_indexed, cosine_sim)
    if params.mode != SearchMode::Lexical && !vector_index.is_empty() {
        let q_vec = embedder.embed(clean);
        let vector_results = vector_index.search(&q_vec, params.candidate_k);
        for (i, res) in vector_results.into_iter().enumerate() {
            vector_map.insert(res.doc_id, (i + 1, res.similarity));
        }
    }

    // 3. Pool all candidate document IDs
    let all_candidates: HashSet<DocId> = lexical_map
        .keys()
        .copied()
        .chain(vector_map.keys().copied())
        .collect();

    if all_candidates.is_empty() {
        return Vec::new();
    }

    // Min-Max normalization bounds for LinearScore strategy
    let (min_bm25, max_bm25) =
        if let HybridFusionStrategy::LinearScore { .. } = params.fusion_strategy {
            let scores: Vec<f64> = lexical_map.values().map(|&(_, s)| s).collect();
            let min = scores.iter().copied().fold(f64::INFINITY, f64::min);
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            (min, max)
        } else {
            (0.0, 1.0)
        };

    let mut results: Vec<ScoredHybridDocument> = Vec::with_capacity(all_candidates.len());

    for doc_id in all_candidates {
        let lex_entry = lexical_map.get(&doc_id);
        let vec_entry = vector_map.get(&doc_id);

        let bm25f_score = lex_entry.map(|&(_, s)| s).unwrap_or(0.0);
        let semantic_similarity = vec_entry.map(|&(_, s)| s).unwrap_or(0.0);
        let lexical_rank = lex_entry.map(|&(r, _)| r);
        let semantic_rank = vec_entry.map(|&(r, _)| r);
        let pr = pagerank_scores.get(&doc_id).copied().unwrap_or(0.0);

        let base_score: f64 = match params.mode {
            SearchMode::Lexical => bm25f_score,
            SearchMode::Semantic => semantic_similarity as f64,
            SearchMode::Hybrid => match params.fusion_strategy {
                HybridFusionStrategy::Rrf { k } => {
                    let k_f = k as f64;
                    let rrf_lex = match lexical_rank {
                        Some(r) => params.lexical_weight / (k_f + r as f64),
                        None => 0.0,
                    };
                    let rrf_sem = match semantic_rank {
                        Some(r) => params.semantic_weight / (k_f + r as f64),
                        None => 0.0,
                    };
                    rrf_lex + rrf_sem
                }
                HybridFusionStrategy::LinearScore { alpha } => {
                    let norm_bm25 = if max_bm25 > min_bm25 {
                        (bm25f_score - min_bm25) / (max_bm25 - min_bm25)
                    } else {
                        0.5
                    };
                    // Map cosine similarity [-1, 1] -> [0, 1]
                    let norm_cosine = ((semantic_similarity + 1.0) / 2.0).clamp(0.0, 1.0) as f64;

                    (alpha as f64) * norm_bm25 + ((1.0 - alpha) as f64) * norm_cosine
                }
            },
        };

        // Integrate PageRank authority signal
        let final_score = if params.include_pagerank && pr > 0.0 {
            base_score * (1.0 + (params.pagerank_weight * pr).ln_1p())
        } else {
            base_score
        };

        results.push(ScoredHybridDocument {
            doc_id,
            score: final_score,
            bm25f_score,
            semantic_similarity,
            pagerank: pr,
            lexical_rank,
            semantic_rank,
        });
    }

    // Sort descending by score, tie-break by DocId ascending
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
    use crate::index::MultiFieldIndex;
    use crate::vector::{SemanticEmbedder, VectorIndex};

    #[test]
    fn test_hybrid_search_synonym_retrieval_and_fusion() {
        let mut multi = MultiFieldIndex::new();
        let mut vec_idx = VectorIndex::new(64);
        let embedder = SemanticEmbedder::new(64);

        // Doc 0: Has exact lexical keywords "automobile maintenance"
        multi.add_document(
            0,
            "Automobile Maintenance Guide",
            "Routine car engine checks, motor mechanics, and tire pressure adjustments.",
            "",
        );
        vec_idx.add_vector(
            0,
            embedder.embed("Automobile Maintenance Guide Routine car engine checks, motor mechanics, and tire pressure adjustments."),
        );

        // Doc 1: Has conceptual match "car repair" but ZERO words matching "automobile maintenance"
        multi.add_document(
            1,
            "Car Repair Manual",
            "Vehicle roadside puncture fixes, mechanic overhaul, and oil fluid servicing.",
            "",
        );
        vec_idx.add_vector(
            1,
            embedder.embed("Car Repair Manual Vehicle roadside puncture fixes, mechanic overhaul, and oil fluid servicing."),
        );

        // Doc 2: Unrelated document
        multi.add_document(
            2,
            "Rust Systems Programming",
            "Bare-metal memory safety and borrow checker in systems software.",
            "",
        );
        vec_idx.add_vector(
            2,
            embedder.embed("Rust Systems Programming Bare-metal memory safety and borrow checker in systems software."),
        );

        let pr_map = HashMap::new();
        let params = HybridSearchParams {
            mode: SearchMode::Hybrid,
            fusion_strategy: HybridFusionStrategy::Rrf { k: 60 },
            candidate_k: 10,
            lexical_weight: 0.5,
            semantic_weight: 0.5,
            include_pagerank: false,
            pagerank_weight: 0.0,
            bm25f_params: BM25FParams::default(),
        };

        // Query: "automobile maintenance"
        let results = rank_hybrid(
            &multi,
            &vec_idx,
            "automobile maintenance",
            &embedder,
            &pr_map,
            &params,
        );
        assert!(!results.is_empty());
        assert_eq!(results[0].doc_id, 0); // Exact lexical match ranks #1

        // Conceptual Query: "car puncture fix" (matches Doc 1 semantically and partially lexically)
        let results_puncture = rank_hybrid(
            &multi,
            &vec_idx,
            "car puncture fix",
            &embedder,
            &pr_map,
            &params,
        );
        assert!(!results_puncture.is_empty());
        assert_eq!(results_puncture[0].doc_id, 1);
        assert!(results_puncture[0].semantic_similarity > 0.5);
    }

    #[test]
    fn test_search_mode_parsing() {
        assert_eq!(SearchMode::parse("lexical"), SearchMode::Lexical);
        assert_eq!(SearchMode::parse("semantic"), SearchMode::Semantic);
        assert_eq!(SearchMode::parse("hybrid"), SearchMode::Hybrid);
        assert_eq!(SearchMode::parse("vector"), SearchMode::Semantic);
        assert_eq!(SearchMode::parse("bm25"), SearchMode::Lexical);
    }
}
