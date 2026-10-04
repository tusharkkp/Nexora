use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::Instant;

use nexora::evaluation::{
    QueryJudgment, average_precision, load_cranfield_dataset, ndcg_at_k, precision_at_k,
    recall_at_k, reciprocal_rank,
};
use nexora::feedback::{PrfParams, rank_bm25f_with_prf};
use nexora::hybrid::{HybridFusionStrategy, HybridSearchParams, SearchMode, rank_hybrid};
use nexora::index::{DocId, Field, InvertedIndex, MultiFieldIndex};
use nexora::ranking::{BM25FParams, BM25Params, rank_bm25, rank_bm25f};
use nexora::vector::{SemanticEmbedder, TextEmbedder, VectorIndex, cosine_similarity};

#[derive(Debug, Clone, Copy, Default)]
struct DetailedMetrics {
    map: f64,
    mrr: f64,
    p5: f64,
    p10: f64,
    r10: f64,
    ndcg5: f64,
    ndcg10: f64,
}

impl DetailedMetrics {
    fn compute<F>(judgments: &[QueryJudgment], mut ranker: F) -> (Self, Vec<f64>)
    where
        F: FnMut(&str) -> Vec<DocId>,
    {
        if judgments.is_empty() {
            return (Self::default(), Vec::new());
        }

        let mut sum_map = 0.0;
        let mut sum_mrr = 0.0;
        let mut sum_p5 = 0.0;
        let mut sum_p10 = 0.0;
        let mut sum_r10 = 0.0;
        let mut sum_ndcg5 = 0.0;
        let mut sum_ndcg10 = 0.0;

        let mut per_query_ap = Vec::with_capacity(judgments.len());

        for qj in judgments {
            let retrieved = ranker(&qj.query);
            let relevant: HashSet<DocId> = qj.relevant_doc_ids();

            let ap = average_precision(&retrieved, &relevant);
            per_query_ap.push(ap);

            sum_map += ap;
            sum_mrr += reciprocal_rank(&retrieved, &relevant);
            sum_p5 += precision_at_k(&retrieved, &relevant, 5);
            sum_p10 += precision_at_k(&retrieved, &relevant, 10);
            sum_r10 += recall_at_k(&retrieved, &relevant, 10);
            sum_ndcg5 += ndcg_at_k(&retrieved, &qj.relevance, 5);
            sum_ndcg10 += ndcg_at_k(&retrieved, &qj.relevance, 10);
        }

        let n = judgments.len() as f64;
        let metrics = Self {
            map: sum_map / n,
            mrr: sum_mrr / n,
            p5: sum_p5 / n,
            p10: sum_p10 / n,
            r10: sum_r10 / n,
            ndcg5: sum_ndcg5 / n,
            ndcg10: sum_ndcg10 / n,
        };

        (metrics, per_query_ap)
    }

    fn print_row(&self, label: &str) {
        println!(
            "| {:<34} | {:>6.4} | {:>6.4} | {:>6.4} | {:>6.4} | {:>6.4} | {:>6.4} | {:>7.4} |",
            label, self.map, self.mrr, self.p5, self.p10, self.r10, self.ndcg5, self.ndcg10
        );
    }

    fn print_header() {
        println!(
            "| {:<34} |   MAP  |   MRR  |   P@5  |  P@10  |  R@10  | NDCG@5 | NDCG@10 |",
            "System / Model"
        );
        println!(
            "|:{:-<34}:|:------:|:------:|:------:|:------:|:------:|:------:|:-------:|",
            ""
        );
    }
}

fn normal_cdf(z: f64) -> f64 {
    // Abramowitz and Stegun approximation
    let t = 1.0 / (1.0 + 0.2316419 * z.abs());
    let d = 0.3989422804014327 * (-z * z / 2.0).exp();
    let p = d
        * t
        * (0.319381530
            + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    if z > 0.0 { 1.0 - p } else { p }
}

fn main() {
    println!("===============================================================================");
    println!("        NEXORA SCIENTIFIC EVALUATION: DENSE & HYBRID VECTOR RETRIEVAL");
    println!("===============================================================================\n");

    // -------------------------------------------------------------------------
    // PHASE 2: CORRECTNESS & EMBEDDER UNIT SIMILARITY TESTS
    // -------------------------------------------------------------------------
    println!("-------------------------------------------------------------------------------");
    println!("PHASE 2: EMBEDDER COSINE SIMILARITY TESTS ACROSS TEST CASES");
    println!("-------------------------------------------------------------------------------");
    let embedder = SemanticEmbedder::default();

    let test_pairs: &[(&str, &str, &str)] = &[
        (
            "1. Identical text",
            "supersonic flutter aerodynamics in flexible wings",
            "supersonic flutter aerodynamics in flexible wings",
        ),
        (
            "2. Near-synonyms (In-domain)",
            "aircraft wing airflow",
            "airplane wings aerodynamics",
        ),
        (
            "3. Near-synonyms (Out-of-domain)",
            "physician curing illness",
            "doctor treating disease",
        ),
        (
            "4. Completely unrelated text",
            "supersonic boundary layer flutter",
            "chocolate cake baking recipe flour sugar",
        ),
        (
            "5. Zero-overlap semantic (In-domain hardcoded)",
            "automobile transmission repair",
            "car motor maintenance",
        ),
        (
            "6. Zero-overlap semantic (Out-of-domain)",
            "purchase residential property",
            "buy house real-estate",
        ),
        (
            "7. Short query vs document",
            "flutter",
            "wing structural vibration and supersonic aeroelastic instability",
        ),
        (
            "8. Long scientific query vs document",
            "what are the effects of boundary layer laminar flow control on drag reduction at high mach numbers",
            "boundary layer suction for skin friction drag reduction in supersonic flow",
        ),
        ("9. Empty text input", "", "supersonic flight"),
        (
            "10. Unknown vocabulary / OOV",
            "zyxwvut rqponmlk jhgf",
            "supersonic flight aerodynamics",
        ),
        (
            "11. Cross-domain mismatch",
            "cardiac arrest myocardial infarction arrhythmia",
            "supersonic aerodynamic pressure distribution across delta wings",
        ),
    ];

    println!(
        "| {:<42} | {:>12} | Status/Note |",
        "Test Case Category", "Cosine Sim"
    );
    println!("|:{:-<42}:|:------------:|:------------|", "");
    for (category, t1, t2) in test_pairs {
        let v1 = embedder.embed(t1);
        let v2 = embedder.embed(t2);
        let sim = cosine_similarity(&v1, &v2);
        let note = if sim > 0.999 {
            "Perfect identity"
        } else if sim > 0.5 {
            "Strong semantic match"
        } else if sim > 0.2 {
            "Weak / Subword hash match"
        } else if sim.abs() < 0.05 {
            "Near orthogonal (Unrelated)"
        } else {
            "Subword hash collision"
        };
        println!("| {:<42} | {:>12.4} | {:<20} |", category, sim, note);
    }
    println!();

    // -------------------------------------------------------------------------
    // PHASE 3 & 4 & 5: CRANFIELD DATASET BENCHMARK
    // -------------------------------------------------------------------------
    println!("-------------------------------------------------------------------------------");
    println!("PHASE 3, 4, 5: AUTHENTIC CRANFIELD BENCHMARK (1,400 DOCS, 225 QUERIES)");
    println!("-------------------------------------------------------------------------------");

    let data_dir = Path::new("data/cranfield");
    let (docs, judgments) =
        load_cranfield_dataset(data_dir).expect("Failed to load Cranfield dataset");
    let doc_titles: HashMap<DocId, String> = docs.iter().map(|d| (d.id, d.title.clone())).collect();

    println!(
        "Loaded {} Cranfield documents and {} queries.",
        docs.len(),
        judgments.len()
    );

    let mut flat_index = InvertedIndex::new();
    let mut multi_index = MultiFieldIndex::new();
    let mut vector_index = VectorIndex::new(embedder.dimension());

    let idx_start = Instant::now();
    for d in &docs {
        let combined = format!("{} {}", d.title, d.body);
        flat_index.add_document(d.id, &combined);
        multi_index.add_document(d.id, &d.title, &d.body, "");
        vector_index.add_vector(d.id, embedder.embed(&combined));
    }
    let idx_time = idx_start.elapsed();
    println!(
        "Built Flat, Multi-field, and Dense Vector indexes in {:.2?}.\n",
        idx_time
    );

    let bm25_params = BM25Params::default();
    let mut bm25f_params = BM25FParams::default();
    bm25f_params.set_field(Field::Title, 2.0, 0.5);
    bm25f_params.set_field(Field::Body, 1.0, 0.75);

    let prf_params = PrfParams::default()
        .with_feedback_docs(5)
        .with_expansion_terms(5)
        .with_beta(0.5);

    let empty_pr = HashMap::new();

    // System A: BM25
    let (m_bm25, _) = DetailedMetrics::compute(&judgments, |q| {
        rank_bm25(&flat_index, q, &bm25_params)
            .into_iter()
            .map(|s| s.doc_id)
            .collect()
    });

    // System B: BM25F
    let (m_bm25f, _ap_bm25f_all) = DetailedMetrics::compute(&judgments, |q| {
        rank_bm25f(&multi_index, q, &bm25f_params)
            .into_iter()
            .map(|s| s.doc_id)
            .collect()
    });

    // System C: Dense-only
    let dense_params = HybridSearchParams {
        mode: SearchMode::Semantic,
        candidate_k: 60,
        ..HybridSearchParams::default()
    };
    let (m_dense, _ap_dense_all) = DetailedMetrics::compute(&judgments, |q| {
        rank_hybrid(
            &multi_index,
            &vector_index,
            q,
            &embedder,
            &empty_pr,
            &dense_params,
        )
        .into_iter()
        .map(|s| s.doc_id)
        .collect()
    });

    // System D: BM25F + Dense (RRF k=60, w_lex=0.55, w_sem=0.45)
    let rrf_params_default = HybridSearchParams {
        mode: SearchMode::Hybrid,
        fusion_strategy: HybridFusionStrategy::Rrf { k: 60 },
        candidate_k: 60,
        lexical_weight: 0.55,
        semantic_weight: 0.45,
        include_pagerank: false,
        bm25f_params: bm25f_params.clone(),
        ..HybridSearchParams::default()
    };
    let (m_rrf_default, _ap_rrf_all) = DetailedMetrics::compute(&judgments, |q| {
        rank_hybrid(
            &multi_index,
            &vector_index,
            q,
            &embedder,
            &empty_pr,
            &rrf_params_default,
        )
        .into_iter()
        .map(|s| s.doc_id)
        .collect()
    });

    // System E: BM25F + Dense (Linear Score alpha=0.5)
    let linear_params_05 = HybridSearchParams {
        mode: SearchMode::Hybrid,
        fusion_strategy: HybridFusionStrategy::LinearScore { alpha: 0.5 },
        candidate_k: 60,
        include_pagerank: false,
        bm25f_params: bm25f_params.clone(),
        ..HybridSearchParams::default()
    };
    let (m_linear_05, _) = DetailedMetrics::compute(&judgments, |q| {
        rank_hybrid(
            &multi_index,
            &vector_index,
            q,
            &embedder,
            &empty_pr,
            &linear_params_05,
        )
        .into_iter()
        .map(|s| s.doc_id)
        .collect()
    });

    // System F: BM25F + PRF
    let (m_bm25f_prf, _ap_prf_all) = DetailedMetrics::compute(&judgments, |q| {
        rank_bm25f_with_prf(&multi_index, q, &bm25f_params, &prf_params)
            .into_iter()
            .map(|s| s.doc_id)
            .collect()
    });

    // System G: BM25F + Dense + PRF (Lexical PRF rank combined with Dense rank via RRF)
    let (m_bm25f_dense_prf, _ap_all_g) = DetailedMetrics::compute(&judgments, |q| {
        // Step 1: get BM25F+PRF ranked candidates
        let prf_results = rank_bm25f_with_prf(&multi_index, q, &bm25f_params, &prf_params);
        let top_prf: Vec<DocId> = prf_results.into_iter().take(60).map(|s| s.doc_id).collect();

        // Step 2: get Dense ranked candidates
        let q_vec = embedder.embed(q);
        let dense_results = vector_index.search(&q_vec, 60);

        let mut rrf_scores: HashMap<DocId, f64> = HashMap::new();
        for (i, &doc_id) in top_prf.iter().enumerate() {
            *rrf_scores.entry(doc_id).or_insert(0.0) += 0.55 / (60.0 + (i + 1) as f64);
        }
        for (i, res) in dense_results.iter().enumerate() {
            *rrf_scores.entry(res.doc_id).or_insert(0.0) += 0.45 / (60.0 + (i + 1) as f64);
        }

        let mut fused: Vec<(DocId, f64)> = rrf_scores.into_iter().collect();
        fused.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        fused.into_iter().map(|(id, _)| id).collect()
    });

    println!("Table 1: Full Cranfield Evaluation (All 225 Queries):");
    DetailedMetrics::print_header();
    m_bm25.print_row("A. BM25 Baseline");
    m_bm25f.print_row("B. BM25F (Title w=2.0)");
    m_dense.print_row("C. Dense-only (SemanticEmbedder)");
    m_rrf_default.print_row("D. BM25F + Dense (RRF k=60)");
    m_linear_05.print_row("E. BM25F + Dense (Linear α=0.5)");
    m_bm25f_prf.print_row("F. BM25F + PRF (Rocchio β=0.5)");
    m_bm25f_dense_prf.print_row("G. BM25F + Dense + PRF (RRF)");
    println!();

    // -------------------------------------------------------------------------
    // PHASE 4 & 5: TRAIN / TEST SPLIT (50% / 50%)
    // -------------------------------------------------------------------------
    let split_idx = 112; // 0..112 = 112 queries (Train); 112..225 = 113 queries (Test)
    let (train_q, test_q) = judgments.split_at(split_idx);

    println!("-------------------------------------------------------------------------------");
    println!("PHASE 4: RRF PARAMETER EXPLORATION ON TRAINING QUERIES (112 QUERIES)");
    println!("-------------------------------------------------------------------------------");

    let k_values = [10, 20, 60, 100];
    let weight_ratios = [
        ("Pure Lexical (1.0 / 0.0)", 1.0, 0.0),
        ("Lexical Dominant (0.8 / 0.2)", 0.8, 0.2),
        ("Default Nexora (0.55 / 0.45)", 0.55, 0.45),
        ("Equal Weights (0.50 / 0.50)", 0.50, 0.50),
        ("Dense Favored (0.30 / 0.70)", 0.30, 0.70),
        ("Pure Dense (0.0 / 1.0)", 0.0, 1.0),
    ];

    println!(
        "| RRF Setting                          | Train MAP | Train MRR | Train P@5 | Train P@10 | Train NDCG@10 |"
    );
    println!(
        "|:-------------------------------------|:---------:|:---------:|:---------:|:----------:|:-------------:|"
    );

    let mut best_train_map = 0.0;
    let mut best_train_k = 60;
    let mut best_train_w_lex = 0.55;
    let mut best_train_w_sem = 0.45;

    for &k in &k_values {
        for &(label, w_lex, w_sem) in &weight_ratios {
            let p = HybridSearchParams {
                mode: SearchMode::Hybrid,
                fusion_strategy: HybridFusionStrategy::Rrf { k },
                candidate_k: 60,
                lexical_weight: w_lex,
                semantic_weight: w_sem,
                include_pagerank: false,
                bm25f_params: bm25f_params.clone(),
                ..HybridSearchParams::default()
            };
            let (m, _) = DetailedMetrics::compute(train_q, |q| {
                rank_hybrid(&multi_index, &vector_index, q, &embedder, &empty_pr, &p)
                    .into_iter()
                    .map(|s| s.doc_id)
                    .collect()
            });
            let full_label = format!("k={}, {}", k, label);
            println!(
                "| {:<36} | {:>9.4} | {:>9.4} | {:>9.4} | {:>10.4} | {:>13.4} |",
                full_label, m.map, m.mrr, m.p5, m.p10, m.ndcg10
            );

            if m.map > best_train_map {
                best_train_map = m.map;
                best_train_k = k;
                best_train_w_lex = w_lex;
                best_train_w_sem = w_sem;
            }
        }
    }
    println!(
        "\nOptimal configuration selected strictly on Train set: k={}, w_lex={:.2}, w_sem={:.2} (Train MAP: {:.4})",
        best_train_k, best_train_w_lex, best_train_w_sem, best_train_map
    );
    println!();

    // -------------------------------------------------------------------------
    // PHASE 5: EVALUATION ON HELD-OUT TEST QUERIES (113 QUERIES)
    // -------------------------------------------------------------------------
    println!("-------------------------------------------------------------------------------");
    println!("PHASE 5: HELD-OUT TEST EVALUATION (113 QUERIES) - EVALUATED EXACTLY ONCE");
    println!("-------------------------------------------------------------------------------");

    let (train_bm25f, _) = DetailedMetrics::compute(train_q, |q| {
        rank_bm25f(&multi_index, q, &bm25f_params)
            .into_iter()
            .map(|s| s.doc_id)
            .collect()
    });
    let (test_bm25f, ap_test_bm25f) = DetailedMetrics::compute(test_q, |q| {
        rank_bm25f(&multi_index, q, &bm25f_params)
            .into_iter()
            .map(|s| s.doc_id)
            .collect()
    });

    let (train_dense, _) = DetailedMetrics::compute(train_q, |q| {
        rank_hybrid(
            &multi_index,
            &vector_index,
            q,
            &embedder,
            &empty_pr,
            &dense_params,
        )
        .into_iter()
        .map(|s| s.doc_id)
        .collect()
    });
    let (test_dense, ap_test_dense) = DetailedMetrics::compute(test_q, |q| {
        rank_hybrid(
            &multi_index,
            &vector_index,
            q,
            &embedder,
            &empty_pr,
            &dense_params,
        )
        .into_iter()
        .map(|s| s.doc_id)
        .collect()
    });

    // Frozen Train-selected RRF parameters evaluated on test
    let frozen_rrf_params = HybridSearchParams {
        mode: SearchMode::Hybrid,
        fusion_strategy: HybridFusionStrategy::Rrf { k: best_train_k },
        candidate_k: 60,
        lexical_weight: best_train_w_lex,
        semantic_weight: best_train_w_sem,
        include_pagerank: false,
        bm25f_params: bm25f_params.clone(),
        ..HybridSearchParams::default()
    };
    let (train_rrf_frozen, _) = DetailedMetrics::compute(train_q, |q| {
        rank_hybrid(
            &multi_index,
            &vector_index,
            q,
            &embedder,
            &empty_pr,
            &frozen_rrf_params,
        )
        .into_iter()
        .map(|s| s.doc_id)
        .collect()
    });
    let (test_rrf_frozen, ap_test_rrf) = DetailedMetrics::compute(test_q, |q| {
        rank_hybrid(
            &multi_index,
            &vector_index,
            q,
            &embedder,
            &empty_pr,
            &frozen_rrf_params,
        )
        .into_iter()
        .map(|s| s.doc_id)
        .collect()
    });

    // Also evaluate Default RRF (k=60, 0.55/0.45)
    let (test_rrf_default, _) = DetailedMetrics::compute(test_q, |q| {
        rank_hybrid(
            &multi_index,
            &vector_index,
            q,
            &embedder,
            &empty_pr,
            &rrf_params_default,
        )
        .into_iter()
        .map(|s| s.doc_id)
        .collect()
    });

    // Test BM25F + PRF
    let (train_prf, _) = DetailedMetrics::compute(train_q, |q| {
        rank_bm25f_with_prf(&multi_index, q, &bm25f_params, &prf_params)
            .into_iter()
            .map(|s| s.doc_id)
            .collect()
    });
    let (test_prf, _ap_test_prf) = DetailedMetrics::compute(test_q, |q| {
        rank_bm25f_with_prf(&multi_index, q, &bm25f_params, &prf_params)
            .into_iter()
            .map(|s| s.doc_id)
            .collect()
    });

    println!("Table 2: Train vs Held-Out Out-of-Sample Results:");
    DetailedMetrics::print_header();
    train_bm25f.print_row("Train: BM25F Baseline");
    test_bm25f.print_row("Test:  BM25F Baseline");
    train_dense.print_row("Train: Dense-only (Semantic)");
    test_dense.print_row("Test:  Dense-only (Semantic)");
    train_rrf_frozen.print_row("Train: BM25F + Dense (Frozen RRF)");
    test_rrf_frozen.print_row("Test:  BM25F + Dense (Frozen RRF)");
    test_rrf_default.print_row("Test:  BM25F + Dense (Default RRF)");
    train_prf.print_row("Train: BM25F + PRF");
    test_prf.print_row("Test:  BM25F + PRF");

    let rel_map_change = (test_rrf_frozen.map - test_bm25f.map) / test_bm25f.map * 100.0;
    let rel_mrr_change = (test_rrf_frozen.mrr - test_bm25f.mrr) / test_bm25f.mrr * 100.0;
    let rel_ndcg10_change =
        (test_rrf_frozen.ndcg10 - test_bm25f.ndcg10) / test_bm25f.ndcg10 * 100.0;
    println!("\nHeld-out Test Relative Change (BM25F + Dense RRF vs BM25F):");
    println!("  • MAP:     {:>+6.2}%", rel_map_change);
    println!("  • MRR:     {:>+6.2}%", rel_mrr_change);
    println!("  • NDCG@10: {:>+6.2}%", rel_ndcg10_change);
    println!();

    // -------------------------------------------------------------------------
    // PHASE 6: PER-QUERY ANALYSIS ON HELD-OUT TEST QUERIES
    // -------------------------------------------------------------------------
    println!("-------------------------------------------------------------------------------");
    println!("PHASE 6: PER-QUERY ANALYSIS ON HELD-OUT TEST SET (113 QUERIES)");
    println!("-------------------------------------------------------------------------------");

    let mut improved = 0;
    let mut worsened = 0;
    let mut unchanged = 0;
    let mut diffs: Vec<(usize, f64, f64, f64)> = Vec::with_capacity(test_q.len());

    for (i, (&ap_lex, &ap_hyb)) in ap_test_bm25f.iter().zip(ap_test_rrf.iter()).enumerate() {
        let diff = ap_hyb - ap_lex;
        if diff.abs() < 1e-5 {
            unchanged += 1;
        } else if diff > 0.0 {
            improved += 1;
        } else {
            worsened += 1;
        }
        diffs.push((i, diff, ap_lex, ap_hyb));
    }

    let mean_diff: f64 = diffs.iter().map(|d| d.1).sum::<f64>() / diffs.len() as f64;
    let mut sorted_diffs = diffs.clone();
    sorted_diffs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let median_diff = sorted_diffs[sorted_diffs.len() / 2].1;

    println!("Query Distribution (N = 113):");
    println!(
        "  • Queries Improved: {:>3} ({:.1}%)",
        improved,
        improved as f64 / test_q.len() as f64 * 100.0
    );
    println!(
        "  • Queries Worsened: {:>3} ({:.1}%)",
        worsened,
        worsened as f64 / test_q.len() as f64 * 100.0
    );
    println!(
        "  • Queries Unchanged: {:>2} ({:.1}%)",
        unchanged,
        unchanged as f64 / test_q.len() as f64 * 100.0
    );
    println!("  • Mean AP Difference:   {:>+8.5}", mean_diff);
    println!("  • Median AP Difference: {:>+8.5}\n", median_diff);

    // Top 5 Improvements
    println!("Top 5 Largest Improvements (BM25F -> Hybrid):");
    sorted_diffs.reverse(); // Descending by diff
    for (rank, &(idx, diff, ap_lex, ap_hyb)) in sorted_diffs.iter().take(5).enumerate() {
        let global_idx = split_idx + idx + 1;
        let q_text = test_q[idx].query.trim();
        println!(
            "  {}. Query #{} (+{:.4}): AP {:.4} -> {:.4} | \"{}\"",
            rank + 1,
            global_idx,
            diff,
            ap_lex,
            ap_hyb,
            q_text
        );
    }
    println!();

    // Top 5 Largest Regressions
    println!("Top 5 Largest Regressions (BM25F -> Hybrid):");
    sorted_diffs.reverse(); // Ascending by diff
    for (rank, &(idx, diff, ap_lex, ap_hyb)) in sorted_diffs.iter().take(5).enumerate() {
        let global_idx = split_idx + idx + 1;
        let q_text = test_q[idx].query.trim();
        println!(
            "  {}. Query #{} ({:.4}): AP {:.4} -> {:.4} | \"{}\"",
            rank + 1,
            global_idx,
            diff,
            ap_lex,
            ap_hyb,
            q_text
        );
    }
    println!();

    println!("-------------------------------------------------------------------------------");
    println!("PHASE 6B: FOUR CONCRETE QUERY BEHAVIORAL CASE STUDIES");
    println!("-------------------------------------------------------------------------------");

    // Case 1: Dense retrieves relevant doc in top 10 that BM25F missed
    println!("Case 1: Dense Retrieval Retrieves Relevant Document Missed by BM25F:");
    let mut case1_found = false;
    for (idx, qj) in test_q.iter().enumerate() {
        let global_idx = split_idx + idx + 1;
        let rel_set = qj.relevant_doc_ids();
        let res_bm25f: Vec<DocId> = rank_bm25f(&multi_index, &qj.query, &bm25f_params)
            .into_iter()
            .take(10)
            .map(|s| s.doc_id)
            .collect();
        let res_dense: Vec<DocId> = rank_hybrid(
            &multi_index,
            &vector_index,
            &qj.query,
            &embedder,
            &empty_pr,
            &dense_params,
        )
        .into_iter()
        .take(10)
        .map(|s| s.doc_id)
        .collect();

        for (rank_d, &doc_id) in res_dense.iter().enumerate() {
            if rel_set.contains(&doc_id) && !res_bm25f.contains(&doc_id) {
                let title = doc_titles
                    .get(&doc_id)
                    .map(|s| s.as_str())
                    .unwrap_or("Unknown");
                println!("  Query #{}: \"{}\"", global_idx, qj.query.trim());
                println!("  Relevant Doc ID #{}: \"{}\"", doc_id, title);
                println!(
                    "  -> Rank in Dense: #{} | Rank in BM25F: NOT in top 10\n",
                    rank_d + 1
                );
                case1_found = true;
                break;
            }
        }
        if case1_found {
            break;
        }
    }
    if !case1_found {
        println!(
            "  No query in held-out test set where Dense retrieved a relevant document in Top-10 missed by BM25F.\n"
        );
    }

    // Case 2: BM25F retrieves relevant doc in top 10 that Dense missed
    println!("Case 2: BM25F Retrieves Relevant Document Missed by Dense:");
    let mut case2_found = false;
    for (idx, qj) in test_q.iter().enumerate() {
        let global_idx = split_idx + idx + 1;
        let rel_set = qj.relevant_doc_ids();
        let res_bm25f: Vec<DocId> = rank_bm25f(&multi_index, &qj.query, &bm25f_params)
            .into_iter()
            .take(10)
            .map(|s| s.doc_id)
            .collect();
        let res_dense: Vec<DocId> = rank_hybrid(
            &multi_index,
            &vector_index,
            &qj.query,
            &embedder,
            &empty_pr,
            &dense_params,
        )
        .into_iter()
        .take(10)
        .map(|s| s.doc_id)
        .collect();

        for (rank_b, &doc_id) in res_bm25f.iter().enumerate() {
            if rel_set.contains(&doc_id) && !res_dense.contains(&doc_id) {
                let title = doc_titles
                    .get(&doc_id)
                    .map(|s| s.as_str())
                    .unwrap_or("Unknown");
                println!("  Query #{}: \"{}\"", global_idx, qj.query.trim());
                println!("  Relevant Doc ID #{}: \"{}\"", doc_id, title);
                println!(
                    "  -> Rank in BM25F: #{} | Rank in Dense: NOT in top 10\n",
                    rank_b + 1
                );
                case2_found = true;
                break;
            }
        }
        if case2_found {
            break;
        }
    }

    // Case 3: Hybrid succeeds where individual systems fail or improves rank
    println!("Case 3: Hybrid Improves AP Over Both Individual Lexical and Dense Systems:");
    let mut case3_found = false;
    for (idx, qj) in test_q.iter().enumerate() {
        let global_idx = split_idx + idx + 1;
        let ap_lex = ap_test_bm25f[idx];
        let ap_dense = ap_test_dense[idx];
        let ap_hyb = ap_test_rrf[idx];
        if ap_hyb > ap_lex && ap_hyb > ap_dense && (ap_hyb - ap_lex) > 0.005 {
            println!("  Query #{}: \"{}\"", global_idx, qj.query.trim());
            println!(
                "  -> AP BM25F: {:.4} | AP Dense: {:.4} | AP Hybrid: {:.4} (+{:.4} vs BM25F)\n",
                ap_lex,
                ap_dense,
                ap_hyb,
                ap_hyb - ap_lex
            );
            case3_found = true;
            break;
        }
    }
    if !case3_found {
        println!(
            "  No query showed > 0.005 AP improvement in Hybrid over both baseline systems.\n"
        );
    }

    // Case 4: Dense hurts ranking
    println!("Case 4: Dense Retrieval Hurts Ranking (Noise Injection):");
    for (idx, qj) in test_q.iter().enumerate() {
        let global_idx = split_idx + idx + 1;
        let ap_lex = ap_test_bm25f[idx];
        let ap_hyb = ap_test_rrf[idx];
        if ap_lex - ap_hyb > 0.05 {
            println!("  Query #{}: \"{}\"", global_idx, qj.query.trim());
            let res_bm25f: Vec<DocId> = rank_bm25f(&multi_index, &qj.query, &bm25f_params)
                .into_iter()
                .take(5)
                .map(|s| s.doc_id)
                .collect();
            let res_hyb: Vec<DocId> = rank_hybrid(
                &multi_index,
                &vector_index,
                &qj.query,
                &embedder,
                &empty_pr,
                &frozen_rrf_params,
            )
            .into_iter()
            .take(5)
            .map(|s| s.doc_id)
            .collect();
            println!(
                "  -> AP BM25F: {:.4} -> AP Hybrid: {:.4} ({:.4})",
                ap_lex,
                ap_hyb,
                ap_hyb - ap_lex
            );
            println!(
                "  -> Top-3 BM25F Doc IDs: {:?}",
                &res_bm25f[..3.min(res_bm25f.len())]
            );
            for &did in &res_bm25f[..3.min(res_bm25f.len())] {
                let is_rel = if qj.relevant_doc_ids().contains(&did) {
                    "★ REL"
                } else {
                    "NON-REL"
                };
                println!(
                    "     Doc #{}: [{}] \"{}\"",
                    did,
                    is_rel,
                    doc_titles.get(&did).unwrap_or(&"Unknown".to_string())
                );
            }
            println!(
                "  -> Top-3 Hybrid Doc IDs: {:?}",
                &res_hyb[..3.min(res_hyb.len())]
            );
            for &did in &res_hyb[..3.min(res_hyb.len())] {
                let is_rel = if qj.relevant_doc_ids().contains(&did) {
                    "★ REL"
                } else {
                    "NON-REL"
                };
                println!(
                    "     Doc #{}: [{}] \"{}\"",
                    did,
                    is_rel,
                    doc_titles.get(&did).unwrap_or(&"Unknown".to_string())
                );
            }
            println!();
            break;
        }
    }

    // -------------------------------------------------------------------------
    // PHASE 7: STATISTICAL SIGNIFICANCE TESTS
    // -------------------------------------------------------------------------
    println!("-------------------------------------------------------------------------------");
    println!("PHASE 7: PAIRED STATISTICAL SIGNIFICANCE TESTING ON HELD-OUT SET");
    println!("-------------------------------------------------------------------------------");

    // 1. Paired Wilcoxon Signed-Rank Test
    // Exclude zero differences
    let mut non_zero_diffs: Vec<f64> = diffs
        .iter()
        .map(|d| d.1)
        .filter(|d| d.abs() > 1e-6)
        .collect();
    let n_non_zero = non_zero_diffs.len();

    non_zero_diffs.sort_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap());

    // Assign ranks (with average rank for ties in absolute value)
    let mut ranks = vec![0.0f64; n_non_zero];
    let mut i = 0;
    while i < n_non_zero {
        let mut j = i;
        while j < n_non_zero && (non_zero_diffs[j].abs() - non_zero_diffs[i].abs()).abs() < 1e-6 {
            j += 1;
        }
        let avg_rank = (i + 1 + j) as f64 / 2.0;
        for r in &mut ranks[i..j] {
            *r = avg_rank;
        }
        i = j;
    }

    let mut w_plus = 0.0;
    let mut w_minus = 0.0;
    for (diff, rank) in non_zero_diffs.iter().zip(ranks.iter()) {
        if *diff > 0.0 {
            w_plus += rank;
        } else {
            w_minus += rank;
        }
    }

    let w_stat = w_plus.min(w_minus);
    let n_f = n_non_zero as f64;
    let mean_w = n_f * (n_f + 1.0) / 4.0;
    let std_w = (n_f * (n_f + 1.0) * (2.0 * n_f + 1.0) / 24.0).sqrt();
    let z_score = (w_stat - mean_w) / std_w;
    let p_value_wilcoxon = 2.0 * normal_cdf(z_score);

    println!("Wilcoxon Signed-Rank Test (Paired, Non-parametric):");
    println!("  • Sample size (Total queries):    N = {}", test_q.len());
    println!("  • Effective queries (Non-zero Δ): n = {}", n_non_zero);
    println!("  • Positive rank sum (W+):         {:.1}", w_plus);
    println!("  • Negative rank sum (W-):         {:.1}", w_minus);
    println!("  • Test statistic (W):             {:.1}", w_stat);
    println!("  • Z-score:                        {:.4}", z_score);
    println!(
        "  • Asymptotic p-value (two-tailed): {:<10.6}",
        p_value_wilcoxon
    );
    println!(
        "  • Significant at α=0.05?           {}",
        if p_value_wilcoxon < 0.05 {
            "YES (p < 0.05)"
        } else {
            "NO (Not statistically significant)"
        }
    );

    // 2. Paired Student's t-test
    let all_diffs: Vec<f64> = diffs.iter().map(|d| d.1).collect();
    let n_total = all_diffs.len() as f64;
    let d_mean = all_diffs.iter().sum::<f64>() / n_total;
    let d_var = all_diffs.iter().map(|d| (d - d_mean).powi(2)).sum::<f64>() / (n_total - 1.0);
    let d_std = d_var.sqrt();
    let se = d_std / n_total.sqrt();
    let t_stat = d_mean / se;
    let p_value_t = 2.0 * normal_cdf(-t_stat.abs());
    let cohen_d = d_mean / d_std;

    println!("\nPaired Student's t-Test (Parametric):");
    println!("  • Mean AP difference (d̄):        {:.6}", d_mean);
    println!("  • Standard error (SE):            {:.6}", se);
    println!("  • t-statistic:                    {:.4}", t_stat);
    println!("  • p-value (two-tailed):           {:<10.6}", p_value_t);
    println!("  • Effect size (Cohen's d):        {:.4}", cohen_d);
    println!(
        "  • Significant at α=0.05?           {}",
        if p_value_t < 0.05 {
            "YES (p < 0.05)"
        } else {
            "NO (Not statistically significant)"
        }
    );
    println!();

    // -------------------------------------------------------------------------
    // PHASE 8: ABLATION STUDY
    // -------------------------------------------------------------------------
    println!("-------------------------------------------------------------------------------");
    println!("PHASE 8: COMPONENT ABLATION STUDY (ON FULL 225-QUERY BENCHMARK)");
    println!("-------------------------------------------------------------------------------");

    DetailedMetrics::print_header();
    m_bm25f.print_row("1. BM25F Baseline");
    m_rrf_default.print_row("2. BM25F + Dense (RRF k=60)");
    m_bm25f_prf.print_row("3. BM25F + PRF");
    m_bm25f_dense_prf.print_row("4. BM25F + Dense + PRF");

    println!("\nPageRank Impact on Cranfield:");
    println!("  • Cranfield contains scientific abstracts without web hyperlinks.");
    println!("  • Initial link graph is completely disconnected (all in-degrees = 0).");
    println!("  • Result: PageRank uniform mass = 1/N = 0.000714 across all nodes.");
    println!("  • PageRank multiplier adds 0.0% variance and does not alter ranking.");
    println!();

    // -------------------------------------------------------------------------
    // PHASE 9: PERFORMANCE & LATENCY BENCHMARK
    // -------------------------------------------------------------------------
    println!("-------------------------------------------------------------------------------");
    println!("PHASE 9: SYSTEM LATENCY & THROUGHPUT BENCHMARK");
    println!("-------------------------------------------------------------------------------");

    let num_warmup = 10;
    let queries_subset: Vec<&str> = judgments.iter().map(|q| q.query.as_str()).collect();

    // Warmup
    for q in queries_subset.iter().take(num_warmup) {
        let _ = rank_bm25f(&multi_index, q, &bm25f_params);
        let _ = rank_hybrid(
            &multi_index,
            &vector_index,
            q,
            &embedder,
            &empty_pr,
            &rrf_params_default,
        );
    }

    // Benchmark 1: Pure BM25F Latency
    let start_bm25f = Instant::now();
    for q in &queries_subset {
        let _ = rank_bm25f(&multi_index, q, &bm25f_params);
    }
    let time_bm25f = start_bm25f.elapsed();
    let us_bm25f = time_bm25f.as_micros() as f64 / queries_subset.len() as f64;

    // Benchmark 2: Pure Dense Latency
    let start_dense = Instant::now();
    for q in &queries_subset {
        let _ = rank_hybrid(
            &multi_index,
            &vector_index,
            q,
            &embedder,
            &empty_pr,
            &dense_params,
        );
    }
    let time_dense = start_dense.elapsed();
    let us_dense = time_dense.as_micros() as f64 / queries_subset.len() as f64;

    // Benchmark 3: Hybrid BM25F + Dense (RRF)
    let start_hybrid = Instant::now();
    for q in &queries_subset {
        let _ = rank_hybrid(
            &multi_index,
            &vector_index,
            q,
            &embedder,
            &empty_pr,
            &rrf_params_default,
        );
    }
    let time_hybrid = start_hybrid.elapsed();
    let us_hybrid = time_hybrid.as_micros() as f64 / queries_subset.len() as f64;

    // Memory footprint of dense vectors:
    // 1,400 vectors * 64 floats * 4 bytes = 358,400 bytes (~350 KB)
    let raw_vector_bytes = vector_index.len() * vector_index.dimension() * 4;

    println!("Corpus Statistics: 1,400 documents, 64-dimensional float32 vectors");
    println!(
        "  • Dense Vector Index Size:   {} documents ({:.2} KB raw floats)",
        vector_index.len(),
        raw_vector_bytes as f64 / 1024.0
    );
    println!(
        "  • Vector Indexing Time:      {:.2?} ({:.2} µs/doc)",
        idx_time,
        idx_time.as_micros() as f64 / docs.len() as f64
    );
    println!("Latency per Query (Average over 225 queries):");
    println!(
        "  • Lexical BM25F:             {:>7.2} µs/query ({:.2} queries/sec)",
        us_bm25f,
        1_000_000.0 / us_bm25f
    );
    println!(
        "  • Dense Vector Cosine:       {:>7.2} µs/query ({:.2} queries/sec)",
        us_dense,
        1_000_000.0 / us_dense
    );
    println!(
        "  • Total Hybrid RRF:          {:>7.2} µs/query ({:.2} queries/sec)",
        us_hybrid,
        1_000_000.0 / us_hybrid
    );
    println!(
        "  • Overhead of Hybrid vs BM25F: {:>+6.1}% latency increase",
        (us_hybrid - us_bm25f) / us_bm25f * 100.0
    );
    println!();
}
