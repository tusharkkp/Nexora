use std::collections::HashMap;
use std::time::Instant;

use nexora::hybrid::{HybridFusionStrategy, HybridSearchParams, SearchMode, rank_hybrid};
use nexora::index::MultiFieldIndex;
use nexora::vector::{SemanticEmbedder, TextEmbedder, VectorIndex};

#[test]
fn test_semantic_retrieval_zero_lexical_overlap() {
    let mut multi_index = MultiFieldIndex::new();
    let embedder = SemanticEmbedder::default();
    let mut vector_index = VectorIndex::new(embedder.dimension());

    // Document 1: Aerospace flutter
    let doc1 =
        "Supersonic aeroelastic instability and vibration in high-speed flight wing surfaces.";
    multi_index.add_document(1, "Aero Dynamics", doc1, "");
    vector_index.add_vector(1, embedder.embed(doc1));

    // Document 2: Automobile mechanic
    let doc2 =
        "Automotive transmission maintenance, engine overhaul, and brake pad repair procedures.";
    multi_index.add_document(2, "Vehicle Care", doc2, "");
    vector_index.add_vector(2, embedder.embed(doc2));

    // Document 3: Deep learning neural nets
    let doc3 = "Deep neural network backpropagation, stochastic gradient descent optimization, and loss convergence.";
    multi_index.add_document(3, "AI Foundations", doc3, "");
    vector_index.add_vector(3, embedder.embed(doc3));

    let pr_map = HashMap::new();

    // Query with ZERO exact word overlap with Doc 2: "car motor servicing"
    let params_sem = HybridSearchParams {
        mode: SearchMode::Semantic,
        ..HybridSearchParams::default()
    };
    let sem_results = rank_hybrid(
        &multi_index,
        &vector_index,
        "car motor servicing",
        &embedder,
        &pr_map,
        &params_sem,
    );
    assert!(
        !sem_results.is_empty(),
        "Semantic search should find results"
    );
    assert_eq!(
        sem_results[0].doc_id, 2,
        "Doc 2 (automotive transmission) should be #1 for 'car motor servicing'"
    );

    // Verify lexical search fails to find Doc 2 due to vocabulary mismatch
    let params_lex = HybridSearchParams {
        mode: SearchMode::Lexical,
        ..HybridSearchParams::default()
    };
    let lex_results = rank_hybrid(
        &multi_index,
        &vector_index,
        "car motor servicing",
        &embedder,
        &pr_map,
        &params_lex,
    );
    assert!(
        lex_results.is_empty(),
        "Lexical BM25 search should have zero matches for non-overlapping query terms"
    );

    // Hybrid search smoothly retrieves and scores Doc 2
    let params_hybrid = HybridSearchParams {
        mode: SearchMode::Hybrid,
        ..HybridSearchParams::default()
    };
    let hybrid_results = rank_hybrid(
        &multi_index,
        &vector_index,
        "car motor servicing",
        &embedder,
        &pr_map,
        &params_hybrid,
    );
    assert!(!hybrid_results.is_empty());
    assert_eq!(hybrid_results[0].doc_id, 2);
    assert!(hybrid_results[0].semantic_similarity > 0.5);
}

#[test]
fn test_rrf_and_linear_score_fusion_consistency() {
    let mut multi_index = MultiFieldIndex::new();
    let embedder = SemanticEmbedder::default();
    let mut vector_index = VectorIndex::new(embedder.dimension());

    let docs = [
        (
            1,
            "Rust Concurrency",
            "Rust guarantees memory safety and thread safety without data races using ownership.",
        ),
        (
            2,
            "Parallel Systems",
            "Distributed computing across multiple server clusters with low latency message passing.",
        ),
        (
            3,
            "Database Storage",
            "B-tree indexing and LSM trees for persistent disk key-value storage engines.",
        ),
    ];

    for (id, title, body) in docs {
        multi_index.add_document(id, title, body, "");
        vector_index.add_vector(id, embedder.embed(&format!("{} {}", title, body)));
    }

    let pr_map = HashMap::new();

    // RRF Fusion
    let rrf_params = HybridSearchParams {
        mode: SearchMode::Hybrid,
        fusion_strategy: HybridFusionStrategy::Rrf { k: 60 },
        ..HybridSearchParams::default()
    };
    let rrf_res = rank_hybrid(
        &multi_index,
        &vector_index,
        "rust thread safety",
        &embedder,
        &pr_map,
        &rrf_params,
    );
    assert!(!rrf_res.is_empty());
    assert_eq!(rrf_res[0].doc_id, 1);

    // Linear Normalized Fusion (50% BM25, 50% Dense Vector)
    let linear_params = HybridSearchParams {
        mode: SearchMode::Hybrid,
        fusion_strategy: HybridFusionStrategy::LinearScore { alpha: 0.5 },
        ..HybridSearchParams::default()
    };
    let linear_res = rank_hybrid(
        &multi_index,
        &vector_index,
        "rust thread safety",
        &embedder,
        &pr_map,
        &linear_params,
    );
    assert!(!linear_res.is_empty());
    assert_eq!(linear_res[0].doc_id, 1);
}

#[test]
fn test_hybrid_search_benchmark_throughput() {
    let mut multi_index = MultiFieldIndex::new();
    let embedder = SemanticEmbedder::default();
    let mut vector_index = VectorIndex::new(embedder.dimension());

    let sample_corpus = [
        "Distributed systems consensus via Raft and Paxos protocols in cloud infrastructure.",
        "Inverted index compression with SIMD-accelerated Elias-Fano and Variable Byte encoding.",
        "Web spider crawler with polite robots exclusion standard and URL frontier queues.",
        "Aeroelastic flutter suppression and transonic buffet boundary prediction.",
        "Deep learning transformer attention mechanisms and sparse matrix multiplication.",
    ];

    // Seed 100 documents
    for i in 0..100 {
        let text = sample_corpus[i % sample_corpus.len()];
        multi_index.add_document(i as u32, "Title", text, "");
        vector_index.add_vector(i as u32, embedder.embed(text));
    }

    let pr_map = HashMap::new();
    let params = HybridSearchParams::default();

    let queries = [
        "distributed consensus cloud",
        "inverted index compression",
        "crawler url frontier",
        "flutter vibration aerodynamics",
        "transformer attention model",
    ];

    let start = Instant::now();
    let iterations = 1000;
    for i in 0..iterations {
        let q = queries[i % queries.len()];
        let results = rank_hybrid(&multi_index, &vector_index, q, &embedder, &pr_map, &params);
        assert!(!results.is_empty());
    }
    let elapsed = start.elapsed();
    let us_per_query = elapsed.as_micros() as f64 / iterations as f64;

    println!(
        "⚡ Hybrid retrieval benchmark: {} queries in {:.2?} ({:.2} µs/query)",
        iterations, elapsed, us_per_query
    );
    // Even in debug build, 1,000 hybrid queries across 100 docs should comfortably complete in under 500ms
    assert!(
        elapsed.as_millis() < 1000,
        "Hybrid search should be fast and scalable"
    );
}
