use nexora::{
    BM25FParams, BM25Params, InvertedIndex, MultiFieldIndex, QueryJudgment, evaluate_bm25f,
    rank_bm25, rank_bm25f,
};

#[test]
fn test_bm25f_vs_bm25_multi_field_benchmark() {
    println!("\n=======================================================");
    println!("   🏆 BM25F MULTI-FIELD WEIGHTED SEARCH BENCHMARK");
    println!("   Evaluating Title, Body, and Anchor Text Relevance");
    println!("=======================================================\n");

    // Corpus scenario:
    // Doc 0: Official "Rust Programming Language" guide (Title: "Rust Programming Language", Body: "Official systems guide")
    // Doc 1: Gardening forum where "rust" is mentioned twice in body, Title: "Gardening & Plant Disease"
    // Doc 2: Wikipedia page on Information Retrieval (Title: "Information Retrieval", Body: "Science of searching documents", Inbound Anchor: "modern web search engine")
    // Doc 3: Blog post casually mentioning "search engine" once in 500 words of body, Title: "My Daily Log"

    let mut flat_index = InvertedIndex::new();
    let mut multi_index = MultiFieldIndex::new();

    // Doc 0: Focused document with "Rust" in title
    flat_index.add_document(0, "Rust Programming Language Official systems guide");
    multi_index.add_document(0, "Rust Programming Language", "Official systems guide", "");

    // Doc 1: Mentions "rust" twice in body, but unrelated title
    let doc1_body = "Gardening tools develop metal rust when left in the rain. Iron rust stains plant pots. To prevent damage, clean and oil your tools thoroughly each autumn.";
    flat_index.add_document(1, &format!("Gardening & Plant Disease {}", doc1_body));
    multi_index.add_document(1, "Gardening & Plant Disease", doc1_body, "");

    // Doc 2: Third-party inbound anchor endorsement
    flat_index.add_document(2, "Information Retrieval Science of searching documents");
    multi_index.add_document(
        2,
        "Information Retrieval",
        "Science of searching documents",
        "",
    );
    multi_index.append_anchor_text(2, "modern web search engine");

    // Doc 3: Long document with incidental mention of search engine
    let doc3_body = format!(
        "Today I walked my dog. I used a search engine to find lunch. {}",
        "irrelevant filler text ".repeat(30)
    );
    flat_index.add_document(3, &format!("My Daily Log {}", doc3_body));
    multi_index.add_document(3, "My Daily Log", &doc3_body, "");

    let bm25_params = BM25Params::default();
    let bm25f_params = BM25FParams::default();

    // Query 1: "rust"
    // Ground truth: Doc 0 is the primary subject (grade 3), Doc 1 is incidental topic (grade 0)
    let q1_bm25 = rank_bm25(&flat_index, "rust", &bm25_params);
    let q1_bm25f = rank_bm25f(&multi_index, "rust", &bm25f_params);

    println!("Query 1: 'rust'");
    println!(
        "  Flat BM25 #1: Doc {} (score: {:.4})",
        q1_bm25[0].doc_id, q1_bm25[0].score
    );
    println!(
        "  BM25F     #1: Doc {} (score: {:.4})",
        q1_bm25f[0].doc_id, q1_bm25f[0].score
    );

    // In BM25F, Doc 0 decisively wins because Title weight (4.0) boosts title matches over body mentions!
    assert_eq!(q1_bm25f[0].doc_id, 0);

    // Query 2: "search engine"
    // Ground truth: Doc 2 is endorsed by anchor text "modern web search engine"
    let q2_bm25f = rank_bm25f(&multi_index, "search engine", &bm25f_params);
    println!("\nQuery 2: 'search engine'");
    println!(
        "  BM25F #1: Doc {} (score: {:.4})",
        q2_bm25f[0].doc_id, q2_bm25f[0].score
    );
    assert_eq!(q2_bm25f[0].doc_id, 2);

    // Benchmark metrics over test suite
    let benchmark = vec![
        QueryJudgment::graded("rust", &[(0, 3), (1, 0)]),
        QueryJudgment::graded("search engine", &[(2, 3), (3, 0)]),
    ];

    let metrics_bm25f = evaluate_bm25f(&multi_index, &benchmark, &bm25f_params, 1);
    println!("\n--- Benchmark Suite Evaluation (k = 1) ---");
    println!("  Mean Precision@1: {:.2}", metrics_bm25f.mean_precision);
    println!(
        "  Mean MRR:         {:.2}",
        metrics_bm25f.mean_reciprocal_rank
    );
    println!("  Mean NDCG@1:      {:.2}", metrics_bm25f.mean_ndcg);

    assert_eq!(metrics_bm25f.mean_precision, 1.0);
    assert_eq!(metrics_bm25f.mean_reciprocal_rank, 1.0);
    assert_eq!(metrics_bm25f.mean_ndcg, 1.0);
    println!("\n✔ BM25F multi-field ranking successfully passed all relevance criteria!\n");
}
