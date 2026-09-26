use std::time::Instant;

use nexora::{
    compute_pagerank, CrawlConfig, Crawler, DocId, InvertedIndex, MockFetcher, PageRankParams,
    UrlFrontier, WebGraph,
};

#[test]
fn test_multi_domain_crawl_and_graph_benchmark() {
    println!("\n=== Starting Multi-Domain Web Crawl & PageRank Benchmark ===");
    let start_time = Instant::now();

    let mut fetcher = MockFetcher::new();

    // 1. Domain: authority.com with robots.txt
    fetcher.add_page(
        "https://authority.com/robots.txt",
        "User-agent: *\nDisallow: /admin/\nDisallow: /private/\n",
    );
    fetcher.add_page(
        "https://authority.com/index.html",
        r#"<html><head><title>Authority Central Portal</title></head>
           <body>
             <h1>Welcome to Authority Portal</h1>
             <p>The definitive knowledge base for search systems and data engineering.</p>
             <a href="/docs/search.html">Search Systems Guide</a>
             <a href="/docs/pagerank.html">PageRank Theory</a>
             <a href="/admin/secret.html">Admin Area</a>
           </body></html>"#,
    );
    fetcher.add_page(
        "https://authority.com/docs/search.html",
        r#"<html><head><title>Search Systems Guide</title></head>
           <body>
             <h1>Search Engine Architecture</h1>
             <p>Inverted index, postings lists, and tokenization techniques.</p>
             <a href="/docs/pagerank.html">PageRank Theory</a>
             <a href="https://tech-forum.io/thread/101">Community Discussion</a>
           </body></html>"#,
    );
    fetcher.add_page(
        "https://authority.com/docs/pagerank.html",
        r#"<html><head><title>PageRank Theory and Math</title></head>
           <body>
             <h1>PageRank Power Iteration</h1>
             <p>Redistributing dangling node mass and dampening factor alpha.</p>
             <a href="/index.html">Back Home</a>
           </body></html>"#,
    );
    fetcher.add_page(
        "https://authority.com/admin/secret.html",
        "<html><body><h1>Admin Secret (Should be blocked by robots.txt)</h1></body></html>",
    );

    // 2. Domain: blog-news.org (frequently cites authority.com)
    fetcher.add_page(
        "https://blog-news.org/robots.txt",
        "User-agent: *\nAllow: /\n",
    );
    fetcher.add_page(
        "https://blog-news.org/index.html",
        r#"<html><head><title>Tech News Daily</title></head>
           <body>
             <h1>Daily Engineering Headlines</h1>
             <a href="/post/search-breakthroughs.html">Search Breakthroughs</a>
             <a href="/post/distributed-crawlers.html">Distributed Crawlers</a>
           </body></html>"#,
    );
    fetcher.add_page(
        "https://blog-news.org/post/search-breakthroughs.html",
        r#"<html><head><title>Breakthroughs in Search Ranking</title></head>
           <body>
             <p>As documented by <a href="https://authority.com/index.html">Authority Central</a>, BM25 remains strong.</p>
             <a href="https://tech-forum.io/thread/101">Discussion on Tech Forum</a>
           </body></html>"#,
    );
    fetcher.add_page(
        "https://blog-news.org/post/distributed-crawlers.html",
        r#"<html><head><title>Distributed Crawlers at Scale</title></head>
           <body>
             <p>Building politeness queues and URL frontiers.</p>
             <a href="https://authority.com/docs/search.html">Read Architecture Docs</a>
             <a href="https://dangling-archive.net/archive1.html">Historical Crawl Archive</a>
           </body></html>"#,
    );

    // 3. Domain: tech-forum.io
    fetcher.add_page(
        "https://tech-forum.io/robots.txt",
        "User-agent: *\nAllow: /\n",
    );
    fetcher.add_page(
        "https://tech-forum.io/thread/101",
        r#"<html><head><title>Forum Thread: How to implement PageRank</title></head>
           <body>
             <h1>Forum Discussion #101</h1>
             <p>Does anyone have good references on power iteration?</p>
             <p>Check out <a href="https://authority.com/docs/pagerank.html">Authority PageRank Guide</a> and <a href="https://authority.com/index.html">Authority Home</a>.</p>
             <a href="/thread/102">Next Thread</a>
           </body></html>"#,
    );
    fetcher.add_page(
        "https://tech-forum.io/thread/102",
        r#"<html><head><title>Forum Thread: Rust vs C++</title></head>
           <body>
             <h1>Forum Discussion #102</h1>
             <p>Rust memory safety guarantees high-performance zero-cost abstractions.</p>
             <a href="https://authority.com/index.html">Authority Portal</a>
           </body></html>"#,
    );

    // 4. Domain: dangling-archive.net (terminal node with 0 outgoing links)
    fetcher.add_page(
        "https://dangling-archive.net/robots.txt",
        "User-agent: *\nAllow: /\n",
    );
    fetcher.add_page(
        "https://dangling-archive.net/archive1.html",
        r#"<html><head><title>Historical Crawl Archive 2026</title></head>
           <body>
             <p>Static archive record without outgoing links.</p>
           </body></html>"#,
    );

    // Setup Crawler with multi-domain seeds (fast 10ms politeness for simulated benchmark)
    let mut frontier = UrlFrontier::new(std::time::Duration::from_millis(10));
    frontier.push("https://authority.com/index.html");
    frontier.push("https://blog-news.org/index.html");

    let config = CrawlConfig {
        max_pages: 20,
        max_wait: std::time::Duration::from_millis(500),
        respect_robots_txt: true,
        user_agent: "NexoraBenchmark/1.0".to_string(),
    };

    let mut crawler = Crawler::new(frontier, fetcher, config);
    let mut index = InvertedIndex::new();

    let summary = crawler.crawl(&mut index);
    let crawl_duration = start_time.elapsed();

    println!("✔ Crawl completed in {:.2?}", crawl_duration);
    println!("  - Pages successfully visited: {}", summary.pages_visited);
    println!("  - Pages blocked by robots.txt: {}", summary.pages_disallowed);
    println!("  - Hyperlinks discovered:      {}", summary.links_discovered);
    println!("  - Indexed vocabulary size:    {}", index.vocabulary_size());

    // Verify robots.txt blocked the /admin/ path
    assert!(
        summary.pages_disallowed >= 1,
        "robots.txt must disallow /admin/secret.html"
    );
    for doc in &summary.documents {
        assert!(
            !doc.url.contains("/admin/"),
            "Admin URL should never have been crawled: {}",
            doc.url
        );
    }

    // Construct WebGraph from crawled pages
    let graph_start = Instant::now();
    let graph = WebGraph::from_crawled_documents(&summary.documents);
    let graph_build_duration = graph_start.elapsed();

    println!("✔ WebGraph constructed in {:.2?}", graph_build_duration);
    println!("  - Graph Node Count: {}", graph.len());
    println!("  - Graph Edge Count: {}", graph.edge_count());

    assert!(graph.len() >= 6, "Must have indexed at least 6 unique pages");
    assert!(graph.edge_count() >= 5, "Must have extracted multiple inter-page links");

    // Compute PageRank with power iteration
    let pr_start = Instant::now();
    let pr_params = PageRankParams {
        damping: 0.85,
        max_iterations: 100,
        tolerance: 1e-6,
    };
    let pr_scores = compute_pagerank(&graph, &pr_params);
    let pr_duration = pr_start.elapsed();

    println!("✔ PageRank converged in {:.2?}", pr_duration);

    // Verify total probability mass conservation: sum(PR) == 1.0 within epsilon
    let total_mass: f64 = pr_scores.values().sum();
    println!("  - Total PageRank Mass: {:.6} (conservation check)", total_mass);
    assert!(
        (total_mass - 1.0).abs() < 1e-4,
        "Total PageRank probability mass must sum to 1.0"
    );

    // Sort documents by PageRank
    let mut ranked_docs: Vec<(&String, &DocId, f64)> = summary
        .documents
        .iter()
        .map(|d| (&d.url, &d.doc_id, pr_scores.get(&d.doc_id).copied().unwrap_or(0.0)))
        .collect();
    ranked_docs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

    println!("\nTop Authority Pages in Link Graph:");
    for (rank, (url, doc_id, score)) in ranked_docs.iter().enumerate().take(5) {
        println!("  {}. [Doc {}] PR: {:.5} -> {}", rank + 1, doc_id, score, url);
    }

    // The authority.com home page should have higher PageRank than the dangling archive
    let authority_home = summary
        .documents
        .iter()
        .find(|d| d.url == "https://authority.com/index.html")
        .expect("Authority home page must be crawled");
    let dangling_doc = summary
        .documents
        .iter()
        .find(|d| d.url == "https://dangling-archive.net/archive1.html");

    let auth_pr = pr_scores[&authority_home.doc_id];
    println!("\nVerification: Authority Home PR ({:.5})", auth_pr);

    if let Some(dangling) = dangling_doc {
        let dangling_pr = pr_scores[&dangling.doc_id];
        println!("Verification: Dangling Archive PR ({:.5})", dangling_pr);
        assert!(
            auth_pr > dangling_pr,
            "Authority central hub must have higher PageRank than dangling page"
        );
    }

    println!("=== Multi-Domain Benchmark Passed Successfully! ===\n");
}
