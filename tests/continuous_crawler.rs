use std::sync::{Arc, RwLock};
use std::thread;
use std::time::Duration;

use nexora::{
    ContinuousCrawler, ContinuousCrawlerConfig, MockFetcher, SearchEngineSink, SearchEngineState,
};

#[test]
fn test_continuous_crawler_concurrent_searching_and_indexing() {
    let mut fetcher = MockFetcher::new();

    // 4 Interlinked pages with rich text content
    fetcher.add_page(
        "https://ai.example.org/index",
        r#"<html>
            <head><title>Artificial Intelligence and Neural Networks</title></head>
            <body>
                <p>Deep learning transforms modern computer vision and natural language processing.</p>
                <a href="/nlp">Natural Language Processing</a>
                <a href="/robotics">Autonomous Robotics</a>
            </body>
        </html>"#,
    );
    fetcher.add_page(
        "https://ai.example.org/nlp",
        r#"<html>
            <head><title>Natural Language Processing and Transformers</title></head>
            <body>
                <p>Large language models understand semantics and token sequences.</p>
                <a href="/index">AI Overview</a>
                <a href="/robotics">Robotics Integration</a>
            </body>
        </html>"#,
    );
    fetcher.add_page(
        "https://ai.example.org/robotics",
        r#"<html>
            <head><title>Autonomous Robotics and Sensor Fusion</title></head>
            <body>
                <p>Mobile robots use SLAM algorithms and neural trajectory planning.</p>
                <a href="/index">Back to AI</a>
            </body>
        </html>"#,
    );

    let state = Arc::new(RwLock::new(SearchEngineState::default_demo()));
    let initial_docs = state.read().unwrap().index.total_documents();

    let sink = Arc::new(SearchEngineSink(Arc::clone(&state)));
    let config = ContinuousCrawlerConfig {
        max_pages: 10,
        politeness_delay: Duration::from_millis(5),
        respect_robots_txt: false,
        user_agent: "NexoraBot/Test".to_string(),
        pagerank_recompute_interval: 2,
    };

    let crawler = ContinuousCrawler::new(Arc::new(fetcher), sink, config);

    // 1. Launch background crawler
    crawler.start(&["https://ai.example.org/index"], Some(3), None);

    // 2. Concurrently run search queries in parallel without interruption
    let reader_state = Arc::clone(&state);
    let search_thread = thread::spawn(move || {
        let mut searches_completed = 0;
        for _ in 0..30 {
            {
                let lock = reader_state.read().unwrap();
                let _ = lock.trie.suggest("neur", 5);
                let _ = lock.spell_checker.suggest("inteligence", 2);
            }
            thread::sleep(Duration::from_millis(5));
            searches_completed += 1;
        }
        searches_completed
    });

    let searches = search_thread
        .join()
        .expect("Search thread should not panic");
    assert_eq!(searches, 30);

    // Wait briefly for crawler to complete 3 pages
    let mut attempts = 0;
    while crawler.is_running() && attempts < 50 {
        thread::sleep(Duration::from_millis(15));
        attempts += 1;
    }

    let tele = crawler.telemetry();
    assert_eq!(tele.pages_visited, 3);

    // 3. Verify state was incrementally updated
    let final_state = state.read().unwrap();
    assert_eq!(final_state.index.total_documents(), initial_docs + 3);

    // Verify search finds newly ingested documents
    let pr_map = final_state
        .doc_metadata
        .iter()
        .map(|(&id, m)| (id, m.pagerank))
        .collect();
    let hybrid = nexora::HybridBM25FParams::default();
    let results = nexora::rank_bm25f_with_pagerank(
        &final_state.multi_index,
        "transformers",
        &pr_map,
        &hybrid,
    );
    assert!(!results.is_empty());
    let best_doc = final_state.doc_metadata.get(&results[0].doc_id).unwrap();
    assert!(best_doc.title.contains("Transformers"));

    // Verify autocomplete trie contains newly indexed terms
    let suggestions = final_state.trie.suggest("robot", 5);
    assert!(!suggestions.is_empty());
}

#[test]
fn test_continuous_crawler_lifecycle_control() {
    let mut fetcher = MockFetcher::new();
    for i in 1..=10 {
        fetcher.add_page(
            &format!("https://example.org/page{}", i),
            &format!(
                r#"<html><head><title>Page {}</title></head><body><p>Content {}</p><a href="/page{}">Next</a></body></html>"#,
                i,
                i,
                i + 1
            ),
        );
    }

    let state = Arc::new(RwLock::new(SearchEngineState::default_demo()));
    let sink = Arc::new(SearchEngineSink(Arc::clone(&state)));
    let config = ContinuousCrawlerConfig {
        max_pages: 10,
        politeness_delay: Duration::from_millis(25),
        respect_robots_txt: false,
        user_agent: "NexoraBot/Lifecycle".to_string(),
        pagerank_recompute_interval: 2,
    };

    let crawler = ContinuousCrawler::new(Arc::new(fetcher), sink, config);

    // Start
    crawler.start(&["https://example.org/page1"], Some(10), None);
    thread::sleep(Duration::from_millis(30));

    // Pause
    crawler.pause();
    assert!(crawler.is_paused());
    let pages_when_paused = crawler.telemetry().pages_visited;

    // Wait and ensure no more pages are crawled while paused
    thread::sleep(Duration::from_millis(60));
    assert_eq!(crawler.telemetry().pages_visited, pages_when_paused);

    // Resume
    crawler.resume();
    assert!(!crawler.is_paused());

    // Stop
    crawler.stop();
    thread::sleep(Duration::from_millis(40));
    assert_eq!(crawler.telemetry().status, nexora::CrawlerStatus::Stopped);
}

#[test]
fn test_continuous_crawler_robots_txt_disallow() {
    let mut fetcher = MockFetcher::new();
    fetcher.add_page(
        "https://secure.org/robots.txt",
        "User-agent: *\nDisallow: /admin\nDisallow: /private\n",
    );
    fetcher.add_page(
        "https://secure.org/public",
        r#"<html><head><title>Public</title></head><body><a href="/admin/secret">Secret</a></body></html>"#,
    );
    fetcher.add_page(
        "https://secure.org/admin/secret",
        r#"<html><head><title>Secret Area</title></head><body>Classified</body></html>"#,
    );

    let state = Arc::new(RwLock::new(SearchEngineState::default_demo()));
    let sink = Arc::new(SearchEngineSink(Arc::clone(&state)));
    let config = ContinuousCrawlerConfig {
        max_pages: 5,
        politeness_delay: Duration::from_millis(10),
        respect_robots_txt: true,
        user_agent: "NexoraBot".to_string(),
        pagerank_recompute_interval: 2,
    };

    let crawler = ContinuousCrawler::new(Arc::new(fetcher), sink, config);
    crawler.start(&["https://secure.org/public"], Some(3), None);

    let mut attempts = 0;
    while crawler.is_running() && attempts < 50 {
        thread::sleep(Duration::from_millis(20));
        attempts += 1;
    }

    let tele = crawler.telemetry();
    assert_eq!(tele.pages_visited, 1);
    assert_eq!(tele.pages_disallowed, 1);
}
