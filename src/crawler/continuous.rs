use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::crawler::engine::CrawledDocument;
use crate::crawler::fetcher::PageFetcher;
use crate::crawler::frontier::UrlFrontier;
use crate::crawler::html::extract_page;
use crate::crawler::robots::RobotsTxt;
use crate::crawler::url::parse_url;
use crate::graph::{PageRankParams, WebGraph, compute_pagerank};
use crate::index::DocId;

/// Current operational state of the background crawler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrawlerStatus {
    Idle,
    Running,
    Paused,
    Stopped,
}

impl CrawlerStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Running => "running",
            Self::Paused => "paused",
            Self::Stopped => "stopped",
        }
    }
}

/// Telemetry record for a crawled or visited page.
#[derive(Debug, Clone)]
pub struct CrawledPageRecord {
    pub doc_id: DocId,
    pub url: String,
    pub title: String,
    pub outgoing_links_count: usize,
    pub timestamp_ms: u64,
    pub status: String,
}

/// Live telemetry and metrics for monitoring the continuous crawler.
#[derive(Debug, Clone)]
pub struct CrawlerTelemetry {
    pub status: CrawlerStatus,
    pub pages_visited: usize,
    pub pages_failed: usize,
    pub pages_disallowed: usize,
    pub links_discovered: usize,
    pub queue_size: usize,
    pub current_url: Option<String>,
    pub current_domain: Option<String>,
    pub recent_pages: VecDeque<CrawledPageRecord>,
    pub last_pagerank_update: Option<u64>,
}

impl Default for CrawlerTelemetry {
    fn default() -> Self {
        Self {
            status: CrawlerStatus::Idle,
            pages_visited: 0,
            pages_failed: 0,
            pages_disallowed: 0,
            links_discovered: 0,
            queue_size: 0,
            current_url: None,
            current_domain: None,
            recent_pages: VecDeque::with_capacity(30),
            last_pagerank_update: None,
        }
    }
}

impl CrawlerTelemetry {
    /// Renders a JSON representation of the current telemetry state.
    pub fn to_json(&self) -> String {
        let current_url_str = match &self.current_url {
            Some(u) => format!("\"{}\"", escape_json(u)),
            None => "null".to_string(),
        };
        let current_dom_str = match &self.current_domain {
            Some(d) => format!("\"{}\"", escape_json(d)),
            None => "null".to_string(),
        };
        let last_pr_str = match self.last_pagerank_update {
            Some(t) => t.to_string(),
            None => "null".to_string(),
        };

        let mut recent_json = String::new();
        for (i, p) in self.recent_pages.iter().enumerate() {
            if i > 0 {
                recent_json.push(',');
            }
            recent_json.push_str(&format!(
                r#"{{"doc_id":{},"url":"{}","title":"{}","outgoing_links_count":{},"timestamp_ms":{},"status":"{}"}}"#,
                p.doc_id,
                escape_json(&p.url),
                escape_json(&p.title),
                p.outgoing_links_count,
                p.timestamp_ms,
                escape_json(&p.status)
            ));
        }

        format!(
            r#"{{"status":"{}","pages_visited":{},"pages_failed":{},"pages_disallowed":{},"links_discovered":{},"queue_size":{},"current_url":{},"current_domain":{},"last_pagerank_update":{},"recent_pages":[{}]}}"#,
            self.status.as_str(),
            self.pages_visited,
            self.pages_failed,
            self.pages_disallowed,
            self.links_discovered,
            self.queue_size,
            current_url_str,
            current_dom_str,
            last_pr_str,
            recent_json
        )
    }
}

fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 16);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

fn current_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Configuration options for the continuous background crawler.
#[derive(Debug, Clone)]
pub struct ContinuousCrawlerConfig {
    /// Maximum number of visited pages before pausing (0 for unlimited)
    pub max_pages: usize,
    /// Minimum politeness cooldown between successive requests to the same domain
    pub politeness_delay: Duration,
    /// Whether to fetch and adhere to robots.txt rules
    pub respect_robots_txt: bool,
    /// User-Agent identifier
    pub user_agent: String,
    /// Interval in terms of newly crawled pages at which PageRank is dynamically recalculated
    pub pagerank_recompute_interval: usize,
}

impl Default for ContinuousCrawlerConfig {
    fn default() -> Self {
        Self {
            max_pages: 50,
            politeness_delay: Duration::from_millis(500),
            respect_robots_txt: true,
            user_agent: "NexoraBot/0.1 (+https://github.com/tusharkkp/Nexora)".to_string(),
            pagerank_recompute_interval: 5,
        }
    }
}

/// Trait defining the sink where crawled documents and updated PageRanks are delivered.
pub trait CrawlSink: Send + Sync + 'static {
    /// Ingests a newly fetched and parsed document into the search engine.
    /// Returns the newly assigned DocId.
    fn on_document_crawled(&self, doc: &CrawledDocument) -> DocId;

    /// Updates PageRank authority scores across the evolving document collection.
    fn on_pagerank_updated(&self, scores: &HashMap<DocId, f64>);
}

/// Continuous, autonomous background crawler service.
///
/// Features:
/// - Runs concurrently in a background worker thread.
/// - Ingests newly discovered pages incrementally without locking out search queries.
/// - Recalculates PageRank on the evolving hyperlink graph periodically.
/// - Exposes real-time telemetry (pages visited, queue size, domain ticker, recent crawl log).
/// - Provides full runtime control: start, pause, resume, stop, and dynamic seed injection.
pub struct ContinuousCrawler {
    frontier: Arc<Mutex<UrlFrontier>>,
    fetcher: Arc<dyn PageFetcher>,
    sink: Arc<dyn CrawlSink>,
    config: Arc<RwLock<ContinuousCrawlerConfig>>,
    telemetry: Arc<RwLock<CrawlerTelemetry>>,
    robots_cache: Arc<Mutex<HashMap<String, RobotsTxt>>>,
    crawled_docs: Arc<RwLock<Vec<CrawledDocument>>>,
    is_running: Arc<AtomicBool>,
    is_paused: Arc<AtomicBool>,
    should_stop: Arc<AtomicBool>,
}

impl ContinuousCrawler {
    /// Creates a new `ContinuousCrawler` with the given fetcher, sink, and configuration.
    pub fn new(
        fetcher: Arc<dyn PageFetcher>,
        sink: Arc<dyn CrawlSink>,
        config: ContinuousCrawlerConfig,
    ) -> Self {
        let frontier = Arc::new(Mutex::new(UrlFrontier::new(config.politeness_delay)));
        Self {
            frontier,
            fetcher,
            sink,
            config: Arc::new(RwLock::new(config)),
            telemetry: Arc::new(RwLock::new(CrawlerTelemetry::default())),
            robots_cache: Arc::new(Mutex::new(HashMap::new())),
            crawled_docs: Arc::new(RwLock::new(Vec::new())),
            is_running: Arc::new(AtomicBool::new(false)),
            is_paused: Arc::new(AtomicBool::new(false)),
            should_stop: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Starts or resumes continuous background crawling.
    ///
    /// If seed URLs are provided, they are queued into the URL frontier.
    /// If the crawler thread is not running, it is spawned immediately.
    pub fn start(
        &self,
        seeds: &[&str],
        max_pages: Option<usize>,
        politeness_delay: Option<Duration>,
    ) -> bool {
        let queue_len = {
            let mut f = self.frontier.lock().unwrap();
            for seed in seeds {
                f.push(seed);
            }
            f.queue_len()
        };

        {
            let mut cfg = self.config.write().unwrap();
            if let Some(mp) = max_pages {
                cfg.max_pages = mp;
            }
            if let Some(delay) = politeness_delay {
                cfg.politeness_delay = delay;
            }
        }

        self.is_paused.store(false, Ordering::SeqCst);
        self.should_stop.store(false, Ordering::SeqCst);

        {
            let mut t = self.telemetry.write().unwrap();
            t.queue_size = queue_len;
            t.status = CrawlerStatus::Running;
        }

        if self
            .is_running
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            let frontier = Arc::clone(&self.frontier);
            let fetcher = Arc::clone(&self.fetcher);
            let sink = Arc::clone(&self.sink);
            let config = Arc::clone(&self.config);
            let telemetry = Arc::clone(&self.telemetry);
            let robots_cache = Arc::clone(&self.robots_cache);
            let crawled_docs = Arc::clone(&self.crawled_docs);
            let is_running = Arc::clone(&self.is_running);
            let is_paused = Arc::clone(&self.is_paused);
            let should_stop = Arc::clone(&self.should_stop);

            thread::Builder::new()
                .name("nexora-continuous-crawler".to_string())
                .spawn(move || {
                    run_crawler_loop(
                        frontier,
                        fetcher,
                        sink,
                        config,
                        telemetry,
                        robots_cache,
                        crawled_docs,
                        is_running,
                        is_paused,
                        should_stop,
                    );
                })
                .expect("Failed to spawn continuous crawler background thread");

            true
        } else {
            // Already running; seeds added and flags updated
            true
        }
    }

    /// Pauses background crawling without terminating the worker thread or losing queue state.
    pub fn pause(&self) {
        self.is_paused.store(true, Ordering::SeqCst);
        let mut t = self.telemetry.write().unwrap();
        t.status = CrawlerStatus::Paused;
    }

    /// Resumes background crawling after being paused.
    pub fn resume(&self) {
        self.is_paused.store(false, Ordering::SeqCst);
        let mut t = self.telemetry.write().unwrap();
        t.status = CrawlerStatus::Running;
    }

    /// Signals the continuous crawler to stop gracefully.
    pub fn stop(&self) {
        self.should_stop.store(true, Ordering::SeqCst);
        let mut t = self.telemetry.write().unwrap();
        t.status = CrawlerStatus::Stopped;
    }

    /// Enqueues additional seed URLs into the active frontier.
    pub fn enqueue_seeds(&self, seeds: &[&str]) -> usize {
        let mut f = self.frontier.lock().unwrap();
        let mut added = 0;
        for seed in seeds {
            if f.push(seed) {
                added += 1;
            }
        }
        self.telemetry.write().unwrap().queue_size = f.queue_len();
        added
    }

    /// Returns a snapshot of the real-time crawler telemetry.
    pub fn telemetry(&self) -> CrawlerTelemetry {
        self.telemetry.read().unwrap().clone()
    }

    /// Returns the active configuration.
    pub fn config(&self) -> ContinuousCrawlerConfig {
        self.config.read().unwrap().clone()
    }

    /// Returns true if the background worker is currently executing.
    pub fn is_running(&self) -> bool {
        self.is_running.load(Ordering::SeqCst)
    }

    /// Returns true if the crawler is paused.
    pub fn is_paused(&self) -> bool {
        self.is_paused.load(Ordering::SeqCst)
    }

    /// Returns a list of all documents crawled so far in this session.
    pub fn crawled_documents(&self) -> Vec<CrawledDocument> {
        self.crawled_docs.read().unwrap().clone()
    }
}

fn run_crawler_loop(
    frontier: Arc<Mutex<UrlFrontier>>,
    fetcher: Arc<dyn PageFetcher>,
    sink: Arc<dyn CrawlSink>,
    config: Arc<RwLock<ContinuousCrawlerConfig>>,
    telemetry: Arc<RwLock<CrawlerTelemetry>>,
    robots_cache: Arc<Mutex<HashMap<String, RobotsTxt>>>,
    crawled_docs: Arc<RwLock<Vec<CrawledDocument>>>,
    is_running: Arc<AtomicBool>,
    is_paused: Arc<AtomicBool>,
    should_stop: Arc<AtomicBool>,
) {
    loop {
        // 1. Check for stop request
        if should_stop.load(Ordering::SeqCst) {
            break;
        }

        // 2. Check for pause request
        if is_paused.load(Ordering::SeqCst) {
            thread::sleep(Duration::from_millis(100));
            continue;
        }

        // 3. Check page limit threshold
        let cfg = config.read().unwrap().clone();
        let pages_visited = telemetry.read().unwrap().pages_visited;
        if cfg.max_pages > 0 && pages_visited >= cfg.max_pages {
            break;
        }

        // 4. Pop next polite URL from frontier
        let (next_url_opt, delay_opt, queue_len) = {
            let mut f = frontier.lock().unwrap();
            let next_url = f.pop_polite();
            let delay = if next_url.is_none() {
                f.next_delay()
            } else {
                None
            };
            let q_len = f.queue_len();
            (next_url, delay, q_len)
        };

        {
            let mut t = telemetry.write().unwrap();
            t.queue_size = queue_len;
        }

        let next_url = match next_url_opt {
            Some(u) => u,
            None => {
                if queue_len == 0 {
                    // Queue empty: crawl session complete
                    break;
                }
                if let Some(delay) = delay_opt {
                    // Politeness cooldown active: sleep in short slices for responsiveness
                    let sleep_slice = delay.min(Duration::from_millis(100));
                    thread::sleep(sleep_slice);
                    continue;
                } else {
                    break;
                }
            }
        };

        // 5. Update telemetry with active URL and domain
        let parsed = parse_url(&next_url);
        let host = parsed.as_ref().map(|p| p.host_key()).unwrap_or_default();
        {
            let mut t = telemetry.write().unwrap();
            t.status = CrawlerStatus::Running;
            t.current_url = Some(next_url.clone());
            t.current_domain = Some(host.clone());
        }

        // 6. Robots.txt compliance check
        if cfg.respect_robots_txt && !host.is_empty() {
            let has_cache = {
                let cache = robots_cache.lock().unwrap();
                cache.contains_key(&host)
            };

            if !has_cache {
                let scheme = parsed
                    .as_ref()
                    .map(|p| p.scheme.as_str())
                    .unwrap_or("https");
                let robots_url = format!("{}://{}/robots.txt", scheme, host);
                let robots = match fetcher.fetch(&robots_url) {
                    Ok(body) => RobotsTxt::parse(&body),
                    Err(_) => RobotsTxt::empty(),
                };
                let mut cache = robots_cache.lock().unwrap();
                cache.insert(host.clone(), robots);
            }

            let is_disallowed = {
                let cache = robots_cache.lock().unwrap();
                if let Some(robots) = cache.get(&host) {
                    let path = parsed
                        .as_ref()
                        .map(|p| {
                            if let Some(ref q) = p.query {
                                format!("{}?{}", p.path, q)
                            } else {
                                p.path.clone()
                            }
                        })
                        .unwrap_or_else(|_| "/".to_string());
                    !robots.is_allowed(&cfg.user_agent, &path)
                } else {
                    false
                }
            };

            if is_disallowed {
                let mut t = telemetry.write().unwrap();
                t.pages_disallowed += 1;
                t.current_url = None;
                t.current_domain = None;
                if t.recent_pages.len() >= 30 {
                    t.recent_pages.pop_back();
                }
                t.recent_pages.push_front(CrawledPageRecord {
                    doc_id: 0,
                    url: next_url,
                    title: "Blocked by robots.txt".to_string(),
                    outgoing_links_count: 0,
                    timestamp_ms: current_time_ms(),
                    status: "Disallowed".to_string(),
                });
                continue;
            }
        }

        // 7. Fetch page content
        match fetcher.fetch(&next_url) {
            Ok(html) => {
                let page = extract_page(&next_url, &html);

                let doc_to_ingest = CrawledDocument {
                    doc_id: 0,
                    url: next_url.clone(),
                    title: page.title.clone(),
                    text: page.text.clone(),
                    outgoing_links: page.outgoing_links.clone(),
                };

                // Ingest incrementally into search engine state
                let assigned_id = sink.on_document_crawled(&doc_to_ingest);

                let completed_doc = CrawledDocument {
                    doc_id: assigned_id,
                    ..doc_to_ingest
                };

                // Enqueue discovered hyperlinks into frontier
                let mut new_links = 0;
                {
                    let mut f = frontier.lock().unwrap();
                    for link in &page.outgoing_links {
                        if f.push(link) {
                            new_links += 1;
                        }
                    }
                }

                // Register document in session archive
                {
                    let mut docs = crawled_docs.write().unwrap();
                    docs.push(completed_doc);
                }

                // Dynamic PageRank re-computation
                let total_crawled = {
                    let docs = crawled_docs.read().unwrap();
                    docs.len()
                };

                if total_crawled > 0 && total_crawled % cfg.pagerank_recompute_interval == 0 {
                    let docs_snap = crawled_docs.read().unwrap().clone();
                    let graph = WebGraph::from_crawled_documents(&docs_snap);
                    let pr_params = PageRankParams::default();
                    let pr_scores = compute_pagerank(&graph, &pr_params);
                    sink.on_pagerank_updated(&pr_scores);
                    telemetry.write().unwrap().last_pagerank_update = Some(current_time_ms());
                }

                // Update telemetry
                {
                    let mut t = telemetry.write().unwrap();
                    t.pages_visited += 1;
                    t.links_discovered += new_links;
                    t.queue_size = frontier.lock().unwrap().queue_len();
                    t.current_url = None;
                    t.current_domain = None;
                    if t.recent_pages.len() >= 30 {
                        t.recent_pages.pop_back();
                    }
                    t.recent_pages.push_front(CrawledPageRecord {
                        doc_id: assigned_id,
                        url: next_url,
                        title: if page.title.is_empty() {
                            "Untitled Page".to_string()
                        } else {
                            page.title
                        },
                        outgoing_links_count: page.outgoing_links.len(),
                        timestamp_ms: current_time_ms(),
                        status: "Indexed".to_string(),
                    });
                }
            }
            Err(err) => {
                let mut t = telemetry.write().unwrap();
                t.pages_failed += 1;
                t.queue_size = frontier.lock().unwrap().queue_len();
                t.current_url = None;
                t.current_domain = None;
                if t.recent_pages.len() >= 30 {
                    t.recent_pages.pop_back();
                }
                t.recent_pages.push_front(CrawledPageRecord {
                    doc_id: 0,
                    url: next_url,
                    title: format!("Error: {}", err),
                    outgoing_links_count: 0,
                    timestamp_ms: current_time_ms(),
                    status: "Failed".to_string(),
                });
            }
        }
    }

    // Final PageRank re-computation pass across all crawled documents
    let docs_snap = crawled_docs.read().unwrap().clone();
    if !docs_snap.is_empty() {
        let graph = WebGraph::from_crawled_documents(&docs_snap);
        let pr_params = PageRankParams::default();
        let pr_scores = compute_pagerank(&graph, &pr_params);
        sink.on_pagerank_updated(&pr_scores);
        telemetry.write().unwrap().last_pagerank_update = Some(current_time_ms());
    }

    is_running.store(false, Ordering::SeqCst);
    let mut t = telemetry.write().unwrap();
    if should_stop.load(Ordering::SeqCst) {
        t.status = CrawlerStatus::Stopped;
    } else {
        t.status = CrawlerStatus::Idle;
    }
    t.current_url = None;
    t.current_domain = None;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crawler::fetcher::MockFetcher;
    use std::sync::atomic::AtomicU32;

    struct TestSink {
        next_id: AtomicU32,
        ingested: Arc<Mutex<Vec<CrawledDocument>>>,
        pageranks: Arc<Mutex<HashMap<DocId, f64>>>,
    }

    impl TestSink {
        fn new() -> Self {
            Self {
                next_id: AtomicU32::new(1),
                ingested: Arc::new(Mutex::new(Vec::new())),
                pageranks: Arc::new(Mutex::new(HashMap::new())),
            }
        }
    }

    impl CrawlSink for TestSink {
        fn on_document_crawled(&self, doc: &CrawledDocument) -> DocId {
            let id = self.next_id.fetch_add(1, Ordering::SeqCst);
            let mut record = doc.clone();
            record.doc_id = id;
            self.ingested.lock().unwrap().push(record);
            id
        }

        fn on_pagerank_updated(&self, scores: &HashMap<DocId, f64>) {
            let mut pr = self.pageranks.lock().unwrap();
            *pr = scores.clone();
        }
    }

    #[test]
    fn test_continuous_crawler_lifecycle_and_telemetry() {
        let mut fetcher = MockFetcher::new();
        fetcher.add_page(
            "https://test.nexora.org/page1",
            r#"<html><head><title>Page One</title></head><body><p>Hello Nexora search!</p><a href="/page2">Link</a></body></html>"#,
        );
        fetcher.add_page(
            "https://test.nexora.org/page2",
            r#"<html><head><title>Page Two</title></head><body><p>Second page content.</p><a href="/page1">Back</a></body></html>"#,
        );

        let sink = Arc::new(TestSink::new());
        let config = ContinuousCrawlerConfig {
            max_pages: 5,
            politeness_delay: Duration::from_millis(10),
            respect_robots_txt: false,
            user_agent: "TestBot".to_string(),
            pagerank_recompute_interval: 2,
        };

        let crawler = ContinuousCrawler::new(Arc::new(fetcher), sink.clone(), config);

        assert_eq!(crawler.telemetry().status, CrawlerStatus::Idle);

        // Start crawling
        crawler.start(&["https://test.nexora.org/page1"], Some(2), None);

        // Give worker thread time to process 2 pages
        let mut attempts = 0;
        while crawler.is_running() && attempts < 50 {
            thread::sleep(Duration::from_millis(20));
            attempts += 1;
        }

        let telemetry = crawler.telemetry();
        assert_eq!(telemetry.pages_visited, 2);
        assert_eq!(sink.ingested.lock().unwrap().len(), 2);
        assert!(!sink.pageranks.lock().unwrap().is_empty());

        let json = telemetry.to_json();
        assert!(json.contains("\"pages_visited\":2"));
        assert!(json.contains("\"status\":"));
    }
}
