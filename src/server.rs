use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;

use tiny_http::{Header, Response, Server, StatusCode};

use crate::crawler::{
    ContinuousCrawler, ContinuousCrawlerConfig, CrawlConfig, CrawlSink, CrawledDocument, Crawler,
    HttpFetcher, UrlFrontier,
};
use crate::graph::{PageRankParams, WebGraph, compute_pagerank};
use crate::index::{DocId, InvertedIndex, MultiFieldIndex};
use crate::ranking::{HybridBM25FParams, ScoredDocument, rank_bm25f_with_pagerank};
use crate::snippet::{HighlightFormat, SnippetConfig, generate_snippet};
use crate::spelling::SpellChecker;
use crate::storage::DocumentMetadata;
use crate::trie::PrefixTrie;

/// Configuration for the HTTP server.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
        }
    }
}

/// Internal shared state for the search engine.
pub struct SearchEngineState {
    pub index: InvertedIndex,
    pub multi_index: MultiFieldIndex,
    pub doc_metadata: HashMap<DocId, DocumentMetadata>,
    pub trie: PrefixTrie,
    pub spell_checker: SpellChecker,
}

impl SearchEngineState {
    pub fn new(index: InvertedIndex, doc_metadata: HashMap<DocId, DocumentMetadata>) -> Self {
        let mut multi_index = MultiFieldIndex::new();
        for (&id, doc) in &doc_metadata {
            multi_index.add_document(id, &doc.title, &doc.body, "");
        }

        let trie = PrefixTrie::from_documents(
            doc_metadata
                .values()
                .map(|d| format!("{} {}", d.title, d.body)),
        );
        let spell_checker = SpellChecker::from_documents(
            doc_metadata
                .values()
                .map(|d| format!("{} {}", d.title, d.body)),
        );

        Self {
            index,
            multi_index,
            doc_metadata,
            trie,
            spell_checker,
        }
    }

    /// Creates default demo state with sample computer science & web search documents.
    pub fn default_demo() -> Self {
        let mut index = InvertedIndex::new();
        let mut multi_index = MultiFieldIndex::new();
        let mut metadata = HashMap::new();

        let docs = &[
            (
                "Introduction to Search Engines and Inverted Indexes",
                "A search engine is an information retrieval system that uses an inverted index to map vocabulary terms to postings lists of matching documents. The postings list contains document identifiers and token positions.",
                Some("https://nexora.dev/docs/inverted-index"),
                0.35,
            ),
            (
                "Rust Systems Programming and Memory Safety",
                "Rust delivers bare-metal performance and memory safety without a garbage collector. Its ownership and borrow checker prevent data races, making it ideal for high-throughput search engine backends.",
                Some("https://www.rust-lang.org/"),
                0.28,
            ),
            (
                "The Okapi BM25 Ranking Algorithm",
                "Okapi BM25 is the gold standard lexical ranking algorithm. It balances term frequency saturation using parameter k1 and document length normalization using parameter b, improving upon classical TF-IDF.",
                Some("https://nexora.dev/algorithms/bm25"),
                0.22,
            ),
            (
                "Web Crawlers and the URL Frontier",
                "A web crawler systematically browses the World Wide Web. It fetches HTML pages over HTTP, obeys robots.txt politeness policies, extracts hyperlinks, and schedules unvisited URLs in a URL frontier.",
                Some("https://nexora.dev/crawler/frontier"),
                0.15,
            ),
            (
                "Distributed Systems, Sharding, and Scalability",
                "When an index grows too large for a single machine, search engines use index partitioning and horizontal sharding. Replication provides high availability, fault tolerance, and load balancing across nodes.",
                Some("https://nexora.dev/architecture/distributed"),
                0.12,
            ),
            (
                "Computer Networks: TCP/IP and HTTP Protocols",
                "Modern search engines interact with network protocols like TCP/IP and HTTP. Devices communicate across IP addresses like 192.168.1.1 and send web requests to URLs like https://nexora.org/search.",
                Some("https://nexora.dev/networks/protocols"),
                0.10,
            ),
        ];

        for (id, &(title, text, url, pr)) in docs.iter().enumerate() {
            let doc_id = id as DocId;
            index.add_document(doc_id, text);
            multi_index.add_document(doc_id, title, text, "");
            metadata.insert(
                doc_id,
                DocumentMetadata {
                    doc_id,
                    url: url.unwrap_or("").to_string(),
                    title: title.to_string(),
                    body: text.to_string(),
                    pagerank: pr,
                },
            );
        }

        let trie = PrefixTrie::from_documents(
            metadata.values().map(|d| format!("{} {}", d.title, d.body)),
        );
        let spell_checker = SpellChecker::from_documents(
            metadata.values().map(|d| format!("{} {}", d.title, d.body)),
        );

        Self {
            index,
            multi_index,
            doc_metadata: metadata,
            trie,
            spell_checker,
        }
    }

    /// Rebuilds multi-field index, trie, and spell checker after a new crawl or index update.
    pub fn rebuild_indexes(&mut self) {
        let mut multi = MultiFieldIndex::new();
        for (&id, doc) in &self.doc_metadata {
            multi.add_document(id, &doc.title, &doc.body, "");
        }
        self.multi_index = multi;

        self.trie = PrefixTrie::from_documents(
            self.doc_metadata
                .values()
                .map(|d| format!("{} {}", d.title, d.body)),
        );
        self.spell_checker = SpellChecker::from_documents(
            self.doc_metadata
                .values()
                .map(|d| format!("{} {}", d.title, d.body)),
        );
    }

    /// Ingests a crawled document incrementally into the live search engine state.
    ///
    /// Updates:
    /// - `InvertedIndex`
    /// - `MultiFieldIndex`
    /// - `PrefixTrie`
    /// - `SpellChecker`
    /// - `doc_metadata`
    ///
    /// Assigns and returns a new unique `DocId`.
    pub fn ingest_crawled_page(&mut self, page: &CrawledDocument, initial_pagerank: f64) -> DocId {
        if let Some((&existing_id, _)) = self.doc_metadata.iter().find(|(_, d)| d.url == page.url) {
            if let Some(meta) = self.doc_metadata.get_mut(&existing_id) {
                if !page.title.is_empty() {
                    meta.title = page.title.clone();
                }
                if !page.text.is_empty() {
                    meta.body = page.text.clone();
                }
            }
            return existing_id;
        }

        let doc_id = self
            .doc_metadata
            .keys()
            .copied()
            .max()
            .map(|max_id| max_id + 1)
            .unwrap_or(0);

        let combined = if page.title.is_empty() {
            page.text.clone()
        } else {
            format!("{} {}", page.title, page.text)
        };

        self.index.add_document(doc_id, &combined);
        self.multi_index
            .add_document(doc_id, &page.title, &page.text, "");
        self.trie.add_document_text(&combined);
        self.spell_checker.add_document_text(&combined);
        self.doc_metadata.insert(
            doc_id,
            DocumentMetadata {
                doc_id,
                url: page.url.clone(),
                title: page.title.clone(),
                body: page.text.clone(),
                pagerank: initial_pagerank,
            },
        );

        doc_id
    }

    /// Bulk updates PageRank scores for all matching documents without reindexing text.
    pub fn update_pageranks(&mut self, scores: &HashMap<DocId, f64>) {
        for (&doc_id, &pr) in scores {
            if let Some(meta) = self.doc_metadata.get_mut(&doc_id) {
                meta.pagerank = pr;
            }
        }
    }
}

/// Adapter bridging the background continuous crawler to the shared SearchEngineState.
#[derive(Clone)]
pub struct SearchEngineSink(pub Arc<RwLock<SearchEngineState>>);

impl CrawlSink for SearchEngineSink {
    fn on_document_crawled(&self, doc: &CrawledDocument) -> DocId {
        let mut state = self.0.write().unwrap();
        state.ingest_crawled_page(doc, 0.0)
    }

    fn on_pagerank_updated(&self, scores: &HashMap<DocId, f64>) {
        let mut state = self.0.write().unwrap();
        state.update_pageranks(scores);
    }
}

/// The embedded search server with integrated continuous background crawler.
pub struct SearchServer {
    config: ServerConfig,
    state: Arc<RwLock<SearchEngineState>>,
    crawler: ContinuousCrawler,
}

impl SearchServer {
    pub fn new(config: ServerConfig, state: SearchEngineState) -> Self {
        Self::with_crawler_config(config, state, ContinuousCrawlerConfig::default())
    }

    pub fn with_crawler_config(
        config: ServerConfig,
        state: SearchEngineState,
        crawler_config: ContinuousCrawlerConfig,
    ) -> Self {
        let shared_state = Arc::new(RwLock::new(state));
        let fetcher = Arc::new(HttpFetcher::default_client());
        let sink = Arc::new(SearchEngineSink(Arc::clone(&shared_state)));
        let crawler = ContinuousCrawler::new(fetcher, sink, crawler_config);
        Self {
            config,
            state: shared_state,
            crawler,
        }
    }

    /// Returns a reference to the shared search engine state.
    pub fn state(&self) -> &Arc<RwLock<SearchEngineState>> {
        &self.state
    }

    /// Returns a reference to the active continuous background crawler.
    pub fn crawler(&self) -> &ContinuousCrawler {
        &self.crawler
    }

    /// Starts the HTTP server listener and serves requests.
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let addr = format!("{}:{}", self.config.host, self.config.port);
        let server = Server::http(&addr)?;
        println!("🚀 Nexora HTTP Search Server listening on http://{}", addr);
        println!(
            "✨ Web SERP Interface available at: http://localhost:{}",
            self.config.port
        );

        for request in server.incoming_requests() {
            let state = Arc::clone(&self.state);
            let url = request.url().to_string();
            let method = request.method().as_str().to_uppercase();

            let path = url.split('?').next().unwrap_or("/").to_string();
            let query_str = url.split('?').nth(1).unwrap_or("");
            let params = parse_query_params(query_str);

            match (method.as_str(), path.as_str()) {
                ("GET", "/") => {
                    let html = render_web_serp_html();
                    let response = Response::from_string(html)
                        .with_header(header("Content-Type", "text/html; charset=utf-8"))
                        .with_header(header("Access-Control-Allow-Origin", "*"));
                    let _ = request.respond(response);
                }
                ("GET", "/api/search") => {
                    let q = params.get("q").cloned().unwrap_or_default();
                    let limit: usize = params.get("k").and_then(|k| k.parse().ok()).unwrap_or(20);

                    let json = {
                        let lock = state.read().unwrap();
                        handle_api_search(&lock, &q, limit)
                    };

                    let response = Response::from_string(json)
                        .with_header(header("Content-Type", "application/json; charset=utf-8"))
                        .with_header(header("Access-Control-Allow-Origin", "*"));
                    let _ = request.respond(response);
                }
                ("GET", "/api/suggest") => {
                    let prefix = params.get("q").cloned().unwrap_or_default();
                    let limit: usize = params.get("k").and_then(|k| k.parse().ok()).unwrap_or(8);

                    let json = {
                        let lock = state.read().unwrap();
                        handle_api_suggest(&lock, &prefix, limit)
                    };

                    let response = Response::from_string(json)
                        .with_header(header("Content-Type", "application/json; charset=utf-8"))
                        .with_header(header("Access-Control-Allow-Origin", "*"));
                    let _ = request.respond(response);
                }
                ("GET", "/api/stats") => {
                    let json = {
                        let lock = state.read().unwrap();
                        handle_api_stats(&lock)
                    };

                    let response = Response::from_string(json)
                        .with_header(header("Content-Type", "application/json; charset=utf-8"))
                        .with_header(header("Access-Control-Allow-Origin", "*"));
                    let _ = request.respond(response);
                }
                ("GET", "/api/crawler/status") => {
                    let json = self.crawler.telemetry().to_json();
                    let response = Response::from_string(json)
                        .with_header(header("Content-Type", "application/json; charset=utf-8"))
                        .with_header(header("Access-Control-Allow-Origin", "*"));
                    let _ = request.respond(response);
                }
                ("POST", "/api/crawler/start") | ("POST", "/api/crawl") => {
                    let raw_url = params.get("url").cloned().unwrap_or_default();
                    let max_pages = params.get("max_pages").and_then(|p| p.parse().ok());
                    let delay_ms = params
                        .get("delay_ms")
                        .and_then(|d| d.parse::<u64>().ok())
                        .map(std::time::Duration::from_millis);

                    let seeds: Vec<&str> = if raw_url.is_empty() {
                        Vec::new()
                    } else {
                        raw_url
                            .split(',')
                            .map(|s| s.trim())
                            .filter(|s| !s.is_empty())
                            .collect()
                    };

                    let started = self.crawler.start(&seeds, max_pages, delay_ms);
                    let telemetry = self.crawler.telemetry();

                    let json = format!(
                        r#"{{"success":{},"status":"{}","pages_visited":{},"queue_size":{}}}"#,
                        started,
                        telemetry.status.as_str(),
                        telemetry.pages_visited,
                        telemetry.queue_size
                    );
                    let response = Response::from_string(json)
                        .with_header(header("Content-Type", "application/json; charset=utf-8"))
                        .with_header(header("Access-Control-Allow-Origin", "*"));
                    let _ = request.respond(response);
                }
                ("POST", "/api/crawler/pause") => {
                    self.crawler.pause();
                    let response = Response::from_string(r#"{"success":true,"status":"paused"}"#)
                        .with_header(header("Content-Type", "application/json; charset=utf-8"))
                        .with_header(header("Access-Control-Allow-Origin", "*"));
                    let _ = request.respond(response);
                }
                ("POST", "/api/crawler/resume") => {
                    self.crawler.resume();
                    let response = Response::from_string(r#"{"success":true,"status":"running"}"#)
                        .with_header(header("Content-Type", "application/json; charset=utf-8"))
                        .with_header(header("Access-Control-Allow-Origin", "*"));
                    let _ = request.respond(response);
                }
                ("POST", "/api/crawler/stop") => {
                    self.crawler.stop();
                    let response = Response::from_string(r#"{"success":true,"status":"stopped"}"#)
                        .with_header(header("Content-Type", "application/json; charset=utf-8"))
                        .with_header(header("Access-Control-Allow-Origin", "*"));
                    let _ = request.respond(response);
                }
                _ => {
                    let not_found = r#"{"error":"Not Found"}"#;
                    let response = Response::from_string(not_found)
                        .with_status_code(StatusCode(404))
                        .with_header(header("Content-Type", "application/json"));
                    let _ = request.respond(response);
                }
            }
        }

        Ok(())
    }
}

fn header(name: &str, value: &str) -> Header {
    Header::from_bytes(name.as_bytes(), value.as_bytes()).unwrap()
}

/// Parses a URL query string `a=1&b=2` into a key-value HashMap with percent-decoding.
pub fn parse_query_params(query: &str) -> HashMap<String, String> {
    let mut params = HashMap::new();
    if query.is_empty() {
        return params;
    }

    for pair in query.split('&') {
        let mut parts = pair.splitn(2, '=');
        if let Some(key) = parts.next() {
            let value = parts.next().unwrap_or("");
            params.insert(percent_decode(key), percent_decode(value));
        }
    }

    params
}

/// Decodes standard URL percent-encoding (e.g. `foo%20bar` -> `foo bar`).
pub fn percent_decode(input: &str) -> String {
    let mut bytes = Vec::with_capacity(input.len());
    let mut chars = input.bytes();

    while let Some(b) = chars.next() {
        if b == b'+' {
            bytes.push(b' ');
        } else if b == b'%' {
            let h1 = chars.next();
            let h2 = chars.next();
            if let (Some(h1), Some(h2)) = (h1, h2) {
                if let (Some(n1), Some(n2)) = (hex_val(h1), hex_val(h2)) {
                    bytes.push((n1 << 4) | n2);
                    continue;
                }
            }
            bytes.push(b'%');
        } else {
            bytes.push(b);
        }
    }

    String::from_utf8_lossy(&bytes).to_string()
}

fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// Escapes a string for safe embedding inside a JSON literal.
pub fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 16);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

fn handle_api_search(state: &SearchEngineState, query: &str, limit: usize) -> String {
    let start = Instant::now();
    let clean = query.trim();

    if clean.is_empty() {
        return format!(
            r#"{{"query":"","total":0,"took_ms":0.0,"did_you_mean":null,"results":[]}}"#
        );
    }

    let pr_map: HashMap<DocId, f64> = state
        .doc_metadata
        .iter()
        .map(|(&id, meta)| (id, meta.pagerank))
        .collect();

    let hybrid_f_params = HybridBM25FParams::default();
    let mut results: Vec<ScoredDocument> =
        rank_bm25f_with_pagerank(&state.multi_index, clean, &pr_map, &hybrid_f_params);

    let mut did_you_mean: Option<String> = None;

    if results.is_empty() {
        if let Some(suggested) = state.spell_checker.suggest_query(clean) {
            did_you_mean = Some(suggested.clone());
            results =
                rank_bm25f_with_pagerank(&state.multi_index, &suggested, &pr_map, &hybrid_f_params);
        }
    }

    let total = results.len();
    let top_results: &[ScoredDocument] = if results.len() > limit {
        &results[..limit]
    } else {
        &results
    };

    let snippet_cfg = SnippetConfig {
        max_chars: 180,
        format: HighlightFormat::Html,
    };

    let mut results_json = String::new();
    for (i, scored) in top_results.iter().enumerate() {
        if i > 0 {
            results_json.push(',');
        }

        let doc = state.doc_metadata.get(&scored.doc_id);
        let title = doc.map(|d| d.title.as_str()).unwrap_or("Untitled");
        let url = doc.map(|d| d.url.as_str()).unwrap_or("");
        let body = doc.map(|d| d.body.as_str()).unwrap_or("");
        let pr = doc.map(|d| d.pagerank).unwrap_or(0.0);

        let active_query = did_you_mean.as_deref().unwrap_or(clean);
        let snippet = generate_snippet(body, active_query, state.index.analyzer(), &snippet_cfg);

        results_json.push_str(&format!(
            r#"{{"doc_id":{},"rank":{},"score":{:.4},"pagerank":{:.4},"title":"{}","url":"{}","snippet":"{}"}}"#,
            scored.doc_id,
            i + 1,
            scored.score,
            pr,
            json_escape(title),
            json_escape(url),
            json_escape(&snippet),
        ));
    }

    let took_ms = start.elapsed().as_secs_f64() * 1000.0;
    let did_you_mean_json = match did_you_mean {
        Some(s) => format!("\"{}\"", json_escape(&s)),
        None => "null".to_string(),
    };

    format!(
        r#"{{"query":"{}","total":{},"took_ms":{:.2},"did_you_mean":{},"results":[{}]}}"#,
        json_escape(clean),
        total,
        took_ms,
        did_you_mean_json,
        results_json
    )
}

fn handle_api_suggest(state: &SearchEngineState, prefix: &str, limit: usize) -> String {
    let suggestions = state.trie.suggest(prefix, limit);
    let mut items_json = String::new();

    for (i, s) in suggestions.iter().enumerate() {
        if i > 0 {
            items_json.push(',');
        }
        items_json.push_str(&format!(
            r#"{{"term":"{}","frequency":{},"doc_frequency":{}}}"#,
            json_escape(&s.term),
            s.term_frequency,
            s.doc_frequency
        ));
    }

    format!(
        r#"{{"prefix":"{}","suggestions":[{}]}}"#,
        json_escape(prefix),
        items_json
    )
}

fn handle_api_stats(state: &SearchEngineState) -> String {
    format!(
        r#"{{"total_documents":{},"vocabulary_size":{},"average_doc_length":{:.2}}}"#,
        state.index.total_documents(),
        state.index.vocabulary_size(),
        state.index.average_doc_length()
    )
}

/// Executes a synchronous crawl session and replaces the entire index.
pub fn handle_api_crawl(state: &mut SearchEngineState, seed_url: &str, max_pages: usize) -> String {
    let start = Instant::now();
    let mut frontier = UrlFrontier::default_polite();
    if !frontier.push(seed_url) {
        return format!(
            r#"{{"success":false,"error":"Invalid seed URL: {}"}}"#,
            json_escape(seed_url)
        );
    }

    let fetcher = HttpFetcher::default_client();
    let config = CrawlConfig {
        max_pages,
        max_wait: std::time::Duration::from_secs(5),
        respect_robots_txt: true,
        user_agent: "NexoraBot/0.1".to_string(),
    };

    let mut crawler = Crawler::new(frontier, fetcher, config);
    let mut new_index = InvertedIndex::new();
    let summary = crawler.crawl(&mut new_index);

    if summary.documents.is_empty() {
        return r#"{"success":false,"error":"No documents could be retrieved"}"#.to_string();
    }

    let graph = WebGraph::from_crawled_documents(&summary.documents);
    let pr_params = PageRankParams::default();
    let pagerank_scores = compute_pagerank(&graph, &pr_params);

    let mut new_metadata = HashMap::with_capacity(summary.documents.len());
    for doc in summary.documents {
        let pr = pagerank_scores.get(&doc.doc_id).copied().unwrap_or(0.0);
        new_metadata.insert(
            doc.doc_id,
            DocumentMetadata {
                doc_id: doc.doc_id,
                url: doc.url,
                title: doc.title,
                body: doc.text,
                pagerank: pr,
            },
        );
    }

    state.index = new_index;
    state.doc_metadata = new_metadata;
    state.rebuild_indexes();

    let took_ms = start.elapsed().as_secs_f64() * 1000.0;
    format!(
        r#"{{"success":true,"pages_visited":{},"links_discovered":{},"took_ms":{:.2}}}"#,
        summary.pages_visited, summary.links_discovered, took_ms
    )
}

/// Returns the embedded modern Web SERP interface with dark mode, live debounced autocomplete, and highlighted snippets.
pub fn render_web_serp_html() -> String {
    r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Nexora Search Engine</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-dark: #0a0e17;
      --card-bg: #121826;
      --card-border: #1f293d;
      --accent: #6366f1;
      --accent-hover: #4f46e5;
      --accent-glow: rgba(99, 102, 241, 0.25);
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
      --badge-pr: #10b981;
      --badge-pr-bg: rgba(16, 185, 129, 0.12);
      --highlight-bg: rgba(99, 102, 241, 0.35);
      --highlight-text: #e0e7ff;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
      background-color: var(--bg-dark);
      color: var(--text-main);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      overflow-x: hidden;
      background-image: 
        radial-gradient(circle at 15% 20%, rgba(99, 102, 241, 0.08) 0%, transparent 40%),
        radial-gradient(circle at 85% 80%, rgba(16, 185, 129, 0.06) 0%, transparent 40%);
    }

    header {
      padding: 1.5rem 2rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--card-border);
      backdrop-filter: blur(12px);
      background: rgba(10, 14, 23, 0.7);
      position: sticky;
      top: 0;
      z-index: 50;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      text-decoration: none;
      color: var(--text-main);
    }

    .brand-icon {
      width: 34px;
      height: 34px;
      background: linear-gradient(135deg, #6366f1, #3b82f6);
      border-radius: 10px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 700;
      font-size: 1.1rem;
      box-shadow: 0 4px 14px var(--accent-glow);
    }

    .brand-name {
      font-size: 1.4rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      background: linear-gradient(135deg, #ffffff, #cbd5e1);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .header-actions {
      display: flex;
      gap: 1rem;
      align-items: center;
    }

    .btn {
      padding: 0.5rem 1rem;
      border-radius: 8px;
      font-weight: 500;
      font-size: 0.9rem;
      cursor: pointer;
      transition: all 0.2s ease;
      border: 1px solid var(--card-border);
      background: var(--card-bg);
      color: var(--text-main);
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
    }

    .btn:hover {
      border-color: var(--accent);
      background: rgba(99, 102, 241, 0.1);
    }

    .btn-primary {
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
      box-shadow: 0 4px 14px var(--accent-glow);
    }

    .btn-primary:hover {
      background: var(--accent-hover);
      transform: translateY(-1px);
    }

    main {
      flex: 1;
      max-width: 860px;
      width: 100%;
      margin: 0 auto;
      padding: 2.5rem 1.5rem;
    }

    .search-section {
      margin-bottom: 2rem;
      position: relative;
    }

    .search-box-wrapper {
      position: relative;
    }

    .search-input {
      width: 100%;
      padding: 1.1rem 3.5rem 1.1rem 1.4rem;
      font-size: 1.1rem;
      font-family: inherit;
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      color: var(--text-main);
      box-shadow: 0 8px 30px rgba(0, 0, 0, 0.25);
      transition: all 0.25s ease;
    }

    .search-input:focus {
      outline: none;
      border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--accent-glow), 0 8px 30px rgba(0, 0, 0, 0.35);
    }

    .search-btn {
      position: absolute;
      right: 12px;
      top: 50%;
      transform: translateY(-50%);
      background: var(--accent);
      border: none;
      width: 40px;
      height: 40px;
      border-radius: 10px;
      color: white;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: background 0.2s;
    }

    .search-btn:hover {
      background: var(--accent-hover);
    }

    /* Autocomplete dropdown */
    .autocomplete-dropdown {
      position: absolute;
      top: calc(100% + 6px);
      left: 0;
      right: 0;
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      overflow: hidden;
      z-index: 100;
      box-shadow: 0 16px 40px rgba(0, 0, 0, 0.4);
      display: none;
    }

    .autocomplete-item {
      padding: 0.85rem 1.25rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      cursor: pointer;
      transition: background 0.15s;
      border-bottom: 1px solid rgba(255, 255, 255, 0.03);
    }

    .autocomplete-item:last-child {
      border-bottom: none;
    }

    .autocomplete-item:hover, .autocomplete-item.active {
      background: rgba(99, 102, 241, 0.15);
    }

    .autocomplete-term {
      font-weight: 500;
      color: var(--text-main);
    }

    .autocomplete-term mark {
      background: transparent;
      color: var(--accent);
      font-weight: 700;
    }

    .autocomplete-freq {
      font-size: 0.8rem;
      color: var(--text-dim);
    }

    /* Meta bar */
    .search-meta {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1.5rem;
      font-size: 0.88rem;
      color: var(--text-muted);
      min-height: 24px;
    }

    .did-you-mean {
      color: var(--text-muted);
    }

    .did-you-mean a {
      color: #38bdf8;
      text-decoration: underline;
      cursor: pointer;
      font-weight: 500;
    }

    /* Results */
    .results-container {
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
    }

    .result-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 1.4rem;
      transition: all 0.2s ease;
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }

    .result-card:hover {
      border-color: rgba(99, 102, 241, 0.4);
      transform: translateY(-2px);
      box-shadow: 0 10px 25px rgba(0, 0, 0, 0.25);
    }

    .result-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 1rem;
    }

    .result-title {
      font-size: 1.15rem;
      font-weight: 600;
      color: #93c5fd;
      text-decoration: none;
      line-height: 1.35;
    }

    .result-title:hover {
      text-decoration: underline;
      color: #bfdbfe;
    }

    .result-badges {
      display: flex;
      gap: 0.5rem;
      flex-shrink: 0;
    }

    .badge {
      font-size: 0.75rem;
      font-weight: 600;
      padding: 0.2rem 0.5rem;
      border-radius: 6px;
      display: inline-flex;
      align-items: center;
      gap: 0.25rem;
    }

    .badge-score {
      background: rgba(99, 102, 241, 0.15);
      color: #a5b4fc;
      border: 1px solid rgba(99, 102, 241, 0.3);
    }

    .badge-pr {
      background: var(--badge-pr-bg);
      color: var(--badge-pr);
      border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .result-url {
      font-size: 0.8rem;
      color: var(--text-dim);
      word-break: break-all;
    }

    .result-snippet {
      font-size: 0.95rem;
      color: var(--text-muted);
      line-height: 1.55;
    }

    mark.highlight {
      background: var(--highlight-bg);
      color: var(--highlight-text);
      padding: 0.1rem 0.25rem;
      border-radius: 4px;
      font-weight: 500;
    }

    .empty-state {
      text-align: center;
      padding: 4rem 1rem;
      color: var(--text-muted);
    }

    .empty-state h3 {
      font-size: 1.3rem;
      margin-bottom: 0.5rem;
      color: var(--text-main);
    }

    /* Modal for continuous crawler */
    .modal-backdrop {
      display: none;
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: rgba(0, 0, 0, 0.75);
      backdrop-filter: blur(8px);
      z-index: 200;
      align-items: center;
      justify-content: center;
    }

    .modal-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 1.75rem;
      max-width: 660px;
      width: 92%;
      max-height: 90vh;
      overflow-y: auto;
      box-shadow: 0 20px 50px rgba(0, 0, 0, 0.6);
    }

    .crawler-pill {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--card-border);
      padding: 0.35rem 0.85rem;
      border-radius: 9999px;
      font-size: 0.82rem;
      font-weight: 500;
      color: var(--text-main);
      cursor: pointer;
      transition: all 0.2s ease;
    }

    .crawler-pill:hover {
      background: rgba(255, 255, 255, 0.08);
      border-color: rgba(99, 102, 241, 0.4);
    }

    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #64748b;
      display: inline-block;
      transition: all 0.3s;
    }

    .status-dot.running {
      background: #10b981;
      box-shadow: 0 0 10px #10b981;
      animation: pulse-dot 1.5s infinite;
    }

    .status-dot.paused {
      background: #f59e0b;
      box-shadow: 0 0 8px #f59e0b;
    }

    @keyframes pulse-dot {
      0%, 100% { transform: scale(1); opacity: 1; }
      50% { transform: scale(1.35); opacity: 0.7; }
    }

    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 0.75rem;
      margin: 1rem 0;
    }

    .metric-card {
      background: var(--bg-dark);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 0.75rem;
      text-align: center;
    }

    .metric-num {
      font-size: 1.35rem;
      font-weight: 700;
      line-height: 1.2;
    }

    .metric-label {
      font-size: 0.72rem;
      color: var(--text-dim);
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-top: 0.25rem;
    }

    .crawler-ticker {
      background: rgba(99, 102, 241, 0.08);
      border: 1px solid rgba(99, 102, 241, 0.2);
      border-radius: 8px;
      padding: 0.6rem 0.9rem;
      font-size: 0.82rem;
      color: var(--text-muted);
      margin-bottom: 1rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      overflow: hidden;
      white-space: nowrap;
      text-overflow: ellipsis;
    }

    .btn-warning {
      background: rgba(245, 158, 11, 0.15);
      border-color: rgba(245, 158, 11, 0.4);
      color: #fbbf24;
    }

    .btn-warning:hover {
      background: rgba(245, 158, 11, 0.25);
    }

    .btn-danger {
      background: rgba(239, 68, 68, 0.15);
      border-color: rgba(239, 68, 68, 0.4);
      color: #f87171;
    }

    .btn-danger:hover {
      background: rgba(239, 68, 68, 0.25);
    }

    .feed-container {
      max-height: 180px;
      overflow-y: auto;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      background: var(--bg-dark);
    }

    .feed-item {
      padding: 0.55rem 0.8rem;
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 0.75rem;
      font-size: 0.82rem;
    }

    .feed-item:last-child {
      border-bottom: none;
    }

    .feed-item:hover {
      background: rgba(255, 255, 255, 0.03);
    }

    .feed-title {
      color: #93c5fd;
      cursor: pointer;
      text-decoration: none;
      font-weight: 500;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      max-width: 320px;
    }

    .feed-title:hover {
      text-decoration: underline;
      color: #bfdbfe;
    }

    .feed-badge {
      font-size: 0.7rem;
      padding: 0.15rem 0.4rem;
      border-radius: 4px;
      font-weight: 600;
    }

    .feed-badge-indexed {
      background: rgba(16, 185, 129, 0.15);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .feed-badge-disallowed {
      background: rgba(245, 158, 11, 0.15);
      color: #fbbf24;
      border: 1px solid rgba(245, 158, 11, 0.3);
    }

    .feed-badge-failed {
      background: rgba(239, 68, 68, 0.15);
      color: #f87171;
      border: 1px solid rgba(239, 68, 68, 0.3);
    }

    .form-group {
      margin-bottom: 1rem;
    }

    .form-label {
      display: block;
      font-size: 0.82rem;
      margin-bottom: 0.35rem;
      color: var(--text-muted);
    }

    .form-input {
      width: 100%;
      padding: 0.65rem 0.85rem;
      font-size: 0.9rem;
      font-family: inherit;
      background: var(--bg-dark);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      color: var(--text-main);
    }

    .form-input:focus {
      outline: none;
      border-color: var(--accent);
    }
  </style>
</head>
<body>

  <header>
    <a href="/" class="brand">
      <div class="brand-icon">N</div>
      <span class="brand-name">Nexora</span>
    </a>
    <div class="header-actions">
      <span id="stats-badge" style="font-size:0.85rem; color: var(--text-dim);">Loading stats...</span>
      <div id="crawler-pill" class="crawler-pill" onclick="openCrawlModal()" title="View autonomous crawler status & controls">
        <span id="crawler-dot" class="status-dot"></span>
        <span id="crawler-pill-text">Crawler: Idle</span>
      </div>
      <button class="btn btn-primary" onclick="openCrawlModal()">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="m4.93 4.93 4.24 4.24"/><path d="m14.83 9.17 4.24-4.24"/><path d="m14.83 14.83 4.24 4.24"/><path d="m9.17 14.83-4.24 4.24"/></svg>
        Live Crawler
      </button>
    </div>
  </header>

  <main>
    <div class="search-section">
      <div class="search-box-wrapper">
        <input 
          type="text" 
          id="query-input" 
          class="search-input" 
          placeholder="Search documents, terms, or enter prefix*..." 
          autocomplete="off"
          autofocus
        />
        <button id="search-btn" class="search-btn" title="Search">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>
        </button>
        <div id="autocomplete" class="autocomplete-dropdown"></div>
      </div>
    </div>

    <div class="search-meta">
      <span id="results-count">Type a query to begin searching.</span>
      <span id="did-you-mean-container"></span>
    </div>

    <div id="results" class="results-container">
      <div class="empty-state">
        <h3>Welcome to Nexora</h3>
        <p>Type your query above or try <b>"inverted index"</b>, <b>"rust"</b>, <b>"bm25"</b>, or <b>"crawler*"</b>.</p>
      </div>
    </div>
  </main>

  <!-- Continuous Crawler Modal -->
  <div id="crawl-modal" class="modal-backdrop">
    <div class="modal-card">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem;">
        <h3 style="font-size:1.25rem; font-weight:600; margin:0;">🕷 Autonomous Background Crawler</h3>
        <button class="btn" style="padding:0.25rem 0.6rem;" onclick="closeCrawlModal()">✕</button>
      </div>
      <p style="font-size:0.85rem; color:var(--text-muted); margin-top:0; margin-bottom:0.75rem;">
        Self-updating crawler traverses links, ingests pages into the index in real-time, and recalculates PageRank across the web graph.
      </p>

      <div class="metrics-grid">
        <div class="metric-card">
          <div id="tele-pages" class="metric-num" style="color:#10b981;">0</div>
          <div class="metric-label">Pages Crawled</div>
        </div>
        <div class="metric-card">
          <div id="tele-queue" class="metric-num" style="color:#818cf8;">0</div>
          <div class="metric-label">Queue Size</div>
        </div>
        <div class="metric-card">
          <div id="tele-disallowed" class="metric-num" style="color:#fbbf24;">0</div>
          <div class="metric-label">Disallowed</div>
        </div>
        <div class="metric-card">
          <div id="tele-failed" class="metric-num" style="color:#f87171;">0</div>
          <div class="metric-label">Failed</div>
        </div>
      </div>

      <div id="crawler-ticker" class="crawler-ticker">
        <b id="ticker-status" style="color:#94a3b8;">IDLE</b>
        <span id="ticker-url" style="color:var(--text-muted);">Frontier ready for seeds</span>
      </div>

      <div style="display:grid; grid-template-columns: 2fr 1fr 1fr; gap:0.75rem; margin-bottom:0.85rem;">
        <div>
          <label class="form-label">Seed URLs (comma-separated)</label>
          <input type="text" id="crawl-url" class="form-input" placeholder="https://example.com" value="https://example.com" />
        </div>
        <div>
          <label class="form-label">Max Pages</label>
          <input type="number" id="crawl-pages" class="form-input" value="15" min="1" max="500" />
        </div>
        <div>
          <label class="form-label">Delay (ms)</label>
          <input type="number" id="crawl-delay" class="form-input" value="400" min="50" step="50" />
        </div>
      </div>

      <div style="display:flex; gap:0.5rem; margin-bottom:1rem;">
        <button id="btn-start" class="btn btn-primary" onclick="crawlerStart()">▶ Start Crawl</button>
        <button id="btn-pause" class="btn btn-warning" onclick="crawlerPause()">⏸ Pause</button>
        <button id="btn-resume" class="btn btn-primary" onclick="crawlerResume()">▶ Resume</button>
        <button id="btn-stop" class="btn btn-danger" onclick="crawlerStop()">⏹ Stop</button>
      </div>

      <div>
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
          <span class="form-label" style="margin:0;">Recent Crawled Documents Feed</span>
          <span id="last-pr-label" style="font-size:0.75rem; color:var(--text-dim);">PageRank: active</span>
        </div>
        <div id="crawler-feed" class="feed-container">
          <div style="padding:1.5rem; text-align:center; color:var(--text-dim); font-size:0.85rem;">No recent crawls in active session. Click Start to begin.</div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const queryInput = document.getElementById('query-input');
    const searchBtn = document.getElementById('search-btn');
    const autocomplete = document.getElementById('autocomplete');
    const resultsContainer = document.getElementById('results');
    const resultsCount = document.getElementById('results-count');
    const didYouMeanContainer = document.getElementById('did-you-mean-container');
    const statsBadge = document.getElementById('stats-badge');
    const crawlerDot = document.getElementById('crawler-dot');
    const crawlerPillText = document.getElementById('crawler-pill-text');

    let debounceTimer = null;
    let selectedIndex = -1;

    // Load initial stats
    function refreshStats() {
      fetch('/api/stats')
        .then(res => res.json())
        .then(stats => {
          statsBadge.textContent = `${stats.total_documents} docs · ${stats.vocabulary_size} terms`;
        })
        .catch(() => {
          statsBadge.textContent = 'Nexora Engine';
        });
    }
    refreshStats();

    // Handle Input Typing & Autocomplete
    queryInput.addEventListener('input', (e) => {
      const q = e.target.value.trim();
      clearTimeout(debounceTimer);

      if (q.length < 2) {
        autocomplete.style.display = 'none';
        return;
      }

      debounceTimer = setTimeout(() => {
        const words = q.split(/\s+/);
        const lastWord = words[words.length - 1];

        fetch(`/api/suggest?q=${encodeURIComponent(lastWord)}&k=6`)
          .then(res => res.json())
          .then(data => {
            if (data.suggestions && data.suggestions.length > 0) {
              renderAutocomplete(data.suggestions, words.slice(0, -1).join(' '), lastWord);
            } else {
              autocomplete.style.display = 'none';
            }
          })
          .catch(() => {
            autocomplete.style.display = 'none';
          });
      }, 150);
    });

    function renderAutocomplete(suggestions, prefixBefore, currentPrefix) {
      selectedIndex = -1;
      autocomplete.innerHTML = '';
      
      suggestions.forEach(item => {
        const div = document.createElement('div');
        div.className = 'autocomplete-item';
        
        const fullTerm = prefixBefore ? `${prefixBefore} ${item.term}` : item.term;
        
        const termSpan = document.createElement('span');
        termSpan.className = 'autocomplete-term';
        termSpan.innerHTML = item.term.replace(new RegExp(`^(${escapeRegExp(currentPrefix)})`, 'i'), '<mark>$1</mark>');

        const freqSpan = document.createElement('span');
        freqSpan.className = 'autocomplete-freq';
        freqSpan.textContent = `${item.doc_frequency} docs`;

        div.appendChild(termSpan);
        div.appendChild(freqSpan);

        div.addEventListener('click', () => {
          queryInput.value = fullTerm;
          autocomplete.style.display = 'none';
          performSearch(fullTerm);
        });

        autocomplete.appendChild(div);
      });

      autocomplete.style.display = 'block';
    }

    function escapeRegExp(string) {
      return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }

    queryInput.addEventListener('keydown', (e) => {
      const items = autocomplete.querySelectorAll('.autocomplete-item');
      if (autocomplete.style.display === 'block' && items.length > 0) {
        if (e.key === 'ArrowDown') {
          e.preventDefault();
          selectedIndex = (selectedIndex + 1) % items.length;
          updateActiveItem(items);
          return;
        } else if (e.key === 'ArrowUp') {
          e.preventDefault();
          selectedIndex = (selectedIndex - 1 + items.length) % items.length;
          updateActiveItem(items);
          return;
        } else if (e.key === 'Enter' && selectedIndex >= 0) {
          e.preventDefault();
          items[selectedIndex].click();
          return;
        }
      }

      if (e.key === 'Enter') {
        autocomplete.style.display = 'none';
        performSearch(queryInput.value.trim());
      } else if (e.key === 'Escape') {
        autocomplete.style.display = 'none';
      }
    });

    function updateActiveItem(items) {
      items.forEach((item, idx) => {
        item.classList.toggle('active', idx === selectedIndex);
      });
    }

    document.addEventListener('click', (e) => {
      if (!e.target.closest('.search-box-wrapper')) {
        autocomplete.style.display = 'none';
      }
    });

    searchBtn.addEventListener('click', () => {
      autocomplete.style.display = 'none';
      performSearch(queryInput.value.trim());
    });

    // Execute Search
    function performSearch(query) {
      if (!query) return;

      resultsCount.textContent = 'Searching index...';
      didYouMeanContainer.innerHTML = '';

      fetch(`/api/search?q=${encodeURIComponent(query)}&k=20`)
        .then(res => res.json())
        .then(data => {
          resultsCount.textContent = `Found ${data.total} results in ${data.took_ms.toFixed(2)} ms`;

          if (data.did_you_mean) {
            didYouMeanContainer.innerHTML = `
              <span class="did-you-mean">Did you mean: <a onclick="setQueryAndSearch('${escapeHtml(data.did_you_mean)}')">${escapeHtml(data.did_you_mean)}</a>?</span>
            `;
          }

          if (!data.results || data.results.length === 0) {
            resultsContainer.innerHTML = `
              <div class="empty-state">
                <h3>No documents matched your query</h3>
                <p>Try searching for broader terms or check spelling.</p>
              </div>
            `;
            return;
          }

          resultsContainer.innerHTML = '';
          data.results.forEach(item => {
            const card = document.createElement('div');
            card.className = 'result-card';

            const header = document.createElement('div');
            header.className = 'result-header';

            const title = document.createElement('a');
            title.className = 'result-title';
            title.href = item.url || '#';
            title.target = item.url ? '_blank' : '_self';
            title.textContent = `${item.rank}. ${item.title}`;

            const badges = document.createElement('div');
            badges.className = 'result-badges';
            badges.innerHTML = `
              <span class="badge badge-score" title="BM25 Hybrid Relevance Score">BM25: ${item.score.toFixed(3)}</span>
              <span class="badge badge-pr" title="PageRank Authority Mass">PR: ${item.pagerank.toFixed(3)}</span>
            `;

            header.appendChild(title);
            header.appendChild(badges);
            card.appendChild(header);

            if (item.url) {
              const urlEl = document.createElement('div');
              urlEl.className = 'result-url';
              urlEl.textContent = item.url;
              card.appendChild(urlEl);
            }

            const snippet = document.createElement('div');
            snippet.className = 'result-snippet';
            snippet.innerHTML = item.snippet;
            card.appendChild(snippet);

            resultsContainer.appendChild(card);
          });
        })
        .catch(err => {
          resultsCount.textContent = 'Search failed';
          resultsContainer.innerHTML = `<div class="empty-state"><p style="color:#ef4444;">Error: ${err.message}</p></div>`;
        });
    }

    function setQueryAndSearch(q) {
      queryInput.value = q;
      performSearch(q);
    }

    function searchDoc(title) {
      closeCrawlModal();
      queryInput.value = title;
      performSearch(title);
    }

    function escapeHtml(str) {
      return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    // Modal logic
    function openCrawlModal() {
      document.getElementById('crawl-modal').style.display = 'flex';
      pollCrawlerTelemetry();
    }

    function closeCrawlModal() {
      document.getElementById('crawl-modal').style.display = 'none';
    }

    // Live Continuous Crawler Controls & Telemetry
    function updateCrawlerUI(tele) {
      document.getElementById('tele-pages').textContent = tele.pages_visited;
      document.getElementById('tele-queue').textContent = tele.queue_size;
      document.getElementById('tele-disallowed').textContent = tele.pages_disallowed;
      document.getElementById('tele-failed').textContent = tele.pages_failed;

      crawlerDot.className = 'status-dot';
      const tickerStatus = document.getElementById('ticker-status');
      const tickerUrl = document.getElementById('ticker-url');

      if (tele.status === 'running') {
        crawlerDot.classList.add('running');
        crawlerPillText.textContent = `Crawler: Running (${tele.pages_visited} indexed)`;
        tickerStatus.textContent = 'RUNNING';
        tickerStatus.style.color = '#10b981';
        tickerUrl.textContent = tele.current_url || 'Fetching next URL from queue...';
      } else if (tele.status === 'paused') {
        crawlerDot.classList.add('paused');
        crawlerPillText.textContent = 'Crawler: Paused';
        tickerStatus.textContent = 'PAUSED';
        tickerStatus.style.color = '#fbbf24';
        tickerUrl.textContent = 'Worker paused; queue preserved.';
      } else {
        crawlerPillText.textContent = tele.pages_visited > 0 ? `Crawler: Idle (${tele.pages_visited} docs)` : 'Crawler: Idle';
        tickerStatus.textContent = 'IDLE';
        tickerStatus.style.color = '#94a3b8';
        tickerUrl.textContent = 'Frontier ready for seeds.';
      }

      if (tele.last_pagerank_update) {
        document.getElementById('last-pr-label').textContent = 'PageRank: freshly computed';
      }

      // Render recent crawled pages feed
      const feedContainer = document.getElementById('crawler-feed');
      if (tele.recent_pages && tele.recent_pages.length > 0) {
        feedContainer.innerHTML = '';
        tele.recent_pages.forEach(p => {
          const item = document.createElement('div');
          item.className = 'feed-item';

          let statusBadgeClass = 'feed-badge-indexed';
          if (p.status === 'Disallowed') statusBadgeClass = 'feed-badge-disallowed';
          if (p.status === 'Failed') statusBadgeClass = 'feed-badge-failed';

          item.innerHTML = `
            <div style="display:flex; align-items:center; gap:0.5rem; overflow:hidden;">
              <span class="badge badge-score" style="font-size:0.7rem;">#${p.doc_id}</span>
              <a class="feed-title" onclick="searchDoc('${escapeHtml(p.title)}')" title="Click to search: ${escapeHtml(p.title)}">${escapeHtml(p.title)}</a>
              <span style="font-size:0.75rem; color:var(--text-dim); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:180px;">${escapeHtml(p.url)}</span>
            </div>
            <div style="display:flex; align-items:center; gap:0.5rem; flex-shrink:0;">
              <span style="font-size:0.72rem; color:var(--text-dim);">${p.outgoing_links_count} links</span>
              <span class="feed-badge ${statusBadgeClass}">${p.status}</span>
            </div>
          `;
          feedContainer.appendChild(item);
        });
      }

      refreshStats();
    }

    function pollCrawlerTelemetry() {
      fetch('/api/crawler/status')
        .then(res => res.json())
        .then(tele => {
          updateCrawlerUI(tele);
        })
        .catch(() => {});
    }

    // Auto poll every 1500ms
    setInterval(pollCrawlerTelemetry, 1500);

    function crawlerStart() {
      const url = document.getElementById('crawl-url').value.trim();
      const pages = document.getElementById('crawl-pages').value;
      const delay = document.getElementById('crawl-delay').value;

      fetch(`/api/crawler/start?url=${encodeURIComponent(url)}&max_pages=${pages}&delay_ms=${delay}`, { method: 'POST' })
        .then(res => res.json())
        .then(data => {
          pollCrawlerTelemetry();
        })
        .catch(err => {
          alert(`Failed to start crawler: ${err.message}`);
        });
    }

    function crawlerPause() {
      fetch('/api/crawler/pause', { method: 'POST' })
        .then(() => pollCrawlerTelemetry());
    }

    function crawlerResume() {
      fetch('/api/crawler/resume', { method: 'POST' })
        .then(() => pollCrawlerTelemetry());
    }

    function crawlerStop() {
      fetch('/api/crawler/stop', { method: 'POST' })
        .then(() => pollCrawlerTelemetry());
    }
  </script>
</body>
</html>
"#
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_percent_decode() {
        assert_eq!(percent_decode("hello%20world"), "hello world");
        assert_eq!(percent_decode("rust+programming"), "rust programming");
        assert_eq!(percent_decode("a%26b%3Dc"), "a&b=c");
    }

    #[test]
    fn test_parse_query_params() {
        let params = parse_query_params("q=rust%20systems&k=10");
        assert_eq!(params.get("q").unwrap(), "rust systems");
        assert_eq!(params.get("k").unwrap(), "10");
    }

    #[test]
    fn test_json_escape() {
        assert_eq!(json_escape("hello \"world\""), "hello \\\"world\\\"");
        assert_eq!(json_escape("line1\nline2"), "line1\\nline2");
    }

    #[test]
    fn test_api_search_output() {
        let state = SearchEngineState::default_demo();
        let json = handle_api_search(&state, "rust", 5);
        assert!(json.contains("\"query\":\"rust\""));
        assert!(json.contains("\"total\":"));
        assert!(json.contains("\"results\":["));
        assert!(json.contains("Rust Systems Programming"));
    }

    #[test]
    fn test_api_suggest_output() {
        let state = SearchEngineState::default_demo();
        let json = handle_api_suggest(&state, "sea", 5);
        assert!(json.contains("\"prefix\":\"sea\""));
        assert!(json.contains("\"suggestions\":["));
        assert!(json.contains("search"));
    }

    #[test]
    fn test_incremental_ingest_crawled_page() {
        let mut state = SearchEngineState::default_demo();
        let initial_doc_count = state.index.total_documents();

        let crawled = CrawledDocument {
            doc_id: 0,
            url: "https://nexora.dev/blog/autonomous-crawler".to_string(),
            title: "Autonomous Web Crawling and Incremental Indexing".to_string(),
            text: "Nexora features continuous real-time crawling with zero index downtime and live PageRank recomputation.".to_string(),
            outgoing_links: vec!["https://nexora.dev/docs/inverted-index".to_string()],
        };

        let new_id = state.ingest_crawled_page(&crawled, 0.45);
        assert_eq!(state.index.total_documents(), initial_doc_count + 1);
        assert_eq!(
            state.doc_metadata.get(&new_id).unwrap().title,
            "Autonomous Web Crawling and Incremental Indexing"
        );
        assert_eq!(state.doc_metadata.get(&new_id).unwrap().pagerank, 0.45);

        // Immediate search retrieval
        let json = handle_api_search(&state, "autonomous crawling", 5);
        assert!(json.contains("Autonomous Web Crawling"));

        // Immediate autocomplete suggestion
        let suggest_json = handle_api_suggest(&state, "auton", 5);
        assert!(suggest_json.contains("autonomous"));
    }

    #[test]
    fn test_update_pageranks_in_state() {
        let mut state = SearchEngineState::default_demo();
        let mut new_scores = HashMap::new();
        new_scores.insert(0, 0.99);
        new_scores.insert(1, 0.88);

        state.update_pageranks(&new_scores);
        assert_eq!(state.doc_metadata.get(&0).unwrap().pagerank, 0.99);
        assert_eq!(state.doc_metadata.get(&1).unwrap().pagerank, 0.88);
    }
}
