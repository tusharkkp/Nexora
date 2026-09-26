use std::thread;
use std::time::Duration;

use crate::crawler::fetcher::PageFetcher;
use crate::crawler::frontier::UrlFrontier;
use crate::crawler::html::extract_page;
use crate::index::{DocId, InvertedIndex};

/// Configuration options controlling crawler behavior.
#[derive(Debug, Clone)]
pub struct CrawlConfig {
    /// Maximum number of successfully visited pages before terminating crawl
    pub max_pages: usize,
    /// Maximum time to wait for domain politeness cooldown before giving up (if blocked)
    pub max_wait: Duration,
}

impl Default for CrawlConfig {
    fn default() -> Self {
        Self {
            max_pages: 50,
            max_wait: Duration::from_secs(10),
        }
    }
}

/// Metadata and content for an indexed page.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrawledDocument {
    /// The document identifier assigned in the search engine index
    pub doc_id: DocId,
    /// The canonical URL where this document was discovered
    pub url: String,
    /// The extracted HTML title (if any)
    pub title: String,
    /// The clean, stripped body text
    pub text: String,
}

/// Execution summary returned upon completing a crawl session.
#[derive(Debug, Clone)]
pub struct CrawlSummary {
    /// Total pages successfully fetched and ingested into the index
    pub pages_visited: usize,
    /// Total network/fetch failures encountered
    pub pages_failed: usize,
    /// Total new hyperlinks discovered and queued
    pub links_discovered: usize,
    /// Registry of all crawled documents mapped to their assigned DocIds
    pub documents: Vec<CrawledDocument>,
}

/// The Crawler engine coordinates graph traversal, politeness scheduling,
/// content fetching, HTML parsing, and inverted index ingestion.
pub struct Crawler<F: PageFetcher> {
    frontier: UrlFrontier,
    fetcher: F,
    config: CrawlConfig,
}

impl<F: PageFetcher> Crawler<F> {
    /// Constructs a new Crawler with a frontier, a page fetcher, and configuration.
    pub fn new(frontier: UrlFrontier, fetcher: F, config: CrawlConfig) -> Self {
        Self {
            frontier,
            fetcher,
            config,
        }
    }

    /// Returns a reference to the internal URL frontier.
    pub fn frontier(&self) -> &UrlFrontier {
        &self.frontier
    }

    /// Returns a mutable reference to the internal URL frontier (e.g. to add seeds).
    pub fn frontier_mut(&mut self) -> &mut UrlFrontier {
        &mut self.frontier
    }

    /// Executes the crawling loop, feeding crawled pages directly into `index`.
    ///
    /// Stops when:
    /// - `max_pages` has been reached, OR
    /// - The frontier has no remaining URLs to crawl.
    pub fn crawl(&mut self, index: &mut InvertedIndex) -> CrawlSummary {
        let mut documents = Vec::new();
        let mut pages_visited = 0;
        let mut pages_failed = 0;
        let mut links_discovered = 0;

        while pages_visited < self.config.max_pages {
            // Check if frontier has any pending URLs
            if self.frontier.is_empty() {
                break;
            }

            // Attempt to pop a URL whose domain politeness cooldown has expired
            let next_url = match self.frontier.pop_polite() {
                Some(url) => url,
                None => {
                    // All pending domains are in cooldown. Check how long to sleep.
                    if let Some(delay) = self.frontier.next_delay() {
                        if delay > self.config.max_wait {
                            // If wait exceeds limit, avoid hanging
                            break;
                        }
                        thread::sleep(delay);
                        continue;
                    } else {
                        break;
                    }
                }
            };

            // Fetch page content
            match self.fetcher.fetch(&next_url) {
                Ok(html) => {
                    pages_visited += 1;
                    let page = extract_page(&next_url, &html);

                    // Allocate next document ID (1-indexed based on current total documents)
                    let doc_id = (index.total_documents() + 1) as DocId;

                    // Combine title and text so that title keywords are searchable
                    let indexed_content = if page.title.is_empty() {
                        page.text.clone()
                    } else {
                        format!("{} {}", page.title, page.text)
                    };

                    index.add_document(doc_id, &indexed_content);

                    // Feed discovered hyperlinks back into the frontier
                    for link in &page.outgoing_links {
                        if self.frontier.push(link) {
                            links_discovered += 1;
                        }
                    }

                    documents.push(CrawledDocument {
                        doc_id,
                        url: next_url,
                        title: page.title,
                        text: page.text,
                    });
                }
                Err(_) => {
                    pages_failed += 1;
                }
            }
        }

        CrawlSummary {
            pages_visited,
            pages_failed,
            links_discovered,
            documents,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crawler::fetcher::MockFetcher;
    use crate::crawler::frontier::UrlFrontier;
    use crate::index::InvertedIndex;
    use crate::ranking::rank_bm25;

    #[test]
    fn test_crawler_multi_page_traversal_and_indexing() {
        let mut fetcher = MockFetcher::new();

        // Page A links to Page B and Page C
        fetcher.add_page(
            "https://rust-lang.org/index.html",
            r#"
            <html>
                <head><title>Rust Programming Language</title></head>
                <body>
                    <p>Rust empowers developers to build reliable, memory-safe software.</p>
                    <a href="/tools.html">Developer Tools</a>
                    <a href="https://crates.io/featured">Package Registry</a>
                </body>
            </html>
            "#,
        );

        // Page B links back to Page A (circular) and has unique tools content
        fetcher.add_page(
            "https://rust-lang.org/tools.html",
            r#"
            <html>
                <head><title>Rust Tools &amp; Cargo</title></head>
                <body>
                    <p>Cargo is the world-class Rust package manager and build system.</p>
                    <a href="/index.html">Back to Home</a>
                </body>
            </html>
            "#,
        );

        // Page C (different domain: crates.io)
        fetcher.add_page(
            "https://crates.io/featured",
            r#"
            <html>
                <head><title>Crates.io Featured Packages</title></head>
                <body>
                    <p>Discover thousands of open-source libraries and crates built in Rust.</p>
                </body>
            </html>
            "#,
        );

        let mut frontier = UrlFrontier::new(Duration::from_millis(0));
        frontier.push("https://rust-lang.org/index.html");

        let mut index = InvertedIndex::new();
        let config = CrawlConfig {
            max_pages: 10,
            max_wait: Duration::from_millis(50),
        };

        let mut crawler = Crawler::new(frontier, fetcher, config);
        let summary = crawler.crawl(&mut index);

        // All 3 pages should be crawled and indexed
        assert_eq!(summary.pages_visited, 3);
        assert_eq!(summary.pages_failed, 0);
        assert_eq!(summary.documents.len(), 3);
        assert_eq!(index.total_documents(), 3);

        // Deduplication verified: Page A was not visited twice despite circular link in Page B
        assert_eq!(crawler.frontier().seen_count(), 3);

        // Execute BM25 search on the resulting index!
        let search_results = rank_bm25(
            &index,
            "cargo package manager",
            &crate::ranking::BM25Params::default(),
        );
        assert!(!search_results.is_empty());

        let top_match = &search_results[0];
        // Doc 2 ("Rust Tools & Cargo") should have the highest BM25 score
        let matched_doc = summary
            .documents
            .iter()
            .find(|d| d.doc_id == top_match.doc_id)
            .expect("document must exist in summary");

        assert_eq!(matched_doc.url, "https://rust-lang.org/tools.html");
        assert_eq!(matched_doc.title, "Rust Tools & Cargo");
    }

    #[test]
    fn test_crawler_max_pages_limit() {
        let mut fetcher = MockFetcher::new();

        // Infinite chain: p1 -> p2 -> p3 -> p4 -> ...
        for i in 1..=10 {
            let current = format!("https://example.com/p{}", i);
            let next = format!("https://example.com/p{}", i + 1);
            let html = format!(
                r#"<html><head><title>Page {}</title></head><body><a href="{}">Next</a></body></html>"#,
                i, next
            );
            fetcher.add_page(&current, &html);
        }

        let mut frontier = UrlFrontier::new(Duration::from_millis(0));
        frontier.push("https://example.com/p1");

        let mut index = InvertedIndex::new();
        let config = CrawlConfig {
            max_pages: 3, // strictly limit to 3 pages
            max_wait: Duration::from_millis(50),
        };

        let mut crawler = Crawler::new(frontier, fetcher, config);
        let summary = crawler.crawl(&mut index);

        assert_eq!(summary.pages_visited, 3);
        assert_eq!(index.total_documents(), 3);
        assert_eq!(summary.documents.len(), 3);
    }
}
