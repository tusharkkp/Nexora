use crate::crawler::url::normalize_url;
use std::collections::HashMap;

/// Trait defining the interface for fetching web documents.
///
/// This abstraction allows the crawler engine to remain completely decoupled
/// from the underlying network transport (e.g. mock testbeds, local disk files,
/// or production HTTP clients like `ureq` / `reqwest`).
pub trait PageFetcher {
    /// Fetches the raw HTML content for a given URL.
    ///
    /// Returns `Ok(html_content)` on success, or `Err(error_message)` on failure.
    fn fetch(&self, url: &str) -> Result<String, String>;
}

/// An in-memory mock fetcher designed for testing crawling workflows,
/// graph traversal algorithms, and offline index construction.
#[derive(Debug, Default, Clone)]
pub struct MockFetcher {
    /// Maps normalized URLs to their raw HTML response content.
    pages: HashMap<String, String>,
}

impl MockFetcher {
    /// Creates a new, empty mock fetcher.
    pub fn new() -> Self {
        Self {
            pages: HashMap::new(),
        }
    }

    /// Registers a mock page with its corresponding HTML content.
    /// The URL is automatically canonicalized.
    pub fn add_page(&mut self, url: &str, html: &str) {
        let canonical = normalize_url(url).unwrap_or_else(|_| url.to_string());
        self.pages.insert(canonical, html.to_string());
    }

    /// Returns the number of mock pages registered.
    pub fn len(&self) -> usize {
        self.pages.len()
    }

    /// Returns true if the mock fetcher has no registered pages.
    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }
}

impl PageFetcher for MockFetcher {
    fn fetch(&self, url: &str) -> Result<String, String> {
        let canonical = normalize_url(url).unwrap_or_else(|_| url.to_string());
        self.pages
            .get(&canonical)
            .cloned()
            .ok_or_else(|| format!("404 Not Found: {}", canonical))
    }
}

use std::io::Read;
use std::time::Duration;

/// Configuration for the live HTTP page fetcher.
#[derive(Debug, Clone)]
pub struct HttpFetcherConfig {
    /// Identification string sent in HTTP request headers.
    pub user_agent: String,
    /// Network timeout for connecting and reading responses.
    pub timeout: Duration,
    /// Maximum response body size in bytes (prevents unbounded memory consumption).
    pub max_body_bytes: u64,
}

impl Default for HttpFetcherConfig {
    fn default() -> Self {
        Self {
            user_agent: "NexoraBot/0.1 (+https://github.com/tusharkkp/Nexora)".to_string(),
            timeout: Duration::from_secs(10),
            max_body_bytes: 2 * 1024 * 1024, // 2 Megabytes default limit
        }
    }
}

/// Production HTTP fetcher powered by `ureq` and pure-Rust TLS (`rustls`).
pub struct HttpFetcher {
    agent: ureq::Agent,
    config: HttpFetcherConfig,
}

impl HttpFetcher {
    /// Creates a new HTTP fetcher with custom configuration.
    pub fn new(config: HttpFetcherConfig) -> Self {
        let ureq_config = ureq::config::Config::builder()
            .timeout_global(Some(config.timeout))
            .build();
        let agent = ureq::Agent::new_with_config(ureq_config);
        Self { agent, config }
    }

    /// Creates an HTTP fetcher with default politeness settings.
    pub fn default_client() -> Self {
        Self::new(HttpFetcherConfig::default())
    }

    /// Returns a reference to the active configuration.
    pub fn config(&self) -> &HttpFetcherConfig {
        &self.config
    }
}

impl Default for HttpFetcher {
    fn default() -> Self {
        Self::default_client()
    }
}

impl PageFetcher for HttpFetcher {
    fn fetch(&self, url: &str) -> Result<String, String> {
        let response = self
            .agent
            .get(url)
            .header("User-Agent", &self.config.user_agent)
            .header(
                "Accept",
                "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.1",
            )
            .call()
            .map_err(|e| format!("HTTP request error for {}: {}", url, e))?;

        let status = response.status();
        if !status.is_success() {
            return Err(format!("HTTP error {}: {}", status.as_u16(), url));
        }

        // Validate Content-Type: only ingest text documents
        if let Some(content_type) = response.headers().get("content-type") {
            if let Ok(ct_str) = content_type.to_str() {
                let ct_lower = ct_str.to_lowercase();
                if !ct_lower.contains("text/html")
                    && !ct_lower.contains("text/plain")
                    && !ct_lower.contains("application/xhtml+xml")
                {
                    return Err(format!(
                        "Skipping non-HTML content-type '{}' for {}",
                        ct_str, url
                    ));
                }
            }
        }

        // Read response body safely up to max_body_bytes
        let mut reader = response
            .into_body()
            .into_reader()
            .take(self.config.max_body_bytes);
        let mut body = String::new();
        reader
            .read_to_string(&mut body)
            .map_err(|e| format!("Failed to read response body for {}: {}", url, e))?;

        Ok(body)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_fetcher_retrieval_and_normalization() {
        let mut fetcher = MockFetcher::new();
        fetcher.add_page(
            "https://example.com/index.html#section",
            "<html><body>Hello World</body></html>",
        );

        // The URL with or without fragment should resolve to canonical URL
        let content = fetcher
            .fetch("https://example.com/index.html")
            .expect("should fetch");
        assert_eq!(content, "<html><body>Hello World</body></html>");

        let err = fetcher.fetch("https://example.com/missing");
        assert!(err.is_err());
    }

    #[test]
    fn test_http_fetcher_instantiation() {
        let fetcher = HttpFetcher::default_client();
        assert_eq!(fetcher.config().max_body_bytes, 2 * 1024 * 1024);
        assert!(fetcher.config().user_agent.contains("NexoraBot"));
    }
}
