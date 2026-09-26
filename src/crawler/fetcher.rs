use std::collections::HashMap;
use crate::crawler::url::normalize_url;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_fetcher_retrieval_and_normalization() {
        let mut fetcher = MockFetcher::new();
        fetcher.add_page("https://example.com/index.html#section", "<html><body>Hello World</body></html>");

        // The URL with or without fragment should resolve to canonical URL
        let content = fetcher.fetch("https://example.com/index.html").expect("should fetch");
        assert_eq!(content, "<html><body>Hello World</body></html>");

        let err = fetcher.fetch("https://example.com/missing");
        assert!(err.is_err());
    }
}
