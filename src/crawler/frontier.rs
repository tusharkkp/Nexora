use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};

use crate::crawler::url::{normalize_url, parse_url};

/// The URL Frontier manages the crawl queue with:
/// 1. Deduplication (never visits or queues the same normalized URL twice).
/// 2. Per-domain politeness (round-robin scheduling respecting domain rate-limits).
#[derive(Debug)]
pub struct UrlFrontier {
    /// Tracks every URL that has ever been queued or crawled (Deduplication Filter)
    seen_urls: HashSet<String>,
    /// Per-domain FIFO queues: maps host_key (e.g. "example.com") -> list of pending URLs
    domain_queues: HashMap<String, VecDeque<String>>,
    /// Active domains with pending URLs for round-robin rotation
    active_domains: VecDeque<String>,
    /// Timestamp of the last crawl dispatched to each domain
    last_access: HashMap<String, Instant>,
    /// Required minimum cooldown between requests to the same domain
    politeness_delay: Duration,
    /// Total count of pending URLs across all domain queues
    total_queued: usize,
}

impl UrlFrontier {
    /// Creates a new URL Frontier with a specified per-domain politeness delay.
    pub fn new(politeness_delay: Duration) -> Self {
        Self {
            seen_urls: HashSet::new(),
            domain_queues: HashMap::new(),
            active_domains: VecDeque::new(),
            last_access: HashMap::new(),
            politeness_delay,
            total_queued: 0,
        }
    }

    /// Default frontier with a 500ms per-domain politeness delay.
    pub fn default_polite() -> Self {
        Self::new(Duration::from_millis(500))
    }

    /// Total number of unique URLs encountered so far (queued + crawled).
    pub fn seen_count(&self) -> usize {
        self.seen_urls.len()
    }

    /// Total number of URLs currently waiting in the frontier queues.
    pub fn queue_len(&self) -> usize {
        self.total_queued
    }

    /// Returns true if there are no pending URLs in any domain queue.
    pub fn is_empty(&self) -> bool {
        self.total_queued == 0
    }

    /// Enqueues a URL into the frontier.
    ///
    /// Normalizes the URL and checks the deduplication set.
    /// Returns `true` if the URL was accepted and enqueued,
    /// or `false` if it was rejected as a duplicate, invalid, or non-web URL.
    pub fn push(&mut self, raw_url: &str) -> bool {
        let normalized = match normalize_url(raw_url) {
            Ok(url) => url,
            Err(_) => return false,
        };

        // Deduplication check: insert into seen set
        if !self.seen_urls.insert(normalized.clone()) {
            return false; // Already queued or visited
        }

        let parsed = match parse_url(&normalized) {
            Ok(p) => p,
            Err(_) => return false,
        };

        let host = parsed.host_key();

        let queue = self.domain_queues.entry(host.clone()).or_default();
        let was_empty = queue.is_empty();
        queue.push_back(normalized);
        self.total_queued += 1;

        if was_empty {
            self.active_domains.push_back(host);
        }

        true
    }

    /// Enqueues multiple seed URLs into the frontier.
    pub fn push_seeds(&mut self, seeds: &[&str]) -> usize {
        seeds.iter().filter(|&&url| self.push(url)).count()
    }

    /// Pops the next eligible URL whose per-domain politeness cooldown has passed.
    ///
    /// Uses round-robin scheduling across active domains. If all pending domains
    /// are currently cooling down, returns `None` without blocking.
    pub fn pop_polite(&mut self) -> Option<String> {
        let now = Instant::now();
        let num_domains = self.active_domains.len();

        for _ in 0..num_domains {
            let domain = self.active_domains.pop_front()?;

            // Check if this domain is ready (politeness delay has elapsed)
            let is_ready = match self.last_access.get(&domain) {
                Some(&last) => now.duration_since(last) >= self.politeness_delay,
                None => true, // First request to this domain is always ready immediately
            };

            if is_ready {
                // Pop the next URL for this domain
                if let Some(queue) = self.domain_queues.get_mut(&domain) {
                    if let Some(url) = queue.pop_front() {
                        self.total_queued -= 1;
                        self.last_access.insert(domain.clone(), now);

                        // If domain still has more pending URLs, put it back at the end of round-robin
                        if !queue.is_empty() {
                            self.active_domains.push_back(domain);
                        }

                        return Some(url);
                    }
                }
            } else {
                // Domain is still cooling down; rotate to back of queue
                self.active_domains.push_back(domain);
            }
        }

        None
    }

    /// Computes how long the crawler should sleep before the earliest domain
    /// becomes ready, or `None` if the frontier is empty.
    pub fn next_delay(&self) -> Option<Duration> {
        if self.is_empty() {
            return None;
        }

        let now = Instant::now();
        let mut min_wait: Option<Duration> = None;

        for domain in &self.active_domains {
            match self.last_access.get(domain) {
                Some(&last) => {
                    let elapsed = now.duration_since(last);
                    if elapsed >= self.politeness_delay {
                        return Some(Duration::ZERO); // A domain is ready right now!
                    }
                    let wait = self.politeness_delay - elapsed;
                    min_wait = Some(min_wait.map_or(wait, |m| m.min(wait)));
                }
                None => return Some(Duration::ZERO), // Never visited domain is ready now
            }
        }

        min_wait
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frontier_deduplication() {
        let mut frontier = UrlFrontier::default_polite();

        // Push URL variations that resolve to the same canonical URL
        assert!(frontier.push("https://example.com/docs/"));
        assert!(!frontier.push("HTTPS://EXAMPLE.COM/docs/")); // Duplicate (case)
        assert!(!frontier.push("https://example.com/docs/#heading1")); // Duplicate (fragment)
        assert!(!frontier.push("https://example.com/docs/?utm_source=twitter")); // Duplicate (tracking query)

        assert_eq!(frontier.seen_count(), 1);
        assert_eq!(frontier.queue_len(), 1);
    }

    #[test]
    fn test_multi_domain_round_robin_politeness() {
        // Set a 100ms politeness delay
        let mut frontier = UrlFrontier::new(Duration::from_millis(100));

        // Enqueue 2 URLs from example.com and 1 URL from rust-lang.org
        frontier.push("https://example.com/page1");
        frontier.push("https://example.com/page2");
        frontier.push("https://rust-lang.org/index");

        assert_eq!(frontier.queue_len(), 3);

        // First pop: immediately succeeds (e.g. example.com/page1)
        let first = frontier.pop_polite().expect("First pop should succeed");
        assert_eq!(first, "https://example.com/page1");

        // Second pop: example.com is now cooling down (100ms delay).
        // But rust-lang.org is a DIFFERENT domain, so it must pop immediately!
        let second = frontier.pop_polite().expect("Second domain should succeed immediately");
        assert_eq!(second, "https://rust-lang.org/index");

        // Third pop: example.com is still cooling down; pop_polite returns None without waiting
        assert!(frontier.pop_polite().is_none());

        // Sleep past the 100ms delay
        std::thread::sleep(Duration::from_millis(110));

        // Now example.com is ready again!
        let third = frontier.pop_polite().expect("Third pop should succeed after cooldown");
        assert_eq!(third, "https://example.com/page2");

        // Frontier is now completely empty
        assert!(frontier.is_empty());
    }
}
