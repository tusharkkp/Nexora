/// URL parsing, normalization, and relative link resolution.
///
/// Designed to canonicalize web addresses, prevent crawler traps,
/// and eliminate duplicate URLs before they enter the URL Frontier.

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct ParsedUrl {
    pub scheme: String,
    pub host: String,
    pub port: Option<u16>,
    pub path: String,
    pub query: Option<String>,
}

impl ParsedUrl {
    /// Formats the parsed URL back into a canonical normalized string.
    pub fn to_string(&self) -> String {
        let mut result = format!("{}://{}", self.scheme, self.host);

        if let Some(port) = self.port {
            result.push_str(&format!(":{}", port));
        }

        result.push_str(&self.path);

        if let Some(ref query) = self.query {
            result.push('?');
            result.push_str(query);
        }

        result
    }

    /// Returns the host with port if non-default (used for per-domain politeness queues).
    pub fn host_key(&self) -> String {
        if let Some(port) = self.port {
            format!("{}:{}", self.host, port)
        } else {
            self.host.clone()
        }
    }
}

/// Normalizes an input URL into canonical form.
///
/// Steps:
/// 1. Verifies scheme is "http" or "https".
/// 2. Lowercases scheme and host.
/// 3. Removes default ports (:80 for http, :443 for https).
/// 4. Strips fragment identifiers (`#heading`).
/// 5. Strips marketing/tracking query parameters (`utm_*`, `ref`, `fbclid`).
/// 6. Sorts remaining query parameters alphabetically.
/// 7. Resolves dot segments (`.` and `..`) in path.
pub fn normalize_url(raw_url: &str) -> Result<String, &'static str> {
    let parsed = parse_url(raw_url)?;
    Ok(parsed.to_string())
}

/// Parses a URL string into structured components.
pub fn parse_url(raw: &str) -> Result<ParsedUrl, &'static str> {
    let trimmed = raw.trim();

    // 1. Strip in-page fragment (#...)
    let without_fragment = match trimmed.find('#') {
        Some(idx) => &trimmed[..idx],
        None => trimmed,
    };

    if without_fragment.is_empty() {
        return Err("URL is empty");
    }

    // 2. Extract scheme (e.g. "https://")
    let scheme_end = without_fragment
        .find("://")
        .ok_or("Missing scheme (must start with http:// or https://)")?;
    let raw_scheme = &without_fragment[..scheme_end];
    let scheme = raw_scheme.to_lowercase();

    if scheme != "http" && scheme != "https" {
        return Err("Unsupported scheme (only http and https are supported)");
    }

    let remainder = &without_fragment[scheme_end + 3..];

    // 3. Separate authority, path, and query
    let (authority_str, path_and_query) = match remainder.find('/') {
        Some(idx) => (&remainder[..idx], &remainder[idx..]),
        None => match remainder.find('?') {
            Some(idx) => (&remainder[..idx], &remainder[idx..]),
            None => (remainder, "/"),
        },
    };

    // Strip credentials (user:pass@) if present in authority
    let host_and_port = match authority_str.rfind('@') {
        Some(idx) => &authority_str[idx + 1..],
        None => authority_str,
    };

    if host_and_port.is_empty() {
        return Err("Missing host in URL");
    }

    // Split host and port
    let (raw_host, port) = match host_and_port.find(':') {
        Some(idx) => {
            let h = &host_and_port[..idx];
            let p_str = &host_and_port[idx + 1..];
            let p: u16 = p_str.parse().map_err(|_| "Invalid port number")?;
            (h, Some(p))
        }
        None => (host_and_port, None),
    };

    let host = raw_host.to_lowercase();
    if host.is_empty() {
        return Err("Host cannot be empty");
    }

    // Strip default ports
    let normalized_port = match (scheme.as_str(), port) {
        ("http", Some(80)) => None,
        ("https", Some(443)) => None,
        (_, other) => other,
    };

    // Split path and query
    let (raw_path, raw_query) = match path_and_query.find('?') {
        Some(idx) => (&path_and_query[..idx], Some(&path_and_query[idx + 1..])),
        None => (path_and_query, None),
    };

    // Normalize path: default empty to "/" and resolve "." and ".."
    let resolved_path = normalize_path(if raw_path.is_empty() { "/" } else { raw_path });

    // Normalize query: filter tracking parameters and sort alphabetically
    let normalized_query = raw_query.and_then(normalize_query_string);

    Ok(ParsedUrl {
        scheme,
        host,
        port: normalized_port,
        path: resolved_path,
        query: normalized_query,
    })
}

/// Resolves a relative URL against an absolute base URL.
///
/// Examples:
/// - Base: `https://example.com/docs/intro`
/// - Relative `/about` -> `https://example.com/about`
/// - Relative `guide.html` -> `https://example.com/docs/guide.html`
/// - Relative `../contact` -> `https://example.com/contact`
pub fn resolve_relative_url(base_url: &str, relative: &str) -> Result<String, &'static str> {
    let rel = relative.trim();
    if rel.is_empty() {
        return Err("Relative URL is empty");
    }

    // 1. If already absolute http/https, normalize directly
    if rel.starts_with("http://") || rel.starts_with("https://") {
        return normalize_url(rel);
    }

    // 2. Reject non-web schemes
    if rel.starts_with("mailto:")
        || rel.starts_with("javascript:")
        || rel.starts_with("tel:")
        || rel.starts_with("data:")
    {
        return Err("Non-web scheme ignored");
    }

    let base = parse_url(base_url)?;

    // 3. Scheme-relative URL: `//cdn.example.com/file.js`
    if rel.starts_with("//") {
        return normalize_url(&format!("{}:{}", base.scheme, rel));
    }

    // 4. Fragment-only: `#heading`
    if rel.starts_with('#') {
        return Ok(base.to_string());
    }

    // 5. Query-only: `?page=2`
    if rel.starts_with('?') {
        let base_no_query = format!("{}://{}{}", base.scheme, base.host_key(), base.path);
        return normalize_url(&format!("{}{}", base_no_query, rel));
    }

    // 6. Root-relative: `/about/index.html`
    if rel.starts_with('/') {
        let full = format!("{}://{}{}", base.scheme, base.host_key(), rel);
        return normalize_url(&full);
    }

    // 7. Path-relative: `page2.html` or `../page2.html`
    let base_dir = match base.path.rfind('/') {
        Some(idx) => &base.path[..=idx],
        None => "/",
    };

    let combined_path = format!("{}{}", base_dir, rel);
    let full = format!("{}://{}{}", base.scheme, base.host_key(), combined_path);
    normalize_url(&full)
}

// -----------------------------------------------------------------------------
// Internal Helpers
// -----------------------------------------------------------------------------

/// Resolves path segments (`.` and `..`).
fn normalize_path(path: &str) -> String {
    let mut segments = Vec::new();
    let is_absolute = path.starts_with('/');

    for part in path.split('/') {
        match part {
            "" | "." => continue,
            ".." => {
                segments.pop();
            }
            other => segments.push(other),
        }
    }

    let mut result = if is_absolute {
        format!("/{}", segments.join("/"))
    } else {
        segments.join("/")
    };

    if path.ends_with('/') && !result.ends_with('/') {
        result.push('/');
    }

    if result.is_empty() {
        "/".to_string()
    } else {
        result
    }
}

/// Normalizes query string: strips tracking params (`utm_*`, `ref`, etc.) and sorts keys.
fn normalize_query_string(query_str: &str) -> Option<String> {
    let mut valid_pairs: Vec<(&str, &str)> = Vec::new();

    for pair in query_str.split('&') {
        if pair.is_empty() {
            continue;
        }
        let (key, val) = match pair.find('=') {
            Some(idx) => (&pair[..idx], &pair[idx + 1..]),
            None => (pair, ""),
        };

        // Filter out analytics and tracking noise
        let lower_key = key.to_lowercase();
        if lower_key.starts_with("utm_")
            || lower_key == "ref"
            || lower_key == "fbclid"
            || lower_key == "gclid"
        {
            continue;
        }

        valid_pairs.push((key, val));
    }

    if valid_pairs.is_empty() {
        return None;
    }

    // Sort query pairs by key for canonical order
    valid_pairs.sort_by(|a, b| a.0.cmp(b.0));

    let formatted: Vec<String> = valid_pairs
        .into_iter()
        .map(|(k, v)| {
            if v.is_empty() {
                k.to_string()
            } else {
                format!("{}={}", k, v)
            }
        })
        .collect();

    Some(formatted.join("&"))
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheme_and_host_lowercasing() {
        let raw = "HTTP://EXAMPLE.COM/Docs/Index.html";
        let normalized = normalize_url(raw).unwrap();
        // Host and scheme lowercased; path casing preserved
        assert_eq!(normalized, "http://example.com/Docs/Index.html");
    }

    #[test]
    fn test_default_port_stripping() {
        assert_eq!(
            normalize_url("http://example.com:80/path").unwrap(),
            "http://example.com/path"
        );
        assert_eq!(
            normalize_url("https://example.com:443/path").unwrap(),
            "https://example.com/path"
        );
        // Non-default ports are preserved
        assert_eq!(
            normalize_url("http://example.com:8080/path").unwrap(),
            "http://example.com:8080/path"
        );
    }

    #[test]
    fn test_fragment_stripping() {
        let raw = "https://example.com/page.html#section-2";
        assert_eq!(normalize_url(raw).unwrap(), "https://example.com/page.html");
    }

    #[test]
    fn test_tracking_query_stripping_and_sorting() {
        let raw = "https://example.com/search?utm_source=newsletter&q=rust&ref=twitter&page=1";
        // utm_source and ref stripped; page=1 and q=rust sorted alphabetically
        assert_eq!(
            normalize_url(raw).unwrap(),
            "https://example.com/search?page=1&q=rust"
        );
    }

    #[test]
    fn test_dot_segment_path_resolution() {
        assert_eq!(
            normalize_url("https://example.com/a/b/../c/./d").unwrap(),
            "https://example.com/a/c/d"
        );
    }

    #[test]
    fn test_relative_url_resolution() {
        let base = "https://example.com/docs/getting_started.html";

        // Root relative
        assert_eq!(
            resolve_relative_url(base, "/about").unwrap(),
            "https://example.com/about"
        );

        // Sibling relative
        assert_eq!(
            resolve_relative_url(base, "advanced.html").unwrap(),
            "https://example.com/docs/advanced.html"
        );

        // Parent relative
        assert_eq!(
            resolve_relative_url(base, "../blog").unwrap(),
            "https://example.com/blog"
        );

        // Protocol relative
        assert_eq!(
            resolve_relative_url(base, "//cdn.example.com/app.js").unwrap(),
            "https://cdn.example.com/app.js"
        );

        // Fragment only
        assert_eq!(
            resolve_relative_url(base, "#overview").unwrap(),
            "https://example.com/docs/getting_started.html"
        );

        // Non-web schemes rejected
        assert!(resolve_relative_url(base, "mailto:admin@example.com").is_err());
        assert!(resolve_relative_url(base, "javascript:void(0)").is_err());
    }
}
