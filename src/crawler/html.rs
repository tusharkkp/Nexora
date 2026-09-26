use crate::crawler::url::resolve_relative_url;

/// Extracted content from a parsed HTML document.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct ExtractedPage {
    /// The page title (extracted from `<title>...</title>`)
    pub title: String,
    /// Clean, readable text content suitable for indexing
    pub text: String,
    /// Canonical, absolute URLs extracted from `<a href="...">` links
    pub outgoing_links: Vec<String>,
}

/// Parses an HTML document, extracting clean text, the page title,
/// and resolving all outgoing hyperlinks against the `base_url`.
pub fn extract_page(base_url: &str, html: &str) -> ExtractedPage {
    let chars: Vec<char> = html.chars().collect();
    let total_len = chars.len();

    let mut title = String::new();
    let mut body_text = String::new();
    let mut outgoing_links = Vec::new();

    let mut i = 0;

    while i < total_len {
        let ch = chars[i];

        // 1. Check for start of an HTML tag or comment: '<'
        if ch == '<' {
            // Check for HTML Comment: "<!--"
            if starts_with_case_insensitive(&chars[i..], "<!--") {
                i += 4;
                while i < total_len && !starts_with_case_insensitive(&chars[i..], "-->") {
                    i += 1;
                }
                if i < total_len {
                    i += 3; // consume "-->"
                }
                continue;
            }

            // Check for <script>...</script>
            if starts_with_tag(&chars[i..], "script") {
                i = skip_tag_content(&chars, i, "script");
                continue;
            }

            // Check for <style>...</style>
            if starts_with_tag(&chars[i..], "style") {
                i = skip_tag_content(&chars, i, "style");
                continue;
            }

            // Check for <title>...</title>
            if starts_with_tag(&chars[i..], "title") {
                // Find closing '>' of opening <title>
                while i < total_len && chars[i] != '>' {
                    i += 1;
                }
                if i < total_len {
                    i += 1; // consume '>'
                }

                let mut raw_title = String::new();
                while i < total_len && !starts_with_case_insensitive(&chars[i..], "</title>") {
                    raw_title.push(chars[i]);
                    i += 1;
                }
                if i < total_len {
                    i += 8; // consume "</title>"
                }
                title = decode_html_entities(&raw_title).trim().to_string();
                continue;
            }

            // Check for anchor tag: <a ... href="..." ...>
            if starts_with_tag(&chars[i..], "a") {
                let tag_start = i;
                // Find closing '>' of <a>
                while i < total_len && chars[i] != '>' {
                    i += 1;
                }
                let tag_end = if i < total_len { i } else { total_len };
                let tag_string: String = chars[tag_start..tag_end].iter().collect();

                if let Some(href) = extract_href_attribute(&tag_string) {
                    let trimmed = href.trim();
                    // Ignore internal fragment jumps, empty links, and javascript: links
                    if !trimmed.is_empty()
                        && !trimmed.starts_with('#')
                        && !trimmed.to_lowercase().starts_with("javascript:")
                    {
                        if let Ok(resolved) = resolve_relative_url(base_url, trimmed) {
                            if !outgoing_links.contains(&resolved) {
                                outgoing_links.push(resolved);
                            }
                        }
                    }
                }

                if i < total_len {
                    i += 1; // consume '>'
                }
                continue;
            }

            // Check if this tag is a block-level boundary (insert space so words don't merge)
            let is_block = is_block_tag(&chars[i..]);

            // Skip any other generic tag until '>'
            while i < total_len && chars[i] != '>' {
                i += 1;
            }
            if i < total_len {
                i += 1; // consume '>'
            }

            if is_block && !body_text.ends_with(' ') && !body_text.is_empty() {
                body_text.push(' ');
            }
            continue;
        }

        // 2. Normal text character
        body_text.push(ch);
        i += 1;
    }

    // Decode HTML entities (e.g. &amp; -> &) and normalize whitespace
    let decoded_text = decode_html_entities(&body_text);
    let clean_text = collapse_whitespace(&decoded_text);

    ExtractedPage {
        title,
        text: clean_text,
        outgoing_links,
    }
}

// -----------------------------------------------------------------------------
// Helper Functions: Tag Parsing & Entity Decoding
// -----------------------------------------------------------------------------

/// Checks if slice starts with case-insensitive needle.
fn starts_with_case_insensitive(chars: &[char], needle: &str) -> bool {
    let needle_chars: Vec<char> = needle.chars().collect();
    if chars.len() < needle_chars.len() {
        return false;
    }
    for (a, b) in chars.iter().zip(needle_chars.iter()) {
        if a.to_lowercase().ne(b.to_lowercase()) {
            return false;
        }
    }
    true
}

/// Checks if slice begins with `<tag_name` followed by whitespace, `>`, or `/`.
fn starts_with_tag(chars: &[char], tag_name: &str) -> bool {
    if chars.is_empty() || chars[0] != '<' {
        return false;
    }
    let after_lt = &chars[1..];
    let tag_chars: Vec<char> = tag_name.chars().collect();
    if after_lt.len() < tag_chars.len() {
        return false;
    }

    for (a, b) in after_lt.iter().zip(tag_chars.iter()) {
        if a.to_lowercase().ne(b.to_lowercase()) {
            return false;
        }
    }

    let next_idx = tag_chars.len();
    if next_idx < after_lt.len() {
        let delimiter = after_lt[next_idx];
        delimiter.is_whitespace() || delimiter == '>' || delimiter == '/'
    } else {
        true
    }
}

/// Checks if tag is a block element (`<p>`, `<div>`, `<h1>`-`<h6>`, `<li>`, `<br>`, etc.).
fn is_block_tag(chars: &[char]) -> bool {
    const BLOCK_TAGS: &[&str] = &[
        "p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "br", "tr", "section", "article",
        "header", "footer", "nav", "main", "blockquote",
    ];
    for &tag in BLOCK_TAGS {
        if starts_with_tag(chars, tag) {
            return true;
        }
        // Also check closing tag </tag>
        if chars.len() > 2 && chars[0] == '<' && chars[1] == '/' && starts_with_tag(&chars[1..], tag)
        {
            return true;
        }
    }
    false
}

/// Skips everything until the closing `</tag_name>`.
fn skip_tag_content(chars: &[char], mut i: usize, tag_name: &str) -> usize {
    let closing_tag = format!("</{}>", tag_name);
    let total_len = chars.len();

    while i < total_len {
        if starts_with_case_insensitive(&chars[i..], &closing_tag) {
            i += closing_tag.chars().count();
            return i;
        }
        i += 1;
    }

    total_len
}

/// Extracts the `href="..."` attribute value from an anchor tag string.
fn extract_href_attribute(tag_string: &str) -> Option<String> {
    let lower = tag_string.to_lowercase();
    let href_idx = lower.find("href")?;

    let remainder = &tag_string[href_idx + 4..].trim_start();
    if !remainder.starts_with('=') {
        return None;
    }

    let after_eq = remainder[1..].trim_start();
    if after_eq.is_empty() {
        return None;
    }

    let quote_char = after_eq.chars().next()?;
    if quote_char == '"' || quote_char == '\'' {
        let content = &after_eq[1..];
        let end_idx = content.find(quote_char)?;
        Some(content[..end_idx].to_string())
    } else {
        // Unquoted attribute: ends at whitespace or '>'
        let end_idx = after_eq
            .find(|c: char| c.is_whitespace() || c == '>')
            .unwrap_or(after_eq.len());
        Some(after_eq[..end_idx].to_string())
    }
}

/// Decodes standard HTML entities into UTF-8 text.
pub fn decode_html_entities(text: &str) -> String {
    text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&nbsp;", " ")
}

/// Collapses runs of whitespace characters into a single space and trims edges.
fn collapse_whitespace(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut prev_is_whitespace = true; // Avoid leading space

    for ch in text.chars() {
        if ch.is_whitespace() {
            if !prev_is_whitespace {
                result.push(' ');
                prev_is_whitespace = true;
            }
        } else {
            result.push(ch);
            prev_is_whitespace = false;
        }
    }

    if result.ends_with(' ') {
        result.pop();
    }

    result
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_script_and_style_stripping() {
        let html = r#"
            <html>
                <head>
                    <style>body { background: red; margin: 0; }</style>
                    <script type="text/javascript">
                        console.log("analytics tracking code");
                    </script>
                </head>
                <body>
                    <p>Clean body content only.</p>
                </body>
            </html>
        "#;

        let page = extract_page("https://example.com/", html);
        assert_eq!(page.text, "Clean body content only.");
        assert!(!page.text.contains("margin"));
        assert!(!page.text.contains("console.log"));
    }

    #[test]
    fn test_title_extraction() {
        let html = r#"
            <html>
                <head>
                    <title>  Nexora Search Engine &amp; Architecture  </title>
                </head>
                <body><p>Content</p></body>
            </html>
        "#;

        let page = extract_page("https://example.com/", html);
        assert_eq!(page.title, "Nexora Search Engine & Architecture");
    }

    #[test]
    fn test_link_extraction_and_resolution() {
        let base = "https://example.com/docs/index.html";
        let html = r##"
            <div>
                <a href="/about">About Us</a>
                <a href="guide.html">User Guide</a>
                <a href="../contact.html">Contact</a>
                <a href="https://external.org/api">API Reference</a>
                <a href="mailto:admin@example.com">Email</a>
                <a href="#section2">Anchor</a>
            </div>
        "##;

        let page = extract_page(base, html);

        assert_eq!(
            page.outgoing_links,
            vec![
                "https://example.com/about",
                "https://example.com/docs/guide.html",
                "https://example.com/contact.html",
                "https://external.org/api",
            ]
        );
    }

    #[test]
    fn test_block_spacing_and_entity_decoding() {
        let html = "<h1>Heading</h1><p>First paragraph &amp; more.</p><p>Second paragraph.</p>";
        let page = extract_page("https://example.com/", html);

        // Ensures words from consecutive tags aren't merged like "HeadingFirst"
        assert_eq!(page.text, "Heading First paragraph & more. Second paragraph.");
    }
}
