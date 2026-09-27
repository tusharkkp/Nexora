use std::collections::HashSet;

use crate::analyzer::{Analyzer, Term};

/// Highlighting style format for matched query keywords in search snippets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HighlightFormat {
    /// Standard HTML bold markup: `<b>keyword</b>`
    Html,
    /// Terminal ANSI bold yellow formatting: `\x1b[1m\x1b[33mkeyword\x1b[0m`
    Ansi,
    /// Custom opening and closing delimiter tags
    Custom { prefix: String, suffix: String },
}

impl HighlightFormat {
    /// Returns the opening and closing tags for this formatting style.
    pub fn tags(&self) -> (&str, &str) {
        match self {
            HighlightFormat::Html => ("<b>", "</b>"),
            HighlightFormat::Ansi => ("\x1b[1m\x1b[33m", "\x1b[0m"),
            HighlightFormat::Custom { prefix, suffix } => (prefix.as_str(), suffix.as_str()),
        }
    }
}

/// Configuration options controlling KWIC snippet extraction and formatting.
#[derive(Debug, Clone)]
pub struct SnippetConfig {
    /// Target maximum snippet length in characters
    pub max_chars: usize,
    /// Highlighting style (HTML, ANSI, or custom)
    pub format: HighlightFormat,
}

impl Default for SnippetConfig {
    fn default() -> Self {
        Self {
            max_chars: 160,
            format: HighlightFormat::Ansi,
        }
    }
}

/// Generates a Keyword-in-Context (KWIC) snippet with query terms highlighted.
///
/// Features:
/// 1. Density-based sliding window: maximizes distinct query term matches.
/// 2. Natural boundary snapping: aligns to sentence or word boundaries, never slicing words.
/// 3. In-place exact preservation: preserves original text casing and punctuation.
/// 4. Ellipsis handling: prepends/appends `...` when context is truncated.
pub fn generate_snippet(
    text: &str,
    query: &str,
    analyzer: &Analyzer,
    config: &SnippetConfig,
) -> String {
    let clean_text = text.trim();
    if clean_text.is_empty() {
        return String::new();
    }

    // 1. Extract normalized query terms
    let query_terms: HashSet<String> = analyzer
        .analyze(query)
        .into_iter()
        .map(|t| t.text)
        .collect();

    // If query terms are empty, return beginning of text up to max_chars
    if query_terms.is_empty() {
        return fallback_prefix(clean_text, config.max_chars);
    }

    // 2. Analyze document into terms with precise byte offsets
    let doc_terms = analyzer.analyze(clean_text);
    let matched_indices: Vec<usize> = doc_terms
        .iter()
        .enumerate()
        .filter(|(_, t)| query_terms.contains(&t.text))
        .map(|(i, _)| i)
        .collect();

    // If no terms matched in text, return beginning excerpt
    if matched_indices.is_empty() {
        return fallback_prefix(clean_text, config.max_chars);
    }

    // If the entire text fits within max_chars, highlight all matches in full text
    if clean_text.chars().count() <= config.max_chars {
        return highlight_window(
            clean_text,
            0,
            clean_text.len(),
            &doc_terms,
            &matched_indices,
            &config.format,
            false,
            false,
        );
    }

    // 3. Find the best scoring window centered around term matches
    let (best_start, best_end) =
        select_best_window(clean_text, &doc_terms, &matched_indices, config.max_chars);

    let has_prefix = best_start > 0;
    let has_suffix = best_end < clean_text.len();

    highlight_window(
        clean_text,
        best_start,
        best_end,
        &doc_terms,
        &matched_indices,
        &config.format,
        has_prefix,
        has_suffix,
    )
}

/// Selects the highest-scoring byte range `[start, end]` in `text`.
fn select_best_window(
    text: &str,
    doc_terms: &[Term],
    matched_indices: &[usize],
    max_chars: usize,
) -> (usize, usize) {
    let mut best_score = -1.0;
    let mut best_range = (0, text.len().min(max_chars));

    for &match_idx in matched_indices {
        let center_term = &doc_terms[match_idx];
        let center_byte = (center_term.start_offset + center_term.end_offset) / 2;

        // Target a window of size roughly max_chars centered around center_byte
        let half_window = max_chars / 2;

        let raw_start = center_byte.saturating_sub(half_window);
        let raw_end = (center_byte + half_window).min(text.len());

        let (snapped_start, snapped_end) = snap_boundaries(text, raw_start, raw_end);

        // Score this window:
        // - Distinct matched query terms in window (weight: 100.0)
        // - Total matched query terms in window (weight: 10.0)
        // - Position bias: prefer matches earlier in document
        let mut distinct_in_window = HashSet::new();
        let mut total_in_window = 0;

        for &m_idx in matched_indices {
            let term = &doc_terms[m_idx];
            if term.start_offset >= snapped_start && term.end_offset <= snapped_end {
                distinct_in_window.insert(&term.text);
                total_in_window += 1;
            }
        }

        let pos_bias = 1.0 - (snapped_start as f64 / (text.len() as f64 + 1.0)) * 0.1;
        let score =
            (distinct_in_window.len() as f64 * 100.0 + total_in_window as f64 * 10.0) * pos_bias;

        if score > best_score {
            best_score = score;
            best_range = (snapped_start, snapped_end);
        }
    }

    best_range
}

/// Snaps raw byte boundaries to clean whitespace or punctuation boundaries.
fn snap_boundaries(text: &str, mut start: usize, mut end: usize) -> (usize, usize) {
    // Ensure valid char boundaries first
    start = floor_char_boundary(text, start);
    end = ceil_char_boundary(text, end);

    // Snap start to next word boundary if not at index 0
    if start > 0 {
        if let Some(space_offset) = text[start..].find(|c: char| c.is_whitespace()) {
            let candidate = start + space_offset + 1;
            if candidate < end {
                start = candidate;
            }
        }
    }

    // Snap end to previous word boundary if not at end
    if end < text.len() {
        if let Some(space_offset) =
            text[..end].rfind(|c: char| c.is_whitespace() || c == '.' || c == '!' || c == '?')
        {
            if space_offset > start {
                end = space_offset;
                // If it snapped on punctuation, include that punctuation
                let trailing = &text[space_offset..];
                if trailing.starts_with('.')
                    || trailing.starts_with('!')
                    || trailing.starts_with('?')
                {
                    end = (space_offset + 1).min(text.len());
                }
            }
        }
    }

    // Ensure valid char boundaries after snapping
    start = floor_char_boundary(text, start);
    end = ceil_char_boundary(text, end);

    if start >= end {
        (0, text.len().min(start + 50))
    } else {
        (start, end)
    }
}

/// Renders the highlighted snippet from the selected window.
fn highlight_window(
    text: &str,
    start: usize,
    end: usize,
    doc_terms: &[Term],
    matched_indices: &[usize],
    format: &HighlightFormat,
    has_prefix: bool,
    has_suffix: bool,
) -> String {
    let (prefix_tag, suffix_tag) = format.tags();
    let mut snippet = String::new();

    if has_prefix {
        snippet.push_str("... ");
    }

    let mut current_pos = start;

    for &m_idx in matched_indices {
        let term = &doc_terms[m_idx];

        // Only highlight terms fully contained within window
        if term.start_offset >= start && term.end_offset <= end {
            if term.start_offset > current_pos {
                snippet.push_str(&text[current_pos..term.start_offset]);
            }

            snippet.push_str(prefix_tag);
            snippet.push_str(&text[term.start_offset..term.end_offset]);
            snippet.push_str(suffix_tag);

            current_pos = term.end_offset;
        }
    }

    if current_pos < end {
        snippet.push_str(&text[current_pos..end]);
    }

    if has_suffix {
        snippet.push_str(" ...");
    }

    snippet
}

/// Fallback for text without matched query terms.
fn fallback_prefix(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        text.to_string()
    } else {
        let mut char_count = 0;
        let mut byte_idx = 0;
        for (idx, _) in text.char_indices() {
            if char_count >= max_chars {
                byte_idx = idx;
                break;
            }
            char_count += 1;
        }
        if byte_idx == 0 {
            byte_idx = text.len();
        }

        // Snap to last whitespace
        if let Some(space_idx) = text[..byte_idx].rfind(char::is_whitespace) {
            format!("{} ...", &text[..space_idx])
        } else {
            format!("{} ...", &text[..byte_idx])
        }
    }
}

fn floor_char_boundary(text: &str, mut idx: usize) -> usize {
    if idx >= text.len() {
        return text.len();
    }
    while !text.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

fn ceil_char_boundary(text: &str, mut idx: usize) -> usize {
    if idx >= text.len() {
        return text.len();
    }
    while !text.is_char_boundary(idx) {
        idx += 1;
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snippet_generation_with_html_tags() {
        let analyzer = Analyzer::new();
        let config = SnippetConfig {
            max_chars: 80,
            format: HighlightFormat::Html,
        };

        let text = "Rust is a systems programming language. It empowers developers to build reliable and memory-safe software with zero-cost abstractions.";
        let query = "memory safe";

        let snippet = generate_snippet(text, query, &analyzer, &config);

        assert!(snippet.contains("<b>memory</b>"));
        assert!(snippet.contains("<b>safe</b>"));
    }

    #[test]
    fn test_snippet_centered_in_long_document() {
        let analyzer = Analyzer::new();
        let config = SnippetConfig {
            max_chars: 60,
            format: HighlightFormat::Html,
        };

        let text = "Introduction to database internals and storage formats. Many chapters precede this section. Inverted indexes provide rapid keyword lookup for search engines. Concluding thoughts and references follow below.";
        let query = "inverted indexes";

        let snippet = generate_snippet(text, query, &analyzer, &config);

        // Should have leading and trailing ellipsis
        assert!(snippet.starts_with("... "));
        assert!(snippet.ends_with(" ..."));
        assert!(snippet.contains("<b>Inverted</b>"));
        assert!(snippet.contains("<b>indexes</b>"));
    }

    #[test]
    fn test_snippet_short_text_no_ellipsis() {
        let analyzer = Analyzer::new();
        let config = SnippetConfig {
            max_chars: 100,
            format: HighlightFormat::Html,
        };

        let text = "Fast search engine written in Rust.";
        let query = "Rust";

        let snippet = generate_snippet(text, query, &analyzer, &config);

        assert_eq!(snippet, "Fast search engine written in <b>Rust</b>.");
        assert!(!snippet.contains("..."));
    }

    #[test]
    fn test_snippet_casing_preservation() {
        let analyzer = Analyzer::new();
        let config = SnippetConfig {
            max_chars: 100,
            format: HighlightFormat::Custom {
                prefix: "[".to_string(),
                suffix: "]".to_string(),
            },
        };

        let text = "RUST offers memory safety without garbage collection.";
        let query = "rust";

        let snippet = generate_snippet(text, query, &analyzer, &config);

        // Original uppercase "RUST" preserved within highlighting brackets
        assert!(snippet.contains("[RUST]"));
    }
}
