use crate::stemmer::stem;
use crate::tokenizer::{tokenize, Token};

/// Represents a fully analyzed, normalized term ready for index storage or query matching.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Term {
    /// The normalized, stemmed text
    pub text: String,
    /// Byte start offset in the original document
    pub start_offset: usize,
    /// Byte end offset in the original document
    pub end_offset: usize,
    /// Sequential token position in the document stream
    pub position: usize,
}

/// The Analyzer orchestrates tokenization, case folding, and stemming
/// into a unified text analysis pipeline.
#[derive(Debug, Default, Clone)]
pub struct Analyzer;

impl Analyzer {
    /// Creates a new default Analyzer.
    pub fn new() -> Self {
        Self
    }

    /// Analyzes raw input text into a sequence of normalized `Term`s.
    ///
    /// Pipeline:
    /// 1. Tokenizes character stream (extracts words, numbers, contractions, URLs).
    /// 2. Words are lowercased and stemmed using the Porter Stemmer.
    /// 3. Numbers, IP addresses, and URLs are preserved intact.
    /// 4. Stop words are intentionally preserved (relying on downstream BM25 scoring).
    /// 5. Precise byte offsets and sequential positions are retained.
    pub fn analyze(&self, text: &str) -> Vec<Term> {
        let tokens: Vec<Token> = tokenize(text);

        tokens
            .into_iter()
            .map(|token| {
                // If the token is a URL, preserve query params / paths; otherwise case-fold
                let is_url = token.text.starts_with("http://")
                    || token.text.starts_with("https://")
                    || token.text.starts_with("ftp://");

                let processed_text = if is_url {
                    token.text
                } else {
                    token.text.to_lowercase()
                };

                let normalized_text = stem(&processed_text);
                Term {
                    text: normalized_text,
                    start_offset: token.start_offset,
                    end_offset: token.end_offset,
                    position: token.position,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sentence_analysis() {
        let analyzer = Analyzer::new();
        let terms = analyzer.analyze("The quick brown foxes jumped over lazy dogs!");
        let words: Vec<&str> = terms.iter().map(|t| t.text.as_str()).collect();

        // Verifies lowercasing, plural reduction ("foxes" -> "fox", "dogs" -> "dog"),
        // and past tense reduction ("jumped" -> "jump")
        assert_eq!(
            words,
            vec!["the", "quick", "brown", "fox", "jump", "over", "lazi", "dog"]
        );

        // Verifies continuous positions 0 through 7
        for (i, term) in terms.iter().enumerate() {
            assert_eq!(term.position, i);
        }
    }

    #[test]
    fn test_stop_words_preserved() {
        let analyzer = Analyzer::new();
        let terms = analyzer.analyze("To be or not to be");
        let words: Vec<&str> = terms.iter().map(|t| t.text.as_str()).collect();

        // Option C check: every word is kept with exact sequential positions
        assert_eq!(words, vec!["to", "be", "or", "not", "to", "be"]);
        assert_eq!(terms[0].position, 0);
        assert_eq!(terms[5].position, 5);
    }

    #[test]
    fn test_technical_and_numeric_preservation() {
        let analyzer = Analyzer::new();
        let terms = analyzer.analyze("Server running at 192.168.1.1 or https://nexora.org/api");
        let words: Vec<&str> = terms.iter().map(|t| t.text.as_str()).collect();

        assert_eq!(
            words,
            vec!["server", "run", "at", "192.168.1.1", "or", "https://nexora.org/api"]
        );
    }
}
