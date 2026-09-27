/// Represents a single extracted token from a document.
///
/// Contains the token's text along with its precise byte boundaries
/// in the source text and its sequential position among tokens.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Token {
    /// The normalized or raw text of the token
    pub text: String,
    /// Byte index in the source text where this token begins (inclusive)
    pub start_offset: usize,
    /// Byte index in the source text where this token ends (exclusive)
    pub end_offset: usize,
    /// 0-based sequential position of this token in the document stream
    pub position: usize,
}

/// Tokenizes an input text string using a character-by-character state-machine scanner.
///
/// Handles:
/// - Words (alphabetic sequences)
/// - Contractions (e.g., "don't", "they're", "O'Connor")
/// - Numbers, decimals, and IP addresses (e.g., "42", "3.1415", "192.168.1.1")
/// - URLs (e.g., "https://nexora.org/search?q=rust")
/// - Preserves exact UTF-8 byte offsets for result snippet generation and highlighting.
pub fn tokenize(text: &str) -> Vec<Token> {
    // Collecting character indices allows O(1) random-access lookahead
    // without complex iterator borrow issues.
    // Each entry is (byte_offset, character).
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let total_chars = chars.len();

    let mut tokens = Vec::new();
    let mut position = 0;
    let mut i = 0;

    while i < total_chars {
        let (byte_offset, ch) = chars[i];

        // 1. Whitespace or punctuation at the start -> skip
        if ch.is_whitespace() || is_standalone_punctuation(ch) {
            i += 1;
            continue;
        }

        // 2. Alphabetic character -> Start of a Word or potentially a URL (e.g. "http://")
        if ch.is_alphabetic() {
            let start_byte = byte_offset;
            let mut word = String::new();
            let mut end_byte = byte_offset + ch.len_utf8();

            word.push(ch);
            i += 1;

            let mut is_url = false;

            while i < total_chars {
                let (curr_byte, curr_ch) = chars[i];

                // Check for URL protocol scheme boundary: e.g. "http" followed by "://"
                if (word == "http" || word == "https" || word == "ftp") && curr_ch == ':' {
                    if i + 2 < total_chars && chars[i + 1].1 == '/' && chars[i + 2].1 == '/' {
                        // Switch into URL mode
                        word.push_str("://");
                        i += 3; // consume ':', '/', '/'
                        is_url = true;
                        break;
                    }
                }

                // Check for contractions: apostrophe inside a word (e.g. "don't")
                if curr_ch == '\'' || curr_ch == '’' {
                    if i + 1 < total_chars && chars[i + 1].1.is_alphabetic() {
                        // Valid contraction: consume apostrophe and next letter
                        word.push(curr_ch);
                        word.push(chars[i + 1].1);
                        end_byte = chars[i + 1].0 + chars[i + 1].1.len_utf8();
                        i += 2;
                        continue;
                    } else {
                        // Trailing quote or quote not followed by letters (e.g. 'hello')
                        break;
                    }
                }

                if curr_ch.is_alphanumeric() {
                    word.push(curr_ch);
                    end_byte = curr_byte + curr_ch.len_utf8();
                    i += 1;
                } else {
                    // Reached boundary of the word
                    break;
                }
            }

            // If we detected a URL scheme, consume remaining URL characters
            if is_url {
                while i < total_chars {
                    let (curr_byte, curr_ch) = chars[i];
                    if curr_ch.is_whitespace() || is_url_terminator(curr_ch) {
                        break;
                    }
                    word.push(curr_ch);
                    end_byte = curr_byte + curr_ch.len_utf8();
                    i += 1;
                }

                // Strip trailing punctuation like '.' or ',' from the end of the URL
                // e.g. "Visit https://nexora.org." -> URL shouldn't include the sentence period
                while word.ends_with('.')
                    || word.ends_with(',')
                    || word.ends_with('?')
                    || word.ends_with('!')
                {
                    word.pop();
                    end_byte -= 1;
                }
            }

            tokens.push(Token {
                text: word,
                start_offset: start_byte,
                end_offset: end_byte,
                position,
            });
            position += 1;
            continue;
        }

        // 3. Digit -> Start of a Number (integer, decimal, or IP address)
        if ch.is_ascii_digit() {
            let start_byte = byte_offset;
            let mut num_str = String::new();
            let mut end_byte = byte_offset + ch.len_utf8();

            num_str.push(ch);
            i += 1;

            while i < total_chars {
                let (curr_byte, curr_ch) = chars[i];

                if curr_ch.is_ascii_digit() {
                    num_str.push(curr_ch);
                    end_byte = curr_byte + curr_ch.len_utf8();
                    i += 1;
                } else if curr_ch == '.' || curr_ch == ',' {
                    // Check if '.' or ',' is followed by another digit (e.g. 3.14 or 192.168.1.1)
                    if i + 1 < total_chars && chars[i + 1].1.is_ascii_digit() {
                        num_str.push(curr_ch);
                        num_str.push(chars[i + 1].1);
                        end_byte = chars[i + 1].0 + chars[i + 1].1.len_utf8();
                        i += 2;
                    } else {
                        // Trailing period (end of sentence), do not consume
                        break;
                    }
                } else {
                    break;
                }
            }

            tokens.push(Token {
                text: num_str,
                start_offset: start_byte,
                end_offset: end_byte,
                position,
            });
            position += 1;
            continue;
        }

        // If character was unhandled (e.g. lone symbol), advance by 1
        i += 1;
    }

    tokens
}

/// Helper: Checks if a character is a punctuation symbol that always breaks tokens
fn is_standalone_punctuation(ch: char) -> bool {
    matches!(
        ch,
        '!' | '"'
            | '#'
            | '$'
            | '%'
            | '&'
            | '('
            | ')'
            | '*'
            | '+'
            | ','
            | '-'
            | '.'
            | '/'
            | ':'
            | ';'
            | '<'
            | '='
            | '>'
            | '?'
            | '@'
            | '['
            | '\\'
            | ']'
            | '^'
            | '_'
            | '`'
            | '{'
            | '|'
            | '}'
            | '~'
    )
}

/// Helper: Checks if a character marks the end of a URL in normal prose
fn is_url_terminator(ch: char) -> bool {
    matches!(ch, '<' | '>' | '"' | '\'' | ')' | ']' | '}')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_words() {
        let text = "The quick brown fox";
        let tokens = tokenize(text);
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0].text, "The");
        assert_eq!(tokens[0].position, 0);
        assert_eq!(tokens[3].text, "fox");
        assert_eq!(tokens[3].position, 3);
    }

    #[test]
    fn test_punctuation_stripping() {
        let text = "Hello, world! Are you ready?";
        let tokens = tokenize(text);
        let words: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(words, vec!["Hello", "world", "Are", "you", "ready"]);
    }

    #[test]
    fn test_contractions() {
        let text = "They're sure it wasn't O'Connor's idea.";
        let tokens = tokenize(text);
        let words: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(
            words,
            vec!["They're", "sure", "it", "wasn't", "O'Connor's", "idea"]
        );
    }

    #[test]
    fn test_numbers_decimals_and_ips() {
        let text = "Version 2 has 3.1415 and runs on 192.168.1.1 now.";
        let tokens = tokenize(text);
        let words: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(
            words,
            vec![
                "Version",
                "2",
                "has",
                "3.1415",
                "and",
                "runs",
                "on",
                "192.168.1.1",
                "now"
            ]
        );
    }

    #[test]
    fn test_sentence_boundary_dots() {
        let text = "Read chapter 42. Then proceed.";
        let tokens = tokenize(text);
        let words: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(words, vec!["Read", "chapter", "42", "Then", "proceed"]);
    }

    #[test]
    fn test_urls() {
        let text = "Check https://nexora.org/search?q=rust for info.";
        let tokens = tokenize(text);
        let words: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(
            words,
            vec!["Check", "https://nexora.org/search?q=rust", "for", "info"]
        );
    }

    #[test]
    fn test_exact_byte_offsets() {
        let text = "fast runner";
        let tokens = tokenize(text);
        assert_eq!(tokens[0].text, "fast");
        assert_eq!(tokens[0].start_offset, 0);
        assert_eq!(tokens[0].end_offset, 4);

        assert_eq!(tokens[1].text, "runner");
        assert_eq!(tokens[1].start_offset, 5);
        assert_eq!(tokens[1].end_offset, 11);

        // Verification: slicing original text using token offsets recovers exact substring
        assert_eq!(&text[tokens[0].start_offset..tokens[0].end_offset], "fast");
        assert_eq!(
            &text[tokens[1].start_offset..tokens[1].end_offset],
            "runner"
        );
    }
}
