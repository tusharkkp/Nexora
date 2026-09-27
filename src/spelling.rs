use std::collections::HashMap;

use crate::tokenizer::tokenize;

/// Computes the Damerau-Levenshtein edit distance between two strings.
///
/// Supports four elementary edit operations:
/// 1. Insertion (cost 1)
/// 2. Deletion (cost 1)
/// 3. Substitution (cost 1)
/// 4. Adjacent Transposition (cost 1) (e.g. "pyhton" -> "python")
pub fn damerau_levenshtein(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();

    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    // Fast path: identical strings
    if a_chars == b_chars {
        return 0;
    }

    // DP matrix of size (m + 1) * (n + 1)
    let stride = n + 1;
    let mut dp = vec![0usize; (m + 1) * (n + 1)];

    for i in 0..=m {
        dp[i * stride] = i;
    }
    for j in 0..=n {
        dp[j] = j;
    }

    for i in 1..=m {
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };

            let deletion = dp[(i - 1) * stride + j] + 1;
            let insertion = dp[i * stride + (j - 1)] + 1;
            let substitution = dp[(i - 1) * stride + (j - 1)] + cost;

            let mut min_val = deletion.min(insertion).min(substitution);

            // Adjacent transposition check: a[i-1] == b[j-2] and a[i-2] == b[j-1]
            if i > 1
                && j > 1
                && a_chars[i - 1] == b_chars[j - 2]
                && a_chars[i - 2] == b_chars[j - 1]
            {
                let transposition = dp[(i - 2) * stride + (j - 2)] + 1;
                min_val = min_val.min(transposition);
            }

            dp[i * stride + j] = min_val;
        }
    }

    dp[m * stride + n]
}

/// A spelling suggestion for a single term.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Suggestion {
    /// The suggested candidate term
    pub term: String,
    /// Damerau-Levenshtein edit distance from the input term
    pub distance: usize,
    /// Number of occurrences / document frequency of this term in the vocabulary
    pub frequency: usize,
}

/// Spell checker trained on vocabulary frequencies.
#[derive(Debug, Clone, Default)]
pub struct SpellChecker {
    /// Maps each normalized vocabulary term to its corpus frequency
    vocabulary: HashMap<String, usize>,
}

impl SpellChecker {
    /// Creates an empty spell checker.
    pub fn new() -> Self {
        Self {
            vocabulary: HashMap::new(),
        }
    }

    /// Inserts or increments a word in the spell checker vocabulary.
    pub fn add_word(&mut self, word: &str, count: usize) {
        let clean = word.to_lowercase();
        *self.vocabulary.entry(clean).or_insert(0) += count;
    }

    /// Returns the number of distinct terms in the vocabulary.
    pub fn len(&self) -> usize {
        self.vocabulary.len()
    }

    /// Returns true if the vocabulary is empty.
    pub fn is_empty(&self) -> bool {
        self.vocabulary.is_empty()
    }

    /// Checks if a word exists in the vocabulary.
    pub fn contains(&self, word: &str) -> bool {
        self.vocabulary.contains_key(&word.to_lowercase())
    }

    /// Builds a spell checker directly from an InvertedIndex vocabulary.
    pub fn from_index(index: &crate::index::InvertedIndex) -> Self {
        let mut checker = Self::new();
        for (term, postings) in index.dictionary() {
            checker.add_word(term, postings.len());
        }
        checker
    }

    /// Builds a spell checker by tokenizing a collection of document texts.
    /// Incrementally updates the spell checker vocabulary from a document's text.
    pub fn add_document_text(&mut self, text: &str) {
        let tokens = tokenize(text);
        for token in tokens {
            if token.text.starts_with("http") || token.text.chars().all(|c| c.is_ascii_digit()) {
                continue;
            }
            if token.text.len() >= 2 {
                self.add_word(&token.text, 1);
            }
        }
    }

    /// Builds a spell checker by tokenizing a collection of document texts.
    /// Preserves full un-stemmed natural words (e.g. "engine", "systems").
    pub fn from_documents<I, S>(documents: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut checker = Self::new();
        for doc in documents {
            checker.add_document_text(doc.as_ref());
        }
        checker
    }

    /// Finds the best candidate correction for `word` within `max_distance`.
    ///
    /// Ranking criteria:
    /// 1. Minimum edit distance (d=1 outranks d=2).
    /// 2. First character match bonus (favors candidates starting with the same letter).
    /// 3. Corpus frequency (higher frequency words win ties).
    pub fn suggest(&self, word: &str, max_distance: usize) -> Option<Suggestion> {
        let clean_word = word.to_lowercase();
        if clean_word.is_empty() {
            return None;
        }

        // Exact match -> distance 0
        if let Some(&freq) = self.vocabulary.get(&clean_word) {
            return Some(Suggestion {
                term: clean_word,
                distance: 0,
                frequency: freq,
            });
        }

        let first_char = clean_word.chars().next().unwrap();
        let mut best: Option<Suggestion> = None;

        for (candidate, &freq) in &self.vocabulary {
            // Fast filter: length difference cannot exceed max_distance
            if candidate.len().abs_diff(clean_word.len()) > max_distance {
                continue;
            }

            let dist = damerau_levenshtein(&clean_word, candidate);
            if dist > max_distance {
                continue;
            }

            let cand_first = candidate.chars().next().unwrap_or('\0');
            let same_first = cand_first == first_char;

            let is_better = match &best {
                None => true,
                Some(current) => {
                    if dist < current.distance {
                        true
                    } else if dist == current.distance {
                        let curr_first = current.term.chars().next().unwrap_or('\0');
                        let curr_same = curr_first == first_char;

                        if same_first && !curr_same {
                            true
                        } else if !same_first && curr_same {
                            false
                        } else {
                            // Frequency tie-breaker
                            freq > current.frequency
                        }
                    } else {
                        false
                    }
                }
            };

            if is_better {
                best = Some(Suggestion {
                    term: candidate.clone(),
                    distance: dist,
                    frequency: freq,
                });
            }
        }

        best
    }

    /// Evaluates a multi-term search query, correcting misspelled terms.
    ///
    /// Returns `Some(corrected_query)` if at least one term was corrected,
    /// or `None` if the query is already correct or cannot be repaired.
    pub fn suggest_query(&self, query: &str) -> Option<String> {
        let words: Vec<&str> = query.split_whitespace().collect();
        if words.is_empty() {
            return None;
        }

        let mut corrected_words = Vec::with_capacity(words.len());
        let mut any_correction = false;

        for word in words {
            let clean = word.to_lowercase();
            if self.contains(&clean) {
                corrected_words.push(word.to_string());
            } else if let Some(suggestion) = self.suggest(&clean, 2) {
                if suggestion.distance > 0 {
                    corrected_words.push(suggestion.term);
                    any_correction = true;
                } else {
                    corrected_words.push(word.to_string());
                }
            } else {
                corrected_words.push(word.to_string());
            }
        }

        if any_correction {
            Some(corrected_words.join(" "))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_damerau_levenshtein_operations() {
        // Exact
        assert_eq!(damerau_levenshtein("search", "search"), 0);

        // Substitution
        assert_eq!(damerau_levenshtein("serch", "search"), 1);

        // Insertion
        assert_eq!(damerau_levenshtein("srch", "serch"), 1);

        // Deletion
        assert_eq!(damerau_levenshtein("searrch", "search"), 1);

        // Adjacent Transposition (Damerau check)
        assert_eq!(damerau_levenshtein("pyhton", "python"), 1);
        assert_eq!(damerau_levenshtein("engien", "engine"), 1);
    }

    #[test]
    fn test_frequency_based_tie_breaking() {
        let mut checker = SpellChecker::new();
        // Both "search" and "seach" are distance 1 from "searh"
        // But "search" has 100 occurrences, while "seach" has 1
        checker.add_word("search", 100);
        checker.add_word("seach", 1);

        let suggestion = checker.suggest("searh", 1).expect("should find suggestion");
        assert_eq!(suggestion.term, "search");
        assert_eq!(suggestion.distance, 1);
    }

    #[test]
    fn test_query_spelling_suggestion() {
        let docs = vec![
            "Rust is a fast modern systems programming language.",
            "Inverted indexes power search engine information retrieval.",
        ];

        let checker = SpellChecker::from_documents(docs);

        // "sytems" -> "systems", "programing" -> "programming"
        let suggestion = checker.suggest_query("sytems programing");
        assert_eq!(suggestion, Some("systems programming".to_string()));

        // Query already correct -> None
        assert_eq!(checker.suggest_query("search engine"), None);

        // Transposition typo: "engien" -> "engine"
        assert_eq!(
            checker.suggest_query("fast engien"),
            Some("fast engine".to_string())
        );
    }
}
