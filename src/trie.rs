use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};

use crate::index::InvertedIndex;
use crate::tokenizer::tokenize;

/// A single autocomplete suggestion ranked by frequency.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixSuggestion {
    /// The suggested vocabulary term
    pub term: String,
    /// Total term frequency across all documents
    pub term_frequency: u64,
    /// Number of unique documents containing this term
    pub doc_frequency: u32,
}

impl Ord for PrefixSuggestion {
    fn cmp(&self, other: &Self) -> Ordering {
        // Highest term_frequency first
        self.term_frequency
            .cmp(&other.term_frequency)
            // Tie-breaker 1: Higher document frequency
            .then_with(|| self.doc_frequency.cmp(&other.doc_frequency))
            // Tie-breaker 2: Alphabetical order (smaller string is better)
            .then_with(|| other.term.cmp(&self.term))
    }
}

impl PartialOrd for PrefixSuggestion {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A node within the Prefix Trie.
#[derive(Debug, Clone)]
pub struct TrieNode {
    /// Ordered map of child characters for deterministic alphabetical traversal
    pub children: BTreeMap<char, TrieNode>,
    /// Whether this node marks the end of a valid vocabulary term
    pub is_terminal: bool,
    /// Total occurrences of this term
    pub term_frequency: u64,
    /// Total documents containing this term
    pub doc_frequency: u32,
    /// Maximum term_frequency among all terminal words in this node's entire subtree.
    /// Used for branch-and-bound pruning during top-K suggestion searches.
    pub max_subtree_freq: u64,
}

impl TrieNode {
    pub fn new() -> Self {
        Self {
            children: BTreeMap::new(),
            is_terminal: false,
            term_frequency: 0,
            doc_frequency: 0,
            max_subtree_freq: 0,
        }
    }
}

impl Default for TrieNode {
    fn default() -> Self {
        Self::new()
    }
}

/// Compact Prefix Trie for fast prefix search and frequency-ranked autocomplete.
#[derive(Debug, Clone)]
pub struct PrefixTrie {
    root: TrieNode,
    total_words: usize,
}

impl PrefixTrie {
    /// Creates a new, empty `PrefixTrie`.
    pub fn new() -> Self {
        Self {
            root: TrieNode::new(),
            total_words: 0,
        }
    }

    /// Returns the number of distinct terms stored in the trie.
    pub fn len(&self) -> usize {
        self.total_words
    }

    /// Returns true if the trie contains no terms.
    pub fn is_empty(&self) -> bool {
        self.total_words == 0
    }

    /// Inserts a word with its term frequency and document frequency.
    /// The word is converted to lowercase for case-insensitive matching.
    pub fn insert(&mut self, word: &str, term_freq: u64, doc_freq: u32) {
        let clean = word.trim().to_lowercase();
        if clean.is_empty() {
            return;
        }

        let mut node = &mut self.root;
        node.max_subtree_freq = node.max_subtree_freq.max(term_freq);

        for ch in clean.chars() {
            node = node.children.entry(ch).or_default();
            node.max_subtree_freq = node.max_subtree_freq.max(term_freq);
        }

        if !node.is_terminal {
            self.total_words += 1;
            node.is_terminal = true;
        }

        node.term_frequency += term_freq;
        node.doc_frequency += doc_freq;
        node.max_subtree_freq = node.max_subtree_freq.max(node.term_frequency);
    }

    /// Checks if a word exists as a terminal term in the trie.
    pub fn contains(&self, word: &str) -> bool {
        let clean = word.trim().to_lowercase();
        let mut node = &self.root;

        for ch in clean.chars() {
            match node.children.get(&ch) {
                Some(next) => node = next,
                None => return false,
            }
        }

        node.is_terminal
    }

    /// Returns the (term_frequency, doc_frequency) of an exact word, if it exists.
    pub fn get_frequency(&self, word: &str) -> Option<(u64, u32)> {
        let clean = word.trim().to_lowercase();
        let mut node = &self.root;

        for ch in clean.chars() {
            node = node.children.get(&ch)?;
        }

        if node.is_terminal {
            Some((node.term_frequency, node.doc_frequency))
        } else {
            None
        }
    }

    /// Finds all terms in the trie that start with the given prefix, in alphabetical order.
    pub fn find_by_prefix(&self, prefix: &str) -> Vec<String> {
        let clean = prefix.trim().to_lowercase();
        let mut node = &self.root;

        for ch in clean.chars() {
            match node.children.get(&ch) {
                Some(next) => node = next,
                None => return Vec::new(),
            }
        }

        let mut results = Vec::new();
        let mut current_term = clean;
        self.collect_all_terms(node, &mut current_term, &mut results);
        results
    }

    fn collect_all_terms(&self, node: &TrieNode, current: &mut String, results: &mut Vec<String>) {
        if node.is_terminal {
            results.push(current.clone());
        }

        for (&ch, child) in &node.children {
            current.push(ch);
            self.collect_all_terms(child, current, results);
            current.pop();
        }
    }

    /// Returns up to `limit` autocomplete suggestions for `prefix`, ranked by frequency.
    ///
    /// Uses branch-and-bound pruning with `max_subtree_freq` to discard entire subtrees
    /// that cannot beat the current top-K candidates.
    pub fn suggest(&self, prefix: &str, limit: usize) -> Vec<PrefixSuggestion> {
        if limit == 0 {
            return Vec::new();
        }

        let clean = prefix.trim().to_lowercase();
        let mut node = &self.root;

        for ch in clean.chars() {
            match node.children.get(&ch) {
                Some(next) => node = next,
                None => return Vec::new(),
            }
        }

        // Min-heap to maintain top-K suggestions
        let mut heap: BinaryHeap<std::cmp::Reverse<PrefixSuggestion>> = BinaryHeap::with_capacity(limit);
        let mut current_term = clean;

        self.collect_top_k(node, &mut current_term, limit, &mut heap);

        // Drain heap and sort in descending order (highest frequency first)
        let mut suggestions: Vec<PrefixSuggestion> = heap.into_iter().map(|rev| rev.0).collect();
        suggestions.sort_by(|a, b| b.cmp(a));
        suggestions
    }

    fn collect_top_k(
        &self,
        node: &TrieNode,
        current: &mut String,
        limit: usize,
        heap: &mut BinaryHeap<std::cmp::Reverse<PrefixSuggestion>>,
    ) {
        // Branch-and-bound pruning:
        // If heap is full and the best possible frequency in this subtree cannot exceed
        // the worst frequency in our current top-K, prune the entire branch!
        if heap.len() == limit {
            if let Some(std::cmp::Reverse(worst)) = heap.peek() {
                if node.max_subtree_freq < worst.term_frequency {
                    return;
                }
            }
        }

        if node.is_terminal {
            let suggestion = PrefixSuggestion {
                term: current.clone(),
                term_frequency: node.term_frequency,
                doc_frequency: node.doc_frequency,
            };

            if heap.len() < limit {
                heap.push(std::cmp::Reverse(suggestion));
            } else if let Some(std::cmp::Reverse(worst)) = heap.peek() {
                if suggestion.cmp(worst) == Ordering::Greater {
                    heap.pop();
                    heap.push(std::cmp::Reverse(suggestion));
                }
            }
        }

        for (&ch, child) in &node.children {
            current.push(ch);
            self.collect_top_k(child, current, limit, heap);
            current.pop();
        }
    }

    /// Constructs a `PrefixTrie` directly from an `InvertedIndex` dictionary.
    ///
    /// The term frequency is calculated as the sum of frequencies across all postings,
    /// and the document frequency is the length of the postings list.
    pub fn from_index(index: &InvertedIndex) -> Self {
        let mut trie = Self::new();
        for (term, postings) in index.dictionary() {
            let total_tf: u64 = postings.iter().map(|p| p.term_frequency as u64).sum();
            let df = postings.len() as u32;
            trie.insert(term, total_tf, df);
        }
        trie
    }

    /// Constructs a `PrefixTrie` from raw document texts, preserving full un-stemmed natural words.
    pub fn from_documents<I, S>(documents: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut trie = Self::new();
        for doc in documents {
            let tokens = tokenize(doc.as_ref());
            for token in tokens {
                if token.text.starts_with("http") || token.text.chars().all(|c| c.is_ascii_digit()) {
                    continue;
                }
                if token.text.len() >= 2 {
                    trie.insert(&token.text, 1, 1);
                }
            }
        }
        trie
    }
}

impl Default for PrefixTrie {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_trie() {
        let trie = PrefixTrie::new();
        assert_eq!(trie.len(), 0);
        assert!(trie.is_empty());
        assert!(!trie.contains("rust"));
        assert!(trie.find_by_prefix("r").is_empty());
        assert!(trie.suggest("r", 5).is_empty());
    }

    #[test]
    fn test_insert_and_contains() {
        let mut trie = PrefixTrie::new();
        trie.insert("search", 10, 3);
        trie.insert("sea", 25, 5);
        trie.insert("season", 5, 2);

        assert_eq!(trie.len(), 3);
        assert!(trie.contains("sea"));
        assert!(trie.contains("search"));
        assert!(trie.contains("season"));
        assert!(!trie.contains("se"));
        assert!(!trie.contains("seat"));

        let (tf, df) = trie.get_frequency("sea").unwrap();
        assert_eq!(tf, 25);
        assert_eq!(df, 5);
    }

    #[test]
    fn test_alphabetical_prefix_search() {
        let mut trie = PrefixTrie::new();
        trie.insert("rust", 5, 2);
        trie.insert("rustacean", 3, 1);
        trie.insert("rustc", 12, 4);
        trie.insert("ruby", 8, 3);

        let matches = trie.find_by_prefix("rust");
        assert_eq!(matches, vec!["rust", "rustacean", "rustc"]);

        let r_matches = trie.find_by_prefix("r");
        assert_eq!(r_matches, vec!["ruby", "rust", "rustacean", "rustc"]);
    }

    #[test]
    fn test_frequency_ranked_suggestions() {
        let mut trie = PrefixTrie::new();
        trie.insert("rust", 10, 3);
        trie.insert("rustc", 50, 8); // highest freq
        trie.insert("rustacean", 2, 1);
        trie.insert("rustproof", 5, 2);

        // Top 2 suggestions for "rust" should be "rustc" (50) and "rust" (10)
        let top2 = trie.suggest("rust", 2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].term, "rustc");
        assert_eq!(top2[0].term_frequency, 50);
        assert_eq!(top2[1].term, "rust");
        assert_eq!(top2[1].term_frequency, 10);

        // Requesting more than available returns all matching
        let all4 = trie.suggest("rust", 10);
        assert_eq!(all4.len(), 4);
        assert_eq!(all4[0].term, "rustc");
        assert_eq!(all4[1].term, "rust");
        assert_eq!(all4[2].term, "rustproof");
        assert_eq!(all4[3].term, "rustacean");
    }

    #[test]
    fn test_from_inverted_index() {
        let mut index = InvertedIndex::new();
        index.add_document(1, "search engines use inverted index data structures");
        index.add_document(2, "modern search engines rank documents with algorithms");
        index.add_document(3, "search query execution");

        let trie = PrefixTrie::from_index(&index);
        assert!(trie.len() > 0);

        // Both documents have stemmed "search"
        let suggestions = trie.suggest("sea", 5);
        assert!(!suggestions.is_empty());
        assert_eq!(suggestions[0].term, "search");
        assert_eq!(suggestions[0].doc_frequency, 3);
        assert_eq!(suggestions[0].term_frequency, 3);
    }

    #[test]
    fn test_from_documents_unstemmed() {
        let docs = vec![
            "Rust programming language is blazing fast",
            "Rustaceans love writing Rust search engines",
        ];
        let trie = PrefixTrie::from_documents(docs);
        assert!(trie.contains("rust"));
        assert!(trie.contains("rustaceans"));

        let suggestions = trie.suggest("rust", 5);
        assert_eq!(suggestions[0].term, "rust");
        assert_eq!(suggestions[0].term_frequency, 2);
        assert_eq!(suggestions[1].term, "rustaceans");
        assert_eq!(suggestions[1].term_frequency, 1);
    }
}
