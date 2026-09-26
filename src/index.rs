use std::collections::HashMap;

use crate::analyzer::{Analyzer, Term};

/// Internal dense identifier for a document in the search engine.
pub type DocId = u32;

/// A Posting represents the occurrence of a term within a specific document.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Posting {
    /// The unique document identifier
    pub doc_id: DocId,
    /// How many times the term occurred in this document (Term Frequency)
    pub term_frequency: u32,
    /// 0-indexed token positions where the term occurred in the document
    pub positions: Vec<u32>,
}

/// The Inverted Index is the core data structure of the search engine.
///
/// It maps normalized vocabulary terms to sorted lists of Postings,
/// and maintains document metadata (lengths, counts) necessary for relevance ranking.
#[derive(Debug, Clone)]
pub struct InvertedIndex {
    /// Maps each normalized term to its sorted list of postings
    dictionary: HashMap<String, Vec<Posting>>,
    /// Maps doc_id to its total token length (needed for BM25 normalization)
    doc_lengths: HashMap<DocId, u32>,
    /// Total number of indexed documents
    total_documents: usize,
    /// The text analysis pipeline used for indexing and query processing
    analyzer: Analyzer,
}

impl InvertedIndex {
    /// Creates a new, empty inverted index using the default Analyzer.
    pub fn new() -> Self {
        Self {
            dictionary: HashMap::new(),
            doc_lengths: HashMap::new(),
            total_documents: 0,
            analyzer: Analyzer::new(),
        }
    }

    /// Returns the total number of documents indexed.
    pub fn total_documents(&self) -> usize {
        self.total_documents
    }

    /// Returns the total term count (length) of a specific document.
    pub fn doc_length(&self, doc_id: DocId) -> Option<u32> {
        self.doc_lengths.get(&doc_id).copied()
    }

    /// Computes the average document length across the entire index.
    pub fn average_doc_length(&self) -> f64 {
        if self.total_documents == 0 {
            return 0.0;
        }
        let total_length: u64 = self.doc_lengths.values().map(|&len| len as u64).sum();
        total_length as f64 / self.total_documents as f64
    }

    /// Indexes a document's text under the given `doc_id`.
    ///
    /// The document is analyzed into terms. Multiple occurrences of a term
    /// within the document are grouped into a single `Posting` with accurate
    /// term frequency and position offsets.
    pub fn add_document(&mut self, doc_id: DocId, text: &str) {
        let terms: Vec<Term> = self.analyzer.analyze(text);
        let doc_length = terms.len() as u32;

        self.doc_lengths.insert(doc_id, doc_length);
        self.total_documents += 1;

        // Step 1: Aggregate terms within this document.
        // Map: term_text -> list of token positions
        let mut term_positions: HashMap<String, Vec<u32>> = HashMap::new();
        for term in terms {
            term_positions
                .entry(term.text)
                .or_default()
                .push(term.position as u32);
        }

        // Step 2: Append a Posting for each unique term to the global dictionary.
        for (term_text, positions) in term_positions {
            let posting = Posting {
                doc_id,
                term_frequency: positions.len() as u32,
                positions,
            };

            let postings_list = self.dictionary.entry(term_text).or_default();

            // Maintain strictly ascending order by doc_id
            if let Some(last) = postings_list.last() {
                if last.doc_id == doc_id {
                    // Document was already indexed; skip duplicate posting
                    continue;
                } else if doc_id < last.doc_id {
                    // Out-of-order insertion: binary search insertion point
                    let idx = postings_list
                        .binary_search_by_key(&doc_id, |p| p.doc_id)
                        .unwrap_or_else(|e| e);
                    postings_list.insert(idx, posting);
                    continue;
                }
            }

            postings_list.push(posting);
        }
    }

    /// Retrieves raw postings list for an already-normalized term string.
    pub fn get_postings(&self, normalized_term: &str) -> Option<&Vec<Posting>> {
        self.dictionary.get(normalized_term)
    }

    /// Searches for a single term, running it through the analyzer first.
    pub fn search_term(&self, query_term: &str) -> Vec<Posting> {
        let analyzed = self.analyzer.analyze(query_term);
        if analyzed.is_empty() {
            return Vec::new();
        }
        self.get_postings(&analyzed[0].text)
            .cloned()
            .unwrap_or_default()
    }

    // -------------------------------------------------------------------------
    // Boolean Retrieval Algorithms
    // -------------------------------------------------------------------------

    /// Intersects two sorted postings lists in O(L1 + L2) time using two pointers.
    /// Returns only postings that appear in both lists.
    pub fn intersect(p1: &[Posting], p2: &[Posting]) -> Vec<Posting> {
        let mut result = Vec::new();
        let (mut i, mut j) = (0, 0);

        while i < p1.len() && j < p2.len() {
            if p1[i].doc_id == p2[j].doc_id {
                // Documents match in both lists
                result.push(p1[i].clone());
                i += 1;
                j += 1;
            } else if p1[i].doc_id < p2[j].doc_id {
                i += 1;
            } else {
                j += 1;
            }
        }

        result
    }

    /// Computes the union of two sorted postings lists in O(L1 + L2) time.
    /// Returns all unique postings sorted by DocId.
    pub fn union(p1: &[Posting], p2: &[Posting]) -> Vec<Posting> {
        let mut result = Vec::new();
        let (mut i, mut j) = (0, 0);

        while i < p1.len() && j < p2.len() {
            if p1[i].doc_id == p2[j].doc_id {
                result.push(p1[i].clone());
                i += 1;
                j += 1;
            } else if p1[i].doc_id < p2[j].doc_id {
                result.push(p1[i].clone());
                i += 1;
            } else {
                result.push(p2[j].clone());
                j += 1;
            }
        }

        // Append any remaining elements
        while i < p1.len() {
            result.push(p1[i].clone());
            i += 1;
        }
        while j < p2.len() {
            result.push(p2[j].clone());
            j += 1;
        }

        result
    }

    /// Performs a Boolean AND search across multiple terms.
    ///
    /// Optimization: Intersects the shortest postings lists first,
    /// drastically reducing intermediate candidate sizes.
    pub fn search_and(&self, query_terms: &[&str]) -> Vec<Posting> {
        if query_terms.is_empty() {
            return Vec::new();
        }

        // 1. Analyze and fetch postings lists for each term
        let mut lists: Vec<Vec<Posting>> = Vec::new();
        for &raw_term in query_terms {
            let postings = self.search_term(raw_term);
            if postings.is_empty() {
                // If any term has 0 matches in an AND query, total matches must be 0
                return Vec::new();
            }
            lists.push(postings);
        }

        // 2. Sort by length ascending (intersect shortest list first!)
        lists.sort_by_key(|list| list.len());

        // 3. Iteratively intersect
        let mut accumulated = lists.remove(0);
        for next_list in lists {
            accumulated = Self::intersect(&accumulated, &next_list);
            if accumulated.is_empty() {
                break;
            }
        }

        accumulated
    }

    /// Performs a Boolean OR search across multiple terms.
    pub fn search_or(&self, query_terms: &[&str]) -> Vec<Posting> {
        let mut accumulated: Vec<Posting> = Vec::new();

        for &raw_term in query_terms {
            let postings = self.search_term(raw_term);
            if !postings.is_empty() {
                accumulated = Self::union(&accumulated, &postings);
            }
        }

        accumulated
    }

    // -------------------------------------------------------------------------
    // Phrase Search Algorithm (Positional Intersection)
    // -------------------------------------------------------------------------

    /// Searches for an exact phrase where terms must appear consecutively in order.
    ///
    /// E.g. "quick brown" requires pos(brown) == pos(quick) + 1.
    pub fn search_phrase(&self, phrase: &str) -> Vec<DocId> {
        let analyzed_terms = self.analyzer.analyze(phrase);
        if analyzed_terms.is_empty() {
            return Vec::new();
        }
        if analyzed_terms.len() == 1 {
            return self
                .search_term(&analyzed_terms[0].text)
                .into_iter()
                .map(|p| p.doc_id)
                .collect();
        }

        // Fetch postings lists for all terms in the phrase
        let mut term_postings: Vec<Vec<Posting>> = Vec::new();
        for term in &analyzed_terms {
            let postings = self.get_postings(&term.text).cloned().unwrap_or_default();
            if postings.is_empty() {
                return Vec::new(); // Phrase cannot exist if any word is missing
            }
            term_postings.push(postings);
        }

        // Find candidate documents containing ALL terms via doc_id intersection
        let mut candidate_docs = term_postings[0].clone();
        for next_postings in &term_postings[1..] {
            candidate_docs = Self::intersect(&candidate_docs, next_postings);
            if candidate_docs.is_empty() {
                return Vec::new();
            }
        }

        // For each candidate document, verify consecutive positional alignment
        let mut matching_doc_ids = Vec::new();

        for candidate in candidate_docs {
            let doc_id = candidate.doc_id;

            // Collect the positions array for each term in this specific document
            let doc_positions: Vec<&[u32]> = term_postings
                .iter()
                .map(|list| {
                    let p = list.iter().find(|p| p.doc_id == doc_id).unwrap();
                    p.positions.as_slice()
                })
                .collect();

            if Self::has_consecutive_phrase(&doc_positions) {
                matching_doc_ids.push(doc_id);
            }
        }

        matching_doc_ids
    }

    /// Verifies if a series of position slices contains at least one sequence
    /// where pos[k] == pos[0] + k for all k.
    fn has_consecutive_phrase(positions_per_term: &[&[u32]]) -> bool {
        if positions_per_term.is_empty() {
            return false;
        }

        let first_positions = positions_per_term[0];

        // For each starting position of term 0
        for &start_pos in first_positions {
            let mut matches_phrase = true;

            // Verify that term k appears at start_pos + k
            for (offset, term_positions) in positions_per_term.iter().enumerate().skip(1) {
                let target_pos = start_pos + offset as u32;
                if term_positions.binary_search(&target_pos).is_err() {
                    matches_phrase = false;
                    break;
                }
            }

            if matches_phrase {
                return true;
            }
        }

        false
    }
}

impl Default for InvertedIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_index() -> InvertedIndex {
        let mut index = InvertedIndex::new();
        index.add_document(0, "The quick brown fox jumps over the lazy dog");
        index.add_document(1, "Quick brown foxes are fast animals");
        index.add_document(2, "Lazy dogs sleep all day in the warm sun");
        index
    }

    #[test]
    fn test_single_term_search() {
        let index = create_test_index();

        // "foxes" and "fox" both stem to "fox"
        let matches = index.search_term("fox");
        let doc_ids: Vec<DocId> = matches.iter().map(|p| p.doc_id).collect();
        assert_eq!(doc_ids, vec![0, 1]);

        // Term frequency verification
        // Doc 0 mentions "the" twice
        let the_matches = index.search_term("the");
        let doc0_posting = the_matches.iter().find(|p| p.doc_id == 0).unwrap();
        assert_eq!(doc0_posting.term_frequency, 2);
    }

    #[test]
    fn test_boolean_and_search() {
        let index = create_test_index();

        // "quick" AND "fox" -> Doc 0 and Doc 1
        let matches = index.search_and(&["quick", "fox"]);
        let doc_ids: Vec<DocId> = matches.iter().map(|p| p.doc_id).collect();
        assert_eq!(doc_ids, vec![0, 1]);

        // "quick" AND "lazy" -> Doc 0 only
        let matches = index.search_and(&["quick", "lazy"]);
        let doc_ids: Vec<DocId> = matches.iter().map(|p| p.doc_id).collect();
        assert_eq!(doc_ids, vec![0]);

        // "quick" AND "sleep" -> no documents match
        let matches = index.search_and(&["quick", "sleep"]);
        assert!(matches.is_empty());
    }

    #[test]
    fn test_boolean_or_search() {
        let index = create_test_index();

        // "fox" (0, 1) OR "sun" (2) -> 0, 1, 2
        let matches = index.search_or(&["fox", "sun"]);
        let doc_ids: Vec<DocId> = matches.iter().map(|p| p.doc_id).collect();
        assert_eq!(doc_ids, vec![0, 1, 2]);
    }

    #[test]
    fn test_exact_phrase_search() {
        let index = create_test_index();

        // "quick brown" appears consecutively in Doc 0 and Doc 1
        let matches = index.search_phrase("quick brown");
        assert_eq!(matches, vec![0, 1]);

        // "lazy dog" appears consecutively in Doc 0 and Doc 2 (dog/dogs stemmed)
        let matches = index.search_phrase("lazy dog");
        assert_eq!(matches, vec![0, 2]);

        // "quick fox" both words are in Doc 0 and 1, but NOT consecutively ("quick brown fox")
        let matches = index.search_phrase("quick fox");
        assert!(matches.is_empty());

        // 3-word phrase "quick brown fox"
        let matches = index.search_phrase("quick brown fox");
        assert_eq!(matches, vec![0, 1]);
    }

    #[test]
    fn test_metadata_tracking() {
        let index = create_test_index();
        assert_eq!(index.total_documents(), 3);
        assert!(index.average_doc_length() > 0.0);
        assert_eq!(index.doc_length(0), Some(9)); // 9 words in doc 0
    }
}
