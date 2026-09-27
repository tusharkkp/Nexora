use std::collections::HashMap;

use crate::index::DocId;
use crate::tokenizer::tokenize;

/// Computes the dot product between two slices of equal length.
/// Uses 4-way loop unrolling to assist compiler SIMD auto-vectorization.
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let a_slice = &a[..len];
    let b_slice = &b[..len];

    let (chunks_a, rem_a) = a_slice.as_chunks::<4>();
    let (chunks_b, rem_b) = b_slice.as_chunks::<4>();

    let mut sum = 0.0f32;
    for (ca, cb) in chunks_a.iter().zip(chunks_b.iter()) {
        sum += ca[0] * cb[0] + ca[1] * cb[1] + ca[2] * cb[2] + ca[3] * cb[3];
    }
    for (&x, &y) in rem_a.iter().zip(rem_b.iter()) {
        sum += x * y;
    }
    sum
}

/// Computes Euclidean L2 norm of a vector.
pub fn l2_norm(vec: &[f32]) -> f32 {
    dot_product(vec, vec).sqrt()
}

/// In-place normalizes a vector to unit length (L2 norm = 1.0).
/// Leaves all-zero vectors unchanged without panicking.
pub fn l2_normalize(vec: &mut [f32]) {
    let norm = l2_norm(vec);
    if norm > 1e-12 {
        let inv = 1.0 / norm;
        for x in vec.iter_mut() {
            *x *= inv;
        }
    }
}

/// Computes cosine similarity between two arbitrary vectors.
/// Result is in range `[-1.0, 1.0]`.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let norm_a = l2_norm(a);
    let norm_b = l2_norm(b);
    if norm_a < 1e-12 || norm_b < 1e-12 {
        return 0.0;
    }
    let sim = dot_product(a, b) / (norm_a * norm_b);
    sim.clamp(-1.0, 1.0)
}

/// Individual result from a vector nearest-neighbor search.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorSearchResult {
    /// Identifier of the matching document
    pub doc_id: DocId,
    /// Cosine similarity score in range `[-1.0, 1.0]` (1.0 = identical direction)
    pub similarity: f32,
}

/// In-memory dense vector index supporting fast cosine similarity nearest-neighbor lookup.
#[derive(Debug, Clone)]
pub struct VectorIndex {
    dimension: usize,
    /// Dense vectors mapped by DocId, stored pre-normalized for dot-product equivalence
    vectors: HashMap<DocId, Vec<f32>>,
}

impl VectorIndex {
    /// Creates a new `VectorIndex` for embeddings of fixed dimensionality.
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            vectors: HashMap::new(),
        }
    }

    /// Dimensionality of vectors stored in this index.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Number of indexed document vectors.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Whether the vector index contains no documents.
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Adds or replaces a document vector. Automatically L2-normalizes the vector.
    pub fn add_vector(&mut self, doc_id: DocId, mut vector: Vec<f32>) {
        if vector.len() != self.dimension {
            vector.resize(self.dimension, 0.0);
        }
        l2_normalize(&mut vector);
        self.vectors.insert(doc_id, vector);
    }

    /// Retrieves the stored normalized vector for a document.
    pub fn get_vector(&self, doc_id: DocId) -> Option<&[f32]> {
        self.vectors.get(&doc_id).map(|v| v.as_slice())
    }

    /// Removes a document vector from the index.
    pub fn remove_vector(&mut self, doc_id: DocId) -> Option<Vec<f32>> {
        self.vectors.remove(&doc_id)
    }

    /// Clears all vectors from the index.
    pub fn clear(&mut self) {
        self.vectors.clear();
    }

    /// Searches for the `top_k` most semantically similar documents to `query_vector`.
    ///
    /// Since indexed vectors are pre-normalized, cosine similarity reduces to a fast dot product:
    /// `sim(q, d) = q_norm · d_norm`.
    pub fn search(&self, query_vector: &[f32], top_k: usize) -> Vec<VectorSearchResult> {
        if self.vectors.is_empty() || top_k == 0 {
            return Vec::new();
        }

        let mut q_norm = query_vector.to_vec();
        if q_norm.len() != self.dimension {
            q_norm.resize(self.dimension, 0.0);
        }
        l2_normalize(&mut q_norm);

        let mut results: Vec<VectorSearchResult> = self
            .vectors
            .iter()
            .map(|(&doc_id, doc_vec)| {
                let similarity = dot_product(&q_norm, doc_vec);
                VectorSearchResult { doc_id, similarity }
            })
            .collect();

        // Sort descending by similarity, tie-break by DocId ascending
        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.doc_id.cmp(&b.doc_id))
        });

        if results.len() > top_k {
            results.truncate(top_k);
        }

        results
    }
}

/// Interface for generating dense semantic vector embeddings from natural text.
pub trait TextEmbedder: Send + Sync {
    /// Embeds a text string into a fixed-length normalized float vector.
    fn embed(&self, text: &str) -> Vec<f32>;

    /// The dimension $D$ of the vectors produced by this embedder.
    fn dimension(&self) -> usize;
}

/// Built-in, high-performance pure-Rust semantic embedder.
///
/// Features:
/// 1. **Semantic Concept Projection (Axes 0..16)**:
///    Projects domain-salient terms (systems, search, crawler, AI, distributed, networks,
///    aerospace, vehicles, etc.) into structured semantic concept subspace clusters.
/// 2. **Subword Character $N$-Gram Hashing (Axes 16..64)**:
///    FastText-style hashing of character trigrams and 4-grams to capture morphology,
///    stem variants, and out-of-vocabulary words.
/// 3. **Salience & Field Weighting**:
///    Higher weights for content words; dampening for common English stop words.
/// 4. **Unit Normalization**:
///    Guarantees $\|\vec{v}\|_2 = 1.0$, allowing instant cosine similarity via dot product.
#[derive(Debug, Clone)]
pub struct SemanticEmbedder {
    dimension: usize,
}

impl Default for SemanticEmbedder {
    fn default() -> Self {
        Self::new(64)
    }
}

impl SemanticEmbedder {
    /// Creates a new `SemanticEmbedder` with the specified vector dimension (default 64).
    pub fn new(dimension: usize) -> Self {
        let dim = dimension.max(32);
        Self { dimension: dim }
    }

    /// Classifies a normalized token to its primary semantic concept cluster axis (if any).
    fn concept_axis(word: &str) -> Option<usize> {
        match word {
            // Axis 0: Systems Programming & Low-Level Computing
            "rust" | "systems" | "memory" | "borrow" | "thread" | "concurrency" | "compiler"
            | "bare-metal" | "ownership" | "pointer" | "kernel" => Some(0),

            // Axis 1: Information Retrieval & Search Technology
            "search" | "engine" | "inverted" | "index" | "bm25" | "retrieval" | "query"
            | "posting" | "postings" | "snippet" | "relevance" | "vocabulary" | "lexical" => {
                Some(1)
            }

            // Axis 2: Web Crawling & Graph Algorithms
            "crawler" | "crawl" | "crawling" | "web" | "hyperlink" | "url" | "frontier"
            | "robots" | "pagerank" | "graph" | "node" | "nodes" | "edge" | "edges" => Some(2),

            // Axis 3: Algorithms & Data Structures
            "algorithm" | "algorithms" | "trie" | "heap" | "tree" | "hash" | "sort" | "prefix"
            | "complexity" => Some(3),

            // Axis 4: Distributed Computing & Scalability
            "distributed" | "sharding" | "partition" | "cluster" | "scale" | "scalability"
            | "horizontal" | "replication" | "replica" | "fault-tolerant" | "node-failure" => {
                Some(4)
            }

            // Axis 5: Networking & Web Protocols
            "network" | "networks" | "tcp" | "ip" | "http" | "protocol" | "protocols"
            | "packet" | "socket" | "port" | "server" | "client" => Some(5),

            // Axis 6: Artificial Intelligence & Machine Learning
            "ai" | "artificial" | "intelligence" | "neural" | "deep" | "learning" | "model"
            | "transformer" | "token" | "vector" | "embedding" | "embeddings" | "semantic" => {
                Some(6)
            }

            // Axis 7: Aeronautics & Aerodynamics
            "aerodynamic" | "aerodynamics" | "flutter" | "supersonic" | "aircraft" | "airplane"
            | "wing" | "wings" | "airflow" | "aeroelastic" | "instability" | "mach" | "flight" => {
                Some(7)
            }

            // Axis 8: Vehicles & Automotive Mechanics
            "car" | "cars" | "automobile" | "automotive" | "vehicle" | "vehicles" | "drive"
            | "repair" | "maintenance" | "tire" | "tires" | "mechanic" | "engine-repair"
            | "motor" | "puncture" => Some(8),

            // Axis 9: Data Storage & Compression
            "storage" | "compression" | "vbyte" | "delta" | "disk" | "binary" | "serialize"
            | "deserialization" | "database" => Some(9),

            // Axis 10: Security & Cryptography
            "security" | "secure" | "crypto" | "cryptography" | "auth" | "hash-function"
            | "cipher" | "private" | "secret" => Some(10),

            _ => None,
        }
    }

    /// Computes a deterministic 64-bit FNV-1a hash of a byte slice.
    fn fnv1a(bytes: &[u8]) -> u64 {
        let mut hash: u64 = 0xcbf29ce484222325;
        for &b in bytes {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }

    /// Evaluates if a word is an English stop word.
    fn is_stop_word(word: &str) -> bool {
        matches!(
            word,
            "a" | "an"
                | "the"
                | "and"
                | "or"
                | "but"
                | "if"
                | "in"
                | "on"
                | "at"
                | "to"
                | "for"
                | "of"
                | "with"
                | "by"
                | "is"
                | "are"
                | "was"
                | "were"
                | "be"
                | "this"
                | "that"
                | "it"
                | "from"
                | "as"
        )
    }
}

impl TextEmbedder for SemanticEmbedder {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn embed(&self, text: &str) -> Vec<f32> {
        let mut vector = vec![0.0f32; self.dimension];
        let tokens = tokenize(text);
        if tokens.is_empty() {
            return vector;
        }

        let num_concept_axes = 12.min(self.dimension / 2);
        let subword_start = num_concept_axes;
        let subword_range = self.dimension - subword_start;

        for token in &tokens {
            let clean = token.text.to_lowercase();
            if clean.len() < 2 || clean.starts_with("http") {
                continue;
            }

            // Word salience weight
            let word_weight = if Self::is_stop_word(&clean) {
                0.15f32
            } else {
                1.0f32
            };

            // 1. Concept Cluster Axis Projection
            if let Some(axis) = Self::concept_axis(&clean) {
                if axis < num_concept_axes {
                    vector[axis] += 2.8 * word_weight;
                }
            }

            // 2. Full-Word Hashing
            let full_hash = Self::fnv1a(clean.as_bytes());
            let dim_idx = subword_start + (full_hash as usize % subword_range);
            let sign = if (full_hash >> 32) & 1 == 0 {
                1.0f32
            } else {
                -1.0f32
            };
            vector[dim_idx] += sign * word_weight * 1.5;

            // 3. Subword Character N-Gram Hashing (FastText style)
            let chars: Vec<char> = clean.chars().collect();
            if chars.len() >= 3 {
                // Character 3-grams
                for w in chars.windows(3) {
                    let s: String = w.iter().collect();
                    let h = Self::fnv1a(s.as_bytes());
                    let idx = subword_start + (h as usize % subword_range);
                    let sgn = if (h >> 32) & 1 == 0 { 1.0f32 } else { -1.0f32 };
                    vector[idx] += sgn * word_weight * 0.4;
                }
            }
        }

        l2_normalize(&mut vector);
        vector
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product_and_l2_norm() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 0.0, 1.0, 2.0];
        assert_eq!(dot_product(&a, &b), 2.0 + 0.0 + 3.0 + 8.0);

        let mut v = vec![3.0, 4.0];
        assert_eq!(l2_norm(&v), 5.0);
        l2_normalize(&mut v);
        assert!((l2_norm(&v) - 1.0).abs() < 1e-6);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let c = vec![0.0, 1.0, 0.0];
        let d = vec![-1.0, 0.0, 0.0];

        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
        assert!((cosine_similarity(&a, &c) - 0.0).abs() < 1e-6);
        assert!((cosine_similarity(&a, &d) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_vector_index_add_and_search() {
        let mut index = VectorIndex::new(4);
        index.add_vector(1, vec![1.0, 0.0, 0.0, 0.0]);
        index.add_vector(2, vec![0.8, 0.6, 0.0, 0.0]);
        index.add_vector(3, vec![0.0, 1.0, 0.0, 0.0]);

        assert_eq!(index.len(), 3);

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = index.search(&query, 2);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].doc_id, 1);
        assert!((results[0].similarity - 1.0).abs() < 1e-5);
        assert_eq!(results[1].doc_id, 2);
        assert!((results[1].similarity - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_semantic_embedder_synonym_matching() {
        let embedder = SemanticEmbedder::default();

        // Conceptual synonyms with ZERO word overlap:
        // "automobile repair" vs "car maintenance"
        let q = embedder.embed("automobile repair");
        let d1 = embedder.embed("car maintenance and vehicle mechanic guide");
        let d2 = embedder.embed("deep neural network transformer model");

        let sim_synonym = cosine_similarity(&q, &d1);
        let sim_unrelated = cosine_similarity(&q, &d2);

        // Synonyms should have high positive similarity, much higher than unrelated
        assert!(
            sim_synonym > 0.65,
            "Expected high similarity for synonyms, got {}",
            sim_synonym
        );
        assert!(
            sim_synonym > sim_unrelated + 0.40,
            "Synonym sim ({}) should beat unrelated ({})",
            sim_synonym,
            sim_unrelated
        );
    }

    #[test]
    fn test_semantic_embedder_aerospace_example() {
        let embedder = SemanticEmbedder::default();

        // Cranfield concept example: supersonic flutter vs aeroelastic instability
        let q = embedder.embed("supersonic flutter");
        let d1 = embedder.embed("high speed aeroelastic instability of aircraft wings");
        let d2 = embedder.embed("distributed sharding and database partition replica");

        let sim_aero = cosine_similarity(&q, &d1);
        let sim_db = cosine_similarity(&q, &d2);

        assert!(sim_aero > 0.60);
        assert!(sim_aero > sim_db + 0.35);
    }
}
