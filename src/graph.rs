use std::collections::{HashMap, HashSet};

use crate::crawler::engine::CrawledDocument;
use crate::index::DocId;

/// A directed graph representing hyperlinks between documents.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WebGraph {
    /// Maps each DocId to its outbound target DocIds
    out_links: HashMap<DocId, HashSet<DocId>>,
    /// Maps each DocId to its inbound source DocIds
    in_links: HashMap<DocId, HashSet<DocId>>,
    /// All node identifiers present in the graph
    nodes: HashSet<DocId>,
}

impl WebGraph {
    /// Creates a new, empty web graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a document node in the graph.
    pub fn add_node(&mut self, node: DocId) {
        self.nodes.insert(node);
        self.out_links.entry(node).or_default();
        self.in_links.entry(node).or_default();
    }

    /// Adds a directed hyperlink edge from `from` to `to`.
    /// Self-loops (`from == to`) are ignored to prevent artificial self-promotion.
    pub fn add_edge(&mut self, from: DocId, to: DocId) {
        if from == to {
            return;
        }

        self.add_node(from);
        self.add_node(to);

        self.out_links.entry(from).or_default().insert(to);
        self.in_links.entry(to).or_default().insert(from);
    }

    /// Returns the set of all unique document IDs in the graph.
    pub fn nodes(&self) -> &HashSet<DocId> {
        &self.nodes
    }

    /// Returns the total number of nodes in the graph.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns true if the graph contains no nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Returns the total number of directed edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.out_links.values().map(|s| s.len()).sum()
    }

    /// Returns the number of outbound hyperlinks originating from `node`.
    pub fn out_degree(&self, node: DocId) -> usize {
        self.out_links.get(&node).map(|s| s.len()).unwrap_or(0)
    }

    /// Returns the number of inbound hyperlinks targeting `node`.
    pub fn in_degree(&self, node: DocId) -> usize {
        self.in_links.get(&node).map(|s| s.len()).unwrap_or(0)
    }

    /// Returns the inbound neighbor nodes that link to `node`.
    pub fn in_neighbors(&self, node: DocId) -> Option<&HashSet<DocId>> {
        self.in_links.get(&node)
    }

    /// Returns the outbound neighbor nodes that `node` links to.
    pub fn out_neighbors(&self, node: DocId) -> Option<&HashSet<DocId>> {
        self.out_links.get(&node)
    }

    /// Constructs a graph from a slice of `(from, to)` directed edges.
    pub fn from_edges(edges: &[(DocId, DocId)]) -> Self {
        let mut graph = Self::new();
        for &(from, to) in edges {
            graph.add_edge(from, to);
        }
        graph
    }

    /// Constructs a WebGraph directly from crawler summary documents,
    /// translating canonical URL hyperlinks into internal DocId graph edges.
    pub fn from_crawled_documents(docs: &[CrawledDocument]) -> Self {
        let mut graph = Self::new();

        // 1. Map canonical URL -> DocId
        let mut url_to_doc_id: HashMap<&str, DocId> = HashMap::with_capacity(docs.len());
        for doc in docs {
            graph.add_node(doc.doc_id);
            url_to_doc_id.insert(&doc.url, doc.doc_id);
        }

        // 2. Add edges for resolved outgoing links
        for doc in docs {
            for target_url in &doc.outgoing_links {
                if let Some(&target_id) = url_to_doc_id.get(target_url.as_str()) {
                    graph.add_edge(doc.doc_id, target_id);
                }
            }
        }

        graph
    }
}

/// Hyperparameters controlling PageRank power iteration convergence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PageRankParams {
    /// Probability that the random surfer follows a hyperlink rather than teleporting (default: 0.85).
    pub damping: f64,
    /// Convergence threshold: iteration stops when L1 norm delta < tolerance (default: 1e-6).
    pub tolerance: f64,
    /// Maximum number of power iterations to execute (default: 100).
    pub max_iterations: usize,
}

impl Default for PageRankParams {
    fn default() -> Self {
        Self {
            damping: 0.85,
            tolerance: 1e-6,
            max_iterations: 100,
        }
    }
}

/// Computes the PageRank authority score for each document in the graph
/// using the standard Power Iteration algorithm with dangling node redistribution.
///
/// Mathematical guarantee:
/// - Sum of all PageRank scores across the graph equals 1.0 (probability distribution).
/// - Dangling nodes (nodes with 0 outbound links) redistribute probability mass evenly.
pub fn compute_pagerank(graph: &WebGraph, params: &PageRankParams) -> HashMap<DocId, f64> {
    let n = graph.len();
    if n == 0 {
        return HashMap::new();
    }

    let initial_val = 1.0 / n as f64;
    if n == 1 {
        let mut single = HashMap::new();
        if let Some(&node) = graph.nodes().iter().next() {
            single.insert(node, 1.0);
        }
        return single;
    }

    let nodes: Vec<DocId> = graph.nodes().iter().copied().collect();

    // Precompute out-degrees and dangling node set
    let mut out_degrees: HashMap<DocId, usize> = HashMap::with_capacity(n);
    let mut dangling_nodes: Vec<DocId> = Vec::new();

    for &node in &nodes {
        let deg = graph.out_degree(node);
        out_degrees.insert(node, deg);
        if deg == 0 {
            dangling_nodes.push(node);
        }
    }

    // Initialize probability distribution vector PR^(0)
    let mut current_pr: HashMap<DocId, f64> = nodes.iter().map(|&node| (node, initial_val)).collect();
    let mut next_pr: HashMap<DocId, f64> = HashMap::with_capacity(n);

    let d = params.damping;
    let teleport_prob = (1.0 - d) / n as f64;

    for _ in 0..params.max_iterations {
        // 1. Calculate total dangling node probability mass to redistribute
        let dangling_sum: f64 = dangling_nodes.iter().map(|node| current_pr[node]).sum();
        let dangling_redistribution = d * (dangling_sum / n as f64);

        let mut delta = 0.0;

        // 2. Compute PR^(k+1) for each node
        for &u in &nodes {
            let mut in_link_sum = 0.0;
            if let Some(in_neighbors) = graph.in_neighbors(u) {
                for &v in in_neighbors {
                    let out_deg = out_degrees[&v];
                    if out_deg > 0 {
                        in_link_sum += current_pr[&v] / out_deg as f64;
                    }
                }
            }

            let new_score = teleport_prob + dangling_redistribution + d * in_link_sum;
            next_pr.insert(u, new_score);
            delta += (new_score - current_pr[&u]).abs();
        }

        // 3. Swap vectors
        std::mem::swap(&mut current_pr, &mut next_pr);

        // 4. Check for L1 convergence
        if delta < params.tolerance {
            break;
        }
    }

    current_pr
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symmetric_two_node_cycle() {
        let mut graph = WebGraph::new();
        // 1 <-> 2
        graph.add_edge(1, 2);
        graph.add_edge(2, 1);

        let pr = compute_pagerank(&graph, &PageRankParams::default());

        assert_eq!(pr.len(), 2);
        // By symmetry, both nodes must receive exactly 0.5 PageRank
        assert!((pr[&1] - 0.5).abs() < 1e-5);
        assert!((pr[&2] - 0.5).abs() < 1e-5);

        let total: f64 = pr.values().sum();
        assert!((total - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_three_node_directed_cycle() {
        let mut graph = WebGraph::new();
        // 1 -> 2 -> 3 -> 1
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph.add_edge(3, 1);

        let pr = compute_pagerank(&graph, &PageRankParams::default());

        assert_eq!(pr.len(), 3);
        let expected = 1.0 / 3.0;
        assert!((pr[&1] - expected).abs() < 1e-5);
        assert!((pr[&2] - expected).abs() < 1e-5);
        assert!((pr[&3] - expected).abs() < 1e-5);

        let total: f64 = pr.values().sum();
        assert!((total - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_authority_hub_star_graph() {
        let mut graph = WebGraph::new();
        // Nodes 2, 3, 4 all link to Node 1 (the Authority)
        graph.add_edge(2, 1);
        graph.add_edge(3, 1);
        graph.add_edge(4, 1);
        // Node 1 links back to Node 2
        graph.add_edge(1, 2);

        let pr = compute_pagerank(&graph, &PageRankParams::default());

        // Node 1 must have the strictly highest authority
        assert!(pr[&1] > pr[&2]);
        assert!(pr[&1] > pr[&3]);
        assert!(pr[&1] > pr[&4]);

        let total: f64 = pr.values().sum();
        assert!((total - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_dangling_node_mass_preservation() {
        let mut graph = WebGraph::new();
        // 1 -> 2 -> 3 (3 is a dead end / dangling node with 0 outgoing links)
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);

        let pr = compute_pagerank(&graph, &PageRankParams::default());

        assert_eq!(pr.len(), 3);
        let total: f64 = pr.values().sum();
        // Mass must be strictly preserved at 1.0 despite dangling node 3
        assert!((total - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_from_crawled_documents() {
        let docs = vec![
            CrawledDocument {
                doc_id: 1,
                url: "https://example.com/home".to_string(),
                title: "Home".to_string(),
                text: "Welcome".to_string(),
                outgoing_links: vec!["https://example.com/about".to_string()],
            },
            CrawledDocument {
                doc_id: 2,
                url: "https://example.com/about".to_string(),
                title: "About".to_string(),
                text: "About us".to_string(),
                outgoing_links: vec![
                    "https://example.com/home".to_string(),
                    "https://external.org/unknown".to_string(), // not in crawl, ignored
                ],
            },
        ];

        let graph = WebGraph::from_crawled_documents(&docs);
        assert_eq!(graph.len(), 2);
        assert_eq!(graph.out_degree(1), 1);
        assert_eq!(graph.in_degree(1), 1);
        assert_eq!(graph.out_degree(2), 1);
        assert_eq!(graph.in_degree(2), 1);
    }
}
