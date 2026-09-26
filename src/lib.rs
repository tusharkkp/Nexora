pub mod analyzer;
pub mod compression;
pub mod crawler;
pub mod evaluation;
pub mod graph;
pub mod index;
pub mod ranking;
pub mod snippet;
pub mod spelling;
pub mod stemmer;
pub mod storage;
pub mod tokenizer;
pub mod trie;
pub mod server;

pub use analyzer::{Analyzer, Term};
pub use compression::{
    compress_sorted_u32, decode_deltas, decode_vbyte, decompress_sorted_u32, encode_deltas,
    encode_vbyte,
};
pub use crawler::{
    decode_html_entities, extract_page, normalize_url, parse_url, resolve_relative_url,
    CrawlConfig, CrawlSummary, CrawledDocument, Crawler, ExtractedPage, HttpFetcher,
    HttpFetcherConfig, MockFetcher, PageFetcher, ParsedUrl, RobotsTxt, UrlFrontier,
};
pub use evaluation::{
    compare_rankers, dcg_at_k, evaluate_bm25, evaluate_hybrid_pagerank, idcg_at_k, ndcg_at_k,
    precision_at_k, recall_at_k, reciprocal_rank, BenchmarkComparison, BenchmarkMetrics,
    QueryJudgment,
};
pub use graph::{compute_pagerank, PageRankParams, WebGraph};
pub use index::{DocId, InvertedIndex, Posting};
pub use ranking::{
    idf, rank_bm25, rank_bm25_with_pagerank, BM25Params, HybridRankingParams, ScoredDocument,
};
pub use server::{SearchEngineState, SearchServer, ServerConfig};
pub use snippet::{generate_snippet, HighlightFormat, SnippetConfig};
pub use spelling::{damerau_levenshtein, SpellChecker, Suggestion};
pub use stemmer::stem;
pub use storage::{
    load_from_file, load_metadata_from_file, save_metadata_to_file, save_to_file, DocumentMetadata,
    StorageError,
};
pub use tokenizer::{tokenize, Token};
pub use trie::{PrefixSuggestion, PrefixTrie, TrieNode};







