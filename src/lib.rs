#![allow(
    clippy::collapsible_if,
    clippy::collapsible_match,
    clippy::too_many_arguments,
    clippy::explicit_counter_loop,
    clippy::needless_range_loop,
    clippy::ptr_arg,
    clippy::inherent_to_string,
    clippy::useless_format,
    clippy::empty_line_after_doc_comments,
    clippy::len_zero
)]

pub mod analyzer;
pub mod compression;
pub mod crawler;
pub mod evaluation;
pub mod feedback;
pub mod graph;
pub mod index;
pub mod query;
pub mod ranking;
pub mod server;
pub mod snippet;
pub mod spelling;
pub mod stemmer;
pub mod storage;
pub mod tokenizer;
pub mod trie;

pub use analyzer::{Analyzer, Term};
pub use compression::{
    compress_sorted_u32, decode_deltas, decode_vbyte, decompress_sorted_u32, encode_deltas,
    encode_vbyte,
};
pub use crawler::{
    CrawlConfig, CrawlSummary, CrawledDocument, Crawler, ExtractedPage, HttpFetcher,
    HttpFetcherConfig, MockFetcher, PageFetcher, ParsedUrl, RobotsTxt, UrlFrontier,
    decode_html_entities, extract_page, normalize_url, parse_url, resolve_relative_url,
};
pub use evaluation::{
    BenchmarkComparison, BenchmarkMetrics, CranfieldDocument, QueryJudgment, average_precision,
    compare_rankers, dcg_at_k, evaluate_bm25, evaluate_bm25f, evaluate_bm25f_with_pagerank,
    evaluate_hybrid_pagerank, idcg_at_k, load_cranfield_dataset, ndcg_at_k,
    parse_cranfield_dataset, parse_cranfield_docs, parse_cranfield_qrels, parse_cranfield_queries,
    precision_at_k, recall_at_k, reciprocal_rank,
};
pub use feedback::{
    ExpandedQuery, PrfParams, default_stop_words, expand_from_feedback_bm25,
    expand_from_feedback_bm25f, expand_query_bm25, expand_query_bm25f, rank_bm25_with_prf,
    rank_bm25f_with_prf,
};
pub use graph::{PageRankParams, WebGraph, compute_pagerank};
pub use index::{DocId, Field, InvertedIndex, MultiFieldIndex, Posting};
pub use query::{
    QueryNode, QueryParseError, QueryToken, execute_query, parse_query, tokenize_query,
};
pub use ranking::{
    BM25FParams, BM25Params, FieldConfig, HybridBM25FParams, HybridRankingParams, ScoredDocument,
    idf, rank_bm25, rank_bm25_weighted, rank_bm25_with_pagerank, rank_bm25f, rank_bm25f_weighted,
    rank_bm25f_with_pagerank,
};
pub use server::{SearchEngineState, SearchServer, ServerConfig};
pub use snippet::{HighlightFormat, SnippetConfig, generate_snippet};
pub use spelling::{SpellChecker, Suggestion, damerau_levenshtein};
pub use stemmer::stem;
pub use storage::{
    DocumentMetadata, StorageError, load_from_file, load_metadata_from_file, save_metadata_to_file,
    save_to_file,
};
pub use tokenizer::{Token, tokenize};
pub use trie::{PrefixSuggestion, PrefixTrie, TrieNode};
