use std::collections::HashMap;
use std::io::{self, Write};
use std::path::Path;
use std::time::{Duration, Instant};

use nexora::{
    compute_pagerank, generate_snippet, load_from_file, load_metadata_from_file,
    rank_bm25_with_pagerank, save_metadata_to_file, save_to_file, CrawlConfig, Crawler,
    DocId, DocumentMetadata, HighlightFormat, HttpFetcher, HybridRankingParams, InvertedIndex,
    PageRankParams, SnippetConfig, SpellChecker, UrlFrontier, WebGraph,
};

/// In-memory representation of an indexed document with optional URL and link authority.
#[derive(Debug, Clone)]
pub struct DocumentEntry {
    pub id: DocId,
    pub title: String,
    pub body: String,
    pub url: Option<String>,
    pub pagerank: f64,
}

const DEFAULT_CORPUS: &[(&str, &str)] = &[
    (
        "Introduction to Search Engines and Inverted Indexes",
        "A search engine is an information retrieval system that uses an inverted index to map vocabulary terms to postings lists of matching documents. The postings list contains document identifiers and token positions.",
    ),
    (
        "Rust Systems Programming and Memory Safety",
        "Rust delivers bare-metal performance and memory safety without a garbage collector. Its ownership and borrow checker prevent data races, making it ideal for high-throughput search engine backends.",
    ),
    (
        "The Okapi BM25 Ranking Algorithm",
        "Okapi BM25 is the gold standard lexical ranking algorithm. It balances term frequency saturation using parameter k1 and document length normalization using parameter b, improving upon classical TF-IDF.",
    ),
    (
        "Web Crawlers and the URL Frontier",
        "A web crawler systematically browses the World Wide Web. It fetches HTML pages over HTTP, obeys robots.txt politeness policies, extracts hyperlinks, and schedules unvisited URLs in a URL frontier.",
    ),
    (
        "Distributed Systems, Sharding, and Scalability",
        "When an index grows too large for a single machine, search engines use index partitioning and horizontal sharding. Replication provides high availability, fault tolerance, and load balancing across nodes.",
    ),
    (
        "Computer Networks: TCP/IP and HTTP Protocols",
        "Modern search engines interact with network protocols like TCP/IP and HTTP. Devices communicate across IP addresses like 192.168.1.1 and send web requests to URLs like https://nexora.org/search.",
    ),
];

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() > 1 {
        match args[1].as_str() {
            "crawl" => handle_cli_crawl(&args[2..]),
            "search" => handle_cli_search(&args[2..]),
            "--help" | "-h" | "help" => print_cli_usage(),
            unknown => {
                eprintln!("Unknown command: '{}'. Run '--help' for usage.", unknown);
                std::process::exit(1);
            }
        }
    } else {
        run_interactive_repl();
    }
}

fn print_cli_usage() {
    println!("Nexora Search Engine - Command Line Interface\n");
    println!("USAGE:");
    println!("  nexora_cli crawl <seed_url> [--max-pages <N>] [--output <file.nex>]");
    println!("  nexora_cli search <query> [--index <file.nex>]");
    println!("  nexora_cli                                  (launches interactive REPL)\n");
    println!("EXAMPLES:");
    println!("  nexora_cli crawl https://example.com --max-pages 10 --output web.nex");
    println!("  nexora_cli search \"search engine\" --index web.nex");
}

fn handle_cli_crawl(args: &[String]) {
    if args.is_empty() {
        eprintln!("Error: Missing seed URL. Usage: nexora_cli crawl <seed_url> [--max-pages <N>] [--output <file.nex>]");
        std::process::exit(1);
    }

    let seed_url = &args[0];
    let mut max_pages = 20;
    let mut output_path = "crawl_index.nex".to_string();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--max-pages" if i + 1 < args.len() => {
                max_pages = args[i + 1].parse().unwrap_or(20);
                i += 2;
            }
            "--output" if i + 1 < args.len() => {
                output_path = args[i + 1].clone();
                i += 2;
            }
            other => {
                eprintln!("Warning: Unrecognized argument '{}'", other);
                i += 1;
            }
        }
    }

    match execute_crawl(seed_url, max_pages, &output_path) {
        Ok(_) => println!("\n✔ Crawl and index build completed successfully."),
        Err(e) => {
            eprintln!("\n✘ Crawl failed: {}", e);
            std::process::exit(1);
        }
    }
}

fn handle_cli_search(args: &[String]) {
    if args.is_empty() {
        eprintln!("Error: Missing query string. Usage: nexora_cli search <query> [--index <file.nex>]");
        std::process::exit(1);
    }

    let query = &args[0];
    let mut index_path: Option<&str> = None;

    let mut i = 1;
    while i < args.len() {
        if args[i] == "--index" && i + 1 < args.len() {
            index_path = Some(&args[i + 1]);
            i += 2;
        } else {
            i += 1;
        }
    }

    let (index, doc_store) = if let Some(path) = index_path {
        load_index_and_metadata(path).unwrap_or_else(|e| {
            eprintln!("Error loading index from '{}': {}", path, e);
            std::process::exit(1);
        })
    } else {
        build_default_corpus()
    };

    handle_bm25_search(&index, &doc_store, query);
}

fn run_interactive_repl() {
    println!("=======================================================");
    println!("   🚀 NEXORA SEARCH ENGINE - INTERACTIVE REPL");
    println!("   Type your search query or ':help' for commands");
    println!("=======================================================\n");

    let (mut index, mut doc_store) = build_default_corpus();

    println!(
        "✔ Loaded {} default documents ({} unique vocabulary terms)\n",
        index.total_documents(),
        index.vocabulary_size()
    );

    print_help();

    let stdin = io::stdin();
    loop {
        print!("\nnexora> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        if stdin.read_line(&mut input).is_err() {
            break;
        }

        let query = input.trim();
        if query.is_empty() {
            continue;
        }

        match query {
            ":exit" | ":quit" | "exit" | "quit" => {
                println!("Goodbye!");
                break;
            }
            ":help" => {
                print_help();
            }
            ":stats" => {
                println!("\n--- Index Statistics ---");
                println!("Total Documents:      {}", index.total_documents());
                println!("Vocabulary Size:      {} unique terms", index.vocabulary_size());
                println!("Average Doc Length:   {:.1} words", index.average_doc_length());
            }
            ":docs" => {
                println!("\n--- Indexed Documents ---");
                let mut docs: Vec<&DocumentEntry> = doc_store.values().collect();
                docs.sort_by_key(|d| d.id);
                for doc in docs {
                    if let Some(ref url) = doc.url {
                        println!("[Doc {}] [PR: {:.4}] {} ({})", doc.id, doc.pagerank, doc.title, url);
                    } else {
                        println!("[Doc {}] {}", doc.id, doc.title);
                    }
                }
            }
            _ => {
                if let Some(args) = query.strip_prefix(":crawl ") {
                    let parts: Vec<&str> = args.split_whitespace().collect();
                    if parts.is_empty() {
                        println!("Usage: :crawl <seed_url> [max_pages] [output_file.nex]");
                        continue;
                    }

                    let seed = parts[0];
                    let max_pages = parts.get(1).and_then(|p| p.parse().ok()).unwrap_or(15);
                    let out_file = parts.get(2).copied().unwrap_or("web.nex");

                    match execute_crawl(seed, max_pages, out_file) {
                        Ok((new_index, new_docs)) => {
                            index = new_index;
                            doc_store = new_docs;
                            println!("✔ Active REPL index switched to newly crawled corpus!");
                        }
                        Err(e) => println!("✘ Crawl error: {}", e),
                    }
                } else if let Some(path) = query.strip_prefix(":save ") {
                    let path = path.trim();
                    match save_to_file(&index, path) {
                        Ok(()) => println!("✔ Successfully persisted index to '{}'", path),
                        Err(e) => println!("✘ Failed to save index: {}", e),
                    }
                } else if let Some(path) = query.strip_prefix(":load ") {
                    let path = path.trim();
                    let start = Instant::now();
                    match load_index_and_metadata(path) {
                        Ok((loaded_index, loaded_docs)) => {
                            index = loaded_index;
                            doc_store = loaded_docs;
                            println!(
                                "✔ Loaded index and metadata from '{}' ({} docs, {} terms) in {:.2?}",
                                path,
                                index.total_documents(),
                                index.vocabulary_size(),
                                start.elapsed()
                            );
                        }
                        Err(e) => println!("✘ Failed to load index: {}", e),
                    }
                } else if let Some(args) = query.strip_prefix(":and ") {
                    handle_boolean_and(&index, &doc_store, args);
                } else if let Some(args) = query.strip_prefix(":or ") {
                    handle_boolean_or(&index, &doc_store, args);
                } else if query.starts_with('"') && query.ends_with('"') && query.len() >= 2 {
                    let phrase = &query[1..query.len() - 1];
                    handle_phrase_search(&index, &doc_store, phrase);
                } else {
                    handle_bm25_search(&index, &doc_store, query);
                }
            }
        }
    }
}

fn print_help() {
    println!("Commands:");
    println!("  <terms>                     Free-text search (BM25 + PageRank hybrid ranking)");
    println!("  \"<phrase>\"                  Exact consecutive phrase search (e.g. '\"inverted index\"')");
    println!("  :and <t1> <t2>              Boolean AND intersection");
    println!("  :or  <t1> <t2>              Boolean OR union");
    println!("  :crawl <url> [N] [out.nex]  Crawl website, calculate PageRank, and save index");
    println!("  :save <path>                Persist active index to binary file (.nex v2)");
    println!("  :load <path>                Load index and metadata from disk (.nex)");
    println!("  :stats                      Display index metadata and vocabulary size");
    println!("  :docs                       List all indexed documents");
    println!("  :exit                       Exit the search engine");
}

fn execute_crawl(
    seed_url: &str,
    max_pages: usize,
    output_path: &str,
) -> Result<(InvertedIndex, HashMap<DocId, DocumentEntry>), String> {
    println!("\n--- Initiating Web Crawl ---");
    println!("Seed URL:     {}", seed_url);
    println!("Max Pages:    {}", max_pages);
    println!("Output File:  {}", output_path);

    let start = Instant::now();

    let mut frontier = UrlFrontier::default_polite();
    if !frontier.push(seed_url) {
        return Err(format!("Invalid seed URL: {}", seed_url));
    }

    let fetcher = HttpFetcher::default_client();
    let config = CrawlConfig {
        max_pages,
        max_wait: Duration::from_secs(5),
        respect_robots_txt: true,
        user_agent: "NexoraBot/0.1".to_string(),
    };

    let mut crawler = Crawler::new(frontier, fetcher, config);
    let mut index = InvertedIndex::new();

    println!("Crawling with RFC 9309 robots.txt compliance and per-domain politeness...");
    let summary = crawler.crawl(&mut index);

    println!(
        "\n✔ Crawl complete in {:.2?}: {} pages visited, {} disallowed, {} links discovered.",
        start.elapsed(),
        summary.pages_visited,
        summary.pages_disallowed,
        summary.links_discovered
    );

    if summary.documents.is_empty() {
        return Err("No pages were successfully crawled.".to_string());
    }

    // Build WebGraph & Compute PageRank
    println!("Constructing WebGraph and calculating PageRank authority...");
    let graph = WebGraph::from_crawled_documents(&summary.documents);
    let pr_params = PageRankParams::default();
    let pagerank_scores = compute_pagerank(&graph, &pr_params);

    println!(
        "✔ PageRank converged across {} nodes in web graph.",
        graph.len()
    );

    // Save index
    save_to_file(&index, output_path)
        .map_err(|e| format!("Failed to save index to '{}': {}", output_path, e))?;

    // Save companion metadata
    let meta_path = format!("{}.meta", output_path);
    let mut metadata_map = HashMap::with_capacity(summary.documents.len());
    let mut doc_store = HashMap::with_capacity(summary.documents.len());

    for doc in summary.documents {
        let pr = pagerank_scores.get(&doc.doc_id).copied().unwrap_or(0.0);

        metadata_map.insert(
            doc.doc_id,
            DocumentMetadata {
                doc_id: doc.doc_id,
                url: doc.url.clone(),
                title: doc.title.clone(),
                body: doc.text.clone(),
                pagerank: pr,
            },
        );

        doc_store.insert(
            doc.doc_id,
            DocumentEntry {
                id: doc.doc_id,
                title: doc.title,
                body: doc.text,
                url: Some(doc.url),
                pagerank: pr,
            },
        );
    }

    save_metadata_to_file(&metadata_map, &meta_path)
        .map_err(|e| format!("Failed to save metadata to '{}': {}", meta_path, e))?;

    println!("✔ Saved compressed index to '{}' and metadata to '{}'", output_path, meta_path);

    Ok((index, doc_store))
}

fn load_index_and_metadata(
    index_path: &str,
) -> Result<(InvertedIndex, HashMap<DocId, DocumentEntry>), String> {
    let index = load_from_file(index_path)
        .map_err(|e| format!("Index load error: {}", e))?;

    let meta_path = format!("{}.meta", index_path);
    let mut doc_store = HashMap::new();

    if Path::new(&meta_path).exists() {
        let meta_map = load_metadata_from_file(&meta_path)
            .map_err(|e| format!("Metadata load error: {}", e))?;

        for (id, meta) in meta_map {
            doc_store.insert(
                id,
                DocumentEntry {
                    id,
                    title: meta.title,
                    body: meta.body,
                    url: Some(meta.url),
                    pagerank: meta.pagerank,
                },
            );
        }
    } else {
        // Fallback placeholder records if no metadata companion file exists
        for &doc_id in index.doc_lengths().keys() {
            doc_store.insert(
                doc_id,
                DocumentEntry {
                    id: doc_id,
                    title: format!("Document #{}", doc_id),
                    body: String::new(),
                    url: None,
                    pagerank: 0.0,
                },
            );
        }
    }

    Ok((index, doc_store))
}

fn build_default_corpus() -> (InvertedIndex, HashMap<DocId, DocumentEntry>) {
    let mut index = InvertedIndex::new();
    let mut doc_store = HashMap::new();

    for (i, &(title, body)) in DEFAULT_CORPUS.iter().enumerate() {
        let doc_id = i as DocId;
        let full_text = format!("{} {}", title, body);
        index.add_document(doc_id, &full_text);

        doc_store.insert(
            doc_id,
            DocumentEntry {
                id: doc_id,
                title: title.to_string(),
                body: body.to_string(),
                url: None,
                pagerank: 1.0 / DEFAULT_CORPUS.len() as f64,
            },
        );
    }

    (index, doc_store)
}

fn handle_bm25_search(
    index: &InvertedIndex,
    doc_store: &HashMap<DocId, DocumentEntry>,
    query: &str,
) {
    let start = Instant::now();

    // Prepare PageRank map
    let pr_map: HashMap<DocId, f64> = doc_store
        .iter()
        .map(|(&id, d)| (id, d.pagerank))
        .collect();

    let hybrid_params = HybridRankingParams::default();
    let results = rank_bm25_with_pagerank(index, query, &pr_map, &hybrid_params);
    let duration = start.elapsed();

    println!(
        "\n--- Search Results for: '{}' ({} matches in {:.2?}) ---",
        query,
        results.len(),
        duration
    );

    if results.is_empty() {
        println!("No matching documents found for '{}'.", query);

        let spell_checker = SpellChecker::from_documents(
            doc_store.values().map(|d| format!("{} {}", d.title, d.body)),
        );

        if let Some(suggestion) = spell_checker.suggest_query(query) {
            println!("\n💡 Did you mean: \x1b[1m\x1b[36m{}\x1b[0m?", suggestion);

            let corrected_results =
                rank_bm25_with_pagerank(index, &suggestion, &pr_map, &hybrid_params);
            if !corrected_results.is_empty() {
                println!(
                    "\n--- Showing Top Results for: '{}' ({} matches) ---",
                    suggestion,
                    corrected_results.len()
                );
                render_scored_results(
                    &corrected_results,
                    doc_store,
                    &suggestion,
                    index.analyzer(),
                );
            }
        }
        return;
    }

    render_scored_results(&results, doc_store, query, index.analyzer());
}

fn render_scored_results(
    results: &[nexora::ScoredDocument],
    doc_store: &HashMap<DocId, DocumentEntry>,
    query: &str,
    analyzer: &nexora::Analyzer,
) {
    let snippet_cfg = SnippetConfig {
        max_chars: 160,
        format: HighlightFormat::Ansi,
    };

    for (rank, scored_doc) in results.iter().enumerate() {
        if let Some(doc) = doc_store.get(&scored_doc.doc_id) {
            let snippet = generate_snippet(&doc.body, query, analyzer, &snippet_cfg);

            if let Some(ref url) = doc.url {
                println!(
                    "{}. [Score: {:.4}] [PR: {:.4}] {}",
                    rank + 1,
                    scored_doc.score,
                    doc.pagerank,
                    doc.title
                );
                println!("   \x1b[34m{}\x1b[0m", url);
            } else {
                println!(
                    "{}. [Score: {:.4}] [Doc {}] {}",
                    rank + 1,
                    scored_doc.score,
                    doc.id,
                    doc.title
                );
            }

            println!("   \"{}\"\n", snippet);
        }
    }
}

fn handle_phrase_search(
    index: &InvertedIndex,
    doc_store: &HashMap<DocId, DocumentEntry>,
    phrase: &str,
) {
    let start = Instant::now();
    let doc_ids = index.search_phrase(phrase);
    let duration = start.elapsed();

    println!(
        "\n--- Exact Phrase Search for: '\"{}\"' ({} matches in {:.2?}) ---",
        phrase,
        doc_ids.len(),
        duration
    );

    if doc_ids.is_empty() {
        println!("No documents contain the exact phrase in consecutive order.");
        return;
    }

    let snippet_cfg = SnippetConfig::default();
    for (rank, &doc_id) in doc_ids.iter().enumerate() {
        if let Some(doc) = doc_store.get(&doc_id) {
            let snippet = generate_snippet(&doc.body, phrase, index.analyzer(), &snippet_cfg);
            println!("{}. [Doc {}] {}", rank + 1, doc.id, doc.title);
            if let Some(ref url) = doc.url {
                println!("   \x1b[34m{}\x1b[0m", url);
            }
            println!("   \"{}\"\n", snippet);
        }
    }
}

fn handle_boolean_and(
    index: &InvertedIndex,
    doc_store: &HashMap<DocId, DocumentEntry>,
    args: &str,
) {
    let terms: Vec<&str> = args.split_whitespace().collect();
    let start = Instant::now();
    let matches = index.search_and(&terms);
    let duration = start.elapsed();

    println!(
        "\n--- Boolean AND Search for: {:?} ({} matches in {:.2?}) ---",
        terms,
        matches.len(),
        duration
    );

    let snippet_cfg = SnippetConfig::default();
    for (rank, posting) in matches.iter().enumerate() {
        if let Some(doc) = doc_store.get(&posting.doc_id) {
            let snippet = generate_snippet(&doc.body, args, index.analyzer(), &snippet_cfg);
            println!("{}. [Doc {}] {}", rank + 1, doc.id, doc.title);
            if let Some(ref url) = doc.url {
                println!("   \x1b[34m{}\x1b[0m", url);
            }
            println!("   \"{}\"\n", snippet);
        }
    }
}

fn handle_boolean_or(
    index: &InvertedIndex,
    doc_store: &HashMap<DocId, DocumentEntry>,
    args: &str,
) {
    let terms: Vec<&str> = args.split_whitespace().collect();
    let start = Instant::now();
    let matches = index.search_or(&terms);
    let duration = start.elapsed();

    println!(
        "\n--- Boolean OR Search for: {:?} ({} matches in {:.2?}) ---",
        terms,
        matches.len(),
        duration
    );

    let snippet_cfg = SnippetConfig::default();
    for (rank, posting) in matches.iter().enumerate() {
        if let Some(doc) = doc_store.get(&posting.doc_id) {
            let snippet = generate_snippet(&doc.body, args, index.analyzer(), &snippet_cfg);
            println!("{}. [Doc {}] {}", rank + 1, doc.id, doc.title);
            if let Some(ref url) = doc.url {
                println!("   \x1b[34m{}\x1b[0m", url);
            }
            println!("   \"{}\"\n", snippet);
        }
    }
}
