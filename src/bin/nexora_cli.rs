use std::collections::HashMap;
use std::io::{self, Write};
use std::time::Instant;

use nexora::{DocId, InvertedIndex};

struct CorpusDocument {
    id: DocId,
    title: &'static str,
    body: &'static str,
}

const CORPUS: &[CorpusDocument] = &[
    CorpusDocument {
        id: 0,
        title: "Introduction to Search Engines and Inverted Indexes",
        body: "A search engine is an information retrieval system that uses an inverted index to map vocabulary terms to postings lists of matching documents. The postings list contains document identifiers and token positions.",
    },
    CorpusDocument {
        id: 1,
        title: "Rust Systems Programming and Memory Safety",
        body: "Rust delivers bare-metal performance and memory safety without a garbage collector. Its ownership and borrow checker prevent data races, making it ideal for high-throughput search engine backends.",
    },
    CorpusDocument {
        id: 2,
        title: "The Okapi BM25 Ranking Algorithm",
        body: "Okapi BM25 is the gold standard lexical ranking algorithm. It balances term frequency saturation using parameter k1 and document length normalization using parameter b, improving upon classical TF-IDF.",
    },
    CorpusDocument {
        id: 3,
        title: "Web Crawlers and the URL Frontier",
        body: "A web crawler systematically browses the World Wide Web. It fetches HTML pages over HTTP, obeys robots.txt politeness policies, extracts hyperlinks, and schedules unvisited URLs in a URL frontier.",
    },
    CorpusDocument {
        id: 4,
        title: "Distributed Systems, Sharding, and Scalability",
        body: "When an index grows too large for a single machine, search engines use index partitioning and horizontal sharding. Replication provides high availability, fault tolerance, and load balancing across nodes.",
    },
    CorpusDocument {
        id: 5,
        title: "Computer Networks: TCP/IP and HTTP Protocols",
        body: "Modern search engines interact with network protocols like TCP/IP and HTTP. Devices communicate across IP addresses like 192.168.1.1 and send web requests to URLs like https://nexora.org/search.",
    },
];

fn main() {
    println!("=======================================================");
    println!("   🚀 NEXORA SEARCH ENGINE - INTERACTIVE REPL");
    println!("   Type your search query or ':help' for commands");
    println!("=======================================================\n");

    let mut index = InvertedIndex::new();
    let mut doc_store: HashMap<DocId, &CorpusDocument> = HashMap::new();

    let index_start = Instant::now();
    for doc in CORPUS {
        let full_text = format!("{} {}", doc.title, doc.body);
        index.add_document(doc.id, &full_text);
        doc_store.insert(doc.id, doc);
    }
    let index_duration = index_start.elapsed();

    println!(
        "✔ Indexed {} documents ({} unique vocabulary terms) in {:.2?}\n",
        index.total_documents(),
        index.vocabulary_size(),
        index_duration
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
                for doc in CORPUS {
                    println!("[Doc {}] {}", doc.id, doc.title);
                }
            }
            _ => {
                if let Some(args) = query.strip_prefix(":and ") {
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
    println!("Search Syntax:");
    println!("  <terms>          Free-text query using Okapi BM25 ranking (e.g. 'rust memory')");
    println!("  \"<phrase>\"       Exact consecutive phrase search (e.g. '\"inverted index\"')");
    println!("  :and <t1> <t2>   Boolean AND intersection (e.g. ':and crawler frontier')");
    println!("  :or  <t1> <t2>   Boolean OR union (e.g. ':or rust python')");
    println!("  :stats           Display index metadata and vocabulary size");
    println!("  :docs            List all indexed documents");
    println!("  :exit            Exit the search engine");
}

fn handle_bm25_search(
    index: &InvertedIndex,
    doc_store: &HashMap<DocId, &CorpusDocument>,
    query: &str,
) {
    let start = Instant::now();
    let results = index.search_bm25_default(query);
    let duration = start.elapsed();

    println!(
        "\n--- BM25 Ranked Results for: '{}' ({} matches in {:.2?}) ---",
        query,
        results.len(),
        duration
    );

    if results.is_empty() {
        println!("No matching documents found.");
        return;
    }

    for (rank, scored_doc) in results.iter().enumerate() {
        let doc = doc_store.get(&scored_doc.doc_id).unwrap();
        println!(
            "{}. [Score: {:.4}] [Doc {}] {}",
            rank + 1,
            scored_doc.score,
            doc.id,
            doc.title
        );
        println!("   \"{}\"\n", truncate_text(doc.body, 120));
    }
}

fn handle_phrase_search(
    index: &InvertedIndex,
    doc_store: &HashMap<DocId, &CorpusDocument>,
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

    for (rank, &doc_id) in doc_ids.iter().enumerate() {
        let doc = doc_store.get(&doc_id).unwrap();
        println!("{}. [Doc {}] {}", rank + 1, doc.id, doc.title);
        println!("   \"{}\"\n", truncate_text(doc.body, 120));
    }
}

fn handle_boolean_and(
    index: &InvertedIndex,
    doc_store: &HashMap<DocId, &CorpusDocument>,
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

    for (rank, posting) in matches.iter().enumerate() {
        let doc = doc_store.get(&posting.doc_id).unwrap();
        println!("{}. [Doc {}] {}", rank + 1, doc.id, doc.title);
        println!("   \"{}\"\n", truncate_text(doc.body, 120));
    }
}

fn handle_boolean_or(
    index: &InvertedIndex,
    doc_store: &HashMap<DocId, &CorpusDocument>,
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

    for (rank, posting) in matches.iter().enumerate() {
        let doc = doc_store.get(&posting.doc_id).unwrap();
        println!("{}. [Doc {}] {}", rank + 1, doc.id, doc.title);
        println!("   \"{}\"\n", truncate_text(doc.body, 120));
    }
}

fn truncate_text(text: &str, max_len: usize) -> String {
    if text.len() <= max_len {
        text.to_string()
    } else {
        let boundary = text.char_indices().nth(max_len).map(|(i, _)| i).unwrap_or(max_len);
        format!("{}...", &text[..boundary])
    }
}
