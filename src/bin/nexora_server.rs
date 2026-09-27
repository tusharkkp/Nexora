use std::collections::HashMap;
use std::env;
use std::path::Path;

use nexora::{
    SearchEngineState, SearchServer, ServerConfig, load_from_file, load_metadata_from_file,
};

fn print_usage() {
    println!("Nexora HTTP Search Server\n");
    println!("USAGE:");
    println!("  nexora_server [--port <PORT>] [--host <HOST>] [--index <file.nex>]\n");
    println!("OPTIONS:");
    println!("  --port <PORT>        Port to bind server (default: 8080)");
    println!("  --host <HOST>        Host address to bind server (default: 127.0.0.1)");
    println!("  --index <file.nex>   Path to saved .nex binary index and companion .meta file");
    println!("  --crawl-seed <URL>   Seed URL to launch continuous background crawler on startup");
    println!("  --crawl-pages <N>    Maximum pages to crawl (default: 50, 0 = unlimited)");
    println!(
        "  --crawl-delay <MS>   Politeness cooldown delay per domain in milliseconds (default: 500)"
    );
    println!("  --help               Display this help text\n");
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<String> = env::args().skip(1).collect();

    let mut port: u16 = 8080;
    let mut host = "127.0.0.1".to_string();
    let mut index_path: Option<String> = None;
    let mut crawl_seed: Option<String> = None;
    let mut crawl_pages: usize = 50;
    let mut crawl_delay_ms: u64 = 500;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                print_usage();
                return Ok(());
            }
            "--port" | "-p" if i + 1 < args.len() => {
                port = args[i + 1].parse().unwrap_or(8080);
                i += 2;
            }
            "--host" if i + 1 < args.len() => {
                host = args[i + 1].clone();
                i += 2;
            }
            "--index" if i + 1 < args.len() => {
                index_path = Some(args[i + 1].clone());
                i += 2;
            }
            "--crawl-seed" if i + 1 < args.len() => {
                crawl_seed = Some(args[i + 1].clone());
                i += 2;
            }
            "--crawl-pages" if i + 1 < args.len() => {
                crawl_pages = args[i + 1].parse().unwrap_or(50);
                i += 2;
            }
            "--crawl-delay" if i + 1 < args.len() => {
                crawl_delay_ms = args[i + 1].parse().unwrap_or(500);
                i += 2;
            }
            other => {
                eprintln!("Unknown argument '{}'. Use --help for usage.", other);
                i += 1;
            }
        }
    }

    println!("=======================================================");
    println!("   🌐 NEXORA EMBEDDED HTTP SEARCH ENGINE & SERP");
    println!("=======================================================");

    let state = if let Some(ref path) = index_path {
        println!("Loading index from '{}'...", path);
        let index = load_from_file(path)
            .map_err(|e| format!("Failed to load index file '{}': {}", path, e))?;

        let meta_path = format!("{}.meta", path);
        let metadata = if Path::new(&meta_path).exists() {
            println!("Loading companion metadata from '{}'...", meta_path);
            load_metadata_from_file(&meta_path)
                .map_err(|e| format!("Failed to load metadata file '{}': {}", meta_path, e))?
        } else {
            HashMap::new()
        };

        println!(
            "✔ Successfully loaded {} documents and {} vocabulary terms",
            index.total_documents(),
            index.vocabulary_size()
        );
        SearchEngineState::new(index, metadata)
    } else {
        println!("✔ Initializing default demo corpus (6 documents, 130 terms)");
        SearchEngineState::default_demo()
    };

    let config = ServerConfig { host, port };
    let server = SearchServer::new(config, state);

    if let Some(ref seed) = crawl_seed {
        let delay = std::time::Duration::from_millis(crawl_delay_ms);
        server
            .crawler()
            .start(&[seed.as_str()], Some(crawl_pages), Some(delay));
        println!(
            "🕷 Autonomous Background Crawler launched with seed: {}",
            seed
        );
    }

    println!("\nEndpoints:");
    println!("  • Web Interface:    http://localhost:{}", port);
    println!(
        "  • Search API:       http://localhost:{}/api/search?q=<query>",
        port
    );
    println!(
        "  • Suggest API:      http://localhost:{}/api/suggest?q=<prefix>",
        port
    );
    println!("  • Index Stats:      http://localhost:{}/api/stats", port);
    println!(
        "  • Crawler Telemetry: GET http://localhost:{}/api/crawler/status",
        port
    );
    println!(
        "  • Crawler Control:   POST http://localhost:{}/api/crawler/start?url=<url>",
        port
    );
    println!("\nPress Ctrl+C to terminate the server.\n");

    server.run()
}
