pub mod frontier;
pub mod html;
pub mod url;

pub use frontier::UrlFrontier;
pub use html::{decode_html_entities, extract_page, ExtractedPage};
pub use url::{normalize_url, parse_url, resolve_relative_url, ParsedUrl};

