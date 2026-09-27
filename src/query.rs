use crate::index::{InvertedIndex, Posting};

/// Lexical tokens for the search query language.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryToken {
    Term(String),
    Phrase(String),
    Prefix(String),
    And,
    Or,
    Not,
    LParen,
    RParen,
}

/// Abstract Syntax Tree (AST) node for structured search queries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryNode {
    /// An analyzed single vocabulary term
    Term(String),
    /// Consecutive phrase search (e.g. `"search engine"`)
    Phrase(String),
    /// Prefix wildcard search (e.g. `rust*`)
    Prefix(String),
    /// Conjunction: all subqueries must match (Intersection / AND)
    And(Vec<QueryNode>),
    /// Disjunction: at least one subquery must match (Union / OR)
    Or(Vec<QueryNode>),
    /// Negation: document must NOT match (NOT / Difference)
    Not(Box<QueryNode>),
}

/// Errors encountered during query parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryParseError {
    EmptyQuery,
    UnclosedQuote(usize),
    UnmatchedLParen,
    UnexpectedToken(String),
    UnexpectedEof,
}

impl std::fmt::Display for QueryParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyQuery => write!(f, "Query string is empty"),
            Self::UnclosedQuote(pos) => write!(f, "Unclosed quote starting at byte offset {}", pos),
            Self::UnmatchedLParen => write!(f, "Unmatched opening parenthesis '('"),
            Self::UnexpectedToken(tok) => write!(f, "Unexpected token '{}'", tok),
            Self::UnexpectedEof => write!(f, "Unexpected end of query"),
        }
    }
}

impl std::error::Error for QueryParseError {}

/// Tokenizes a query string into a stream of `QueryToken`s.
pub fn tokenize_query(input: &str) -> Result<Vec<QueryToken>, QueryParseError> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        let ch = chars[i];

        // Skip whitespace
        if ch.is_whitespace() {
            i += 1;
            continue;
        }

        // Parentheses
        if ch == '(' {
            tokens.push(QueryToken::LParen);
            i += 1;
            continue;
        }
        if ch == ')' {
            tokens.push(QueryToken::RParen);
            i += 1;
            continue;
        }

        // Quoted phrase
        if ch == '"' {
            let start = i;
            i += 1;
            let mut phrase_chars = Vec::new();
            let mut closed = false;
            while i < len {
                if chars[i] == '"' {
                    closed = true;
                    i += 1;
                    break;
                }
                phrase_chars.push(chars[i]);
                i += 1;
            }
            if !closed {
                return Err(QueryParseError::UnclosedQuote(start));
            }
            let phrase: String = phrase_chars.into_iter().collect();
            tokens.push(QueryToken::Phrase(phrase.trim().to_string()));
            continue;
        }

        // Operators: &&, ||, !
        if ch == '&' && i + 1 < len && chars[i + 1] == '&' {
            tokens.push(QueryToken::And);
            i += 2;
            continue;
        }
        if ch == '|' && i + 1 < len && chars[i + 1] == '|' {
            tokens.push(QueryToken::Or);
            i += 2;
            continue;
        }
        if ch == '!' || ch == '-' {
            tokens.push(QueryToken::Not);
            i += 1;
            continue;
        }

        // Word or keyword (AND, OR, NOT, or terms with optional wildcard *)
        let mut word_chars = Vec::new();
        while i < len {
            let c = chars[i];
            if c.is_whitespace() || c == '(' || c == ')' || c == '"' || c == '!' {
                break;
            }
            if c == '&' && i + 1 < len && chars[i + 1] == '&' {
                break;
            }
            if c == '|' && i + 1 < len && chars[i + 1] == '|' {
                break;
            }
            word_chars.push(c);
            i += 1;
        }

        let word: String = word_chars.into_iter().collect();
        let upper = word.to_uppercase();

        if upper == "AND" {
            tokens.push(QueryToken::And);
        } else if upper == "OR" {
            tokens.push(QueryToken::Or);
        } else if upper == "NOT" {
            tokens.push(QueryToken::Not);
        } else if let Some(prefix) = word.strip_suffix('*') {
            tokens.push(QueryToken::Prefix(prefix.to_string()));
        } else {
            tokens.push(QueryToken::Term(word));
        }
    }

    Ok(tokens)
}

/// Recursive Descent Parser for structured boolean search queries.
pub struct QueryParser<'a> {
    tokens: &'a [QueryToken],
    pos: usize,
}

impl<'a> QueryParser<'a> {
    pub fn new(tokens: &'a [QueryToken]) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> Option<&QueryToken> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) -> Option<&QueryToken> {
        if self.pos < self.tokens.len() {
            let tok = &self.tokens[self.pos];
            self.pos += 1;
            Some(tok)
        } else {
            None
        }
    }

    /// Parses the entire token stream into a QueryNode AST.
    pub fn parse(&mut self) -> Result<QueryNode, QueryParseError> {
        if self.tokens.is_empty() {
            return Err(QueryParseError::EmptyQuery);
        }

        let expr = self.parse_or()?;

        if let Some(tok) = self.peek() {
            return Err(QueryParseError::UnexpectedToken(format!("{:?}", tok)));
        }

        Ok(expr)
    }

    // OrExpr := AndExpr ("OR" AndExpr)*
    fn parse_or(&mut self) -> Result<QueryNode, QueryParseError> {
        let mut left = self.parse_and()?;

        while let Some(QueryToken::Or) = self.peek() {
            self.advance(); // consume OR
            let right = self.parse_and()?;

            left = match left {
                QueryNode::Or(mut list) => {
                    list.push(right);
                    QueryNode::Or(list)
                }
                _ => QueryNode::Or(vec![left, right]),
            };
        }

        Ok(left)
    }

    // AndExpr := UnaryExpr (("AND"? | "&&") UnaryExpr)*
    // Supports both explicit AND and implicit whitespace AND!
    fn parse_and(&mut self) -> Result<QueryNode, QueryParseError> {
        let mut left = self.parse_unary()?;

        loop {
            let is_explicit_and = matches!(self.peek(), Some(QueryToken::And));
            let is_implicit_and = matches!(
                self.peek(),
                Some(
                    QueryToken::Term(_)
                        | QueryToken::Phrase(_)
                        | QueryToken::Prefix(_)
                        | QueryToken::Not
                        | QueryToken::LParen
                )
            );

            if is_explicit_and {
                self.advance(); // consume explicit AND
                let right = self.parse_unary()?;
                left = self.combine_and(left, right);
            } else if is_implicit_and {
                let right = self.parse_unary()?;
                left = self.combine_and(left, right);
            } else {
                break;
            }
        }

        Ok(left)
    }

    fn combine_and(&self, left: QueryNode, right: QueryNode) -> QueryNode {
        match left {
            QueryNode::And(mut list) => {
                list.push(right);
                QueryNode::And(list)
            }
            _ => QueryNode::And(vec![left, right]),
        }
    }

    // UnaryExpr := ("NOT" | "!") UnaryExpr | PrimaryExpr
    fn parse_unary(&mut self) -> Result<QueryNode, QueryParseError> {
        if let Some(QueryToken::Not) = self.peek() {
            self.advance(); // consume NOT
            let operand = self.parse_unary()?;
            return Ok(QueryNode::Not(Box::new(operand)));
        }

        self.parse_primary()
    }

    // PrimaryExpr := LParen OrExpr RParen | Term | Phrase | Prefix
    fn parse_primary(&mut self) -> Result<QueryNode, QueryParseError> {
        match self.advance() {
            Some(QueryToken::LParen) => {
                let expr = self.parse_or()?;
                match self.advance() {
                    Some(QueryToken::RParen) => Ok(expr),
                    _ => Err(QueryParseError::UnmatchedLParen),
                }
            }
            Some(QueryToken::Term(t)) => Ok(QueryNode::Term(t.clone())),
            Some(QueryToken::Phrase(p)) => Ok(QueryNode::Phrase(p.clone())),
            Some(QueryToken::Prefix(p)) => Ok(QueryNode::Prefix(p.clone())),
            Some(other) => Err(QueryParseError::UnexpectedToken(format!("{:?}", other))),
            None => Err(QueryParseError::UnexpectedEof),
        }
    }
}

/// Convenience function to compile a query string directly into a `QueryNode` AST.
pub fn parse_query(query_str: &str) -> Result<QueryNode, QueryParseError> {
    let tokens = tokenize_query(query_str)?;
    let mut parser = QueryParser::new(&tokens);
    parser.parse()
}

/// Executes a compiled `QueryNode` AST against an `InvertedIndex`.
///
/// Uses $O(L_1 + L_2)$ two-pointer algorithms for Intersection, Union, and Set Difference.
pub fn execute_query(node: &QueryNode, index: &InvertedIndex) -> Vec<Posting> {
    match node {
        QueryNode::Term(term) => index.search_term(term),
        QueryNode::Phrase(phrase) => {
            let doc_ids = index.search_phrase(phrase);
            doc_ids
                .into_iter()
                .map(|doc_id| Posting {
                    doc_id,
                    term_frequency: 1,
                    positions: vec![],
                })
                .collect()
        }
        QueryNode::Prefix(prefix) => index.search_prefix(prefix),
        QueryNode::Or(children) => {
            let mut accumulated: Vec<Posting> = Vec::new();
            for child in children {
                let res = execute_query(child, index);
                accumulated = InvertedIndex::union(&accumulated, &res);
            }
            accumulated
        }
        QueryNode::And(children) => {
            if children.is_empty() {
                return Vec::new();
            }

            // Separate positive and negative clauses
            let mut positive_nodes = Vec::new();
            let mut negative_nodes = Vec::new();

            for child in children {
                if let QueryNode::Not(inner) = child {
                    negative_nodes.push(inner.as_ref());
                } else {
                    positive_nodes.push(child);
                }
            }

            // Evaluate positive clauses
            let mut positives = if !positive_nodes.is_empty() {
                let mut accumulated = execute_query(positive_nodes[0], index);
                for &next in &positive_nodes[1..] {
                    if accumulated.is_empty() {
                        return Vec::new();
                    }
                    let res = execute_query(next, index);
                    accumulated = InvertedIndex::intersect(&accumulated, &res);
                }
                accumulated
            } else {
                // If query is purely negative (e.g. `NOT legacy`), universe is all docs
                universal_postings(index)
            };

            // Evaluate and subtract negative clauses: Positives \ Negatives
            if !negative_nodes.is_empty() {
                let mut negatives: Vec<Posting> = Vec::new();
                for neg in &negative_nodes {
                    let res = execute_query(neg, index);
                    negatives = InvertedIndex::union(&negatives, &res);
                }
                positives = InvertedIndex::difference(&positives, &negatives);
            }

            positives
        }
        QueryNode::Not(inner) => {
            let all = universal_postings(index);
            let excluded = execute_query(inner, index);
            InvertedIndex::difference(&all, &excluded)
        }
    }
}

fn universal_postings(index: &InvertedIndex) -> Vec<Posting> {
    index
        .all_doc_ids()
        .into_iter()
        .map(|doc_id| Posting {
            doc_id,
            term_frequency: 1,
            positions: vec![],
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DocId;

    fn create_test_index() -> InvertedIndex {
        let mut index = InvertedIndex::new();
        // Doc 0
        index.add_document(0, "The quick brown fox jumps over the lazy dog");
        // Doc 1
        index.add_document(1, "Rust delivers bare-metal performance and memory safety");
        // Doc 2
        index.add_document(2, "Python is an interpreted programming language");
        // Doc 3
        index.add_document(
            3,
            "Modern web search engines use inverted index data structures",
        );
        // Doc 4
        index.add_document(4, "Web crawler systematically explores web pages and links");
        index
    }

    #[test]
    fn test_lexer_tokens() {
        let tokens =
            tokenize_query("(rust OR python) AND \"memory safety\" AND NOT legacy").unwrap();
        assert_eq!(
            tokens,
            vec![
                QueryToken::LParen,
                QueryToken::Term("rust".to_string()),
                QueryToken::Or,
                QueryToken::Term("python".to_string()),
                QueryToken::RParen,
                QueryToken::And,
                QueryToken::Phrase("memory safety".to_string()),
                QueryToken::And,
                QueryToken::Not,
                QueryToken::Term("legacy".to_string()),
            ]
        );
    }

    #[test]
    fn test_parser_precedence_and_grouping() {
        let ast = parse_query("(rust OR python) AND web").unwrap();
        assert_eq!(
            ast,
            QueryNode::And(vec![
                QueryNode::Or(vec![
                    QueryNode::Term("rust".to_string()),
                    QueryNode::Term("python".to_string())
                ]),
                QueryNode::Term("web".to_string())
            ])
        );
    }

    #[test]
    fn test_implicit_and_parsing() {
        // Space between terms acts as implicit AND
        let ast = parse_query("rust safety").unwrap();
        assert_eq!(
            ast,
            QueryNode::And(vec![
                QueryNode::Term("rust".to_string()),
                QueryNode::Term("safety".to_string())
            ])
        );
    }

    #[test]
    fn test_execute_and_or_not_queries() {
        let index = create_test_index();

        // 1. Single term
        let q = parse_query("rust").unwrap();
        let results = execute_query(&q, &index);
        let docs: Vec<DocId> = results.iter().map(|p| p.doc_id).collect();
        assert_eq!(docs, vec![1]);

        // 2. OR query: rust OR python -> Doc 1, 2
        let q = parse_query("rust OR python").unwrap();
        let results = execute_query(&q, &index);
        let docs: Vec<DocId> = results.iter().map(|p| p.doc_id).collect();
        assert_eq!(docs, vec![1, 2]);

        // 3. AND query: web AND crawler -> Doc 4
        let q = parse_query("web AND crawler").unwrap();
        let results = execute_query(&q, &index);
        let docs: Vec<DocId> = results.iter().map(|p| p.doc_id).collect();
        assert_eq!(docs, vec![4]);

        // 4. AND NOT query: web AND NOT crawler -> Doc 3
        let q = parse_query("web AND NOT crawler").unwrap();
        let results = execute_query(&q, &index);
        let docs: Vec<DocId> = results.iter().map(|p| p.doc_id).collect();
        assert_eq!(docs, vec![3]);

        // 5. Exact phrase search
        let q = parse_query("\"inverted index\"").unwrap();
        let results = execute_query(&q, &index);
        let docs: Vec<DocId> = results.iter().map(|p| p.doc_id).collect();
        assert_eq!(docs, vec![3]);

        // 6. Prefix wildcard search: crawl* -> Doc 4
        let q = parse_query("crawl*").unwrap();
        let results = execute_query(&q, &index);
        let docs: Vec<DocId> = results.iter().map(|p| p.doc_id).collect();
        assert_eq!(docs, vec![4]);

        // 7. Compound nested query: (rust OR python) AND NOT interpreted -> Doc 1
        let q = parse_query("(rust OR python) AND NOT interpreted").unwrap();
        let results = execute_query(&q, &index);
        let docs: Vec<DocId> = results.iter().map(|p| p.doc_id).collect();
        assert_eq!(docs, vec![1]);
    }
}
