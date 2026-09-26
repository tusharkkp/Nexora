use std::time::Duration;

/// The type of access rule: Allow or Disallow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuleType {
    Allow,
    Disallow,
}

/// A specific path matching rule with its pattern.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rule {
    pub rule_type: RuleType,
    pub pattern: String,
}

/// A group of rules associated with one or more User-Agents.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AgentGroup {
    /// User-agent names this group applies to (lowercased)
    pub agents: Vec<String>,
    /// Sequential list of Allow and Disallow rules
    pub rules: Vec<Rule>,
    /// Optional crawl delay requested by the webmaster
    pub crawl_delay: Option<Duration>,
}

/// Parsed `robots.txt` document complying with RFC 9309.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RobotsTxt {
    /// Rule groups defined in the document
    pub groups: Vec<AgentGroup>,
    /// Sitemaps referenced in the document
    pub sitemaps: Vec<String>,
}

impl RobotsTxt {
    /// Creates an empty `robots.txt` granting unrestricted access.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Parses a raw `robots.txt` file content according to RFC 9309.
    pub fn parse(content: &str) -> Self {
        let mut groups: Vec<AgentGroup> = Vec::new();
        let mut sitemaps = Vec::new();

        let mut current_agents: Vec<String> = Vec::new();
        let mut current_rules: Vec<Rule> = Vec::new();
        let mut current_delay: Option<Duration> = None;
        let mut in_rule_block = false;

        for line in content.lines() {
            // Strip comments
            let clean = match line.find('#') {
                Some(idx) => &line[..idx],
                None => line,
            }
            .trim();

            if clean.is_empty() {
                continue;
            }

            let Some((key, value)) = clean.split_once(':') else {
                continue;
            };

            let directive = key.trim().to_lowercase();
            let val = value.trim();

            match directive.as_str() {
                "user-agent" => {
                    // If we were already building rules for a previous set of agents,
                    // commit that group first before starting the new agent list.
                    if in_rule_block && !current_agents.is_empty() {
                        groups.push(AgentGroup {
                            agents: std::mem::take(&mut current_agents),
                            rules: std::mem::take(&mut current_rules),
                            crawl_delay: current_delay.take(),
                        });
                        in_rule_block = false;
                    }

                    if !val.is_empty() {
                        current_agents.push(val.to_lowercase());
                    }
                }
                "disallow" => {
                    in_rule_block = true;
                    if val.is_empty() {
                        // "Disallow:" with an empty value means nothing is disallowed (allow all)
                        current_rules.push(Rule {
                            rule_type: RuleType::Allow,
                            pattern: "/".to_string(),
                        });
                    } else {
                        current_rules.push(Rule {
                            rule_type: RuleType::Disallow,
                            pattern: val.to_string(),
                        });
                    }
                }
                "allow" => {
                    in_rule_block = true;
                    if !val.is_empty() {
                        current_rules.push(Rule {
                            rule_type: RuleType::Allow,
                            pattern: val.to_string(),
                        });
                    }
                }
                "crawl-delay" => {
                    in_rule_block = true;
                    if let Ok(secs) = val.parse::<f64>() {
                        if secs >= 0.0 {
                            current_delay = Some(Duration::from_secs_f64(secs));
                        }
                    }
                }
                "sitemap" => {
                    if !val.is_empty() {
                        sitemaps.push(val.to_string());
                    }
                }
                _ => {
                    // Ignore unrecognized extension directives
                }
            }
        }

        // Commit final group
        if !current_agents.is_empty() {
            groups.push(AgentGroup {
                agents: current_agents,
                rules: current_rules,
                crawl_delay: current_delay,
            });
        }

        Self { groups, sitemaps }
    }

    /// Determines whether the given `path` is allowed for the specified `user_agent`.
    ///
    /// Evaluates rules according to RFC 9309:
    /// 1. Uses the most specific matching User-Agent block (or falls back to `*`).
    /// 2. If multiple rules match, the longest matching pattern wins.
    /// 3. If matching lengths are equal, `Allow` takes precedence over `Disallow`.
    /// 4. If no rules match, access is allowed by default.
    pub fn is_allowed(&self, user_agent: &str, path: &str) -> bool {
        let normalized_path = if path.is_empty() { "/" } else { path };
        let ua_lower = user_agent.to_lowercase();

        // 1. Find matching agent group (specific agent has priority over '*')
        let selected_group = self
            .groups
            .iter()
            .find(|g| g.agents.iter().any(|a| ua_lower.contains(a) || a.contains(&ua_lower)))
            .or_else(|| {
                self.groups
                    .iter()
                    .find(|g| g.agents.iter().any(|a| a == "*"))
            });

        let Some(group) = selected_group else {
            return true; // No group applies -> allow by default
        };

        // 2. Evaluate all rules in group and find the best match
        let mut best_match: Option<(usize, RuleType)> = None;

        for rule in &group.rules {
            if pattern_matches(&rule.pattern, normalized_path) {
                let match_len = rule.pattern.len();

                match best_match {
                    None => {
                        best_match = Some((match_len, rule.rule_type));
                    }
                    Some((best_len, best_type)) => {
                        if match_len > best_len {
                            // Longest matching rule wins
                            best_match = Some((match_len, rule.rule_type));
                        } else if match_len == best_len && rule.rule_type == RuleType::Allow {
                            // Equal length tie-breaker: Allow takes precedence
                            best_match = Some((match_len, RuleType::Allow));
                        } else {
                            // Keep previous best
                            best_match = Some((best_len, best_type));
                        }
                    }
                }
            }
        }

        // If a rule matched, return based on Allow vs Disallow
        match best_match {
            Some((_, RuleType::Allow)) => true,
            Some((_, RuleType::Disallow)) => false,
            None => true, // No rule matched -> allow by default
        }
    }

    /// Returns the crawl delay specified for the given user-agent, if any.
    pub fn crawl_delay(&self, user_agent: &str) -> Option<Duration> {
        let ua_lower = user_agent.to_lowercase();

        let specific = self
            .groups
            .iter()
            .find(|g| g.agents.iter().any(|a| ua_lower.contains(a) || a.contains(&ua_lower)))
            .and_then(|g| g.crawl_delay);

        if specific.is_some() {
            return specific;
        }

        self.groups
            .iter()
            .find(|g| g.agents.iter().any(|a| a == "*"))
            .and_then(|g| g.crawl_delay)
    }
}

/// Matches a `robots.txt` path pattern against a URL path according to RFC 9309.
///
/// Syntax rules:
/// - Patterns are case-sensitive path prefixes by default (e.g. `/foo` matches `/foo`, `/foobar`, `/foo/bar`).
/// - `*` matches zero or more characters.
/// - `$` at the end of the pattern anchors the match to the end of the path.
pub fn pattern_matches(pattern: &str, path: &str) -> bool {
    let pat_chars: Vec<char> = pattern.chars().collect();
    let path_chars: Vec<char> = path.chars().collect();

    if pat_chars.is_empty() {
        return false;
    }

    let has_end_anchor = *pat_chars.last().unwrap() == '$';

    let effective_pattern: Vec<char> = if has_end_anchor {
        // Strip the trailing '$' and enforce exact end match
        pat_chars[..pat_chars.len() - 1].to_vec()
    } else {
        // Without '$', the pattern has an implicit '*' at the end (prefix match)
        let mut p = pat_chars.clone();
        if p.last() != Some(&'*') {
            p.push('*');
        }
        p
    };

    wildcard_match(&effective_pattern, &path_chars)
}

/// Linear-time wildcard matching algorithm supporting `*`.
fn wildcard_match(pattern: &[char], text: &[char]) -> bool {
    let mut p_idx = 0;
    let mut t_idx = 0;
    let mut star_idx: Option<usize> = None;
    let mut match_idx = 0;

    while t_idx < text.len() {
        if p_idx < pattern.len() && pattern[p_idx] == text[t_idx] {
            p_idx += 1;
            t_idx += 1;
        } else if p_idx < pattern.len() && pattern[p_idx] == '*' {
            star_idx = Some(p_idx);
            match_idx = t_idx;
            p_idx += 1;
        } else if let Some(star) = star_idx {
            p_idx = star + 1;
            match_idx += 1;
            t_idx = match_idx;
        } else {
            return false;
        }
    }

    while p_idx < pattern.len() && pattern[p_idx] == '*' {
        p_idx += 1;
    }

    p_idx == pattern.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_prefix_and_wildcard_matching() {
        // Basic prefix
        assert!(pattern_matches("/fish", "/fish"));
        assert!(pattern_matches("/fish", "/fish.html"));
        assert!(pattern_matches("/fish", "/fish/salmon.html"));
        assert!(!pattern_matches("/fish", "/Fisherman"));

        // Wildcard '*'
        assert!(pattern_matches("/fish*.php", "/fish.php"));
        assert!(pattern_matches("/fish*.php", "/fish_heads/catfood.php"));
        assert!(pattern_matches("/fish*.php", "/fish_heads/catfood.php?more"));
        assert!(!pattern_matches("/fish*.php", "/fish.html"));

        // End anchor '$'
        assert!(pattern_matches("/fish$", "/fish"));
        assert!(!pattern_matches("/fish$", "/fish.html"));
        assert!(!pattern_matches("/fish$", "/fish/"));

        // Wildcard with end anchor
        assert!(pattern_matches("/fish*.php$", "/fish.php"));
        assert!(!pattern_matches("/fish*.php$", "/fish.php?id=1"));
    }

    #[test]
    fn test_robots_txt_parse_and_specificity() {
        let content = r#"
        # Global rules
        User-agent: *
        Disallow: /admin
        Disallow: /private/
        Allow: /private/public-report.html
        Crawl-delay: 2.5
        Sitemap: https://example.com/sitemap.xml

        # Specific rules for NexoraBot
        User-agent: NexoraBot
        Disallow: /crawler-trap/
        Allow: /admin/read-only
        "#;

        let robots = RobotsTxt::parse(content);
        assert_eq!(robots.sitemaps, vec!["https://example.com/sitemap.xml"]);

        // Wildcard crawler checking rules
        assert!(!robots.is_allowed("GenericBot", "/admin"));
        assert!(!robots.is_allowed("GenericBot", "/admin/dashboard"));
        assert!(!robots.is_allowed("GenericBot", "/private/secret.pdf"));
        // Specificity: /private/public-report.html is longer than /private/ -> Allow wins
        assert!(robots.is_allowed("GenericBot", "/private/public-report.html"));
        assert!(robots.is_allowed("GenericBot", "/articles/index.html"));

        // Crawl delay
        assert_eq!(robots.crawl_delay("GenericBot"), Some(Duration::from_millis(2500)));

        // NexoraBot specific rules
        assert!(!robots.is_allowed("NexoraBot", "/crawler-trap/loop"));
        // NexoraBot has explicit Allow for /admin/read-only
        assert!(robots.is_allowed("NexoraBot", "/admin/read-only"));
        // NexoraBot has no Disallow for /admin in its block -> allowed
        assert!(robots.is_allowed("NexoraBot", "/admin/dashboard"));
    }

    #[test]
    fn test_empty_disallow_allows_all() {
        let content = r#"
        User-agent: *
        Disallow:
        "#;

        let robots = RobotsTxt::parse(content);
        assert!(robots.is_allowed("NexoraBot", "/anything"));
    }
}
