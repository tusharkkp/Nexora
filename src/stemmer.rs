/// Implementation of Martin Porter's 1980 Stemming Algorithm.
///
/// Reduces English words to their morphological roots (stems)
/// using rule-based suffix transformations without requiring a dictionary.
///
/// Reference: Porter, M. "An algorithm for suffix stripping." Program 14.3 (1980): 130-137.

/// Stems an input word using the 5-phase Porter Stemming Algorithm.
///
/// Non-alphabetic tokens (e.g., numbers, URLs, IP addresses) or words
/// with 2 or fewer characters are returned unmodified.
pub fn stem(word: &str) -> String {
    // Guard 1: Words with 2 or fewer characters cannot be safely stemmed.
    if word.chars().count() <= 2 {
        return word.to_string();
    }

    // Guard 2: Skip tokens containing non-alphabetic characters (e.g. URLs, numbers, IPs)
    if !word.chars().all(|c| c.is_alphabetic() || c == '\'' || c == '’') {
        return word.to_string();
    }

    // Work on lowercase characters
    let mut chars: Vec<char> = word.to_lowercase().chars().collect();

    // Execute the 5 sequential steps of the Porter Stemmer
    step_1a(&mut chars);
    step_1b(&mut chars);
    step_1c(&mut chars);
    step_2(&mut chars);
    step_3(&mut chars);
    step_4(&mut chars);
    step_5(&mut chars);

    chars.into_iter().collect()
}

// -----------------------------------------------------------------------------
// Core Linguistic Primitives: Consonants, Vowels, and Measure (m)
// -----------------------------------------------------------------------------

/// Checks if character at index `i` is a consonant.
/// In Porter's algorithm:
/// - 'a', 'e', 'i', 'o', 'u' are always vowels.
/// - 'y' is a consonant if it is the first letter, or preceded by a vowel.
/// - 'y' is a vowel if preceded by a consonant.
fn is_consonant(chars: &[char], i: usize) -> bool {
    match chars[i] {
        'a' | 'e' | 'i' | 'o' | 'u' => false,
        'y' => {
            if i == 0 {
                true
            } else {
                !is_consonant(chars, i - 1)
            }
        }
        _ => true,
    }
}

/// Checks if character at index `i` is a vowel.
fn is_vowel(chars: &[char], i: usize) -> bool {
    !is_consonant(chars, i)
}

/// Computes the Porter Measure `m` for a slice of characters.
/// A word has the form: [C](VC)^m[V]
/// `m` represents the number of transitions from a vowel sequence to a consonant sequence.
fn measure(chars: &[char]) -> usize {
    let len = chars.len();
    let mut i = 0;
    let mut m = 0;

    // Skip leading consonants [C]
    while i < len && is_consonant(chars, i) {
        i += 1;
    }

    loop {
        // Skip vowels (V)
        while i < len && is_vowel(chars, i) {
            i += 1;
        }
        if i >= len {
            break;
        }

        // Skip consonants (C)
        while i < len && is_consonant(chars, i) {
            i += 1;
        }
        m += 1;
    }

    m
}

/// Condition: `*v*` - Does the stem contain at least one vowel?
fn contains_vowel(chars: &[char]) -> bool {
    (0..chars.len()).any(|i| is_vowel(chars, i))
}

/// Condition: `*d` - Does the stem end with a double consonant? (e.g. -tt, -ss)
fn ends_with_double_consonant(chars: &[char]) -> bool {
    let len = chars.len();
    if len < 2 {
        return false;
    }
    chars[len - 1] == chars[len - 2] && is_consonant(chars, len - 1)
}

/// Condition: `*o` - Does the stem end with consonant-vowel-consonant (cvc),
/// where the second consonant is not 'w', 'x', or 'y'?
fn ends_with_cvc(chars: &[char]) -> bool {
    let len = chars.len();
    if len < 3 {
        return false;
    }
    let last = chars[len - 1];
    is_consonant(chars, len - 1)
        && is_vowel(chars, len - 2)
        && is_consonant(chars, len - 3)
        && last != 'w'
        && last != 'x'
        && last != 'y'
}

/// Helper: Checks if `chars` ends with `suffix`.
fn ends_with(chars: &[char], suffix: &str) -> bool {
    let suffix_chars: Vec<char> = suffix.chars().collect();
    if chars.len() < suffix_chars.len() {
        return false;
    }
    &chars[chars.len() - suffix_chars.len()..] == suffix_chars.as_slice()
}

/// Helper: Replaces `suffix` with `replacement` if the stem matches condition.
fn replace_suffix(chars: &mut Vec<char>, suffix: &str, replacement: &str, condition: impl FnOnce(&[char]) -> bool) -> bool {
    if ends_with(chars, suffix) {
        let stem_len = chars.len() - suffix.chars().count();
        if condition(&chars[..stem_len]) {
            chars.truncate(stem_len);
            chars.extend(replacement.chars());
            return true;
        }
    }
    false
}

// -----------------------------------------------------------------------------
// Step 1: Plurals and Past Participles
// -----------------------------------------------------------------------------

fn step_1a(chars: &mut Vec<char>) {
    if ends_with(chars, "sses") {
        chars.truncate(chars.len() - 2); // sses -> ss
    } else if ends_with(chars, "ies") {
        chars.truncate(chars.len() - 2); // ies -> i
    } else if ends_with(chars, "ss") {
        // ss -> ss (no change)
    } else if ends_with(chars, "s") {
        chars.pop(); // s -> empty
    }
}

fn step_1b(chars: &mut Vec<char>) {
    let mut extra = false;

    if ends_with(chars, "eed") {
        let stem_len = chars.len() - 3;
        if measure(&chars[..stem_len]) > 0 {
            chars.pop(); // eed -> ee
        }
    } else if ends_with(chars, "ed") {
        let stem_len = chars.len() - 2;
        if contains_vowel(&chars[..stem_len]) {
            chars.truncate(stem_len);
            extra = true;
        }
    } else if ends_with(chars, "ing") {
        let stem_len = chars.len() - 3;
        if contains_vowel(&chars[..stem_len]) {
            chars.truncate(stem_len);
            extra = true;
        }
    }

    if extra {
        if ends_with(chars, "at") || ends_with(chars, "bl") || ends_with(chars, "iz") {
            chars.push('e');
        } else if ends_with_double_consonant(chars) {
            let last = chars[chars.len() - 1];
            if last != 'l' && last != 's' && last != 'z' {
                chars.pop();
            }
        } else if measure(chars) == 1 && ends_with_cvc(chars) {
            chars.push('e');
        }
    }
}

fn step_1c(chars: &mut Vec<char>) {
    if ends_with(chars, "y") {
        let stem_len = chars.len() - 1;
        if contains_vowel(&chars[..stem_len]) {
            chars[stem_len] = 'i';
        }
    }
}

// -----------------------------------------------------------------------------
// Step 2: Derivational / Complex Suffixes (if m > 0)
// -----------------------------------------------------------------------------

fn step_2(chars: &mut Vec<char>) {
    const RULES: &[(&str, &str)] = &[
        ("ational", "ate"),
        ("tional", "tion"),
        ("enci", "ence"),
        ("anci", "ance"),
        ("izer", "ize"),
        ("abli", "able"),
        ("alli", "al"),
        ("entli", "ent"),
        ("eli", "e"),
        ("ousli", "ous"),
        ("ization", "ize"),
        ("ation", "ate"),
        ("ator", "ate"),
        ("alism", "al"),
        ("iveness", "ive"),
        ("fulness", "ful"),
        ("ousness", "ous"),
        ("aliti", "al"),
        ("iviti", "ive"),
        ("biliti", "ble"),
    ];

    for &(suffix, replacement) in RULES {
        if replace_suffix(chars, suffix, replacement, |stem| measure(stem) > 0) {
            break;
        }
    }
}

// -----------------------------------------------------------------------------
// Step 3: More Suffix Replacements (if m > 0)
// -----------------------------------------------------------------------------

fn step_3(chars: &mut Vec<char>) {
    const RULES: &[(&str, &str)] = &[
        ("icate", "ic"),
        ("ative", ""),
        ("alize", "al"),
        ("iciti", "ic"),
        ("ical", "ic"),
        ("ful", ""),
        ("ness", ""),
    ];

    for &(suffix, replacement) in RULES {
        if replace_suffix(chars, suffix, replacement, |stem| measure(stem) > 0) {
            break;
        }
    }
}

// -----------------------------------------------------------------------------
// Step 4: Suffix Stripping (if m > 1)
// -----------------------------------------------------------------------------

fn step_4(chars: &mut Vec<char>) {
    const SUFFIXES: &[&str] = &[
        "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment", "ent", "ou",
        "ism", "ate", "iti", "ous", "ive", "ize",
    ];

    // Special check for -sion / -tion: only stripped if preceded by 's' or 't'
    if ends_with(chars, "ion") {
        let stem_len = chars.len() - 3;
        if stem_len > 0 && (chars[stem_len - 1] == 's' || chars[stem_len - 1] == 't') {
            if measure(&chars[..stem_len]) > 1 {
                chars.truncate(stem_len);
                return;
            }
        }
    }

    for &suffix in SUFFIXES {
        if replace_suffix(chars, suffix, "", |stem| measure(stem) > 1) {
            break;
        }
    }
}

// -----------------------------------------------------------------------------
// Step 5: Tidy-up / Clean-up
// -----------------------------------------------------------------------------

fn step_5(chars: &mut Vec<char>) {
    // Step 5a: Remove trailing 'e' if m > 1, or if m == 1 and not *o (cvc)
    if ends_with(chars, "e") {
        let stem_len = chars.len() - 1;
        let stem = &chars[..stem_len];
        let m = measure(stem);
        if m > 1 || (m == 1 && !ends_with_cvc(stem)) {
            chars.pop();
        }
    }

    // Step 5b: Remove double 'l' if m > 1 (e.g. controll -> control)
    let len = chars.len();
    if len >= 2 && chars[len - 1] == 'l' && chars[len - 2] == 'l' && measure(&chars[..len - 1]) > 1 {
        chars.pop();
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plurals_and_participles() {
        assert_eq!(stem("caresses"), "caress");
        assert_eq!(stem("ponies"), "poni");
        assert_eq!(stem("cats"), "cat");
        assert_eq!(stem("feed"), "feed");
        assert_eq!(stem("agreed"), "agre");
        assert_eq!(stem("plastered"), "plaster");
        assert_eq!(stem("bled"), "bled");
        assert_eq!(stem("motoring"), "motor");
        assert_eq!(stem("sing"), "sing");
    }

    #[test]
    fn test_cleanup_after_ed_and_ing() {
        assert_eq!(stem("conflated"), "conflat");
        assert_eq!(stem("troubled"), "troubl");
        assert_eq!(stem("sized"), "size");

        assert_eq!(stem("hopping"), "hop");
        assert_eq!(stem("tanned"), "tan");
        assert_eq!(stem("falling"), "fall");
        assert_eq!(stem("hissing"), "hiss");
        assert_eq!(stem("fizzing"), "fizz");
        assert_eq!(stem("filing"), "file");
    }

    #[test]
    fn test_step_1c_y_to_i() {
        assert_eq!(stem("happy"), "happi");
        assert_eq!(stem("sky"), "sky");
    }

    #[test]
    fn test_complex_and_derivational_suffixes() {
        assert_eq!(stem("relational"), "relat");
        assert_eq!(stem("conditional"), "condit");
        assert_eq!(stem("rational"), "ration");
        assert_eq!(stem("valency"), "valenc");
        assert_eq!(stem("hesitancy"), "hesit");
        assert_eq!(stem("digitizer"), "digit");
        assert_eq!(stem("conformabli"), "conform");
        assert_eq!(stem("radicalli"), "radic");
        assert_eq!(stem("differentli"), "differ");
        assert_eq!(stem("vileli"), "vile");
        assert_eq!(stem("analogousli"), "analog");
        assert_eq!(stem("vietnamization"), "vietnam");
        assert_eq!(stem("predication"), "predic");
        assert_eq!(stem("operator"), "oper");
        assert_eq!(stem("feudalism"), "feudal");
        assert_eq!(stem("decisiveness"), "decis");
        assert_eq!(stem("hopefulness"), "hope");
        assert_eq!(stem("callousness"), "callous");
        assert_eq!(stem("formaliti"), "formal");
        assert_eq!(stem("sensitiviti"), "sensit");
        assert_eq!(stem("sensibiliti"), "sensibl");
    }

    #[test]
    fn test_generalization_trace() {
        assert_eq!(stem("generalizations"), "gener");
        assert_eq!(stem("general"), "gener");
        assert_eq!(stem("generally"), "gener");
    }

    #[test]
    fn test_short_words_and_non_words() {
        assert_eq!(stem("to"), "to");
        assert_eq!(stem("an"), "an");
        assert_eq!(stem("192.168.1.1"), "192.168.1.1");
        assert_eq!(stem("42"), "42");
        assert_eq!(stem("https://nexora.org"), "https://nexora.org");
    }
}
