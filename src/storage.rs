use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::analyzer::Analyzer;
use crate::index::{InvertedIndex, Posting};

/// Magic signature bytes at the start of every Nexora index file.
pub const MAGIC_BYTES: &[u8; 8] = b"NEXORA01";

/// Errors that can occur during index serialization or deserialization.
#[derive(Debug)]
pub enum StorageError {
    /// Underlying I/O error (disk read/write, permissions, EOF)
    Io(io::Error),
    /// File does not start with expected magic bytes "NEXORA01"
    InvalidMagicBytes([u8; 8]),
    /// Corrupt UTF-8 term string encountered
    InvalidUtf8(std::string::FromUtf8Error),
    /// File ended unexpectedly during parsing
    UnexpectedEof,
}

impl std::fmt::Display for StorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StorageError::Io(err) => write!(f, "I/O error: {}", err),
            StorageError::InvalidMagicBytes(bytes) => {
                write!(f, "Invalid magic bytes: {:?}", bytes)
            }
            StorageError::InvalidUtf8(err) => write!(f, "Corrupted UTF-8 term: {}", err),
            StorageError::UnexpectedEof => write!(f, "Unexpected end of file while reading index"),
        }
    }
}

impl std::error::Error for StorageError {}

impl From<io::Error> for StorageError {
    fn from(err: io::Error) -> Self {
        StorageError::Io(err)
    }
}

impl From<std::string::FromUtf8Error> for StorageError {
    fn from(err: std::string::FromUtf8Error) -> Self {
        StorageError::InvalidUtf8(err)
    }
}

/// Serializes an `InvertedIndex` into a binary stream adhering to the `.nex` v1 format.
pub fn save_index<W: Write>(index: &InvertedIndex, writer: &mut W) -> Result<(), StorageError> {
    // 1. Magic Header
    writer.write_all(MAGIC_BYTES)?;

    // 2. Metadata: total_documents and doc_lengths
    let total_docs = index.total_documents() as u32;
    writer.write_all(&total_docs.to_le_bytes())?;

    let doc_lengths = index.doc_lengths();
    let doc_lengths_count = doc_lengths.len() as u32;
    writer.write_all(&doc_lengths_count.to_le_bytes())?;

    for (&doc_id, &len) in doc_lengths {
        writer.write_all(&doc_id.to_le_bytes())?;
        writer.write_all(&len.to_le_bytes())?;
    }

    // 3. Vocabulary & Postings
    let dictionary = index.dictionary();
    let vocab_size = dictionary.len() as u32;
    writer.write_all(&vocab_size.to_le_bytes())?;

    for (term, postings) in dictionary {
        let term_bytes = term.as_bytes();
        let term_len = term_bytes.len() as u16;

        // Write term string (length-prefixed)
        writer.write_all(&term_len.to_le_bytes())?;
        writer.write_all(term_bytes)?;

        // Write postings list for this term
        let postings_count = postings.len() as u32;
        writer.write_all(&postings_count.to_le_bytes())?;

        for posting in postings {
            writer.write_all(&posting.doc_id.to_le_bytes())?;
            writer.write_all(&posting.term_frequency.to_le_bytes())?;

            let positions_count = posting.positions.len() as u32;
            writer.write_all(&positions_count.to_le_bytes())?;

            for &pos in &posting.positions {
                writer.write_all(&pos.to_le_bytes())?;
            }
        }
    }

    writer.flush()?;
    Ok(())
}

/// Deserializes a binary stream adhering to the `.nex` v1 format into an `InvertedIndex`.
pub fn load_index<R: Read>(reader: &mut R) -> Result<InvertedIndex, StorageError> {
    // 1. Verify Magic Header
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != MAGIC_BYTES {
        return Err(StorageError::InvalidMagicBytes(magic));
    }

    // 2. Metadata: total_documents and doc_lengths
    let total_documents = read_u32(reader)? as usize;
    let doc_lengths_count = read_u32(reader)? as usize;

    let mut doc_lengths = HashMap::with_capacity(doc_lengths_count);
    for _ in 0..doc_lengths_count {
        let doc_id = read_u32(reader)?;
        let length = read_u32(reader)?;
        doc_lengths.insert(doc_id, length);
    }

    // 3. Vocabulary & Postings
    let vocab_size = read_u32(reader)? as usize;
    let mut dictionary = HashMap::with_capacity(vocab_size);

    for _ in 0..vocab_size {
        // Read term string
        let term_len = read_u16(reader)? as usize;
        let mut term_bytes = vec![0u8; term_len];
        reader.read_exact(&mut term_bytes)?;
        let term = String::from_utf8(term_bytes)?;

        // Read postings list
        let postings_count = read_u32(reader)? as usize;
        let mut postings = Vec::with_capacity(postings_count);

        for _ in 0..postings_count {
            let doc_id = read_u32(reader)?;
            let term_frequency = read_u32(reader)?;

            let positions_count = read_u32(reader)? as usize;
            let mut positions = Vec::with_capacity(positions_count);

            for _ in 0..positions_count {
                let pos = read_u32(reader)?;
                positions.push(pos);
            }

            postings.push(Posting {
                doc_id,
                term_frequency,
                positions,
            });
        }

        dictionary.insert(term, postings);
    }

    Ok(InvertedIndex::from_raw_parts(
        dictionary,
        doc_lengths,
        total_documents,
        Analyzer::new(),
    ))
}

/// Helper: Saves an InvertedIndex to a disk file path.
pub fn save_to_file<P: AsRef<Path>>(index: &InvertedIndex, path: P) -> Result<(), StorageError> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    save_index(index, &mut writer)
}

/// Helper: Loads an InvertedIndex from a disk file path.
pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<InvertedIndex, StorageError> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    load_index(&mut reader)
}

// -----------------------------------------------------------------------------
// Low-Level Byte Reading Helpers
// -----------------------------------------------------------------------------

fn read_u32<R: Read>(reader: &mut R) -> Result<u32, io::Error> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u16<R: Read>(reader: &mut R) -> Result<u16, io::Error> {
    let mut buf = [0u8; 2];
    reader.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn build_sample_index() -> InvertedIndex {
        let mut index = InvertedIndex::new();
        index.add_document(0, "The quick brown fox jumps over the lazy dog");
        index.add_document(1, "Quick brown foxes are fast animals");
        index.add_document(2, "Lazy dogs sleep all day in the warm sun");
        index
    }

    #[test]
    fn test_memory_roundtrip() {
        let original_index = build_sample_index();

        // 1. Serialize index to memory buffer
        let mut buffer = Vec::new();
        save_index(&original_index, &mut buffer).expect("save_index should succeed");

        assert!(!buffer.is_empty());
        assert_eq!(&buffer[0..8], MAGIC_BYTES);

        // 2. Deserialize from buffer
        let mut cursor = Cursor::new(buffer);
        let loaded_index = load_index(&mut cursor).expect("load_index should succeed");

        // 3. Verify metadata equality
        assert_eq!(
            loaded_index.total_documents(),
            original_index.total_documents()
        );
        assert_eq!(
            loaded_index.vocabulary_size(),
            original_index.vocabulary_size()
        );
        assert_eq!(
            loaded_index.average_doc_length(),
            original_index.average_doc_length()
        );

        // 4. Verify search query parity
        // Single term
        assert_eq!(
            original_index.search_term("fox"),
            loaded_index.search_term("fox")
        );

        // Boolean AND
        assert_eq!(
            original_index.search_and(&["quick", "fox"]),
            loaded_index.search_and(&["quick", "fox"])
        );

        // Exact phrase
        assert_eq!(
            original_index.search_phrase("quick brown"),
            loaded_index.search_phrase("quick brown")
        );

        // BM25 ranking
        let orig_bm25 = original_index.search_bm25_default("quick fox");
        let load_bm25 = loaded_index.search_bm25_default("quick fox");
        assert_eq!(orig_bm25, load_bm25);
    }

    #[test]
    fn test_invalid_magic_bytes_rejected() {
        let corrupt_data = b"BADMAGIC\x01\x00\x00\x00";
        let mut cursor = Cursor::new(corrupt_data);

        match load_index(&mut cursor) {
            Err(StorageError::InvalidMagicBytes(bytes)) => {
                assert_eq!(&bytes, b"BADMAGIC");
            }
            other => panic!("Expected InvalidMagicBytes, got: {:?}", other),
        }
    }

    #[test]
    fn test_file_persistence_roundtrip() {
        let original_index = build_sample_index();
        let temp_file = std::env::temp_dir().join("nexora_test_index.nex");

        // Save to temporary file
        save_to_file(&original_index, &temp_file).expect("save_to_file should succeed");

        // Load back from file
        let loaded_index = load_from_file(&temp_file).expect("load_from_file should succeed");

        assert_eq!(
            loaded_index.total_documents(),
            original_index.total_documents()
        );
        assert_eq!(
            original_index.search_phrase("lazy dog"),
            loaded_index.search_phrase("lazy dog")
        );

        // Clean up temp file
        let _ = std::fs::remove_file(temp_file);
    }
}
