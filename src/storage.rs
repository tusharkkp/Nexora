use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::analyzer::Analyzer;
use crate::compression::{compress_sorted_u32, decode_vbyte, decompress_sorted_u32, encode_vbyte};
use crate::index::{InvertedIndex, Posting};

/// Magic signature bytes for uncompressed v1 format.
pub const MAGIC_V1: &[u8; 8] = b"NEXORA01";

/// Magic signature bytes for modern compressed v2 format.
pub const MAGIC_V2: &[u8; 8] = b"NEXORA02";

/// Errors that can occur during index serialization or deserialization.
#[derive(Debug)]
pub enum StorageError {
    /// Underlying I/O error (disk read/write, permissions, EOF)
    Io(io::Error),
    /// File does not start with expected magic bytes
    InvalidMagicBytes([u8; 8]),
    /// Corrupt UTF-8 term string encountered
    InvalidUtf8(std::string::FromUtf8Error),
    /// Corrupt compressed byte stream encountered during decompression
    DecompressionError(&'static str),
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
            StorageError::DecompressionError(msg) => write!(f, "Decompression error: {}", msg),
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

// -----------------------------------------------------------------------------
// Serialization (Writes v2 by default)
// -----------------------------------------------------------------------------

/// Serializes an `InvertedIndex` into a compressed binary stream adhering to the `.nex` v2 format.
pub fn save_index<W: Write>(index: &InvertedIndex, writer: &mut W) -> Result<(), StorageError> {
    save_index_v2(index, writer)
}

/// Serializes an index using the compressed `.nex` v2 format.
pub fn save_index_v2<W: Write>(index: &InvertedIndex, writer: &mut W) -> Result<(), StorageError> {
    // 1. Magic Header v2
    writer.write_all(MAGIC_V2)?;

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

    // 3. Compressed Vocabulary & Postings
    let dictionary = index.dictionary();
    let vocab_size = dictionary.len() as u32;
    writer.write_all(&vocab_size.to_le_bytes())?;

    for (term, postings) in dictionary {
        let term_bytes = term.as_bytes();
        let term_len = term_bytes.len() as u16;

        writer.write_all(&term_len.to_le_bytes())?;
        writer.write_all(term_bytes)?;

        let postings_count = postings.len() as u32;
        writer.write_all(&postings_count.to_le_bytes())?;

        // Extract and compress sorted doc_ids
        let doc_ids: Vec<u32> = postings.iter().map(|p| p.doc_id).collect();
        let compressed_doc_ids = compress_sorted_u32(&doc_ids);

        let doc_ids_len = compressed_doc_ids.len() as u32;
        writer.write_all(&doc_ids_len.to_le_bytes())?;
        writer.write_all(&compressed_doc_ids)?;

        // For each document, compress term frequency and positions
        for posting in postings {
            // Encode term frequency using Variable-Byte
            let mut tf_bytes = Vec::new();
            encode_vbyte(posting.term_frequency, &mut tf_bytes);
            let tf_len = tf_bytes.len() as u8;
            writer.write_all(&[tf_len])?;
            writer.write_all(&tf_bytes)?;

            // Compress sorted positions using Delta + Variable-Byte
            let compressed_positions = compress_sorted_u32(&posting.positions);
            let pos_len = compressed_positions.len() as u32;
            writer.write_all(&pos_len.to_le_bytes())?;
            writer.write_all(&compressed_positions)?;
        }
    }

    writer.flush()?;
    Ok(())
}

/// Serializes an index using the uncompressed `.nex` v1 format (kept for testing & fallback).
pub fn save_index_v1<W: Write>(index: &InvertedIndex, writer: &mut W) -> Result<(), StorageError> {
    writer.write_all(MAGIC_V1)?;

    let total_docs = index.total_documents() as u32;
    writer.write_all(&total_docs.to_le_bytes())?;

    let doc_lengths = index.doc_lengths();
    let doc_lengths_count = doc_lengths.len() as u32;
    writer.write_all(&doc_lengths_count.to_le_bytes())?;

    for (&doc_id, &len) in doc_lengths {
        writer.write_all(&doc_id.to_le_bytes())?;
        writer.write_all(&len.to_le_bytes())?;
    }

    let dictionary = index.dictionary();
    let vocab_size = dictionary.len() as u32;
    writer.write_all(&vocab_size.to_le_bytes())?;

    for (term, postings) in dictionary {
        let term_bytes = term.as_bytes();
        let term_len = term_bytes.len() as u16;

        writer.write_all(&term_len.to_le_bytes())?;
        writer.write_all(term_bytes)?;

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

// -----------------------------------------------------------------------------
// Deserialization (Auto-detects v1 vs v2)
// -----------------------------------------------------------------------------

/// Loads an `InvertedIndex` from a binary stream, automatically detecting whether
/// it was stored in v1 (uncompressed) or v2 (compressed) format.
pub fn load_index<R: Read>(reader: &mut R) -> Result<InvertedIndex, StorageError> {
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;

    if &magic == MAGIC_V2 {
        load_index_v2(reader)
    } else if &magic == MAGIC_V1 {
        load_index_v1(reader)
    } else {
        Err(StorageError::InvalidMagicBytes(magic))
    }
}

/// Deserializer for `.nex` v2 (Compressed).
fn load_index_v2<R: Read>(reader: &mut R) -> Result<InvertedIndex, StorageError> {
    let total_documents = read_u32(reader)? as usize;
    let doc_lengths_count = read_u32(reader)? as usize;

    let mut doc_lengths = HashMap::with_capacity(doc_lengths_count);
    for _ in 0..doc_lengths_count {
        let doc_id = read_u32(reader)?;
        let length = read_u32(reader)?;
        doc_lengths.insert(doc_id, length);
    }

    let vocab_size = read_u32(reader)? as usize;
    let mut dictionary = HashMap::with_capacity(vocab_size);

    for _ in 0..vocab_size {
        let term_len = read_u16(reader)? as usize;
        let mut term_bytes = vec![0u8; term_len];
        reader.read_exact(&mut term_bytes)?;
        let term = String::from_utf8(term_bytes)?;

        let postings_count = read_u32(reader)? as usize;

        // Read and decompress doc_ids
        let doc_ids_len = read_u32(reader)? as usize;
        let mut compressed_doc_ids = vec![0u8; doc_ids_len];
        reader.read_exact(&mut compressed_doc_ids)?;

        let doc_ids = decompress_sorted_u32(&compressed_doc_ids)
            .map_err(StorageError::DecompressionError)?;

        if doc_ids.len() != postings_count {
            return Err(StorageError::DecompressionError(
                "Postings count mismatch during doc_ids decompression",
            ));
        }

        let mut postings = Vec::with_capacity(postings_count);

        for &doc_id in &doc_ids {
            // Read term frequency (Variable-Byte)
            let mut tf_len_buf = [0u8; 1];
            reader.read_exact(&mut tf_len_buf)?;
            let tf_len = tf_len_buf[0] as usize;

            let mut tf_bytes = vec![0u8; tf_len];
            reader.read_exact(&mut tf_bytes)?;

            let (term_frequency, _) =
                decode_vbyte(&tf_bytes).ok_or(StorageError::DecompressionError(
                    "Failed to decode term frequency in v2 index",
                ))?;

            // Read and decompress positions
            let pos_len = read_u32(reader)? as usize;
            let mut compressed_positions = vec![0u8; pos_len];
            reader.read_exact(&mut compressed_positions)?;

            let positions = decompress_sorted_u32(&compressed_positions)
                .map_err(StorageError::DecompressionError)?;

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

/// Deserializer for `.nex` v1 (Uncompressed backward compatibility).
fn load_index_v1<R: Read>(reader: &mut R) -> Result<InvertedIndex, StorageError> {
    let total_documents = read_u32(reader)? as usize;
    let doc_lengths_count = read_u32(reader)? as usize;

    let mut doc_lengths = HashMap::with_capacity(doc_lengths_count);
    for _ in 0..doc_lengths_count {
        let doc_id = read_u32(reader)?;
        let length = read_u32(reader)?;
        doc_lengths.insert(doc_id, length);
    }

    let vocab_size = read_u32(reader)? as usize;
    let mut dictionary = HashMap::with_capacity(vocab_size);

    for _ in 0..vocab_size {
        let term_len = read_u16(reader)? as usize;
        let mut term_bytes = vec![0u8; term_len];
        reader.read_exact(&mut term_bytes)?;
        let term = String::from_utf8(term_bytes)?;

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

// -----------------------------------------------------------------------------
// File I/O Helpers
// -----------------------------------------------------------------------------

/// Saves an InvertedIndex to a disk file path using the v2 compressed format.
pub fn save_to_file<P: AsRef<Path>>(index: &InvertedIndex, path: P) -> Result<(), StorageError> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    save_index_v2(index, &mut writer)
}

/// Loads an InvertedIndex from a disk file path (auto-detecting v1 or v2).
pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<InvertedIndex, StorageError> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    load_index(&mut reader)
}

// -----------------------------------------------------------------------------
// Low-Level Byte Helpers
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
    fn test_v2_compressed_memory_roundtrip() {
        let original_index = build_sample_index();

        // Serialize using v2 compression
        let mut buffer = Vec::new();
        save_index_v2(&original_index, &mut buffer).expect("save_index_v2 should succeed");

        assert_eq!(&buffer[0..8], MAGIC_V2);

        // Deserialize v2 buffer
        let mut cursor = Cursor::new(buffer);
        let loaded_index = load_index(&mut cursor).expect("load_index should succeed for v2");

        // Verify full equality
        assert_eq!(
            loaded_index.total_documents(),
            original_index.total_documents()
        );
        assert_eq!(
            loaded_index.vocabulary_size(),
            original_index.vocabulary_size()
        );

        assert_eq!(
            original_index.search_phrase("quick brown"),
            loaded_index.search_phrase("quick brown")
        );
        assert_eq!(
            original_index.search_bm25_default("quick fox"),
            loaded_index.search_bm25_default("quick fox")
        );
    }

    #[test]
    fn test_v1_backward_compatibility() {
        let original_index = build_sample_index();

        // 1. Serialize in old v1 uncompressed format
        let mut v1_buffer = Vec::new();
        save_index_v1(&original_index, &mut v1_buffer).expect("save_index_v1 should succeed");
        assert_eq!(&v1_buffer[0..8], MAGIC_V1);

        // 2. Load it back using the modern unified load_index
        let mut cursor = Cursor::new(v1_buffer);
        let loaded_index = load_index(&mut cursor).expect("load_index must support v1 files");

        // 3. Verify it loaded perfectly
        assert_eq!(
            loaded_index.total_documents(),
            original_index.total_documents()
        );
        assert_eq!(
            original_index.search_term("fox"),
            loaded_index.search_term("fox")
        );
    }

    #[test]
    fn test_v1_vs_v2_file_size_savings() {
        // Build a slightly larger corpus to measure real disk savings
        let mut index = InvertedIndex::new();
        for i in 0..50 {
            index.add_document(
                i,
                "Rust systems programming provides zero cost abstractions and memory safety without garbage collection",
            );
        }

        let mut v1_buffer = Vec::new();
        save_index_v1(&index, &mut v1_buffer).unwrap();

        let mut v2_buffer = Vec::new();
        save_index_v2(&index, &mut v2_buffer).unwrap();

        println!(
            "\nIndex Size Comparison (50 documents):\n  Uncompressed (v1): {} bytes\n  Compressed   (v2): {} bytes\n  Disk Savings:      {:.2}%",
            v1_buffer.len(),
            v2_buffer.len(),
            ((v1_buffer.len() - v2_buffer.len()) as f64 / v1_buffer.len() as f64) * 100.0
        );

        // Compressed v2 must be strictly smaller than uncompressed v1
        assert!(v2_buffer.len() < v1_buffer.len());
    }
}
