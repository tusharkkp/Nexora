/// Implementation of Delta (Gap) Encoding and Variable-Byte (VByte) Compression.
///
/// Designed to compress sorted arrays of 32-bit integers (such as postings list
/// document IDs and token positions) with high compression ratios and fast decoding.

// -----------------------------------------------------------------------------
// 1. Variable-Byte (VByte) Integer Encoding / Decoding
// -----------------------------------------------------------------------------

/// Encodes a single `u32` into Variable-Byte format, appending the bytes to `out`.
///
/// Each byte holds 7 bits of data.
/// Bit 7 (the MSB) is the termination flag:
/// - `0`: More bytes follow for this number.
/// - `1`: Final byte of this number.
pub fn encode_vbyte(mut val: u32, out: &mut Vec<u8>) {
    while val >= 128 {
        // Take lower 7 bits; MSB is 0 (more bytes follow)
        out.push((val & 0x7F) as u8);
        val >>= 7;
    }
    // Final byte; set MSB to 1 (terminal flag)
    out.push((val as u8) | 0x80);
}

/// Decodes a single Variable-Byte encoded `u32` from a byte slice.
///
/// Returns `Some((value, bytes_consumed))` or `None` if the slice is empty
/// or ended before finding the terminal byte.
pub fn decode_vbyte(bytes: &[u8]) -> Option<(u32, usize)> {
    let mut val: u32 = 0;
    let mut shift: u32 = 0;
    let mut bytes_consumed = 0;

    for &byte in bytes {
        bytes_consumed += 1;
        let payload = (byte & 0x7F) as u32;

        val |= payload << shift;

        // If MSB is set, this is the terminal byte
        if (byte & 0x80) != 0 {
            return Some((val, bytes_consumed));
        }

        shift += 7;
        if shift > 35 {
            // Guard against malformed or malicious multi-byte overflow
            return None;
        }
    }

    // EOF reached before terminal byte
    None
}

// -----------------------------------------------------------------------------
// 2. Delta (Gap) Encoding / Decoding
// -----------------------------------------------------------------------------

/// Computes differences (deltas) between consecutive sorted numbers.
///
/// Input:  [100, 104, 105, 112]
/// Output: [100,   4,   1,   7]
pub fn encode_deltas(sorted: &[u32]) -> Vec<u32> {
    if sorted.is_empty() {
        return Vec::new();
    }

    let mut deltas = Vec::with_capacity(sorted.len());
    deltas.push(sorted[0]);

    for i in 1..sorted.len() {
        deltas.push(sorted[i] - sorted[i - 1]);
    }

    deltas
}

/// Reconstructs original sorted numbers from deltas via running prefix sums.
///
/// Input:  [100,   4,   1,   7]
/// Output: [100, 104, 105, 112]
pub fn decode_deltas(deltas: &[u32]) -> Vec<u32> {
    if deltas.is_empty() {
        return Vec::new();
    }

    let mut reconstructed = Vec::with_capacity(deltas.len());
    let mut current = deltas[0];
    reconstructed.push(current);

    for &gap in &deltas[1..] {
        current += gap;
        reconstructed.push(current);
    }

    reconstructed
}

// -----------------------------------------------------------------------------
// 3. End-to-End Postings List Compression
// -----------------------------------------------------------------------------

/// Compresses a sorted slice of integers using Delta Encoding followed by Variable-Byte.
pub fn compress_sorted_u32(sorted: &[u32]) -> Vec<u8> {
    if sorted.is_empty() {
        return Vec::new();
    }

    let deltas = encode_deltas(sorted);
    let mut compressed = Vec::with_capacity(deltas.len() * 2);

    for gap in deltas {
        encode_vbyte(gap, &mut compressed);
    }

    compressed
}

/// Decompresses a Variable-Byte compressed byte slice back into the original sorted integers.
pub fn decompress_sorted_u32(bytes: &[u8]) -> Result<Vec<u32>, &'static str> {
    if bytes.is_empty() {
        return Ok(Vec::new());
    }

    let mut deltas = Vec::new();
    let mut cursor = 0;

    while cursor < bytes.len() {
        match decode_vbyte(&bytes[cursor..]) {
            Some((gap, bytes_consumed)) => {
                deltas.push(gap);
                cursor += bytes_consumed;
            }
            None => return Err("Corrupted compressed byte stream or premature EOF"),
        }
    }

    Ok(decode_deltas(&deltas))
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vbyte_roundtrip_edge_cases() {
        let test_cases = vec![
            0,
            1,
            42,
            127,
            128,
            129,
            255,
            256,
            16383,
            16384,
            1_000_000,
            u32::MAX,
        ];

        for &val in &test_cases {
            let mut encoded = Vec::new();
            encode_vbyte(val, &mut encoded);

            let (decoded, consumed) = decode_vbyte(&encoded).expect("Should decode successfully");
            assert_eq!(decoded, val);
            assert_eq!(consumed, encoded.len());
        }
    }

    #[test]
    fn test_vbyte_byte_sizes() {
        // Values < 128 must take exactly 1 byte
        let mut b1 = Vec::new();
        encode_vbyte(127, &mut b1);
        assert_eq!(b1.len(), 1);

        // Values from 128 to 16383 must take exactly 2 bytes
        let mut b2 = Vec::new();
        encode_vbyte(128, &mut b2);
        assert_eq!(b2.len(), 2);

        let mut b3 = Vec::new();
        encode_vbyte(16383, &mut b3);
        assert_eq!(b3.len(), 2);

        // 16384 takes 3 bytes
        let mut b4 = Vec::new();
        encode_vbyte(16384, &mut b4);
        assert_eq!(b4.len(), 3);
    }

    #[test]
    fn test_delta_roundtrip() {
        let original = vec![1000, 1004, 1005, 1012, 1015, 1025];
        let deltas = encode_deltas(&original);

        assert_eq!(deltas, vec![1000, 4, 1, 7, 3, 10]);

        let recovered = decode_deltas(&deltas);
        assert_eq!(recovered, original);
    }

    #[test]
    fn test_compress_sorted_u32_roundtrip() {
        let doc_ids = vec![5, 12, 19, 23, 45, 90, 120, 500, 10000];
        let compressed = compress_sorted_u32(&doc_ids);

        let decompressed = decompress_sorted_u32(&compressed).expect("Should decompress");
        assert_eq!(decompressed, doc_ids);
    }

    #[test]
    fn test_compression_ratio_simulation() {
        // Simulate a dense postings list: 10,000 documents with small gaps
        let mut doc_ids = Vec::with_capacity(10_000);
        let mut curr = 10;
        for i in 0..10_000 {
            // Small realistic gaps between 1 and 5
            curr += (i % 5) + 1;
            doc_ids.push(curr);
        }

        let raw_bytes_size = doc_ids.len() * std::mem::size_of::<u32>();
        assert_eq!(raw_bytes_size, 40_000); // 40 KB uncompressed

        let compressed = compress_sorted_u32(&doc_ids);
        let compressed_size = compressed.len();

        let savings_percentage =
            ((raw_bytes_size - compressed_size) as f64 / raw_bytes_size as f64) * 100.0;

        println!(
            "Raw size: {} bytes, Compressed: {} bytes, Savings: {:.2}%",
            raw_bytes_size, compressed_size, savings_percentage
        );

        // Almost all gaps are < 128, meaning each number compresses from 4 bytes to 1 byte!
        // Savings should be ~75%!
        assert!(savings_percentage > 70.0);

        // Verification of correctness
        let decompressed = decompress_sorted_u32(&compressed).expect("Decompression must work");
        assert_eq!(decompressed, doc_ids);
    }
}
