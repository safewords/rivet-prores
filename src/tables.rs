//! Block scanning (RDD 36 §7.2.2) and quantisation (§7.3).

/// Figure 4: `PROGRESSIVE_SCAN[8 * v + u]` is the scanned index of the
/// coefficient at row `v`, column `u` of a block of a frame picture.
#[rustfmt::skip]
pub(crate) const PROGRESSIVE_SCAN: [u8; 64] = [
     0,  1,  4,  5, 16, 17, 21, 22,
     2,  3,  6,  7, 18, 20, 23, 28,
     8,  9, 12, 13, 19, 24, 27, 29,
    10, 11, 14, 15, 25, 26, 30, 31,
    32, 33, 37, 38, 45, 46, 53, 54,
    34, 36, 39, 44, 47, 52, 55, 60,
    35, 40, 43, 48, 51, 56, 59, 61,
    41, 42, 49, 50, 57, 58, 62, 63,
];

/// Figure 5: the same for a block of a field picture (`interlace_mode` ≠ 0).
#[rustfmt::skip]
pub(crate) const INTERLACED_SCAN: [u8; 64] = [
     0,  2,  8, 10, 32, 34, 35, 41,
     1,  3,  9, 11, 33, 36, 40, 42,
     4,  6, 12, 14, 37, 39, 43, 49,
     5,  7, 13, 15, 38, 44, 48, 50,
    16, 18, 19, 25, 45, 47, 51, 57,
    17, 20, 24, 26, 46, 52, 56, 58,
    21, 23, 27, 30, 53, 55, 59, 62,
    22, 28, 29, 31, 54, 60, 61, 63,
];

/// §7.3: with `load_luma_quantization_matrix` 0, every weight is 4.
pub(crate) const DEFAULT_MATRIX: [u8; 64] = [4; 64];

/// Table 15: qScale for `quantization_index` 1..=224.
#[inline]
pub(crate) fn qscale(quantization_index: u8) -> u32 {
    let q = quantization_index as u32;
    if q <= 128 { q } else { 128 + 4 * (q - 128) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scans_are_permutations_and_each_others_transpose() {
        for scan in [PROGRESSIVE_SCAN, INTERLACED_SCAN] {
            let mut seen = [false; 64];
            for &s in &scan {
                assert!(!seen[s as usize]);
                seen[s as usize] = true;
            }
        }
        for v in 0..8 {
            for u in 0..8 {
                assert_eq!(PROGRESSIVE_SCAN[8 * v + u], INTERLACED_SCAN[8 * u + v]);
            }
        }
        // The DC is first and the highest frequency last in both.
        assert_eq!(PROGRESSIVE_SCAN[0], 0);
        assert_eq!(PROGRESSIVE_SCAN[63], 63);
    }

    #[test]
    fn qscale_is_table_15() {
        let table = [(1, 1), (2, 2), (126, 126), (127, 127), (128, 128), (129, 132), (130, 136), (223, 508), (224, 512)];
        for (i, q) in table {
            assert_eq!(qscale(i), q);
        }
    }
}
