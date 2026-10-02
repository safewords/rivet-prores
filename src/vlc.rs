//! The variable-length codes of RDD 36 §7.1: Golomb-Rice / exponential-
//! Golomb combination codes (§7.1.1.1), the signed mapping (§7.1.1.2), the
//! adaptive codebook choices for DC differences, runs and levels (Tables
//! 9–11), and the alpha channel's run and difference codes (Tables 12–14).

use crate::bits::{BitReader, BitSink};
use crate::error::{Result, invalid};

/// One codebook. `ExpGolomb(k)` is `EXP_GOLOMB_CODE(k)`; `Combo(q, kr, ke)`
/// is `RICE_EXP_COMBO_CODE(lastRiceQ, kRice, kExp)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Codebook {
    ExpGolomb(u32),
    Combo(u32, u32, u32),
}

use Codebook::{Combo, ExpGolomb};

/// Longest `0` prefix accepted. A 12-bit DCT coefficient quantised at the
/// finest step is at most 2^14 in magnitude, so its symbols need at most 16
/// prefix bits under any codebook here; 32 is a generous bound that keeps
/// every read inside the reader's 57-bit window.
const MAX_PREFIX: u32 = 32;

/// Largest symbol [`Codebook::read`] accepts.
pub(crate) const MAX_SYMBOL: u32 = 1 << 30;

/// Table 9: codebook for `dc_coeff_difference` by |previousDCDiff| (≥ 3
/// shares the last entry).
pub(crate) const DC_CODEBOOKS: [Codebook; 4] =
    [ExpGolomb(0), ExpGolomb(1), Combo(1, 2, 3), ExpGolomb(3)];

/// Table 10: codebook for `run` by previousRun (≥ 15 shares the last entry).
pub(crate) const RUN_CODEBOOKS: [Codebook; 16] = [
    Combo(2, 0, 1),
    Combo(2, 0, 1),
    Combo(1, 0, 1),
    Combo(1, 0, 1),
    ExpGolomb(0),
    Combo(1, 1, 2),
    Combo(1, 1, 2),
    Combo(1, 1, 2),
    Combo(1, 1, 2),
    ExpGolomb(1),
    ExpGolomb(1),
    ExpGolomb(1),
    ExpGolomb(1),
    ExpGolomb(1),
    ExpGolomb(1),
    ExpGolomb(2),
];

/// Table 11: codebook for `abs_level_minus_1` by previousLevelSymbol (≥ 8
/// shares the last entry).
pub(crate) const LEVEL_CODEBOOKS: [Codebook; 9] = [
    Combo(2, 0, 2),
    Combo(1, 0, 1),
    Combo(2, 0, 1),
    ExpGolomb(0),
    ExpGolomb(1),
    ExpGolomb(1),
    ExpGolomb(1),
    ExpGolomb(1),
    ExpGolomb(2),
];

/// §7.1.1.3: the codebook of the first DC coefficient.
pub(crate) const FIRST_DC_CODEBOOK: Codebook = ExpGolomb(5);

#[inline]
pub(crate) fn dc_codebook(prev_dc_diff: i32) -> Codebook {
    DC_CODEBOOKS[(prev_dc_diff.unsigned_abs() as usize).min(3)]
}

#[inline]
pub(crate) fn run_codebook(prev_run: u32) -> Codebook {
    RUN_CODEBOOKS[(prev_run as usize).min(15)]
}

#[inline]
pub(crate) fn level_codebook(prev_level_symbol: u32) -> Codebook {
    LEVEL_CODEBOOKS[(prev_level_symbol as usize).min(8)]
}

/// Number of bits in `v` (0 for 0).
#[inline]
fn bit_len(v: u64) -> u32 {
    64 - v.leading_zeros()
}

/// Order-`k` exponential-Golomb: `q = floor(log2(n + 2^k)) - k` zeros, then
/// `n + 2^k` in `q + k + 1` bits.
#[inline]
fn put_exp_golomb<S: BitSink>(s: &mut S, n: u64, k: u32) {
    let v = n + (1u64 << k);
    let len = bit_len(v);
    let q = len - k - 1;
    if len + q <= 57 {
        s.put(v, len + q);
    } else {
        s.put(0, q);
        s.put(v, len);
    }
}

impl Codebook {
    /// Writes symbol `n`.
    #[inline]
    pub(crate) fn put<S: BitSink>(self, s: &mut S, n: u32) {
        let n = n as u64;
        match self {
            ExpGolomb(k) => put_exp_golomb(s, n, k),
            Combo(last_q, k_rice, k_exp) => {
                let threshold = ((last_q + 1) as u64) << k_rice;
                if n < threshold {
                    // Order-kRice Golomb-Rice: q zeros, a 1, k bits of n mod 2^k.
                    let q = (n >> k_rice) as u32;
                    let r = n & ((1u64 << k_rice) - 1);
                    s.put((1u64 << k_rice) | r, q + 1 + k_rice);
                } else {
                    s.put(0, last_q + 1);
                    put_exp_golomb(s, n - threshold, k_exp);
                }
            }
        }
    }

    /// Length in bits of the codeword for symbol `n`.
    #[cfg(test)]
    pub(crate) fn len(self, n: u32) -> u32 {
        let eg = |n: u64, k: u32| {
            let len = bit_len(n + (1u64 << k));
            2 * len - k - 1
        };
        let n = n as u64;
        match self {
            ExpGolomb(k) => eg(n, k),
            Combo(last_q, k_rice, k_exp) => {
                let threshold = ((last_q + 1) as u64) << k_rice;
                if n < threshold {
                    (n >> k_rice) as u32 + 1 + k_rice
                } else {
                    last_q + 1 + eg(n - threshold, k_exp)
                }
            }
        }
    }

    /// Reads one symbol (§7.1.1.1's decoding procedure).
    #[inline]
    pub(crate) fn read(self, r: &mut BitReader) -> Result<u32> {
        let q = r.leading_zeros(MAX_PREFIX)?;
        let (q_exp, k_exp, offset) = match self {
            ExpGolomb(k) => (q, k, 0u64),
            Combo(last_q, k_rice, k_exp) => {
                if q <= last_q {
                    // Skip the separator `1`, then k bits of remainder.
                    let v = r.read(1 + k_rice)? & ((1u64 << k_rice) - 1);
                    return Ok((((q as u64) << k_rice) | v) as u32);
                }
                (q - (last_q + 1), k_exp, ((last_q + 1) as u64) << k_rice)
            }
        };
        // The exp-Golomb part: the `1` already under the reader starts the
        // (q + k + 1)-bit value n + 2^k.
        let v = r.read(q_exp + k_exp + 1)?;
        let n = v - (1u64 << k_exp) + offset;
        // No coefficient, run or level comes near 2^30; the bound keeps the
        // signed mapping and the sums built from symbols clear of overflow.
        if n > MAX_SYMBOL as u64 {
            return Err(invalid("a variable-length code's value is out of range"));
        }
        Ok(n as u32)
    }
}

/// §7.1.1.2 S(n): 0, -1, 1, -2, 2, … ↦ 0, 1, 2, 3, 4, …
#[inline]
pub(crate) fn signed_to_symbol(n: i32) -> u32 {
    if n >= 0 { 2 * n as u32 } else { 2 * n.unsigned_abs() - 1 }
}

/// The inverse of [`signed_to_symbol`].
#[inline]
pub(crate) fn symbol_to_signed(s: u32) -> i32 {
    if s & 1 == 0 { (s >> 1) as i32 } else { -(s.div_ceil(2) as i32) }
}

/// Table 12: alpha run lengths 1..=2048.
pub(crate) fn put_alpha_run<S: BitSink>(s: &mut S, run: u32) {
    debug_assert!((1..=2048).contains(&run));
    match run {
        1 => s.put(1, 1),
        2..=16 => s.put((run - 1) as u64, 5),
        _ => s.put((run - 1) as u64, 16), // five 0 bits, then 11 bits of run - 1
    }
}

pub(crate) fn read_alpha_run(r: &mut BitReader) -> Result<u32> {
    if r.read_bit()? == 1 {
        return Ok(1);
    }
    let v = r.read(4)? as u32;
    if v != 0 {
        return Ok(v + 1);
    }
    Ok(r.read(11)? as u32 + 1)
}

/// The alpha difference code of Table 13 (`bits` = 8) or Table 14
/// (`bits` = 16): small differences as `0`, |d| - 1, sign; anything else
/// as an escape `1` then the difference modulo 2^bits.
pub(crate) fn put_alpha_difference<S: BitSink>(s: &mut S, diff: i32, bits: u32) {
    let (mag_bits, max) = if bits == 8 { (3, 8) } else { (6, 64) };
    let a = diff.unsigned_abs();
    if (1..=max).contains(&a) {
        let code = (((a - 1) << 1) | (diff < 0) as u32) as u64;
        s.put(code, 1 + mag_bits + 1);
    } else {
        let m = (diff as u32) & ((1u32 << bits) - 1);
        s.put((1u64 << bits) | m as u64, 1 + bits);
    }
}

/// Reads an alpha difference: `(difference, is_modulo)`. A modulo
/// difference is returned as its raw unsigned value.
pub(crate) fn read_alpha_difference(r: &mut BitReader, bits: u32) -> Result<(i32, bool)> {
    let mag_bits = if bits == 8 { 3 } else { 6 };
    if r.read_bit()? == 1 {
        return Ok((r.read(bits)? as i32, true));
    }
    let a = r.read(mag_bits)? as i32 + 1;
    let neg = r.read_bit()? == 1;
    Ok((if neg { -a } else { a }, false))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::{BitCounter, BitWriter};

    /// The codeword as a string of '0'/'1'.
    fn codeword(cb: Codebook, n: u32) -> String {
        let mut v = Vec::new();
        let mut w = BitWriter::new(&mut v);
        cb.put(&mut w, n);
        let bits = w.bits();
        w.finish();
        let s: String = v.iter().map(|b| format!("{b:08b}")).collect();
        s[..bits].to_string()
    }

    #[test]
    fn exp_golomb_order_0_and_1_by_hand() {
        // §7.1.1.1: q = floor(log2(n + 2^k)) - k zeros, then n + 2^k.
        let eg0 = ["1", "010", "011", "00100", "00101", "00110", "00111", "0001000"];
        for (n, want) in eg0.iter().enumerate() {
            assert_eq!(codeword(ExpGolomb(0), n as u32), *want, "EG0({n})");
        }
        let eg1 = ["10", "11", "0100", "0101", "0110", "0111", "001000"];
        for (n, want) in eg1.iter().enumerate() {
            assert_eq!(codeword(ExpGolomb(1), n as u32), *want, "EG1({n})");
        }
        assert_eq!(codeword(ExpGolomb(5), 0), "100000");
        assert_eq!(codeword(ExpGolomb(5), 32), "01000000");
    }

    #[test]
    fn combination_code_by_hand() {
        // RICE_EXP_COMBO_CODE(2, 0, 1): Rice order 0 for n < 3 (q + 1 bits),
        // then three 0s and EG1(n - 3).
        assert_eq!(codeword(Combo(2, 0, 1), 0), "1");
        assert_eq!(codeword(Combo(2, 0, 1), 1), "01");
        assert_eq!(codeword(Combo(2, 0, 1), 2), "001");
        assert_eq!(codeword(Combo(2, 0, 1), 3), "00010");
        assert_eq!(codeword(Combo(2, 0, 1), 4), "00011");
        assert_eq!(codeword(Combo(2, 0, 1), 5), "0000100");
        // RICE_EXP_COMBO_CODE(1, 2, 3): Rice order 2 for n < 8.
        assert_eq!(codeword(Combo(1, 2, 3), 0), "100");
        assert_eq!(codeword(Combo(1, 2, 3), 5), "0101");
        assert_eq!(codeword(Combo(1, 2, 3), 7), "0111");
        assert_eq!(codeword(Combo(1, 2, 3), 8), "001000");
        assert_eq!(codeword(Combo(1, 2, 3), 15), "001111");
        assert_eq!(codeword(Combo(1, 2, 3), 16), "00010000");
    }

    #[test]
    fn exp_golomb_is_the_combination_code_with_last_rice_q_0() {
        // The note at the end of §7.1.1.1.
        for k in 0..6 {
            for n in 0..2000 {
                assert_eq!(codeword(ExpGolomb(k), n), codeword(Combo(0, k, k + 1), n));
            }
        }
    }

    #[test]
    fn every_codebook_round_trips_and_counts_its_length() {
        let mut books: Vec<Codebook> = DC_CODEBOOKS.to_vec();
        books.extend(RUN_CODEBOOKS);
        books.extend(LEVEL_CODEBOOKS);
        books.push(FIRST_DC_CODEBOOK);
        let values: Vec<u32> = (0..3000).chain([65535, 1 << 20, (1 << 24) + 7, MAX_SYMBOL]).collect();
        for cb in books {
            let mut v = Vec::new();
            let mut w = BitWriter::new(&mut v);
            let mut c = BitCounter::default();
            for &n in &values {
                let before = w.bits();
                cb.put(&mut w, n);
                cb.put(&mut c, n);
                assert_eq!((w.bits() - before) as u32, cb.len(n), "{cb:?} {n}");
            }
            assert_eq!(w.bits(), c.0);
            w.finish();
            let mut r = BitReader::new(&v);
            for &n in &values {
                assert_eq!(cb.read(&mut r).unwrap(), n, "{cb:?}");
            }
        }
    }

    #[test]
    fn signed_mapping_is_table_8() {
        let table = [(0, 0), (-1, 1), (1, 2), (-2, 3), (2, 4), (-3, 5), (3, 6)];
        for (n, s) in table {
            assert_eq!(signed_to_symbol(n), s);
            assert_eq!(symbol_to_signed(s), n);
        }
        for n in -70000..70000 {
            assert_eq!(symbol_to_signed(signed_to_symbol(n)), n);
        }
    }

    fn bits_of(f: impl FnOnce(&mut BitWriter)) -> String {
        let mut v = Vec::new();
        let mut w = BitWriter::new(&mut v);
        f(&mut w);
        let bits = w.bits();
        w.finish();
        let s: String = v.iter().map(|b| format!("{b:08b}")).collect();
        s[..bits].to_string()
    }

    #[test]
    fn alpha_runs_are_table_12() {
        let table = [
            (1, "1"),
            (2, "00001"),
            (3, "00010"),
            (15, "01110"),
            (16, "01111"),
            (17, "0000000000010000"),
            (18, "0000000000010001"),
            (2047, "0000011111111110"),
            (2048, "0000011111111111"),
        ];
        for (run, want) in table {
            assert_eq!(bits_of(|w| put_alpha_run(w, run)), want, "run {run}");
            let mut v = Vec::new();
            let mut w = BitWriter::new(&mut v);
            put_alpha_run(&mut w, run);
            w.finish();
            assert_eq!(read_alpha_run(&mut BitReader::new(&v)).unwrap(), run);
        }
    }

    #[test]
    fn alpha_differences_are_tables_13_and_14() {
        let t13 = [(1, "00000"), (-1, "00001"), (2, "00010"), (-2, "00011"), (8, "01110"), (-8, "01111")];
        for (d, want) in t13 {
            assert_eq!(bits_of(|w| put_alpha_difference(w, d, 8)), want);
        }
        assert_eq!(bits_of(|w| put_alpha_difference(w, 9, 8)), "100001001");
        assert_eq!(bits_of(|w| put_alpha_difference(w, -9, 8)), "111110111");
        assert_eq!(bits_of(|w| put_alpha_difference(w, 0, 8)), "100000000");
        let t14 = [(1, "00000000"), (-1, "00000001"), (2, "00000010"), (64, "01111110"), (-64, "01111111")];
        for (d, want) in t14 {
            assert_eq!(bits_of(|w| put_alpha_difference(w, d, 16)), want);
        }
        assert_eq!(bits_of(|w| put_alpha_difference(w, 65, 16)), "10000000001000001");
        for bits in [8, 16] {
            for d in -300..300 {
                let mut v = Vec::new();
                let mut w = BitWriter::new(&mut v);
                put_alpha_difference(&mut w, d, bits);
                w.finish();
                let (got, modulo) = read_alpha_difference(&mut BitReader::new(&v), bits).unwrap();
                let mask = (1i32 << bits) - 1;
                if modulo {
                    assert_eq!(got & mask, d & mask);
                } else {
                    assert_eq!(got, d);
                }
            }
        }
    }
}
