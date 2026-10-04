//! Bit-level reading and writing, most significant bit first (RDD 36 §5:
//! "bit strings and variable-length codes appear in the bitstream left bit
//! first; numerical values appear most-significant bit first").

use crate::error::{Result, invalid};

/// Reads one syntax structure of known size (a colour component's or the
/// alpha channel's coded data). Reads past its end are errors, never a
/// panic.
pub(crate) struct BitReader<'a> {
    data: &'a [u8],
    /// Position in bits from the start of `data`.
    pos: usize,
}

impl<'a> BitReader<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        BitReader { data, pos: 0 }
    }

    /// Bits left before the end of the structure.
    #[inline]
    pub(crate) fn remaining(&self) -> usize {
        self.data.len() * 8 - self.pos
    }

    /// The next 64 bits, MSB-aligned, zero past the end of the data. At
    /// least 57 of them are real bits when that many remain.
    #[inline]
    fn peek64(&self) -> u64 {
        let byte = self.pos >> 3;
        let mut buf = [0u8; 8];
        if byte + 8 <= self.data.len() {
            buf.copy_from_slice(&self.data[byte..byte + 8]);
        } else if byte < self.data.len() {
            let n = self.data.len() - byte;
            buf[..n].copy_from_slice(&self.data[byte..]);
        }
        u64::from_be_bytes(buf) << (self.pos & 7)
    }

    /// The next 64 bits, MSB-aligned, when the 8 bytes from the current
    /// one are all data: then at least 57 of them are real bits. The
    /// decoder's fast path; near the end it is `None`.
    #[inline]
    pub(crate) fn peek_fast(&self) -> Option<u64> {
        let byte = self.pos >> 3;
        let bytes: [u8; 8] = self.data.get(byte..byte + 8)?.try_into().ok()?;
        Some(u64::from_be_bytes(bytes) << (self.pos & 7))
    }

    /// Consumes `n` bits the caller has seen through [`Self::peek_fast`].
    #[inline]
    pub(crate) fn skip(&mut self, n: u32) {
        debug_assert!(n as usize <= self.remaining());
        self.pos += n as usize;
    }

    /// `n` bits (0..=57) as an unsigned number.
    #[inline]
    pub(crate) fn read(&mut self, n: u32) -> Result<u64> {
        debug_assert!(n <= 57);
        if n == 0 {
            return Ok(0);
        }
        if (n as usize) > self.remaining() {
            return Err(invalid("a codeword runs past the end of its data"));
        }
        let v = self.peek64() >> (64 - n);
        self.pos += n as usize;
        Ok(v)
    }

    #[inline]
    pub(crate) fn read_bit(&mut self) -> Result<u32> {
        Ok(self.read(1)? as u32)
    }

    /// Counts the `0` bits before the next `1` and consumes them, leaving
    /// the reader on the `1`. A prefix longer than `max` is an error: no
    /// codeword this crate decodes has one, and the bound keeps every later
    /// read within 57 bits.
    #[inline]
    pub(crate) fn leading_zeros(&mut self, max: u32) -> Result<u32> {
        let lz = self.peek64().leading_zeros();
        if lz > max {
            return Err(invalid("a variable-length code's prefix is too long"));
        }
        if lz as usize >= self.remaining() {
            return Err(invalid("a codeword runs past the end of its data"));
        }
        self.pos += lz as usize;
        Ok(lz)
    }

    /// RDD 36 §5 `endOfData()`: 31 or fewer bits remain and all are `0`.
    #[inline]
    pub(crate) fn end_of_data(&self) -> bool {
        let r = self.remaining();
        r <= 31 && (r == 0 || self.peek64() >> (64 - r) == 0)
    }
}

/// Where coded bits go: a real writer, or a counter for rate control (the
/// encoder sizes a slice at several quantisers before writing it once).
pub(crate) trait BitSink {
    /// The low `n` bits of `v` (`n` ≤ 57), most significant first.
    fn put(&mut self, v: u64, n: u32);
    /// Bits written so far.
    fn bits(&self) -> usize;
    /// `0` bits up to the next byte boundary.
    fn align(&mut self) {
        let r = self.bits() % 8;
        if r != 0 {
            self.put(0, 8 - r as u32);
        }
    }
}

/// Counts bits without storing them.
#[cfg(test)]
#[derive(Default)]
pub(crate) struct BitCounter(pub(crate) usize);

#[cfg(test)]
impl BitSink for BitCounter {
    #[inline]
    fn put(&mut self, _v: u64, n: u32) {
        self.0 += n as usize;
    }
    fn bits(&self) -> usize {
        self.0
    }
}

/// Appends bits to a byte vector, 64 at a time. [`BitWriter::finish`]
/// (or dropping the writer) byte-aligns and writes what is held.
pub(crate) struct BitWriter<'a> {
    out: &'a mut Vec<u8>,
    start: usize,
    /// The last `nacc` bits written, in the low bits.
    acc: u64,
    /// Bits held in `acc`, always < 64.
    nacc: u32,
}

impl<'a> BitWriter<'a> {
    pub(crate) fn new(out: &'a mut Vec<u8>) -> Self {
        let start = out.len();
        BitWriter { out, start, acc: 0, nacc: 0 }
    }

    /// Byte-aligns with `0` bits and returns the bytes written.
    pub(crate) fn finish(mut self) -> usize {
        self.flush();
        self.out.len() - self.start
    }

    fn flush(&mut self) {
        self.align();
        if self.nacc > 0 {
            let bytes = (self.acc << (64 - self.nacc)).to_be_bytes();
            self.out.extend_from_slice(&bytes[..self.nacc as usize / 8]);
            (self.acc, self.nacc) = (0, 0);
        }
    }
}

impl Drop for BitWriter<'_> {
    fn drop(&mut self) {
        self.flush();
    }
}

impl BitSink for BitWriter<'_> {
    #[inline]
    fn put(&mut self, v: u64, n: u32) {
        debug_assert!(n <= 57);
        let v = v & ((1u64 << n) - 1);
        let free = 64 - self.nacc;
        if n < free {
            self.acc = (self.acc << n) | v;
            self.nacc += n;
        } else {
            // Fill the accumulator (free ≤ n, so free < 64), write it, and
            // keep the bits that did not fit.
            let rest = n - free;
            let full = (self.acc << free) | (v >> rest);
            self.out.extend_from_slice(&full.to_be_bytes());
            self.acc = v & ((1u64 << rest) - 1);
            self.nacc = rest;
        }
    }
    fn bits(&self) -> usize {
        (self.out.len() - self.start) * 8 + self.nacc as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_then_read() {
        let mut v = Vec::new();
        let mut w = BitWriter::new(&mut v);
        w.put(0b101, 3);
        w.put(0, 4);
        w.put(1, 1);
        w.put(0x1_2345_6789, 33);
        assert_eq!(w.bits(), 41);
        assert_eq!(w.finish(), 6);
        let mut r = BitReader::new(&v);
        assert_eq!(r.read(3).unwrap(), 0b101);
        assert_eq!(r.leading_zeros(32).unwrap(), 4);
        assert_eq!(r.read_bit().unwrap(), 1);
        assert_eq!(r.read(33).unwrap(), 0x1_2345_6789);
        assert_eq!(r.remaining(), 7);
        assert!(r.end_of_data());
        assert!(r.read(8).is_err());
    }

    #[test]
    fn end_of_data_needs_fewer_than_32_zero_bits() {
        assert!(BitReader::new(&[]).end_of_data());
        assert!(BitReader::new(&[0, 0, 0]).end_of_data());
        assert!(!BitReader::new(&[0, 0, 0, 0]).end_of_data());
        assert!(!BitReader::new(&[0, 0, 1]).end_of_data());
    }
}
