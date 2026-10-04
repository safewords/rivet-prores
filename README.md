# rivet-prores

[![CI](https://github.com/safewords/rivet-prores/actions/workflows/ci.yml/badge.svg)](https://github.com/safewords/rivet-prores/actions/workflows/ci.yml)

An **Apple ProRes** decoder and encoder in Rust: no C, no system libraries,
no build script, nothing to install on a build host. Written from SMPTE
RDD 36:2022, *Apple ProRes Bitstream Syntax and Decoding Process*, not
translated from any other implementation. The decoder reads Apple's own
encoder's frames (the figures are [below](#how-it-is-checked)).

Written for the **[rivet](https://github.com/safewords/rivet)**
transcoder, where it is the ProRes codec on both sides: the decoder that
lets a ProRes master be transcoded, and the encoder behind ProRes output.
Usable on its own by anything that has MOV / Matroska samples and wants
planar Y′CbCr back, or planar Y′CbCr and wants ProRes.

Published as `rivet-prores`; **imported as `prores`** (`use prores::…`). One
dependency (`thiserror`), no features, no build script.

```toml
[dependencies]
prores = { package = "rivet-prores", git = "https://github.com/safewords/rivet-prores", branch = "develop" }
```

## What it decodes

Everything RDD 36 describes. The six profiles share one syntax and differ
in chroma format, alpha and bit rate, so there is nothing per-profile in a
decoder.

| | supported | refused |
|---|---|---|
| **Bitstream versions** | 0 and 1 | 2 and above (`Error::Unsupported`, as RDD 36 §6.4 requires) |
| **Profiles** | Proxy (`apco`), LT (`apcs`), 422 (`apcn`), HQ (`apch`), 4444 (`ap4h`), 4444 XQ (`ap4x`) | — |
| **Chroma** | 4:2:2, 4:4:4 | the reserved codes 0 and 1 |
| **Scanning** | progressive frames; interlaced frames as two field pictures, either field first | `interlace_mode` 3 |
| **Quantisation** | the default matrix, loaded luma and chroma matrices, quantisation indices 1–224 | index 0 and 225–255 (reserved) |
| **Alpha** | 8- and 16-bit, lossless; each slice's alpha covers its whole macroblocks (16 rows, those below the picture discarded), and alpha that stops at the picture's last row is read too | `alpha_channel_type` 3–15 |
| **Sizes** | 1×1 to 65535×65535, any slice size (1, 2, 4, 8 macroblocks) | — |
| **Skipped** | version-variant bytes in any header (by its size field), stuffing, bytes after `frame_size` | — |

Output is a [`Frame`](src/frame.rs): planar `u16` samples in one buffer —
Y′, then Cb, then Cr, each tightly packed — the layout of `yuv422p10le` /
`yuv444p12le` once written little-endian (`Frame::to_le_bytes`), with
the alpha plane separate. 4:2:2 comes out at 10 bits and 4:4:4 at 12 by
default; `Decoder::with_bit_depth` gives any depth from 8 to 16 (colour and
alpha, by RDD 36 §7.5's conversions — 16 keeps 16-bit alpha exact).
Interlaced frames come out woven, top field on the even rows, with
`Frame::interlace` saying which field came first. The frame header's
aspect ratio, frame rate and colour codes (H.273 numbering) are in
`Frame::metadata`. Samples are clamped to the full 0…2^b − 1 range.

Every frame stands alone, so `Decoder::decode` takes `&self` and frames can
be decoded on as many threads as the caller likes. A frame is decoded on
one thread: 1080p 422 HQ in about 40 ms, 4444 XQ in about 100 ms (one core
of a desktop x86-64, release build).

## What it encodes

| profile | chroma | input | target per 1920×1080 frame |
|---|---|---|---|
| Proxy | 4:2:2 | 8–16-bit | 187 687 bytes (45 Mb/s at 29.97) |
| LT | 4:2:2 | 8–16-bit | 425 425 bytes (102 Mb/s) |
| 422 | 4:2:2 | 8–16-bit | 613 112 bytes (147 Mb/s) |
| HQ | 4:2:2 | 8–16-bit | 917 583 bytes (220 Mb/s) |
| 4444 | 4:4:4 (+ alpha) | 8–16-bit | 1 376 375 bytes (330 Mb/s) |
| 4444 XQ | 4:4:4 (+ alpha) | 8–16-bit | 2 085 416 bytes (500 Mb/s) |

The rates are Apple's published figures for 1920×1080 at 29.97 frames per
second. The per-frame target (`Profile::target_frame_bytes`) gives every
frame that rate's coded data per macroblock — the 1080 frame's bytes less
its headers, over its 8160 macroblocks — times its own macroblocks, plus
its own headers (frame and picture header, and per slice a table entry
and a slice header); 1920×1080 comes out at the reference rate, or
`Config::target_frame_bytes` says. Alpha is coded losslessly on top of the
target, as 16-bit samples by default (exact for any input depth) or 8-bit.

Until 2026-10-03 the target was the 1080 frame's bytes scaled by area,
headers included. At tiny sizes the headers, fixed per frame and per
slice, then took most of it: a 16×16 422 HQ frame (one macroblock) had
113 bytes of which 44 were headers, and sat at the coarsest quantisers;
it now has 155, a macroblock's share of data plus its headers (35.2 dB
against 30.4 dB on the round-trip test picture; `tests/roundtrip.rs`,
`tiny_frames_get_their_share_per_macroblock`).

A constant rate per macroblock is an approximation away from 1080: Apple's
own 422 HQ frames at 720×486 (the sample below) are about 255 000 bytes,
where it gives about 157 000. Apple publishes rates per frame size; until
those are tabulated here, a caller matching Apple's sizes at SD or UHD sets
`Config::target_frame_bytes`.

Progressive or interlaced (by `Frame::interlace`, as two field pictures),
slices of 1–8 macroblocks (8 by default), the default quantisation matrix
or loaded ones. Each picture's blocks go through the forward DCT once; a
binary search then finds the finest quantiser that, used for every slice,
fits the picture's budget, and the bytes it leaves are shared out slice by
slice, each taking the finest quantiser that fits its share. A frame
therefore lands just under its target (unless even the coarsest quantiser
cannot reach it, or a simple picture needs far less), with near-uniform
quality across it.

Not done (an encoder is free to leave them out): perceptual quantisation
matrices (the default flat matrix is used unless the caller loads one) and
adaptive quantisation by content.

## Speed

Slices are independent (§4), so both directions run slice-parallel on a
pool of worker threads shared by every encoder and decoder in the process:
`Decoder::with_threads(n)` and `Config::threads` (0, the default, is one
per CPU; 1 keeps everything on the calling thread). The block arithmetic —
inverse quantisation, IDCT and sample conversion; sample conversion and
FDCT; quantisation — has SSE4.1, AVX2 and NEON versions picked at run time,
which do the scalar code's single-precision operations in the same order
(multiply, then add; never fused), so the output is bit-identical whatever
the CPU, the SIMD level or the thread count. `PRORES_FORCE_SCALAR=1` forces
the scalar code; CI runs the tests both ways on x86-64 (aarch64 by hand,
see "NEON on ARM hardware").

Frames per second on a Ryzen 9 9950X (16 cores, 32 threads), synthetic
pictures with grain (`examples/bench.rs`), measured 2026-10-04; "before" is
the single-threaded scalar code of 64544da, whose output is byte-for-byte
the same:

| | before | 1 thread | 32 threads |
|---|---|---|---|
| 422 HQ 1280×720, encode | 12 | 79 | 425 |
| 422 HQ 1280×720, decode | 60 | 203 | 913 |
| 422 HQ 1920×1080, encode | 4.8 | 34 | 244 |
| 422 HQ 1920×1080, decode | 22 | 82 | 501 |
| 4444 + 16-bit alpha 1280×720, encode | 8.2 | 55 | 353 |
| 4444 + 16-bit alpha 1280×720, decode | 30 | 112 | 552 |
| 4444 + 16-bit alpha 1920×1080, encode | 3.1 | 23 | 179 |
| 4444 + 16-bit alpha 1920×1080, decode | 14 | 46 | 332 |

```sh
cargo run --release --example bench -- 1080p hq 0 5     # size, hq|4444a, threads, runs
cargo run --release --features bench --example kernels  # each kernel at each SIMD level
```

## How it is checked

No other ProRes implementation is used anywhere — not its code, and not as
a black box.

- **Spec-derived unit tests.** Every codebook of Tables 9–11 against
  codewords worked out by hand from §7.1.1.1, the identity of
  exp-Golomb(k) with the combination code (0, k, k+1) the section notes,
  the signed mapping of Table 8, the alpha codes of Tables 12–14 codeword
  for codeword, both scan orders (permutations, each other's transpose),
  Table 15's qScale, the slice layout of §4's 720-wide example, the field
  heights of §6.2.
- **IDCT accuracy.** The Annex A qualification, run as written — IEEE
  1180's generator, 10 000 blocks at each of the three ranges, both signs,
  against a double-precision IDCT of §7.4's formula — passes all five
  criteria.
- **Hand-built frames** (`tests/handbuilt.rs`): frames assembled bit by bit
  from the syntax tables — DC differences through the sign rule, a run and
  a level through the adaptive codebooks, an AC coefficient through the
  scan into the right block, field order, alpha runs and escapes,
  version-variant header bytes, stuffing — decode to the samples §7 gives,
  computed independently in the test; and malformed headers are refused.
- **Round trips** (`tests/roundtrip.rs`, `tests/fuzz.rs`): every profile;
  sizes from 1×1 up, every slice size, both field orders, input depths 8 to
  16, loaded matrices, lossless alpha. Measured 2026-10-02 on a synthetic
  1920×1080 picture with heavy grain (gradients, hard edges, fine stripes,
  ±24/1023 noise):

  | profile | frame | of target | PSNR |
  |---|---|---|---|
  | Proxy | 187 677 bytes | 99.99% | 37.54 dB |
  | LT | 425 420 bytes | 100.00% | 38.82 dB |
  | 422 | 613 107 bytes | 100.00% | 40.04 dB |
  | HQ | 917 564 bytes | 100.00% | 42.85 dB |
  | 4444 (12-bit) | 1 376 337 bytes | 100.00% | 43.03 dB |
  | 4444 XQ (12-bit) | 2 085 373 bytes | 100.00% | 49.31 dB |

  The same picture without grain at 1280×720: Proxy 62.5 dB, LT 69.5 dB,
  422 and HQ 73.6 dB (both reach the finest quantiser under target). At the
  finest quantiser with the smallest weights (2) the error is at most one
  LSB at 10 bits (90 dB).
- **Malformed input**: property tests feed arbitrary bytes, and valid
  frames with bytes changed, bits flipped, cut and spliced, to the header
  parser and the decoder — errors, never a panic — and a frame too small
  for the slices its dimensions imply is refused before anything is
  allocated.
- **Apple's encoder** (`tests/sample.rs`, run by CI): eleven frames of
  Probe.dev's public sample `AppleProRes422.mov` — 422 HQ, 720×486, bottom
  field first, both matrices loaded, encoder `apl0` — decode to the picture
  (a ColorChecker, colours right), fields in order; re-encoded at Apple's
  own frame size (about 255 000 bytes) they come back at 65 dB against
  Apple's decode. The file is fetched by CI, not committed: it is 100 MB and
  its host gives no licence beyond "freely accessible for testing".

### NEON on ARM hardware

CI runs on x86-64 Linux only, so the NEON (aarch64) code paths are not tested
there. They are verified by hand on ARM hardware (an aarch64 Linux machine,
or Apple silicon) after a change to them and before a release:

```sh
PRORES_REQUIRE_SIMD=1 cargo test --release
PRORES_FORCE_SCALAR=1 cargo test --release
```

The first run checks the NEON kernels bit-exact against the scalar ones; the
second runs everything on the scalar kernels.

## Provenance and licensing

Written from SMPTE RDD 36:2022, which SMPTE publishes free of charge
(<https://pub.smpte.org/doc/rdd36/>); **no ProRes implementation's source
was read** — not FFmpeg's, not Apple's, not any other — and none was run.
The tables are reproduced from RDD 36; the target rates are Apple's
published ones.

**Trademarks and patents.** Apple and ProRes are trademarks of Apple Inc.;
this project is not affiliated with or endorsed by Apple. Apple runs a
licensing programme for ProRes products. Nothing here is a licence to any
Apple patent or trademark, and the authors make no claim about whether
anyone needs one.

## Using it

```rust
use prores::{ChromaFormat, Config, Decoder, Encoder, Frame, Profile};

// Decoding: one MOV sample in, one frame out.
let frame = Decoder::new().decode(&sample)?;
// frame.plane(0), plane(1), plane(2): Y′, Cb, Cr (u16); frame.alpha;
// frame.bit_depth, frame.chroma, frame.interlace, frame.metadata.

// Encoding: a frame in the profile's chroma format, one MOV sample out.
let mut frame = Frame::new(1920, 1080, ChromaFormat::Yuv422, 10)?;
frame.plane_mut(0).copy_from_slice(&luma);
let sample = Encoder::new(Config::new(Profile::Hq)).encode(&frame)?;
// The sample entry code for the MOV: Profile::Hq.fourcc() == *b"apch".
```

## License

Open Encoding Attribution License v1.0 — a source-available (not OSI open-source)
license, royalty-free, with a commercial-attribution requirement. See
[LICENSE.md](LICENSE.md) and [NOTICE](NOTICE).
