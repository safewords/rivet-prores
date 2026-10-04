//! Per-kernel timings at every SIMD level this CPU has (best of five).
//!
//! ```sh
//! cargo run --release --features bench --example prores_kernels
//! ```

use prores::bench_api as b;

fn main() {
    let n = 200_000;
    println!("kernel                                   level    ns/call");
    for level in b::levels() {
        println!("idct+dequant+samples (1 block)           {level:8} {:7.1}", b::idct_put(level, n));
    }
    for level in b::levels() {
        println!("samples+fdct (1 block)                   {level:8} {:7.1}", b::fdct_load(level, n));
    }
    for level in b::levels() {
        println!("quantise (32 blocks)                     {level:8} {:7.1}", b::quantise(level, n / 10));
    }
    println!("count bits (32 blocks)                   -        {:7.1}", b::count_bits(n / 10));
    println!("write coefficients (32 blocks)           -        {:7.1}", b::write_coefficients(n / 10));
    println!("decode coefficients (32 blocks)          -        {:7.1}", b::decode_coefficients(n / 10));
    println!("({} coded bytes per 32-block component)", b::component_bytes());
    println!("Frame::to_le_bytes (1080p 4:2:2)         -        {:7.1}", b::to_le_bytes(200));
}
