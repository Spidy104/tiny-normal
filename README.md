# tiny-normal

Small, allocation-free standard-normal sampling for Rust. Standard library only,
no dependencies.

- **`NormalSampler`**: 32 bytes of state. A 128-slot modified undergraph Ziggurat
  with an exact packed alias and chord-triangle overhangs, plus a Box–Muller reference.
- **`NormalSampler4`**: four jump-separated streams in 256 bytes. A branch-free
  AVX2 kernel (AVX-512VL rotates when present) with deferred per-lane repair, and a
  safe scalar fallback. **Every backend is bit-identical**, including final RNG state.
- **`Xoshiro256PlusPlus`**: SplitMix64 seeding and 2^128 jumps for parallel streams.
- 3,264 bytes of shared tables. Generation never allocates. Not cryptographic.

```rust
use tiny_normal::{NormalSampler, NormalSampler4};

let mut s = NormalSampler::from_seed(42);
let [a, b] = s.normal_pair();

let mut bulk = NormalSampler4::from_seed(42);
let mut rows = [[0.0; 4]; 128]; // row k = value k of streams 0, 1, 2, 3
bulk.fill_values(&mut rows);
```

## Contracts

- Seeded output is deterministic. A fill may be split at any point with identical
  results, and an empty fill consumes nothing.
- `NormalSampler4` lane `i` uses jump segment `i` (main) and `4 + i` (repair), so one
  sampler occupies `NormalSampler4::SEGMENTS` = 8 segments. To give each thread its
  own sampler, pass `from_rng` an engine advanced by 8 jumps per thread.
- Runtime CPU detection runs once per `fill_values`. With
  `-C target-cpu=native`, detection compiles away and the kernel is inlined.

## Performance (Ryzen 7 7840HS, one core)

| Sampler | Throughput |
|---|---|
| `NormalSampler4::fill_values` | ~1.0-1.3 G normals/s |
| `NormalSampler::fill_pairs` | ~0.56-0.74 G normals/s |
| Box–Muller reference | ~0.08-0.11 G normals/s |

Run `cargo run --release --example bench` (env: `TINY_BENCH_METHOD`, `TINY_BENCH_MODE`,
`TINY_BENCH_ROUNDS`, `TINY_BENCH_SEED`).

## Validation

- Tables were generated at 70-80 digits, and the alias masses agree with independent
  quadrature to within 8e-17.
- A 1e8-sample chi² test of the uncommon path against its exact density passes. A
  deliberately broken variant fails it.
- Each SIMD backend matches an independent scalar reference bit for bit on 16.8M
  values, including all repair positions and split fills.
- Miri passes on both the SIMD and the scalar paths. A standalone allocation audit is
  in `tests/memory.rs`.
- PractRand 0.95, 256 GB per stream through Φ: no persistent anomaly. A defective
  control fails at 16 GB.

These checks are evidence, not proofs. Validation ran on x86-64 Windows; aarch64 is
compile-checked only. Finite precision means discrete output.

## License

MIT, see [LICENSE](LICENSE). The Ziggurat tables are independently generated;
earlier versions used tables from the Rand project (MIT), which are no longer included.
