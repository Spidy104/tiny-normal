# tiny-normal

A small Rust crate for generating normally distributed (Gaussian) random numbers.
It has no dependencies, just the standard library.

Heads up: this was AI generated. Claude wrote most of the code, and I steered it
and made the design calls.

## Usage

```rust
use tiny_normal::{NormalSampler, NormalSampler4};

// Two normal values at a time
let mut s = NormalSampler::from_seed(42);
let [a, b] = s.normal_pair();

// Or fill a buffer from four streams at once (faster)
let mut bulk = NormalSampler4::from_seed(42);
let mut rows = [[0.0; 4]; 128];
bulk.fill_values(&mut rows);
```

## What's inside

- `NormalSampler`: a Ziggurat sampler, plus a plain Box–Muller version to compare against.
- `NormalSampler4`: four independent streams at once. It uses AVX2 when your CPU has
  it and plain code otherwise, and both give exactly the same numbers.
- `Xoshiro256PlusPlus`: the random number generator underneath. It can `jump` ahead
  so that separate threads get separate streams.

The same seed always gives the same numbers. It isn't meant for cryptography.

## Speed

On my laptop (Ryzen 7 7840HS, one core):

| | Numbers per second |
|---|---|
| `NormalSampler4` | about 1 billion |
| `NormalSampler` | about 600 million |
| Box–Muller | about 90 million |

Try it yourself with `cargo run --release --example bench`.

## Testing

I checked it a fair bit: the output matches the normal distribution in statistical
tests, the fast and plain versions agree bit for bit, it runs clean under Miri, and
it passed PractRand up to 256 GB. It's only been run on x86-64 Windows so far.

There's also an MCMC example (`examples/mcmc.rs`) that uses the sampler for real
sampling problems.

## License

MIT
