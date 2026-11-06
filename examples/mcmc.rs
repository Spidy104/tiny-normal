//! Sixteen-chain Metropolis–Hastings on 2D targets whose answers are known exactly.
//!
//! The question this answers: does a faster Gaussian sampler make MCMC faster?
//! It measures acceptance, mode coverage and effective samples per second for three
//! proposal-noise sources: Box–Muller (independent reference), the scalar ziggurat
//! and the four-lane bulk sampler.
//!
//! ```text
//! cargo run --release --example mcmc -- [--target gauss|banana|twin|all]
//!     [--kernel rwm|pt|all] [--source box-muller|ziggurat|bulk4|all]
//!     [--workers 1|2|4|8|all] [--steps N] [--warmup N] [--seed S] [--rounds R] [--quick]
//! ```
//!
//! Scalar RWM uses paired jobs (and a wide serial ziggurat task); bulk/PT keep
//! four-chain groups. `all` respects source/job limits and available concurrency.
//! Explicit supported counts can be used for oversubscription experiments.
//!
//! One CSV row per run goes to stdout; the summary goes to stderr. Chains are never
//! stored: estimates, batch-means errors and split R-hat all stream in fixed memory.
//! Worker warmup/sampling loops allocate nothing; TLS allocator counters check
//! these windows. Thread startup, coordination and result processing are separate.
#![allow(clippy::needless_range_loop)] // lockstep loops index several parallel arrays

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};
use std::sync::{Condvar, Mutex, MutexGuard};
use std::thread;
use std::time::Instant;
use tiny_normal::{NormalSampler, NormalSampler4, Xoshiro256PlusPlus};

/// Sixteen logical chains, in two-chain jobs or four-chain SIMD/tempering groups.
const C: usize = 16;
/// Tracked test functions per target.
const F: usize = 4;
/// Batch-means slots. When full, neighbours merge and the batch size doubles.
const SLOTS: usize = 64;
/// `NormalSampler4` rows per refill: 64 rows serve 32 steps of two normals per lane.
const ROWS: usize = 64;
/// Heuristic acceptance target, motivated by 2D Gaussian random walks.
/// Neither 0.35 nor the asymptotic 0.234 rule is optimal for every target.
const TARGET_ACCEPT: f64 = 0.35;
/// Warmup adapts the proposal every this many steps; measurement never adapts.
const ADAPT_EVERY: u64 = 256;
/// Parallel-tempering inverse temperatures, cold to hot (geometric-ish ladder).
const BETAS: [f64; 4] = [1.0, 0.35, 0.12, 0.04];
/// Steps between replica-exchange rounds.
const SWAP_EVERY: u64 = 8;
/// Pass gate: pooled minimum ESS per estimating chain (50 per split half).
const MIN_ESS_PER_CHAIN: f64 = 100.0;
/// Requested stack per worker thread.
const STACK: usize = 256 * 1024;

// ------------------------------------------------------------------------------
// Allocation audit: sampling must not touch the heap.

struct Counting;
static ARMED: AtomicBool = AtomicBool::new(false);
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
#[global_allocator]
static GLOBAL: Counting = Counting;
thread_local! {
    static LOCAL_ARMED: Cell<bool> = const { Cell::new(false) };
    static LOCAL_ALLOCS: Cell<usize> = const { Cell::new(0) };
}
fn count() {
    let _ = LOCAL_ARMED.try_with(|armed| {
        if armed.get() {
            LOCAL_ALLOCS.with(|n| n.set(n.get() + 1));
        }
    });
    if ARMED.load(Relaxed) {
        ALLOCS.fetch_add(1, Relaxed);
    }
}
// SAFETY: every method forwards the unchanged allocator contract to System.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count();
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        count();
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        count();
        unsafe { System.realloc(ptr, layout, size) }
    }
}

// ------------------------------------------------------------------------------
// Targets. Each has closed-form expectations, so every estimate can be graded.

struct Target {
    name: &'static str,
    funcs: [&'static str; F],
    truth: [f64; F],
    /// Half-widths of the overdispersed start box.
    spread: [f64; 2],
}
const RHO: f64 = 0.95;
const BEND: f64 = 0.1;
/// Twin mode centres at -MODE and +MODE: far enough apart (barrier ~ e^-18) that
/// a random walk cannot cross on its own.
const MODE: f64 = 6.0;
/// 0.7 * Phi(6) + 0.3 * Phi(-6): the twin's mass at x > 0.
const P_RIGHT: f64 = 0.6999999996053649;
const TARGETS: [Target; 3] = [
    Target {
        name: "gauss",
        funcs: ["x", "y", "x^2", "xy"],
        truth: [0.0, 0.0, 1.0, RHO],
        spread: [4.0, 4.0],
    },
    Target {
        name: "banana",
        funcs: ["x", "y", "x^2", "y^2"],
        truth: [0.0, 0.0, 100.0, 1.0 + 2.0 * BEND * BEND * 1e4],
        spread: [30.0, 60.0],
    },
    Target {
        name: "twin",
        funcs: ["x", "y", "x>0", "x^2"],
        truth: [0.4 * MODE, 0.0, P_RIGHT, 1.0 + MODE * MODE],
        spread: [1.5 * MODE, 4.0],
    },
];
const TWIN: usize = 2;

/// Unnormalized log densities: correlated Gaussian; Haario's twisted "banana"
/// (x ~ N(0, 100), y - b(x^2 - 100) ~ N(0, 1)); and 0.3 N((-6,0), I) + 0.7 N((6,0), I).
#[inline(always)]
fn log_density<const T: usize>(x: f64, y: f64) -> f64 {
    match T {
        0 => -(x * x - 2.0 * RHO * x * y + y * y) / (2.0 * (1.0 - RHO * RHO)),
        1 => {
            let z = y - BEND * (x * x - 100.0);
            -x * x / 200.0 - 0.5 * z * z
        }
        _ => {
            let left = -1.2039728043259361 - 0.5 * (x + MODE) * (x + MODE); // ln 0.3
            let right = -0.35667494393873245 - 0.5 * (x - MODE) * (x - MODE); // ln 0.7
            let base = -0.5 * y * y + left.max(right);
            let distance = (left - right).abs();
            // base <= ln(0.7) < -0.25, so its half-ULP is at least 2^-55.
            // ln(1+exp(-40)) < 4.25e-18 < 2^-55: adding this positive
            // correction rounds back to base. Keep the full formula near modes' overlap.
            if distance >= 40.0 {
                base
            } else {
                base + (-distance).exp().ln_1p()
            }
        }
    }
}

#[inline(always)]
fn test_functions<const T: usize>(x: f64, y: f64) -> [f64; F] {
    match T {
        0 => [x, y, x * x, x * y],
        1 => [x, y, x * x, y * y],
        _ => [x, y, if x > 0.0 { 1.0 } else { 0.0 }, x * x],
    }
}

// ------------------------------------------------------------------------------
// Proposal noise. Each source is a const generic, so the hot loop has no dispatch.

const BOX_MULLER: u8 = 0;
const ZIGGURAT: u8 = 1;
const BULK4: u8 = 2;
const SOURCES: [&str; 3] = ["box-muller", "ziggurat", "bulk4"];

/// Every chain draws from its own jump segment of one seeded engine:
/// uniforms 0..16, scalar normals 16..32, bulk samplers 32..64, start points 64.
struct BulkNoise {
    sampler: NormalSampler4,
    rows: [[f64; 4]; ROWS],
    row: usize,
}
struct Noise<const N: usize> {
    scalar: [NormalSampler; N],
    bulk: Option<Box<BulkNoise>>,
}
impl<const N: usize> Noise<N> {
    #[inline(always)]
    fn draw<const S: u8>(&mut self, z: &mut [[f64; 2]; N]) {
        match S {
            BOX_MULLER => {
                for (sampler, pair) in self.scalar.iter_mut().zip(z) {
                    *pair = sampler.box_muller_pair();
                }
            }
            ZIGGURAT => {
                for (sampler, pair) in self.scalar.iter_mut().zip(z) {
                    *pair = sampler.normal_pair();
                }
            }
            _ => {
                let b = self.bulk.as_mut().expect("bulk task");
                if b.row == ROWS {
                    b.sampler.fill_values(&mut b.rows);
                    b.row = 0;
                }
                for c in 0..N {
                    z[c] = [b.rows[b.row][c], b.rows[b.row + 1][c]];
                }
                b.row += 2;
            }
        }
    }
}

// ------------------------------------------------------------------------------
// Chains and the Metropolis–Hastings step.

struct Chains<const N: usize> {
    x: [f64; N],
    y: [f64; N],
    /// Untempered log density at (x, y).
    lp: [f64; N],
    beta: [f64; N],
    /// Proposal offset is (p0 z1, p1 z1 + p2 z2): scale times a Cholesky factor.
    prop: [[f64; 3]; N],
    uniform: [Xoshiro256PlusPlus; N],
    accepted: [u64; N],
}
impl<const N: usize> Chains<N> {
    /// One random-walk proposal per chain. Returns a bit mask of acceptances.
    #[inline(always)]
    fn step<const T: usize>(&mut self, z: &[[f64; 2]; N]) -> u32 {
        let mut mask = 0;
        for c in 0..N {
            let [p0, p1, p2] = self.prop[c];
            let [z1, z2] = z[c];
            let (px, py) = (self.x[c] + p0 * z1, self.y[c] + p1 * z1 + p2 * z2);
            let lp = log_density::<T>(px, py);
            let delta = self.beta[c] * (lp - self.lp[c]);
            // Uphill moves skip exp; the uniform is drawn anyway to keep streams aligned.
            let u = self.uniform[c].uniform();
            if delta >= 0.0 || u < delta.exp() {
                (self.x[c], self.y[c], self.lp[c]) = (px, py, lp);
                self.accepted[c] += 1;
                mask |= 1 << c;
            }
        }
        mask
    }
}

/// Deterministic even/odd replica exchange (non-reversible parallel tempering,
/// Syed et al. 2021) plus round-trip counting for tagged walkers.
struct Tempering<const N: usize> {
    round: u64,
    walker: [u8; N],
    /// Per walker: 0 = last touched the cold rung, 1 = the hot rung, 2 = neither yet.
    phase: [u8; N],
    trips: u64,
    tries: [u64; 3],
    swaps: [u64; 3],
}
impl<const N: usize> Tempering<N> {
    fn new() -> Self {
        Self {
            round: 0,
            walker: std::array::from_fn(|c| c as u8),
            phase: std::array::from_fn(|c| if c % 4 == 0 { 0 } else { 2 }),
            trips: 0,
            tries: [0; 3],
            swaps: [0; 3],
        }
    }
    fn exchange(&mut self, ch: &mut Chains<N>) {
        let parity = (self.round % 2) as usize;
        self.round += 1;
        for replica in 0..N / 4 {
            for rung in (parity..3).step_by(2) {
                let (a, b) = (4 * replica + rung, 4 * replica + rung + 1);
                let log_alpha = (ch.beta[a] - ch.beta[b]) * (ch.lp[b] - ch.lp[a]);
                let u = ch.uniform[a].uniform();
                self.tries[rung] += 1;
                if log_alpha >= 0.0 || u < log_alpha.exp() {
                    self.swaps[rung] += 1;
                    ch.x.swap(a, b);
                    ch.y.swap(a, b);
                    ch.lp.swap(a, b);
                    self.walker.swap(a, b);
                }
            }
            let (cold, hot) = (self.walker[4 * replica], self.walker[4 * replica + 3]);
            if self.phase[cold as usize] == 1 {
                self.trips += 1;
            }
            self.phase[cold as usize] = 0;
            // A first hot visit is not yet half of a cold -> hot -> cold trip.
            if self.phase[hot as usize] == 0 {
                self.phase[hot as usize] = 1;
            }
        }
    }
}

/// Warmup-only adaptation: Haario-style empirical covariance (Cholesky) and a
/// Robbins–Monro walk of the log scale toward the target acceptance.
#[derive(Clone, Copy)]
struct Adapt {
    n: f64,
    mean: [f64; 2],
    m2: [f64; 3],
    log_scale: f64,
    chol: [f64; 3],
    hits: u64,
    rounds: f64,
}
impl Adapt {
    fn new() -> Self {
        Self {
            n: 0.0,
            mean: [0.0; 2],
            m2: [0.0; 3],
            log_scale: (2.38 / std::f64::consts::SQRT_2).ln(),
            chol: [1.0, 0.0, 1.0],
            hits: 0,
            rounds: 0.0,
        }
    }
    fn observe(&mut self, x: f64, y: f64, accepted: bool) {
        self.hits += u64::from(accepted);
        self.n += 1.0;
        let (dx, dy) = (x - self.mean[0], y - self.mean[1]);
        self.mean[0] += dx / self.n;
        self.mean[1] += dy / self.n;
        self.m2[0] += dx * (x - self.mean[0]);
        self.m2[1] += dx * (y - self.mean[1]);
        self.m2[2] += dy * (y - self.mean[1]);
    }
    fn forget_history(&mut self) {
        (self.n, self.mean, self.m2) = (0.0, [0.0; 2], [0.0; 3]);
    }
    fn proposal(&mut self) -> [f64; 3] {
        self.rounds += 1.0;
        let rate = self.hits as f64 / ADAPT_EVERY as f64;
        self.hits = 0;
        self.log_scale += (rate - TARGET_ACCEPT) / self.rounds.powf(0.6);
        if self.n > 16.0 {
            let [sxx, sxy, syy] = self.m2.map(|m| m / (self.n - 1.0));
            let eps = 1e-9 * (sxx + syy) + f64::MIN_POSITIVE;
            let l11 = (sxx + eps).sqrt();
            let l21 = sxy / l11;
            self.chol = [l11, l21, (syy + eps - l21 * l21).max(eps).sqrt()];
        }
        self.scaled()
    }
    fn scaled(&self) -> [f64; 3] {
        self.chol.map(|l| l * self.log_scale.exp())
    }
}

// ------------------------------------------------------------------------------
// Streaming estimates: shifted sums per chain half, plus folding batch means.

struct Stats<const N: usize> {
    shift: [[f64; F]; N],
    sum: [[[f64; F]; N]; 2],
    sq: [[[f64; F]; N]; 2],
    open: [[f64; F]; N],
    slots: Box<[[[f64; SLOTS]; F]]>,
    stride: usize,
    size: u64,
    filled: usize,
    in_batch: u64,
    /// Twin only: mode side with +-1 hysteresis, switches, time at x > 0, sides seen.
    side: [i8; N],
    switches: [u64; N],
    seen: [u8; N],
}
#[derive(Clone, Copy, Default)]
struct Estimate {
    mean: f64,
    mcse: f64,
    ess: f64,
    rhat: f64,
    z: f64,
}
impl<const N: usize> Stats<N> {
    fn begin<const T: usize>(&mut self, ch: &Chains<N>) {
        for c in 0..N {
            self.shift[c] = test_functions::<T>(ch.x[c], ch.y[c]);
            self.side[c] = side(ch.x[c], 0);
            self.seen[c] = side_bit(self.side[c]);
        }
    }
    #[inline(always)]
    fn record<const T: usize, const STRIDE: usize>(&mut self, ch: &Chains<N>, half: usize) {
        for c in (0..N).step_by(STRIDE) {
            let f = test_functions::<T>(ch.x[c], ch.y[c]);
            for k in 0..F {
                let v = f[k] - self.shift[c][k];
                self.sum[half][c][k] += v;
                self.sq[half][c][k] += v * v;
                self.open[c][k] += v;
            }
            if T == TWIN {
                let s = side(ch.x[c], self.side[c]);
                self.switches[c] += u64::from(s != self.side[c] && self.side[c] != 0);
                self.side[c] = s;
                self.seen[c] |= side_bit(s);
            }
        }
        self.in_batch += 1;
        if self.in_batch == self.size {
            self.close_batch();
        }
    }
    #[cold]
    fn close_batch(&mut self) {
        for c in (0..N).step_by(self.stride) {
            for k in 0..F {
                self.slots[c / self.stride][k][self.filled] = self.open[c][k];
                self.open[c][k] = 0.0;
            }
        }
        self.in_batch = 0;
        self.filled += 1;
        if self.filled == SLOTS {
            for slots in self.slots.iter_mut().flatten() {
                for i in 0..SLOTS / 2 {
                    slots[i] = slots[2 * i] + slots[2 * i + 1];
                }
            }
            self.filled = SLOTS / 2;
            self.size *= 2;
        }
    }
}
/// Pooled estimates in global chain order, without copying or duplicating slots.
fn pooled_estimates<const N: usize>(
    tasks: &[Task<N>],
    chains: &[usize],
    n: u64,
    truth: &[f64; F],
) -> [Estimate; F] {
    let (a, nf, nh) = (chains.len() as f64, n as f64, n as f64 / 2.0);
    std::array::from_fn(|k| {
        let (mut mean, mut within, mut bm, mut half_w) = (0.0, 0.0, 0.0, 0.0);
        let mut chain_means = [0.0; C];
        let mut half_means = [0.0; 2 * C];
        for (j, &global) in chains.iter().enumerate() {
            let stats = &tasks[global / N].stats;
            let c = global % N;
            let m = (stats.sum[0][c][k] + stats.sum[1][c][k]) / nf;
            chain_means[j] = stats.shift[c][k] + m;
            within += (stats.sq[0][c][k] + stats.sq[1][c][k] - nf * m * m) / (nf - 1.0);
            for h in 0..2 {
                let mh = stats.sum[h][c][k] / nh;
                half_means[2 * j + h] = stats.shift[c][k] + mh;
                half_w += (stats.sq[h][c][k] - nh * mh * mh) / (nh - 1.0);
            }
            let (b, size) = (
                &stats.slots[c / stats.stride][k][..stats.filled],
                stats.size as f64,
            );
            let mb = b.iter().sum::<f64>() / (b.len() as f64 * size);
            let vb = b.iter().map(|s| (s / size - mb).powi(2)).sum::<f64>();
            bm += size * vb / (b.len() as f64 - 1.0);
            mean += chain_means[j];
        }
        mean /= a;
        let between = sample_variance(&chain_means[..chains.len()]);
        let total = within / a + between * (a - 1.0) / a;
        let mcse = (bm / a / (nf * a)).sqrt();
        let w = half_w / (2.0 * a);
        let b_over_n = sample_variance(&half_means[..2 * chains.len()]);
        let rhat = if w > 0.0 {
            (((nh - 1.0) / nh * w + b_over_n) / w).sqrt()
        } else if b_over_n > 0.0 {
            f64::INFINITY
        } else {
            1.0
        };
        let err = mean - truth[k];
        let z = if mcse > 0.0 {
            err / mcse
        } else if err == 0.0 {
            0.0
        } else {
            f64::INFINITY
        };
        let ess = if mcse > 0.0 {
            total / (mcse * mcse)
        } else {
            0.0
        };
        Estimate {
            mean,
            mcse,
            ess,
            rhat,
            z,
        }
    })
}
fn side(x: f64, previous: i8) -> i8 {
    if x > 1.0 {
        1
    } else if x < -1.0 {
        -1
    } else {
        previous
    }
}
fn side_bit(side: i8) -> u8 {
    match side {
        1 => 2,
        -1 => 1,
        _ => 0,
    }
}
fn sample_variance(v: &[f64]) -> f64 {
    let m = v.iter().sum::<f64>() / v.len() as f64;
    v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() as f64 - 1.0)
}

// ------------------------------------------------------------------------------
// One run: setup, warmup (adapting), frozen measurement, summary.

#[derive(Clone, Copy)]
struct Run {
    target: usize,
    pt: bool,
    source: u8,
    steps: u64,
    warmup: u64,
    seed: u64,
    workers: usize,
}
struct Outcome {
    run: Run,
    seconds: f64,
    accept: f64,
    est: [Estimate; F],
    /// Twin: fraction of chains that visited both modes, and switches per chain.
    both: f64,
    switches: f64,
    trips: u64,
    swap_rate: [f64; 3],
    allocs: usize,
    bytes: usize,
    /// Chains per task (SoA width): 2, 4 or 16, chosen by source/kernel/workers.
    width: usize,
    /// Thread startup plus warmup wall time, before the timed release.
    startup_warmup: f64,
}
impl Outcome {
    fn min_ess(&self) -> f64 {
        self.est.iter().map(|e| e.ess).fold(f64::INFINITY, f64::min)
    }
    fn worst_rhat(&self) -> f64 {
        self.est
            .iter()
            .map(|e| {
                if e.rhat.is_finite() {
                    e.rhat
                } else {
                    f64::INFINITY
                }
            })
            .fold(0.0, f64::max)
    }
    fn worst_z(&self) -> f64 {
        self.est
            .iter()
            .map(|e| {
                if e.z.is_finite() {
                    e.z.abs()
                } else {
                    f64::INFINITY
                }
            })
            .fold(0.0, f64::max)
    }
    /// Screening gate (not a convergence proof): every estimate finite with positive
    /// MCSE and ESS, |z| < 4 against the exact truth, classic split R-hat < 1.01, and
    /// pooled minimum ESS >= `MIN_ESS_PER_CHAIN` per estimating (cold) chain.
    fn passed(&self) -> bool {
        let chains = if self.run.pt { C / 4 } else { C };
        self.min_ess() >= MIN_ESS_PER_CHAIN * chains as f64
            && self.est.iter().all(|e| {
                e.mean.is_finite()
                    && e.mcse.is_finite()
                    && e.mcse > 0.0
                    && e.ess.is_finite()
                    && e.ess > 0.0
                    && e.rhat.is_finite()
                    && e.rhat > 0.0
                    && e.rhat < 1.01
                    && e.z.is_finite()
                    && e.z.abs() < 4.0
            })
    }
}

fn run(r: Run) -> Outcome {
    fn by_source<const T: usize>(r: Run) -> Outcome {
        match (r.source, r.pt) {
            (BOX_MULLER, false) => sample::<T, BOX_MULLER, false>(r),
            (ZIGGURAT, false) => sample::<T, ZIGGURAT, false>(r),
            (_, false) => sample::<T, BULK4, false>(r),
            (BOX_MULLER, true) => sample::<T, BOX_MULLER, true>(r),
            (ZIGGURAT, true) => sample::<T, ZIGGURAT, true>(r),
            (_, true) => sample::<T, BULK4, true>(r),
        }
    }
    match r.target {
        0 => by_source::<0>(r),
        1 => by_source::<1>(r),
        _ => by_source::<TWIN>(r),
    }
}
struct Task<const N: usize> {
    noise: Noise<N>,
    ch: Chains<N>,
    stats: Box<Stats<N>>,
    adapt: [Adapt; N],
    tempering: Tempering<N>,
    z: [[f64; 2]; N],
    trips_before: u64,
    tries_before: [u64; 3],
    swaps_before: [u64; 3],
}
impl<const N: usize> Task<N> {
    fn new<const T: usize, const S: u8, const PT: bool>(r: Run, offset: usize) -> Self {
        let mut rng = Xoshiro256PlusPlus::from_seed(r.seed);
        for _ in 0..offset {
            rng.jump();
        }
        let uniform = std::array::from_fn(|_| {
            let e = rng.clone();
            rng.jump();
            e
        });
        let mut normal = NormalSampler::from_seed(r.seed);
        for _ in 0..C + offset {
            normal.jump();
        }
        let scalar = std::array::from_fn(|_| {
            let e = normal.clone();
            normal.jump();
            e
        });
        let bulk = if S == BULK4 {
            assert_eq!(N, 4);
            let mut e = Xoshiro256PlusPlus::from_seed(r.seed);
            for _ in 0..2 * C + offset / 4 * NormalSampler4::SEGMENTS {
                e.jump();
            }
            Some(Box::new(BulkNoise {
                sampler: NormalSampler4::from_rng(e),
                rows: [[0.; 4]; ROWS],
                row: ROWS,
            }))
        } else {
            None
        };
        let mut start = Xoshiro256PlusPlus::from_seed(r.seed);
        for _ in 0..2 * C + (C / 4) * NormalSampler4::SEGMENTS {
            start.jump();
        }
        for _ in 0..2 * offset {
            start.uniform();
        }
        let adapt = [Adapt::new(); N];
        let mut ch = Chains {
            x: [0.; N],
            y: [0.; N],
            lp: [0.; N],
            beta: std::array::from_fn(|c| if PT { BETAS[c % 4] } else { 1. }),
            prop: [[0.; 3]; N],
            uniform,
            accepted: [0; N],
        };
        for c in 0..N {
            ch.x[c] = TARGETS[T].spread[0] * (2. * start.uniform() - 1.);
            ch.y[c] = TARGETS[T].spread[1] * (2. * start.uniform() - 1.);
            ch.lp[c] = log_density::<T>(ch.x[c], ch.y[c]);
            ch.prop[c] = adapt[c].scaled();
        }
        Self {
            noise: Noise { scalar, bulk },
            ch,
            stats: Box::new(Stats {
                shift: [[0.; F]; N],
                sum: [[[0.; F]; N]; 2],
                sq: [[[0.; F]; N]; 2],
                open: [[0.; F]; N],
                slots: vec![[[0.; SLOTS]; F]; if PT { 1 } else { N }].into_boxed_slice(),
                stride: if PT { 4 } else { 1 },
                size: 1,
                filled: 0,
                in_batch: 0,
                side: [0; N],
                switches: [0; N],
                seen: [0; N],
            }),
            adapt,
            tempering: Tempering::new(),
            z: [[0.; 2]; N],
            trips_before: 0,
            tries_before: [0; 3],
            swaps_before: [0; 3],
        }
    }
    fn warm<const T: usize, const S: u8, const PT: bool>(&mut self, r: Run) {
        for t in 0..r.warmup {
            self.noise.draw::<S>(&mut self.z);
            let mask = self.ch.step::<T>(&self.z);
            for c in 0..N {
                self.adapt[c].observe(self.ch.x[c], self.ch.y[c], mask & (1 << c) != 0);
            }
            if PT && t % SWAP_EVERY == SWAP_EVERY - 1 {
                self.tempering.exchange(&mut self.ch);
            }
            if t == r.warmup / 4 {
                self.adapt.iter_mut().for_each(Adapt::forget_history);
            }
            if t % ADAPT_EVERY == ADAPT_EVERY - 1 {
                for c in 0..N {
                    self.ch.prop[c] = self.adapt[c].proposal();
                }
            }
        }
        self.ch.accepted = [0; N];
        self.trips_before = self.tempering.trips;
        self.tries_before = self.tempering.tries;
        self.swaps_before = self.tempering.swaps;
        self.stats.begin::<T>(&self.ch);
    }
    fn measure<const T: usize, const S: u8, const PT: bool>(&mut self, r: Run) {
        for t in 0..r.steps {
            self.noise.draw::<S>(&mut self.z);
            self.ch.step::<T>(&self.z);
            if PT && t % SWAP_EVERY == SWAP_EVERY - 1 {
                self.tempering.exchange(&mut self.ch);
            }
            let half = usize::from(t >= r.steps / 2);
            if PT {
                self.stats.record::<T, 4>(&self.ch, half);
            } else {
                self.stats.record::<T, 1>(&self.ch, half);
            }
        }
    }
    fn payload(&self) -> usize {
        size_of::<Self>()
            + size_of::<Stats<N>>()
            + self.stats.slots.len() * size_of::<[[f64; SLOTS]; F]>()
            + if self.noise.bulk.is_some() {
                size_of::<BulkNoise>()
            } else {
                0
            }
    }
}
/// Start gate: workers arrive after warmup and wait for one release. Any failure
/// aborts it, so nobody waits forever on a dead peer.
struct Gate {
    state: Mutex<GateState>,
    cv: Condvar,
}
struct GateState {
    ready: usize,
    go: bool,
    error: Option<String>,
}
impl Gate {
    fn new() -> Self {
        Self {
            state: Mutex::new(GateState {
                ready: 0,
                go: false,
                error: None,
            }),
            cv: Condvar::new(),
        }
    }
    fn lock(&self) -> MutexGuard<'_, GateState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
    fn wait<'a>(&self, g: MutexGuard<'a, GateState>) -> MutexGuard<'a, GateState> {
        self.cv.wait(g).unwrap_or_else(|e| e.into_inner())
    }
    /// Records the first failure and wakes every waiter.
    fn abort(&self, message: String) {
        self.lock().error.get_or_insert(message);
        self.cv.notify_all();
    }
    fn error(&self) -> Option<String> {
        self.lock().error.clone()
    }
    /// Worker side: announce readiness, then wait for release or abort.
    fn arrive_and_wait(&self) -> Result<(), String> {
        let mut g = self.lock();
        g.ready += 1;
        self.cv.notify_all();
        while !g.go && g.error.is_none() {
            g = self.wait(g);
        }
        g.error.clone().map_or(Ok(()), Err)
    }
    /// Coordinator side: wait for `workers` arrivals (or an abort), then release.
    fn release_when_ready(&self, workers: usize) -> Result<Instant, String> {
        let mut g = self.lock();
        while g.ready < workers && g.error.is_none() {
            g = self.wait(g);
        }
        if let Some(e) = &g.error {
            return Err(e.clone());
        }
        g.go = true;
        self.cv.notify_all();
        Ok(Instant::now())
    }
}
struct WorkerEnd {
    end: Instant,
    allocs: usize,
}
/// Proves this thread's allocation counter intercepts, then zeroes it.
fn allocation_control() {
    LOCAL_ALLOCS.set(0);
    LOCAL_ARMED.set(true);
    drop(std::hint::black_box(Box::new(0_u64)));
    LOCAL_ARMED.set(false);
    assert!(
        LOCAL_ALLOCS.get() >= 2,
        "worker allocation counter failed its control"
    );
    LOCAL_ALLOCS.set(0);
}
/// One worker: audited warmup, the start gate, audited measurement. A failure or
/// panic aborts the gate for everyone.
fn worker<const T: usize, const S: u8, const PT: bool, const N: usize>(
    group: &mut [Task<N>],
    gate: &Gate,
    r: Run,
) -> Result<WorkerEnd, String> {
    let body = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<_, String> {
        allocation_control();
        LOCAL_ARMED.set(true);
        for task in group.iter_mut() {
            task.warm::<T, S, PT>(r);
        }
        LOCAL_ARMED.set(false);
        gate.arrive_and_wait()?;
        LOCAL_ARMED.set(true);
        for task in group.iter_mut() {
            task.measure::<T, S, PT>(r);
        }
        let end = Instant::now();
        LOCAL_ARMED.set(false);
        Ok(WorkerEnd {
            end,
            allocs: LOCAL_ALLOCS.get(),
        })
    }));
    LOCAL_ARMED.set(false);
    let result = body.unwrap_or_else(|_| Err("MCMC worker panicked".into()));
    if let Err(e) = &result {
        gate.abort(e.clone());
    }
    result
}
fn sample<const T: usize, const S: u8, const PT: bool>(r: Run) -> Outcome {
    if PT || S == BULK4 {
        sample_tasks::<T, S, PT, 4>(r)
    } else if S == ZIGGURAT && r.workers == 1 {
        // Preserve the fast serial SoA width for cheap scalar proposal noise.
        sample_tasks::<T, S, PT, C>(r)
    } else {
        sample_tasks::<T, S, PT, 2>(r)
    }
}
fn sample_tasks<const T: usize, const S: u8, const PT: bool, const N: usize>(r: Run) -> Outcome {
    let mut tasks: Vec<_> = (0..C / N)
        .map(|j| Task::<N>::new::<T, S, PT>(r, j * N))
        .collect();
    let bytes = tasks.iter().map(Task::payload).sum::<usize>();
    let gate = Gate::new();
    let startup = Instant::now();
    let execution = thread::scope(|scope| -> Result<(f64, f64, usize), String> {
        let mut handles = Vec::with_capacity(r.workers);
        for group in tasks.chunks_mut((C / N) / r.workers) {
            let gate = &gate;
            let spawned = thread::Builder::new()
                .stack_size(STACK)
                .spawn_scoped(scope, move || worker::<T, S, PT, N>(group, gate, r));
            match spawned {
                Ok(h) => handles.push(h),
                Err(e) => {
                    gate.abort(e.to_string());
                    break;
                }
            }
        }
        let release = gate.release_when_ready(r.workers);
        let mut ends = Vec::with_capacity(r.workers);
        for h in handles {
            ends.push(h.join().map_err(|_| "worker join panicked".to_string())?);
        }
        let release = release?;
        let ends = ends.into_iter().collect::<Result<Vec<_>, _>>()?;
        if let Some(e) = gate.error() {
            return Err(e);
        }
        let last = ends
            .iter()
            .map(|e| e.end)
            .max()
            .ok_or("no worker results")?;
        Ok((
            (last - release).as_secs_f64(),
            (release - startup).as_secs_f64(),
            ends.iter().map(|e| e.allocs).sum(),
        ))
    });
    let (seconds, startup_warmup, allocs) =
        execution.unwrap_or_else(|e| panic!("MCMC execution failed: {e}"));
    let cold: Vec<_> = (0..C).step_by(if PT { 4 } else { 1 }).collect();
    let est = pooled_estimates(&tasks, &cold, r.steps, &TARGETS[T].truth);
    let both = cold
        .iter()
        .filter(|&&c| tasks[c / N].stats.seen[c % N] == 3)
        .count() as f64
        / cold.len() as f64;
    let switches = cold
        .iter()
        .map(|&c| tasks[c / N].stats.switches[c % N])
        .sum::<u64>() as f64
        / cold.len() as f64;
    let accept =
        tasks.iter().flat_map(|t| t.ch.accepted).sum::<u64>() as f64 / (C as f64 * r.steps as f64);
    let trips = tasks
        .iter()
        .map(|t| t.tempering.trips - t.trips_before)
        .sum();
    let swap_rate = std::array::from_fn(|i| {
        let tried: u64 = tasks
            .iter()
            .map(|t| t.tempering.tries[i] - t.tries_before[i])
            .sum();
        let swapped: u64 = tasks
            .iter()
            .map(|t| t.tempering.swaps[i] - t.swaps_before[i])
            .sum();
        swapped as f64 / tried.max(1) as f64
    });
    drop(tasks);
    Outcome {
        run: r,
        seconds,
        accept,
        est,
        both,
        switches,
        trips,
        swap_rate,
        allocs,
        bytes,
        width: N,
        startup_warmup,
    }
}
fn supported_workers(pt: bool, source: u8) -> &'static [usize] {
    if pt || source == BULK4 {
        &[1, 2, 4]
    } else {
        &[1, 2, 4, 8]
    }
}
fn worker_counts(o: &Options, pt: bool, source: u8) -> Vec<usize> {
    match o.workers {
        Some(w) => vec![w],
        None => {
            let available = thread::available_parallelism().map_or(1, |n| n.get());
            supported_workers(pt, source)
                .iter()
                .copied()
                .filter(|&w| w <= available)
                .collect()
        }
    }
}

// ------------------------------------------------------------------------------
// Command line, benchmark matrix and reports.

struct Options {
    targets: Vec<usize>,
    kernels: Vec<bool>,
    sources: Vec<u8>,
    steps: u64,
    warmup: u64,
    seeds: Vec<u64>,
    rounds: usize,
    workers: Option<usize>,
}
fn parse() -> Result<Options, String> {
    let mut o = Options {
        targets: vec![0, 1, 2],
        kernels: vec![false, true],
        sources: vec![0, 1, 2],
        steps: 4_000_000,
        warmup: 400_000,
        seeds: vec![73, 1_234_567, 987_654_321],
        rounds: 1,
        workers: Some(1),
    };
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--quick" {
            (o.steps, o.warmup, o.seeds) = (200_000, 50_000, vec![73]);
            continue;
        }
        let value = args.next().ok_or(format!("{flag} needs a value"))?;
        let number = || value.parse::<u64>().map_err(|e| format!("{flag}: {e}"));
        let pick = |names: &[&str]| -> Result<Vec<usize>, String> {
            if value == "all" {
                return Ok((0..names.len()).collect());
            }
            let i = names.iter().position(|n| *n == value);
            Ok(vec![i.ok_or(format!("{flag}: unknown {value}"))?])
        };
        match flag.as_str() {
            "--target" => o.targets = pick(&TARGETS.map(|t| t.name))?,
            "--kernel" => o.kernels = pick(&["rwm", "pt"])?.iter().map(|&k| k == 1).collect(),
            "--source" => o.sources = pick(&SOURCES)?.iter().map(|&s| s as u8).collect(),
            "--steps" => o.steps = number()?,
            "--warmup" => o.warmup = number()?,
            "--seed" => o.seeds = vec![number()?],
            "--rounds" => o.rounds = number()? as usize,
            "--workers" => {
                o.workers = if value == "all" {
                    None
                } else {
                    Some(number()? as usize)
                }
            }
            _ => return Err(format!("unknown flag {flag}")),
        }
    }
    if o.steps < 1024 || o.steps % 2 == 1 || o.rounds == 0 {
        return Err("--steps must be even and >= 1024; --rounds >= 1".into());
    }
    for &pt in &o.kernels {
        for &source in &o.sources {
            if let Some(w) = o.workers
                && !supported_workers(pt, source).contains(&w)
            {
                return Err(format!(
                    "unsupported workers={w} for {}/{}, supported {:?}",
                    kernel_name(pt),
                    SOURCES[source as usize],
                    supported_workers(pt, source)
                ));
            }
        }
    }
    Ok(o)
}

fn main() {
    let o = parse().unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(2);
    });
    // Positive control: the allocation counter must see an allocation.
    ARMED.store(true, Relaxed);
    drop(std::hint::black_box(Box::new(0_u64)));
    ARMED.store(false, Relaxed);
    assert!(
        ALLOCS.load(Relaxed) >= 2,
        "allocation counter failed its control"
    );

    println!(
        "target,kernel,source,seed,round,seconds,mproposals_s,accept,min_ess,min_ess_s,\
         worst_rhat,worst_z,pass,both_modes,switches,round_trips,swap01,swap12,swap23,bytes,workers,startup_warmup_seconds,task_width"
    );
    let mut all = Vec::new();
    for round in 0..o.rounds {
        for (seed_index, &seed) in o.seeds.iter().enumerate() {
            for &target in &o.targets {
                for &pt in &o.kernels {
                    // Rotate source order between rounds and seeds to spread drift.
                    let shift = round + seed_index;
                    for i in 0..o.sources.len() {
                        let source = o.sources[(i + shift) % o.sources.len()];
                        for workers in worker_counts(&o, pt, source) {
                            let r = Run {
                                target,
                                pt,
                                source,
                                steps: o.steps,
                                warmup: o.warmup,
                                seed,
                                workers,
                            };
                            let out = run(r);
                            assert_eq!(out.allocs, 0, "heap traffic during sampling");
                            print_row(&out, round);
                            all.push(out);
                        }
                    }
                }
            }
        }
    }
    summarize(&o, &all);
}

fn kernel_name(pt: bool) -> &'static str {
    if pt { "pt" } else { "rwm" }
}

fn print_row(o: &Outcome, round: usize) {
    let r = o.run;
    let proposals = (C as u64 * r.steps) as f64;
    println!(
        "{},{},{},{},{},{:.6},{:.3},{:.4},{:.1},{:.1},{:.5},{:.3},{},{:.3},{:.1},{},{:.3},{:.3},{:.3},{},{},{:.6},{}",
        TARGETS[r.target].name,
        kernel_name(r.pt),
        SOURCES[r.source as usize],
        r.seed,
        round,
        o.seconds,
        proposals / o.seconds / 1e6,
        o.accept,
        o.min_ess(),
        o.min_ess() / o.seconds,
        o.worst_rhat(),
        o.worst_z(),
        if o.passed() { "yes" } else { "NO" },
        o.both,
        o.switches,
        o.trips,
        o.swap_rate[0],
        o.swap_rate[1],
        o.swap_rate[2],
        o.bytes,
        r.workers,
        o.startup_warmup,
        o.width,
    );
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

fn summarize(o: &Options, all: &[Outcome]) {
    let group = |t: usize, pt: bool, s: u8, workers: usize| -> Vec<&Outcome> {
        all.iter()
            .filter(|x| {
                x.run.target == t && x.run.pt == pt && x.run.source == s && x.run.workers == workers
            })
            .collect()
    };
    eprintln!(
        "\n{} measured steps x {C} chains per run, {} warmup; medians over {} run(s) per row.",
        o.steps,
        o.warmup,
        o.rounds * o.seeds.len()
    );
    eprintln!(
        "ESS/s is a performance ranking only when pass=yes; requires finite diagnostics, exact-truth checks, classic split R-hat <1.01 and >=50 ESS per split chain on average."
    );

    eprintln!(
        "{:<7}{:<5}{:<11}{:>9}{:>8}{:>12}{:>10}{:>8}{:>7}  extra",
        "target", "kern", "source", "Mprop/s", "accept", "minESS/s", "R-hat", "|z|", "pass"
    );
    for &t in &o.targets {
        for &pt in &o.kernels {
            for &s in &o.sources {
                for workers in worker_counts(o, pt, s) {
                    let g = group(t, pt, s, workers);
                    if g.is_empty() {
                        continue;
                    }
                    let med =
                        |f: &dyn Fn(&Outcome) -> f64| median(g.iter().map(|x| f(x)).collect());
                    let passed = g.iter().filter(|x| x.passed()).count();
                    let mut extra = format!("workers={workers}");
                    let serial = group(t, pt, s, 1);
                    if serial.len() == g.len() && !serial.is_empty() {
                        let speedup = median(
                            g.iter()
                                .zip(&serial)
                                .map(|(x, base)| base.seconds / x.seconds)
                                .collect(),
                        );
                        extra += &format!(
                            "; timing speedup {speedup:.2}x, efficiency {:.1}%",
                            100.0 * speedup / workers as f64
                        );
                    }
                    if t == TWIN {
                        extra += &format!(
                            "; both-modes {:.2}, switches/chain {:.0}",
                            med(&|x| x.both),
                            med(&|x| x.switches)
                        );
                    }
                    if pt {
                        extra += &format!(
                            "; round trips {:.0}, swap {:.2}/{:.2}/{:.2}",
                            med(&|x| x.trips as f64),
                            med(&|x| x.swap_rate[0]),
                            med(&|x| x.swap_rate[1]),
                            med(&|x| x.swap_rate[2]),
                        );
                    }
                    eprintln!(
                        "{:<7}{:<5}{:<11}{:>9.1}{:>8.3}{:>12.0}{:>10.4}{:>8.2}{:>7}  {extra}",
                        TARGETS[t].name,
                        kernel_name(pt),
                        SOURCES[s as usize],
                        med(&|x| (C as u64 * x.run.steps) as f64 / x.seconds / 1e6),
                        med(&|x| x.accept),
                        med(&|x| x.min_ess() / x.seconds),
                        g.iter().map(|x| x.worst_rhat()).fold(0.0, f64::max),
                        g.iter().map(|x| x.worst_z()).fold(0.0, f64::max),
                        format!("{passed}/{}", g.len()),
                    );
                }
            }
        }
    }
    // Truth table for the first valid run of each target/kernel (bulk4 if present).
    eprintln!("\nEstimates vs exact truth (bulk4 when run, else first source), pooled ESS:");
    for &t in &o.targets {
        for &pt in &o.kernels {
            let Some(x) = all.iter().find(|x| {
                x.run.target == t
                    && x.run.pt == pt
                    && (x.run.source == BULK4 || !o.sources.contains(&BULK4))
            }) else {
                continue;
            };
            let target = &TARGETS[t];
            let cells: Vec<String> = (0..F)
                .map(|k| {
                    let e = x.est[k];
                    format!(
                        "{}={:.4}+-{:.4} (truth {:.4}, z {:+.2}, ESS {:.0})",
                        target.funcs[k], e.mean, e.mcse, target.truth[k], e.z, e.ess
                    )
                })
                .collect();
            eprintln!(
                "  {} {} workers={}: {}",
                target.name,
                kernel_name(pt),
                x.run.workers,
                cells.join("; ")
            );
        }
    }
    let bytes = all.iter().map(|x| x.bytes).max().unwrap_or(0);
    eprintln!(
        "\nLargest per-run accounted sampler + diagnostics payload (excludes coordinator/allocator/CRT overhead): {bytes} bytes; heap calls while sampling: 0."
    );
}
