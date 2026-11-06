use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};
use tiny_normal::{NormalSampler, NormalSampler4, Xoshiro256PlusPlus};

struct Audit;
static ACTIVE: AtomicBool = AtomicBool::new(false);
static CALLS: AtomicUsize = AtomicUsize::new(0);
#[global_allocator]
static ALLOCATOR: Audit = Audit;

thread_local! {
    static GENERATING: Cell<bool> = const { Cell::new(false) };
    static THREAD_CALLS: Cell<usize> = const { Cell::new(0) };
}
fn record() {
    if ACTIVE.load(Relaxed) {
        CALLS.fetch_add(1, Relaxed);
    }
    let _ = GENERATING.try_with(|active| {
        if active.get() {
            THREAD_CALLS.with(|n| n.set(n.get() + 1));
        }
    });
}
// SAFETY: every operation forwards the unchanged allocator contract to System.
unsafe impl GlobalAlloc for Audit {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record();
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        record();
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record();
        unsafe { System.realloc(ptr, layout, size) }
    }
}

fn main() {
    assert_eq!(size_of::<NormalSampler>(), 32);
    assert_eq!(size_of::<NormalSampler4>(), 256);
    assert!(!std::mem::needs_drop::<NormalSampler4>());
    assert_eq!(size_of::<Xoshiro256PlusPlus>(), 32);
    assert!(!std::mem::needs_drop::<NormalSampler>());
    assert!(!std::mem::needs_drop::<Xoshiro256PlusPlus>());
    ACTIVE.store(true, Relaxed);
    let probe = std::hint::black_box(Box::new(0_u64));
    drop(probe);
    ACTIVE.store(false, Relaxed);
    assert!(
        CALLS.load(Relaxed) >= 2,
        "allocator counter failed its positive control"
    );
    CALLS.store(0, Relaxed);
    let mut out = [[0.0; 2]; 257];
    ACTIVE.store(true, Relaxed);
    let (seeds, steps) = if cfg!(miri) { (2, 16) } else { (32, 4096) };
    for seed in 0..seeds {
        let mut s = NormalSampler::from_seed(seed);
        s.jump();
        for _ in 0..steps {
            std::hint::black_box(s.box_muller_pair());
            std::hint::black_box(s.normal_pair());
        }
        s.fill_pairs(&mut out);
        std::hint::black_box(&out);
    }
    let mut rows = [[0.0; 4]; 257];
    for seed in 0..seeds {
        let mut s = NormalSampler4::from_seed(seed);
        for _ in 0..if cfg!(miri) { 1 } else { 32 } {
            s.fill_values(&mut rows);
            std::hint::black_box(&rows);
        }
        std::hint::black_box(s);
    }
    ACTIVE.store(false, Relaxed);
    assert_eq!(CALLS.load(Relaxed), 0);
    std::thread::scope(|scope| {
        for seed in 0..4 {
            scope.spawn(move || {
                THREAD_CALLS.set(0);
                GENERATING.set(true);
                let probe = std::hint::black_box(Box::new(1_u64));
                drop(probe);
                GENERATING.set(false);
                assert!(THREAD_CALLS.get() >= 2);
                THREAD_CALLS.set(0);
                let mut rng = Xoshiro256PlusPlus::from_seed(seed);
                rng.jump();
                let mut sampler = NormalSampler4::from_rng(rng);
                let mut out = [[0.; 4]; 128];
                GENERATING.set(true);
                for _ in 0..if cfg!(miri) { 1 } else { 1024 } {
                    sampler.fill_values(&mut out);
                    std::hint::black_box(&out);
                }
                GENERATING.set(false);
                assert_eq!(THREAD_CALLS.get(), 0);
            });
        }
    });
    println!(
        "scalar=32 bytes; four-stream=256 bytes; allocator calls during generation=0; four worker TLS audits passed"
    );
}
