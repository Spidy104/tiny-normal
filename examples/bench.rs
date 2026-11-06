use std::{hint::black_box, time::Instant};
use tiny_normal::{NormalSampler, NormalSampler4};
fn run<const METHOD: usize, const MODE: usize>(n: usize) -> f64 {
    let seed = std::env::var("TINY_BENCH_SEED").map_or(42, |s| s.parse().expect("integer seed"));
    let mut sampler = NormalSampler::from_seed(seed);
    let mut streams = NormalSampler4::from_seed(seed);
    let mut out = vec![[0.0; 2]; if MODE == 2 { 262144 } else { 256 }];
    let mut sum = [0.0; 2];
    let start = Instant::now();
    for _ in 0..n {
        if METHOD == 2 && MODE != 1 {
            streams.fill_values(out.as_flattened_mut().as_chunks_mut::<4>().0);
        } else if METHOD == 1 && MODE != 1 {
            sampler.fill_pairs(&mut out);
        } else {
            for p in &mut out {
                let pair = match METHOD {
                    0 => sampler.box_muller_pair(),
                    _ => sampler.normal_pair(),
                };
                if MODE == 1 {
                    sum[0] += pair[0];
                    sum[1] += pair[1];
                } else {
                    *p = pair;
                }
            }
        }
        black_box(&out);
    }
    black_box(sampler);
    black_box(streams);
    black_box(sum);
    start.elapsed().as_secs_f64()
}
fn bench<const MODE: usize>() {
    let funcs = [run::<0, MODE>, run::<1, MODE>, run::<2, MODE>];
    let names = ["box-muller", "ziggurat", "streams4"];
    let n = if MODE == 2 { 512 } else { 524288 };
    let rounds: usize =
        std::env::var("TINY_BENCH_ROUNDS").map_or(7, |s| s.parse().expect("integer rounds"));
    assert!((1..=7).contains(&rounds));
    let method = std::env::var("TINY_BENCH_METHOD").unwrap_or_default();
    assert!(
        ["", "bulk", "box-muller", "ziggurat", "streams4"].contains(&method.as_str()),
        "unknown TINY_BENCH_METHOD: {method}"
    );
    assert!(
        MODE != 1 || !matches!(method.as_str(), "streams4" | "bulk"),
        "bulk comparison requires fill or stream mode"
    );
    let selected = |i: usize| {
        (MODE != 1 || i != 2)
            && match method.as_str() {
                "" => true,
                "bulk" => i != 0,
                name => name == names[i],
            }
    };
    for (i, f) in funcs.iter().enumerate() {
        if selected(i) {
            f(if MODE == 2 { 1 } else { 512 });
        }
    }
    let mut times = [[0.0; 7]; 3];
    for round in [0, 1, 2, 3, 4, 5, 6] {
        if round >= rounds {
            break;
        }
        for j in 0..3 {
            let i = (round + j) % 3;
            if !selected(i) {
                continue;
            }
            times[i][round] = funcs[i](n);
        }
    }
    for i in 0..3 {
        if !selected(i) {
            continue;
        }
        times[i][..rounds].sort_by(f64::total_cmp);
        println!(
            "{} median={:.6}s range={:.6}..{:.6}s Mvalues/s={:.3}",
            names[i],
            times[i][rounds / 2],
            times[i][0],
            times[i][rounds - 1],
            268435456.0 / times[i][rounds / 2] / 1e6
        );
    }
}
fn main() {
    match std::env::var("TINY_BENCH_MODE")
        .as_deref()
        .unwrap_or("fill")
    {
        "fill" => bench::<0>(),
        "pair" => bench::<1>(),
        "stream" => bench::<2>(),
        other => panic!("unknown TINY_BENCH_MODE: {other}"),
    }
}
