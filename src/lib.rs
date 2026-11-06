//! Small, allocation-free standard-normal sampling.
//!
//! - [`Xoshiro256PlusPlus`]: SplitMix64-seeded xoshiro256++ with 2^128 jumps.
//! - [`NormalSampler`]: 32 bytes of state; a 128-slot modified undergraph Ziggurat
//!   (exact packed alias, chord-triangle overhangs) plus a Box–Muller reference.
//! - [`NormalSampler4`]: four jump-separated streams in 256 bytes, filled by a
//!   branch-free AVX2 kernel when available. Every backend is bit-identical.
//!
//! Tables total 3,264 bytes and are shared. Generation never allocates. The
//! generator is not cryptographic.
//!
//! ```
//! use tiny_normal::{NormalSampler, NormalSampler4};
//!
//! let mut s = NormalSampler::from_seed(42);
//! let [a, b] = s.normal_pair();
//! assert!(a.is_finite() && b.is_finite());
//!
//! let mut bulk = NormalSampler4::from_seed(42);
//! let mut rows = [[0.0; 4]; 128]; // one value per stream per row
//! bulk.fill_values(&mut rows);
//! ```
#![deny(unsafe_code, unsafe_op_in_unsafe_fn, missing_docs)]

#[allow(clippy::excessive_precision)]
mod tables {
    // 126 equal-area rectangles in 128 index slots (MX has one padding entry). Independently generated
    // at 70 decimal digits; geometry and alias validation are recorded in plan.md.
    // ALIAS packs ceil(probability * 2^53) above an 8-bit region index.
    #[rustfmt::skip]
    pub(super) static MX: [f64; 128] = [
        3.4224543379391648, 3.203053018561586, 3.065336759857423, 2.9626600760731963,
        2.879811646023978, 2.809821891610007, 2.7488887066234917, 2.6946995494370993,
        2.6457383604039975, 2.600955011039593, 2.5595907532673716, 2.5210787275315645,
        2.484983768010903, 2.4509641941167075, 2.4187465816177074, 2.3881085441502057,
        2.358866646543615, 2.3308677123995043, 2.3039824394462323, 2.2781006222124245,
        2.253127518175781, 2.2289810428768138, 2.205589576205124, 2.1828902261688135,
        2.1608274398330973, 2.1393518810182877, 2.118419515316837, 2.097990857922785,
        2.0780303505538695, 2.0585058416403053, 2.039388149800631, 2.020650695003432,
        2.0022691851267775, 1.984221348158309, 1.9664867022299428, 1.9490463571977805,
        1.931882842665994, 1.914979958291196, 1.8983226429490707, 1.8818968599411712,
        1.8656894958996089, 1.8496882714357832, 1.8338816618954852, 1.8182588268414193,
        1.802809547096924, 1.7875241683604055, 1.7723935505458106, 1.757409022125955,
        1.7425623388571356, 1.7278456463487493, 1.713251446013499, 1.6987725639944653,
        1.6844021227167663, 1.670133514755225, 1.6559603787466532, 1.6418765771070862,
        1.627876175341378, 1.6139534227557015, 1.6001027344032588, 1.5863186741103519,
        1.5725959384443156, 1.5589293414969558, 1.5453138003673605, 1.5317443212364694,
        1.5182159859327615, 1.5047239388940088, 1.491263374434344, 1.4778295242289805,
        1.4644176449308623, 1.451023005834341, 1.43764087650066, 1.424266514258601,
        1.410895151490995, 1.3975219826139353, 1.3841421506502762, 1.3707507332922635,
        1.3573427283397508, 1.3439130383901559, 1.3304564546438782, 1.316967639673956,
        1.303441108990901, 1.2898712112123905, 1.2762521066221946, 1.2625777438726054,
        1.2488418345487544, 1.2350378252703784, 1.2211588669553535, 1.207197780807828,
        1.1931470205197463, 1.1789986300850617, 1.164744196517321, 1.150374796628854,
        1.1358809368675065, 1.1212524850069443, 1.1064785922388845, 1.0915476039069412,
        1.0764469567345818, 1.0611630599106547, 1.0456811567737305, 1.0299851630386077,
        1.0140574764765276, 0.9978787516143067, 0.9814276312437638, 0.9646804241713,
        0.9476107154579937, 0.9301888910672852, 0.9123815528528005, 0.8941507914335323,
        0.875453272566826, 0.856239075338567, 0.8364501949555673, 0.8160185844004639,
        0.7948635496744424, 0.7728882188267926, 0.7499746502007633, 0.725976882818071,
        0.7007107679744493, 0.6739385610684355, 0.6453445635676288, 0.6144945464906132,
        0.580763501790702, 0.5431951724071683, 0.5001933197390409, 0.44870442926391,
        0.38120216911170124, 0.25169140223915126, 0.0,
        0.0, // index 127 is never a strip; it keeps masked SIMD lookups in bounds
    ];
    #[rustfmt::skip]
    pub(super) static MF: [f64; 127] = [
        0.0028609634288571167, 0.005917895874569327, 0.009112166998709297, 0.012417141733021487,
        0.015817196285650315, 0.019301942677232765, 0.022863933628043006, 0.026497554563565378,
        0.030198418127520433, 0.03396300331112333, 0.037788426060943105, 0.04167228600980095,
        0.04561255991574076, 0.049607525130361554, 0.0536557031297403, 0.05775581688787,
        0.06190675806639088, 0.06610756133192562, 0.07035738395649857, 0.07465548940553185,
        0.07900123398447381, 0.08339405586560565, 0.08783346599132105, 0.09231904047435972,
        0.09685041420521795, 0.10142727544279836, 0.10604936121333702, 0.11071645337954021,
        0.11542837526998206, 0.12018498878047619, 0.12498619187599018, 0.1298319164349104,
        0.13472212638795295, 0.139656816112396, 0.14463600904905508, 0.14965975651489524,
        0.15472813668864588, 0.1598412537504634, 0.16499923715973577, 0.1702022410576664,
        0.17545044378341312, 0.1807440474943654, 0.18608327788268658, 0.19146838398157276,
        0.19689963805582852, 0.20237733557236534, 0.20790179524711536, 0.21347335916564608,
        0.2190923929754776, 0.2247592861487588, 0.23047445231456642, 0.23623832966066258,
        0.2420513814050908, 0.2479140963385208, 0.25382698943877013, 0.25979060255945274,
        0.2658055051952285, 0.271872295326668, 0.2779916003483103, 0.2841640780840771,
        0.2903904178948384, 0.29667134188359023, 0.3030076062044354, 0.3094000024823425,
        0.3158493593515245, 0.32235654412122666, 0.32892246457876584, 0.33554807094082967,
        0.3422343579653483, 0.348982367237708, 0.3557931896467165, 0.3626679680675757,
        0.3696079002712054, 0.3766142420816313, 0.3836883108058423, 0.39083148896359543,
        0.3980452283481604, 0.4053310544530298, 0.41269057130425835, 0.42012546674345186,
        0.4276375182126262, 0.43522859909935596, 0.44290068570902447, 0.4506558649407956,
        0.45849634275543844, 0.4664244535366882, 0.4744426704638518, 0.48255361703237937,
        0.49076007988178755, 0.49906502311744755, 0.5074716043453728, 0.5159831926785587,
        0.5246033890213022, 0.5333360489963807, 0.5421853089517539, 0.5511556155721257,
        0.5602517597309424, 0.5694789153563953, 0.5788426842589424, 0.5883491480888591,
        0.5980049288753675, 0.6078172599646279, 0.6177940696509097, 0.6279440804238438,
        0.638276927592045, 0.648803302172312, 0.6595351244753404, 0.6704857569543199,
        0.6816702678866609, 0.6931057617572316, 0.704811798478638, 0.7168109329162087,
        0.7291294204123177, 0.7417981562776933, 0.7548539531571695, 0.7683413202201109,
        0.7823150125697829, 0.7968438097976107, 0.8120163505619543, 0.8279506119051946,
        0.8448103433924292, 0.8628361234345939, 0.8824115881826752, 0.9042333400067882,
        0.9299192292570884, 0.9688220944748429, 1.0,
    ];
    #[rustfmt::skip]
    pub(super) static ALIAS: [u64; 128] = [
        2305843009213693952, 2122527244513342976, 1259320191710396161, 1832458663618814722,
        2047241350395324672, 1784538723044567808, 1595428041511826688, 1452282414428583168,
        1339905796237251328, 1249210899263492608, 1174412224565039616, 1111642193296922880,
        1058212151246271744, 1012192009357537792, 972158541703887361, 937038115854236674,
        906004831498073461, 878412485672346743, 853747904793271672, 831598178604799608,
        811627177055501433, 793558409810097273, 777162309031533690, 762246654054755962,
        748649265056172411, 736232360143460731, 724878148727949691, 714485355313317755,
        704966451621094268, 696245433742193276, 688256022801478524, 680940197734165884,
        674246990725702524, 668131492049409660, 662554023089037948, 657479445397338237,
        652876580519900797, 648717720577449853, 644978213660717693, 641636111248615037,
        638671867331685757, 636068080870767741, 633809274766190461, 631881705746544253,
        630273200576836989, 628973014786180989, 627971710765302397, 627261052615313789,
        626833915565538429, 626684208138714493, 626806805541570429, 627197493009384061,
        627852918044165245, 628770550665107069, 629948650943057277, 631386243222953085,
        633083096553548669, 635039710945733757, 637257309172152190, 639737833904138366,
        642483950059287934, 645499052306173054, 648787277743488126, 652353523840900222,
        656203471799597438, 660343615563560318, 664781296789489790, 669524746165827710,
        674583131561244286, 679966613582410110, 685686409232166270, 691754864485081470,
        698185536741033342, 704993288282632574, 712194392053547902, 719806651297518462,
        727849534858606974, 736344330250099070, 745314316962153342, 754784962908961406,
        764784147429686654, 775342414872377982, 786493263529552766, 798273475587304574,
        810723494833251198, 823887860188921470, 837815704748518270, 852561331994392446,
        868184883317956990, 884753114030288254, 902340298865218686, 921029292777534334,
        940912778908701566, 962094743317627518, 984692225972108670, 1008837410269800574,
        1034680129965610878, 1062390894150230142, 1092164559698902398, 1124224818992100990,
        1158829722406813054, 1196278525432282750, 1236920247081255294, 1281164461102322558,
        1329495031716686974, 1382487777723961982, 1440833444057283710, 1505367943479212926,
        1577112708598787710, 1657329340217270910, 1747594847951019390, 1849907168625740926,
        1966836243587651710, 2101745461544228734, 2259125063373850494, 1698254672154404355,
        1558987777326679923, 1196517651803409524, 1960971467351195765, 984109915574055542,
        994749935164100727, 1915429494348813432, 2085631256403679353, 1056814952010260602,
        1076028212633705339, 1388100991438483836, 1833580968743658621, 126,
    ];
    // Concave strips 102..=126: round(2^53 * chord-triangle area / region area),
    // integrated at 80 digits over the rounded endpoints above.
    #[rustfmt::skip]
    pub(super) static TRI: [u64; 25] = [
        9006685218851363, 9005825455842477, 9004900375179035, 9003900078849697,
        9002812699882333, 9001623883222521, 9000316093840889, 8998867681041645,
        8997251592483506, 8995433574697400, 8993369603702915, 8991002131611266,
        8988254458970731, 8985022040051110, 8981158571729540, 8976452798037702,
        8970587866545311, 8963065641977728, 8953054527497355, 8939051247195772,
        8918018169575116, 8882681406084538, 8809428502089168, 8485862652037653,
        6776880568828748,
    ];
}
use tables::{ALIAS, MF, MX, TRI};
/// Strips 1..CROSSING lie at x >= 1 (convex density); CROSSING contains x = 1;
/// strips above it lie at x <= 1 (concave density).
const CROSSING: usize = 101;
/// 53-bit grid scale and its integer one.
const GRID: f64 = 1.0 / 9007199254740992.0;
const ONE: u64 = 1 << 53;

/// Non-cryptographic generator; streams consume fewer than 2^128 words per jump.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Xoshiro256PlusPlus {
    state: [u64; 4],
}

impl Xoshiro256PlusPlus {
    /// Expands seed into the 256-bit state with SplitMix64.
    pub fn from_seed(mut seed: u64) -> Self {
        let state = std::array::from_fn(|_| {
            seed = seed.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^ (z >> 31)
        });
        Self { state }
    }
    /// Next 64-bit output.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.state;
        let result = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }
    /// Advances by 2^128 steps, starting a non-overlapping segment.
    pub fn jump(&mut self) {
        let mut state = [0; 4];
        for word in [
            0x180ec6d33cfd0abau64,
            0xd5a61266f0c9392c,
            0xa9582618e03fc9aa,
            0x39abdc4529b1661c,
        ] {
            for bit in 0..64 {
                if word & (1 << bit) != 0 {
                    for (out, s) in state.iter_mut().zip(self.state) {
                        *out ^= s;
                    }
                }
                self.next_u64();
            }
        }
        self.state = state;
    }
    /// Uniform 53-bit grid in [0,1).
    #[inline]
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }
    #[inline]
    fn open(&mut self) -> f64 {
        loop {
            let u = self.uniform();
            if u != 0.0 {
                return u;
            }
        }
    }
}

/// Fixed-size sampler without cached outputs or owned output storage.
#[derive(Clone, Debug)]
pub struct NormalSampler {
    rng: Xoshiro256PlusPlus,
}
impl NormalSampler {
    /// Sampler on the SplitMix64-seeded xoshiro256++ stream of seed.
    pub fn from_seed(seed: u64) -> Self {
        Self {
            rng: Xoshiro256PlusPlus::from_seed(seed),
        }
    }
    /// Moves to the next 2^128-step segment, e.g. one per thread.
    pub fn jump(&mut self) {
        self.rng.jump();
    }
    /// Standard-normal pair from the modified undergraph Ziggurat.
    /// Version 0.3 uses 128 slots; 0.5 samples overhangs from chord triangles,
    /// so seeded Gaussian sequences differ from earlier versions.
    #[inline]
    pub fn normal_pair(&mut self) -> [f64; 2] {
        [self.ziggurat(), self.ziggurat()]
    }
    /// Fills out exactly as repeated [Self::normal_pair] calls would.
    pub fn fill_pairs(&mut self, out: &mut [[f64; 2]]) {
        for pair in out {
            *pair = self.normal_pair();
        }
    }
    /// Independent Box–Muller reference pair (slower; uses ln, sin and cos).
    #[inline]
    pub fn box_muller_pair(&mut self) -> [f64; 2] {
        let radius = (-2.0 * self.rng.open().ln()).sqrt();
        let (s, c) = (std::f64::consts::TAU * self.rng.uniform()).sin_cos();
        [radius * c, radius * s]
    }
    #[inline]
    fn ziggurat(&mut self) -> f64 {
        let bits = self.rng.next_u64();
        let i = (bits as u8) & 127;
        if i < 126 {
            let u = f64::from_bits((bits >> 12) | (1024u64 << 52)) - 3.0;
            u * MX[usize::from(i)]
        } else {
            self.rng.modified_slow(bits)
        }
    }
}
impl Xoshiro256PlusPlus {
    /// Uncommon Ziggurat path for a primary word with index 126 or 127; the
    /// primary word supplies only the sign (bit 8). Draws further words from `self`.
    #[cold]
    #[inline(never)]
    fn modified_slow(&mut self, bits: u64) -> f64 {
        let word = self.next_u64();
        let j = ((word as u8) & 127) as usize;
        // Integer cutoffs preserve the corresponding 53-bit uniform decisions.
        let packed = ALIAS[j];
        let i = if (word >> 11) < (packed >> 8) {
            j
        } else {
            usize::from(packed as u8)
        };
        let sign = if bits & 256 == 0 { 1.0 } else { -1.0 };
        if i == 0 {
            loop {
                let t = -self.open().ln() / MX[0];
                let y = -self.open().ln();
                if 2.0 * y >= t * t {
                    return sign * (MX[0] + t);
                }
            }
        }
        let (a, b, fa, fb) = (MX[i], MX[i - 1], MF[i], MF[i - 1]);
        // Order statistics approximate a uniform triangle without reflection.
        // On this finite grid diagonal pairs have half the off-diagonal weight;
        // this is not an exactly uniform distribution over triangle grid points.
        if i < CROSSING {
            // Convex: the region lies below the chord, inside the triangle
            // (a, fb), (a, fa), (b, fb). Tangents bound the density from below.
            loop {
                let (p, q) = (self.next_u64() >> 11, self.next_u64() >> 11);
                let x = a + (b - a) * (p.min(q) as f64 * GRID);
                let y = fb + (fa - fb) * ((ONE - p.max(q)) as f64 * GRID);
                let lo = (fa - a * fa * (x - a)).max(fb - b * fb * (x - b));
                if y < lo - 1e-12 || y < (-0.5 * x * x).exp() {
                    return sign * x;
                }
            }
        }
        if i > CROSSING {
            // Concave: the region is the chord triangle plus a sliver above the
            // chord. The triangle's x marginal is the minimum of two grid uniforms.
            if self.next_u64() >> 11 < TRI[i - CROSSING - 1] {
                let m = (self.next_u64() >> 11).min(self.next_u64() >> 11);
                return sign * (a + (b - a) * (m as f64 * GRID));
            }
            // Sliver: rejection from the triangle chord/tangent-intersection,
            // whose apex is raised by 1e-12 to absorb rounding.
            let (sa, sb) = (-a * fa, -b * fb);
            let xc = (fb - fa + sa * a - sb * b) / (sa - sb);
            let yc = fa + sa * (xc - a) + 1e-12;
            loop {
                let (p, q) = (self.next_u64() >> 11, self.next_u64() >> 11);
                let (s, t) = (p.min(q) as f64 * GRID, (ONE - p.max(q)) as f64 * GRID);
                let x = a + s * (b - a) + t * (xc - a);
                let y = fa + s * (fb - fa) + t * (yc - fa);
                if y < (-0.5 * x * x).exp() {
                    return sign * x;
                }
            }
        }
        // The strip containing x = 1: bounding rectangle with the exact test.
        loop {
            let x = a + (b - a) * self.uniform();
            let y = fb + (fa - fb) * self.uniform();
            if y < (-0.5 * x * x).exp() {
                return sign * x;
            }
        }
    }
}
/// Four jump-separated normal streams with 256 bytes of state and no output cache.
///
/// Lane `i` owns a main engine at jump segment `i` and a repair engine at segment
/// `4 + i` of the seeded state (segment `k` = `k * 2^128` steps). Each output takes
/// exactly one main-engine word. If that word selects the uncommon path, its sign
/// bit is kept and every further draw comes from the lane's repair engine. A lane's
/// values thus depend only on its own two engines; fills may be split arbitrarily.
/// Rows interleave lanes `[0, 1, 2, 3]` and differ from `NormalSampler`'s sequence.
/// Each engine must consume fewer than `2^128` words. Jump spacing alone is not a
/// proof of statistical independence.
///
/// Uses AVX2 when detected on x86-64, with AVX-512F/VL packed rotations when
/// available; vector width remains four lanes. Otherwise uses safe scalar code. All
/// backends produce identical values and final states.
#[derive(Clone, Debug)]
pub struct NormalSampler4 {
    state: [[u64; 4]; 4],
    repair: [Xoshiro256PlusPlus; 4],
}
impl NormalSampler4 {
    /// Jump segments occupied by one sampler, starting at the engine given to `from_rng`.
    pub const SEGMENTS: usize = 8;
    /// Four streams starting at the seeded engine; see [Self::from_rng].
    pub fn from_seed(seed: u64) -> Self {
        Self::from_rng(Xoshiro256PlusPlus::from_seed(seed))
    }
    /// Uses `rng` and its next seven jump segments; see [`Self::SEGMENTS`].
    pub fn from_rng(mut rng: Xoshiro256PlusPlus) -> Self {
        let engines: [Xoshiro256PlusPlus; 8] = std::array::from_fn(|i| {
            if i != 0 {
                rng.jump();
            }
            rng.clone()
        });
        Self {
            state: std::array::from_fn(|j| std::array::from_fn(|i| engines[i].state[j])),
            repair: std::array::from_fn(|i| engines[4 + i].clone()),
        }
    }
    /// Fills rows in lane order `[0, 1, 2, 3]`, without allocating.
    ///
    /// Splitting a fill across calls preserves the sequence. An empty slice consumes
    /// no randomness. CPU dispatch occurs once per call; prefer bulk slices.
    #[allow(unsafe_code)]
    pub fn fill_values(&mut self, out: &mut [[f64; 4]]) {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: each branch verifies its kernel's target-feature contract.
            unsafe {
                if std::is_x86_feature_detected!("avx512f")
                    && std::is_x86_feature_detected!("avx512vl")
                {
                    self.bulk_rotate(out);
                } else {
                    self.bulk_avx2(out);
                }
            };
            return;
        }
        for row in out {
            for i in 0..4 {
                let s = &mut self.state;
                let bits = s[0][i]
                    .wrapping_add(s[3][i])
                    .rotate_left(23)
                    .wrapping_add(s[0][i]);
                let t = s[1][i] << 17;
                s[2][i] ^= s[0][i];
                s[3][i] ^= s[1][i];
                s[1][i] ^= s[2][i];
                s[0][i] ^= s[3][i];
                s[2][i] ^= t;
                s[3][i] = s[3][i].rotate_left(45);
                let j = bits & 127;
                row[i] = if j < 126 {
                    (f64::from_bits((bits >> 12) | (1024u64 << 52)) - 3.0) * MX[j as usize]
                } else {
                    self.repair[i].modified_slow(bits)
                };
            }
        }
    }
    #[allow(unsafe_code)]
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn bulk_avx2(&mut self, out: &mut [[f64; 4]]) {
        // SAFETY: the dispatcher guarantees AVX2; false uses shift/or rotations.
        unsafe { self.bulk::<false>(out) }
    }
    #[allow(unsafe_code)]
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,avx512f,avx512vl")]
    unsafe fn bulk_rotate(&mut self, out: &mut [[f64; 4]]) {
        // SAFETY: the dispatcher also guarantees AVX-512F/VL for packed rotations.
        unsafe { self.bulk::<true>(out) }
    }
    /// Branch-free over 16-row blocks: rare slots temporarily hold their raw word,
    /// flagged in a 64-bit mask (bit `4 * row + lane`), and are repaired after the block.
    #[allow(unsafe_code)]
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn bulk<const ROTATE: bool>(&mut self, out: &mut [[f64; 4]]) {
        use std::arch::x86_64::*;
        // SAFETY: each state row and output row is exactly four u64/f64 values, so
        // every unaligned 32-byte load/store stays inside its array. The wrappers
        // guarantee AVX2 and, when ROTATE is true, AVX-512F/VL. This helper must
        // inline into the corresponding target-feature wrapper. Width indices are
        // masked to 0..127 and MX has 128 entries, so safe indexing cannot panic.
        unsafe {
            let mut a = _mm256_loadu_si256(self.state[0].as_ptr().cast());
            let mut b = _mm256_loadu_si256(self.state[1].as_ptr().cast());
            let mut c = _mm256_loadu_si256(self.state[2].as_ptr().cast());
            let mut d = _mm256_loadu_si256(self.state[3].as_ptr().cast());
            for block in out.chunks_mut(16) {
                let mut rare = 0_u64;
                for (k, row) in block.iter_mut().enumerate() {
                    let sum = _mm256_add_epi64(a, d);
                    let r = _mm256_add_epi64(
                        if ROTATE {
                            _mm256_rol_epi64::<23>(sum)
                        } else {
                            _mm256_or_si256(
                                _mm256_slli_epi64::<23>(sum),
                                _mm256_srli_epi64::<41>(sum),
                            )
                        },
                        a,
                    );
                    let t = _mm256_slli_epi64::<17>(b);
                    c = _mm256_xor_si256(c, a);
                    d = _mm256_xor_si256(d, b);
                    b = _mm256_xor_si256(b, c);
                    a = _mm256_xor_si256(a, d);
                    c = _mm256_xor_si256(c, t);
                    d = if ROTATE {
                        _mm256_rol_epi64::<45>(d)
                    } else {
                        _mm256_or_si256(_mm256_slli_epi64::<45>(d), _mm256_srli_epi64::<19>(d))
                    };
                    let index = _mm256_and_si256(r, _mm256_set1_epi64x(127));
                    let slow = _mm256_cmpgt_epi64(index, _mm256_set1_epi64x(125));
                    let widths = _mm256_set_pd(
                        MX[_mm256_extract_epi64::<3>(index) as usize & 127],
                        MX[_mm256_extract_epi64::<2>(index) as usize & 127],
                        MX[_mm256_extract_epi64::<1>(index) as usize & 127],
                        MX[_mm256_cvtsi256_si32(index) as usize & 127],
                    );
                    let u = _mm256_sub_pd(
                        _mm256_castsi256_pd(_mm256_or_si256(
                            _mm256_srli_epi64::<12>(r),
                            _mm256_set1_epi64x(1024_i64 << 52),
                        )),
                        _mm256_set1_pd(3.0),
                    );
                    let value = _mm256_blendv_pd(
                        _mm256_mul_pd(u, widths),
                        _mm256_castsi256_pd(r),
                        _mm256_castsi256_pd(slow),
                    );
                    _mm256_storeu_pd(row.as_mut_ptr(), value);
                    rare |= (_mm256_movemask_pd(_mm256_castsi256_pd(slow)) as u64) << (4 * k);
                }
                while rare != 0 {
                    let bit = rare.trailing_zeros() as usize;
                    rare &= rare - 1;
                    let slot = &mut block[bit / 4][bit % 4];
                    *slot = self.repair[bit % 4].modified_slow(slot.to_bits());
                }
            }
            _mm256_storeu_si256(self.state[0].as_mut_ptr().cast(), a);
            _mm256_storeu_si256(self.state[1].as_mut_ptr().cast(), b);
            _mm256_storeu_si256(self.state[2].as_mut_ptr().cast(), c);
            _mm256_storeu_si256(self.state[3].as_mut_ptr().cast(), d);
        }
    }
}
