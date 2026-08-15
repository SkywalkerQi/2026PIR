#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <seal/seal.h>
#include <seal/util/polyarithsmallmod.h>

using namespace seal;
using u64 = std::uint64_t;
using Poly = std::vector<u64>;

#ifndef SMALLPIR_THREAD_COUNT
#define SMALLPIR_THREAD_COUNT 1
#endif

constexpr int kThreadCount = SMALLPIR_THREAD_COUNT;

// Returns true when verbose debugging output is enabled through SMALLPIR_VERBOSE.
bool SmallPirVerbose()
{
    return std::getenv("SMALLPIR_VERBOSE") != nullptr;
}

// One logical item stored in a Vacuum Filter slot.
// fp is the client-side verification fingerprint, route drives the alternate
// bucket, value is the raw payload limb, and packed combines fp||value.
struct VfEntry
{
    u64 fp{};
    u64 route{};
    u64 value{};
    u64 packed{};
};

// Fixed-size Vacuum Filter bucket. Empty slots are represented by nullopt.
struct Bucket
{
    std::vector<std::optional<VfEntry>> slots;
};

// Two-level server-side layout: c chunks, each with b buckets, each bucket with s slots.
struct VacuumFilter
{
    std::size_t c{};
    std::size_t b{};
    std::size_t s{};
    std::vector<std::vector<Bucket>> chunks;
};

// Tunable SmallPIR parameters. N is poly_modulus_degree; c,b,s describe the
// chunk/bucket/slot geometry; the remaining fields control plaintext packing
// and Vacuum Filter insertion robustness.
struct SmallPirParams
{
    std::size_t poly_modulus_degree{8192};
    std::vector<int> coeff_modulus_bits{55, 55, 55, 55, 55};
    int plain_modulus_bits{37};

    std::size_t c{274};
    std::size_t b{1024};
    std::size_t s{4};
    std::array<std::size_t, 4> vf_alt_masks{1023, 511, 255, 63};

    u64 plain_modulus{};
    u64 beta_scalar{1};

    std::size_t fp_bits{4};
    std::size_t value_bits{32};
    std::size_t slot_degree{4};
    bool gbfv_fp4_slots{false};
    bool gbfv_binomial_t{false};

    std::size_t vf_max_kicks{32768};
    std::size_t vf_build_retries{16};
    std::size_t vf_seed_trials{64};
    u64 vf_primary_seed{0};
};

// Toy GBFV encoding metadata used to emulate GBFV slots inside a SEAL BFV ring.
// slot_moduli are CRT factors for slots, basis_polys are CRT basis elements,
// and beta/beta_inv model the BFV<->GBFV conversion described in GBFV.
struct GbfvEncoding
{
    std::vector<Poly> slot_moduli;
    Poly phi_poly;
    Poly t_poly;
    std::vector<Poly> basis_polys;
    Poly beta_poly{1, 0};
    Poly beta_inv_mod_t{1};
    Plaintext beta_plain;
    Plaintext beta_inv_plain;
    bool beta_is_one{true};
    bool beta_inv_is_one{true};
};

// Client query state. It carries the encrypted chunk/slice selectors and the
// encrypted bucket mask, plus plaintext indices needed by the local verifier.
struct Query
{
    std::size_t target_chunk{};
    std::size_t target_slice{};
    std::size_t local_chunk{};
    std::size_t slice_len{};
    std::size_t slice_count{};
    std::size_t bucket1{};
    std::size_t bucket2{};
    u64 fp{};
    Ciphertext q_sl_bfv;
    Ciphertext q_ch_bfv;
    Ciphertext q_bk_gbfv;
};

// Server response plus a timing breakdown for the online answer phase.
struct Response
{
    Ciphertext ans;
    double expand_ms{0.0};
    double select_ms{0.0};
    double ctct_ms{0.0};
};

// SplitMix-style mixer used as a deterministic synthetic hash source.
u64 Mix64(u64 x)
{
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// Domain-separated 64-bit hash used for fingerprints, routing, and synthetic data.
u64 HashWithSeed(u64 seed, u64 x)
{
    return Mix64(seed ^ (x + 0x9e3779b97f4a7c15ULL));
}

// Rounds x up to the next power of two for selector expansion lengths.
std::size_t NextPow2(std::size_t x)
{
    std::size_t p = 1;
    while (p < x) p <<= 1U;
    return p;
}

// Integer ceil(a / b), with a guard for accidental zero divisors.
std::size_t CeilDiv(std::size_t a, std::size_t b)
{
    if (b == 0) throw std::invalid_argument("division by zero");
    return (a + b - 1U) / b;
}

// Chooses the number of chunks inside each slice. A power near sqrt(c) keeps
// slice and intra-slice selector expansions balanced.
std::size_t ChooseSliceLen(std::size_t chunk_count)
{
    const auto root = static_cast<std::size_t>(
        std::ceil(std::sqrt(static_cast<long double>(chunk_count))));
    return NextPow2(std::max<std::size_t>(root, 1U));
}

// Computes the chunk count needed to reach the target Vacuum Filter load.
std::size_t AutoChunkCount(std::size_t n, std::size_t b, std::size_t s, double target_load)
{
    const long double denom = static_cast<long double>(b) * static_cast<long double>(s) * target_load;
    return static_cast<std::size_t>(std::ceil(static_cast<long double>(n) / denom));
}

// Builds masks used by the alternate bucket rule; b must be a power of two so
// XOR routing stays within the same chunk-sized segment.
std::array<std::size_t, 4> MakeAltMasks(std::size_t b)
{
    if ((b & (b - 1U)) != 0 || b < 16)
    {
        throw std::invalid_argument("b must be a power of two and at least 16");
    }
    return {
        b - 1,
        (b >> 1U) - 1U,
        (b >> 2U) - 1U,
        std::max<std::size_t>((b >> 4U), 1U) - 1U
    };
}

// Modular exponentiation by repeated squaring.
u64 ModPow(u64 base, u64 exp, u64 mod)
{
    u64 result = 1ULL;
    base %= mod;
    while (exp > 0)
    {
        if (exp & 1ULL)
        {
            result = static_cast<u64>((static_cast<unsigned __int128>(result) * base) % mod);
        }
        base = static_cast<u64>((static_cast<unsigned __int128>(base) * base) % mod);
        exp >>= 1ULL;
    }
    return result;
}

// 64x64->128 modular multiplication, avoiding overflow before reduction.
u64 MulMod128(u64 a, u64 b, u64 mod)
{
    return static_cast<u64>((static_cast<unsigned __int128>(a) * b) % mod);
}

// Deterministic Miller-Rabin-style primality test for the small range used here.
bool IsPrime64(u64 n)
{
    if (n < 2) return false;
    static constexpr u64 small_primes[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
    for (u64 p : small_primes)
    {
        if (n == p) return true;
        if (n % p == 0) return false;
    }

    u64 d = n - 1;
    unsigned s = 0;
    while ((d & 1ULL) == 0)
    {
        d >>= 1ULL;
        ++s;
    }

    auto witness = [&](u64 a) {
        if (a % n == 0) return false;
        u64 x = ModPow(a % n, d, n);
        if (x == 1 || x == n - 1) return false;
        for (unsigned r = 1; r < s; ++r)
        {
            x = MulMod128(x, x, n);
            if (x == n - 1) return false;
        }
        return true;
    };

    static constexpr u64 bases[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
    for (u64 a : bases)
    {
        if (witness(a)) return false;
    }
    return true;
}

// Finds a plaintext prime compatible with the simulated F_{p^4} slot layout.
// The residue condition gives the desired factorization of the BFV cyclotomic ring.
u64 FindFp4PlainModulus(std::size_t degree, int bits)
{
    if (degree < 8 || (degree & (degree - 1U)) != 0)
    {
        throw std::runtime_error("Fp4 GBFV mode expects power-of-two N >= 8");
    }
    if (bits <= 2 || bits >= 62)
    {
        throw std::runtime_error("unsupported plaintext modulus bit length for Fp4 GBFV mode");
    }

    const u64 two_n = static_cast<u64>(2ULL * degree);
    const u64 residue = static_cast<u64>(1ULL + degree / 2ULL);
    const u64 upper = (1ULL << bits) - 1ULL;
    const u64 lower = 1ULL << (bits - 1);

    u64 candidate = upper - ((upper + two_n - residue) % two_n);
    if ((candidate & 1ULL) == 0) candidate -= two_n;
    while (candidate >= lower)
    {
        if (candidate % two_n == residue && IsPrime64(candidate))
        {
            return candidate;
        }
        if (candidate < lower + two_n) break;
        candidate -= two_n;
    }
    throw std::runtime_error("failed to find an Fp4 plaintext prime; increase plain_modulus_bits");
}

// Multiplicative inverse modulo a prime, via Fermat's little theorem.
u64 ModInvPrime(u64 x, u64 mod)
{
    if (x % mod == 0) throw std::runtime_error("inverse of zero does not exist");
    return ModPow(x, mod - 2, mod);
}

// Encodes a vector into BFV batching slots, padding remaining slots with zero.
Plaintext EncodeVector(const BatchEncoder &encoder, const std::vector<u64> &values)
{
    std::vector<u64> slots(encoder.slot_count(), 0ULL);
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        slots[i] = values[i];
    }
    Plaintext pt;
    encoder.encode(slots, pt);
    return pt;
}

// Decodes BFV batching slots from a plaintext.
std::vector<u64> DecodePlaintextSlots(const BatchEncoder &encoder, const Plaintext &pt)
{
    std::vector<u64> slots;
    encoder.decode(pt, slots);
    return slots;
}

// Creates a batching plaintext where every slot contains the same value.
Plaintext EncodeAllSlots(const BatchEncoder &encoder, u64 value)
{
    std::vector<u64> slots(encoder.slot_count(), value);
    Plaintext pt;
    encoder.encode(slots, pt);
    return pt;
}

// Convenience helper: encode a vector and encrypt it as a BFV ciphertext.
Ciphertext EncryptVector(const BatchEncoder &encoder,
                         Encryptor &encryptor,
                         const std::vector<u64> &values)
{
    Plaintext pt = EncodeVector(encoder, values);
    Ciphertext ct;
    encryptor.encrypt(pt, ct);
    return ct;
}

// Convenience helper: decrypt a BFV ciphertext and decode batching slots.
std::vector<u64> DecodeCiphertext(const BatchEncoder &encoder,
                                  Decryptor &decryptor,
                                  const Ciphertext &ct)
{
    Plaintext pt;
    decryptor.decrypt(ct, pt);
    std::vector<u64> slots;
    encoder.decode(pt, slots);
    return slots;
}

// Decrypts a ciphertext but leaves it as a SEAL Plaintext polynomial.
Plaintext DecryptPlain(Decryptor &decryptor, const Ciphertext &ct)
{
    Plaintext pt;
    decryptor.decrypt(ct, pt);
    return pt;
}

// Creates the plaintext polynomial x^exponent.
Plaintext MakeMonomialPlain(std::size_t exponent)
{
    Plaintext pt(exponent + 1);
    pt.set_zero();
    pt[exponent] = 1;
    return pt;
}

// Creates a constant plaintext polynomial.
Plaintext MakeConstantPlain(u64 value)
{
    Plaintext pt(1);
    pt[0] = value;
    return pt;
}

// Removes trailing zero coefficients while keeping the zero polynomial as {0}.
Poly TrimPoly(Poly a)
{
    while (!a.empty() && a.back() == 0) a.pop_back();
    if (a.empty()) a.push_back(0);
    return a;
}

// Tests whether a polynomial is the constant one polynomial.
bool IsOnePoly(const Poly &poly)
{
    Poly trimmed = TrimPoly(poly);
    return trimmed.size() == 1 && trimmed[0] == 1ULL;
}

// Computes a-b modulo mod without underflow.
u64 ModSub(u64 a, u64 b, u64 mod)
{
    return (a >= b) ? (a - b) : (a + mod - b);
}

// Adds two coefficient vectors in F_mod[x].
Poly AddPolyMod(const Poly &a, const Poly &b, u64 mod)
{
    Poly out(std::max(a.size(), b.size()), 0ULL);
    for (std::size_t i = 0; i < out.size(); ++i)
    {
        u64 av = (i < a.size()) ? a[i] : 0ULL;
        u64 bv = (i < b.size()) ? b[i] : 0ULL;
        out[i] = (av + bv) % mod;
    }
    return TrimPoly(std::move(out));
}

// Returns Phi_{2N}(x)=x^N+1 for power-of-two cyclotomic BFV rings.
Poly PowerOfTwoCyclotomicPoly(std::size_t degree)
{
    Poly phi(degree + 1, 0ULL);
    phi[0] = 1ULL;
    phi[degree] = 1ULL;
    return phi;
}

// Subtracts two coefficient vectors in F_mod[x].
Poly SubPolyMod(const Poly &a, const Poly &b, u64 mod)
{
    Poly out(std::max(a.size(), b.size()), 0ULL);
    for (std::size_t i = 0; i < out.size(); ++i)
    {
        u64 av = (i < a.size()) ? a[i] : 0ULL;
        u64 bv = (i < b.size()) ? b[i] : 0ULL;
        out[i] = ModSub(av, bv, mod);
    }
    return TrimPoly(std::move(out));
}

// Multiplies two polynomials over F_mod without reducing by another modulus.
Poly MulPolyMod(const Poly &a, const Poly &b, u64 mod)
{
    Poly out(a.size() + b.size() - 1, 0ULL);
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        for (std::size_t j = 0; j < b.size(); ++j)
        {
            unsigned __int128 acc =
                static_cast<unsigned __int128>(a[i]) * b[j] + out[i + j];
            out[i + j] = static_cast<u64>(acc % mod);
        }
    }
    return TrimPoly(std::move(out));
}

// Polynomial long division over F_mod[x]. The divisor is treated as monic up to
// an invertible leading coefficient, and both quotient and remainder are returned.
std::pair<Poly, Poly> DivModMonic(Poly num, const Poly &den, u64 mod)
{
    num = TrimPoly(std::move(num));
    const Poly d = TrimPoly(den);
    if (d.size() == 1 && d[0] == 0) throw std::runtime_error("divide by zero polynomial");
    u64 lead_inv = ModInvPrime(d.back(), mod);

    Poly quot((num.size() >= d.size()) ? (num.size() - d.size() + 1) : 1, 0ULL);
    while (!(num.size() == 1 && num[0] == 0) && num.size() >= d.size())
    {
        std::size_t shift = num.size() - d.size();
        u64 lead = static_cast<u64>((static_cast<unsigned __int128>(num.back()) * lead_inv) % mod);
        quot[shift] = lead;
        for (std::size_t i = 0; i < d.size(); ++i)
        {
            std::size_t idx = i + shift;
            u64 sub = static_cast<u64>((static_cast<unsigned __int128>(lead) * d[i]) % mod);
            num[idx] = ModSub(num[idx], sub, mod);
        }
        num = TrimPoly(std::move(num));
    }
    return {TrimPoly(std::move(quot)), TrimPoly(std::move(num))};
}

// Reduces a polynomial modulo another polynomial over F_mod.
Poly ModPoly(const Poly &a, const Poly &modulus, u64 mod)
{
    return DivModMonic(a, modulus, mod).second;
}

// Multiplies two polynomials and immediately reduces them modulo modulus.
Poly MulPolyModModulus(const Poly &a, const Poly &b, const Poly &modulus, u64 mod)
{
    return ModPoly(MulPolyMod(a, b, mod), modulus, mod);
}

// Polynomial exponentiation modulo a target polynomial.
Poly PolyPowMod(Poly base, unsigned __int128 exp, const Poly &modulus, u64 mod)
{
    Poly result{1};
    base = ModPoly(base, modulus, mod);
    while (exp > 0)
    {
        if (exp & 1) result = MulPolyModModulus(result, base, modulus, mod);
        base = MulPolyModModulus(base, base, modulus, mod);
        exp >>= 1;
    }
    return TrimPoly(std::move(result));
}

// Euclidean gcd over F_mod[x], normalized to monic output.
Poly GcdPolyMod(Poly a, Poly b, u64 mod)
{
    a = TrimPoly(std::move(a));
    b = TrimPoly(std::move(b));
    while (!(b.size() == 1 && b[0] == 0))
    {
        Poly r = ModPoly(a, b, mod);
        a = std::move(b);
        b = std::move(r);
    }
    u64 inv = ModInvPrime(a.back(), mod);
    for (u64 &coef : a)
    {
        coef = static_cast<u64>((static_cast<unsigned __int128>(coef) * inv) % mod);
    }
    return TrimPoly(std::move(a));
}

// Computes x^(mod^power) modulo a polynomial. This is used for finite-field
// irreducibility tests via Frobenius powers.
Poly XPowPowersMod(std::size_t power, const Poly &modulus, u64 mod)
{
    Poly x{0, 1};
    unsigned __int128 exp = 1;
    for (std::size_t i = 0; i < power; ++i) exp *= mod;
    return PolyPowMod(x, exp, modulus, mod);
}

// Checks whether a monic degree-4 polynomial is irreducible over F_mod.
bool IsIrreducibleDegree4(const Poly &f, u64 mod)
{
    if (f.size() != 5 || f.back() != 1) return false;

    Poly x{0, 1};
    Poly xp1 = XPowPowersMod(1, f, mod);
    Poly g1 = GcdPolyMod(SubPolyMod(xp1, x, mod), f, mod);
    if (!(g1.size() == 1 && g1[0] == 1)) return false;

    Poly xp2 = XPowPowersMod(2, f, mod);
    Poly g = GcdPolyMod(SubPolyMod(xp2, x, mod), f, mod);
    if (!(g.size() == 1 && g[0] == 1)) return false;

    Poly xp4 = XPowPowersMod(4, f, mod);
    Poly r = ModPoly(SubPolyMod(xp4, x, mod), f, mod);
    return r.size() == 1 && r[0] == 0;
}

// Extended Euclidean inverse of a polynomial modulo another polynomial.
Poly InversePolyMod(const Poly &a, const Poly &modulus, u64 mod)
{
    Poly r0 = TrimPoly(modulus), r1 = ModPoly(a, modulus, mod);
    Poly s0{0}, s1{1};

    while (!(r1.size() == 1 && r1[0] == 0))
    {
        auto [q, r2] = DivModMonic(r0, r1, mod);
        Poly s2 = SubPolyMod(s0, MulPolyMod(q, s1, mod), mod);
        r0 = std::move(r1);
        r1 = std::move(r2);
        s0 = std::move(s1);
        s1 = ModPoly(s2, modulus, mod);
    }

    if (!(r0.size() == 1 && r0[0] != 0))
    {
        throw std::runtime_error("polynomial is not invertible modulo target");
    }

    u64 inv = ModInvPrime(r0[0], mod);
    for (u64 &coef : s0)
    {
        coef = static_cast<u64>((static_cast<unsigned __int128>(coef) * inv) % mod);
    }
    return ModPoly(s0, modulus, mod);
}

// Converts the local Poly representation into a SEAL plaintext polynomial.
Plaintext PlainFromPoly(const Poly &poly)
{
    Poly trimmed = TrimPoly(poly);
    Plaintext pt(trimmed.size());
    pt.set_zero();
    for (std::size_t i = 0; i < trimmed.size(); ++i) pt[i] = trimmed[i];
    return pt;
}

// Converts a SEAL plaintext polynomial back into the local Poly representation.
Poly PolyFromPlain(const Plaintext &pt, u64 mod)
{
    std::size_t deg = std::max<std::size_t>(1, pt.significant_coeff_count());
    Poly poly(deg, 0ULL);
    for (std::size_t i = 0; i < deg; ++i) poly[i] = pt[i] % mod;
    return TrimPoly(std::move(poly));
}

// Substitutes x -> x - shift in poly over F_mod.
Poly TranslatePoly(const Poly &poly, u64 shift, u64 mod)
{
    Poly result{0ULL};
    const u64 neg_shift = (mod - (shift % mod)) % mod;
    for (std::size_t i = 0; i < poly.size(); ++i)
    {
        if (poly[i] == 0) continue;
        Poly term{1ULL};
        Poly base{neg_shift, 1ULL};
        std::size_t exp = i;
        while (exp > 0)
        {
            if (exp & 1U)
            {
                term = MulPolyMod(term, base, mod);
            }
            exp >>= 1U;
            if (exp) base = MulPolyMod(base, base, mod);
        }
        for (u64 &coef : term)
        {
            coef = static_cast<u64>((static_cast<unsigned __int128>(coef) * poly[i]) % mod);
        }
        result = AddPolyMod(result, term, mod);
    }
    return TrimPoly(std::move(result));
}

// Searches for a deterministic random irreducible polynomial of the requested
// degree. The current GBFV simulation only needs degree 4.
Poly FindBaseIrreducible(std::size_t degree, u64 mod)
{
    if (degree == 0) throw std::invalid_argument("degree must be positive");
    if (degree != 4) throw std::runtime_error("current toy implementation expects slot_degree = 4");
    std::mt19937_64 rng(0xC0DEC0FFEEULL ^ mod);
    std::uniform_int_distribution<u64> dist(0, mod - 1);
    for (std::size_t trial = 0; trial < 200000; ++trial)
    {
        Poly candidate(degree + 1, 0ULL);
        candidate[0] = dist(rng);
        if (candidate[0] == 0) candidate[0] = 1;
        candidate[1] = dist(rng);
        candidate[2] = dist(rng);
        candidate[3] = dist(rng);
        candidate[degree] = 1ULL;
        if (IsIrreducibleDegree4(candidate, mod))
        {
            return candidate;
        }
    }
    throw std::runtime_error("failed to find a base irreducible polynomial");
}

// Builds the simulated GBFV CRT encoding.
// Key steps:
// 1. Choose slot factors t_i(x) either as linear BFV batching slots or as
//    degree-4 F_{p^4} slot factors.
// 2. Multiply them into t(x) and compute beta=Phi_m(x)/t(x) when needed.
// 3. Build CRT basis polynomials so bucket payloads can be packed into slots.
GbfvEncoding BuildToyGbfvEncoding(const SmallPirParams &pp)
{
    GbfvEncoding enc;
    const std::size_t total_slots = pp.gbfv_fp4_slots ? pp.b : (pp.b * pp.s);
    if ((!pp.gbfv_fp4_slots && total_slots > pp.poly_modulus_degree) ||
        (pp.gbfv_fp4_slots && total_slots * pp.s > pp.poly_modulus_degree))
    {
        throw std::runtime_error("GBFV bucket count exceeds the number of available plaintext slots");
    }

    if (pp.gbfv_fp4_slots)
    {
        // F_{p^4} mode selects degree-4 slot factors x^4-a. In binomial mode
        // these factors are chosen so t(x)=x^{N/2}-B matches the GBFV sketch.
        if (pp.s != 4 || pp.slot_degree != 4)
        {
            throw std::runtime_error("Fp4 GBFV mode expects s=4 and slot_degree=4");
        }
        if (pp.gbfv_binomial_t && pp.b != pp.poly_modulus_degree / 8)
        {
            throw std::runtime_error("binomial Fp4 GBFV mode expects b=N/8");
        }

        const std::size_t order = pp.poly_modulus_degree / 2;
        u64 omega = 0;
        for (u64 g = 2; g < 1000000ULL; ++g)
        {
            const u64 candidate = ModPow(g, (pp.plain_modulus - 1) / order, pp.plain_modulus);
            if (candidate != 1 && ModPow(candidate, order, pp.plain_modulus) == 1 &&
                ModPow(candidate, order / 2, pp.plain_modulus) != 1)
            {
                omega = candidate;
                break;
            }
        }
        if (omega == 0)
        {
            throw std::runtime_error("failed to construct an order N/2 element for Fp4 slots");
        }

        enc.slot_moduli.clear();
        enc.slot_moduli.reserve(total_slots);
        const u64 binomial_b = pp.gbfv_binomial_t
            ? ModPow(omega, order / 4, pp.plain_modulus)
            : 0ULL;
        for (std::size_t i = 0; i < order; ++i)
        {
            const u64 a = ModPow(omega, 2ULL * i + 1ULL, pp.plain_modulus);
            if (pp.gbfv_binomial_t &&
                ModPow(a, pp.poly_modulus_degree / 8, pp.plain_modulus) != binomial_b)
            {
                continue;
            }
            enc.slot_moduli.push_back(Poly{(pp.plain_modulus - a) % pp.plain_modulus, 0ULL, 0ULL, 0ULL, 1ULL});
            if (enc.slot_moduli.size() == total_slots) break;
        }
        if (enc.slot_moduli.size() != total_slots)
        {
            throw std::runtime_error("failed to collect the requested Fp4 slot factors");
        }
    }
    else
    {
        // Plain BFV batching mode uses linear factors x-root for ordinary slots.
        const u64 n = pp.plain_modulus - 1;
        std::vector<u64> prime_factors;
        u64 x = n;
        for (u64 d = 2; d * d <= x; d += (d == 2 ? 1 : 2))
        {
            if (x % d != 0) continue;
            prime_factors.push_back(d);
            while (x % d == 0) x /= d;
        }
        if (x > 1) prime_factors.push_back(x);

        u64 primitive_root = 0;
        for (u64 g = 2; g < pp.plain_modulus; ++g)
        {
            bool ok = true;
            for (u64 q : prime_factors)
            {
                if (ModPow(g, n / q, pp.plain_modulus) == 1)
                {
                    ok = false;
                    break;
                }
            }
            if (ok)
            {
                primitive_root = g;
                break;
            }
        }
        if (primitive_root == 0)
        {
            throw std::runtime_error("failed to find a primitive root for the chosen plaintext modulus");
        }

        const u64 two_n = pp.poly_modulus_degree * 2ULL;
        if ((pp.plain_modulus - 1) % two_n != 0)
        {
            throw std::runtime_error("plaintext modulus is not a batching prime for the chosen N");
        }

        const u64 psi = ModPow(primitive_root, (pp.plain_modulus - 1) / two_n, pp.plain_modulus);
        if (ModPow(psi, pp.poly_modulus_degree, pp.plain_modulus) != (pp.plain_modulus - 1))
        {
            throw std::runtime_error("failed to construct a primitive 2N-th root of unity");
        }

        enc.slot_moduli.clear();
        enc.slot_moduli.reserve(total_slots);
        for (std::size_t i = 0; i < total_slots; ++i)
        {
            const u64 root = ModPow(psi, 2ULL * i + 1ULL, pp.plain_modulus);
            enc.slot_moduli.push_back(Poly{(pp.plain_modulus - root) % pp.plain_modulus, 1ULL});
        }
    }

    if (pp.gbfv_binomial_t)
    {
        // Fast special case for t(x)=x^{N/2}-B. beta is represented directly
        // instead of dividing Phi_m by t(x).
        const std::size_t half_n = pp.poly_modulus_degree / 2;
        const u64 binomial_b = (pp.plain_modulus - enc.slot_moduli[0][0]) % pp.plain_modulus;
        const u64 root_b = ModPow(binomial_b, pp.poly_modulus_degree / 8, pp.plain_modulus);
        if (MulMod128(root_b, root_b, pp.plain_modulus) != pp.plain_modulus - 1)
        {
            throw std::runtime_error("binomial Fp4 GBFV mode failed to construct B^2=-1");
        }
        enc.t_poly.assign(half_n + 1, 0ULL);
        enc.t_poly[0] = (pp.plain_modulus - root_b) % pp.plain_modulus;
        enc.t_poly[half_n] = 1ULL;
        enc.beta_poly.assign(half_n + 1, 0ULL);
        enc.beta_poly[0] = root_b;
        enc.beta_poly[half_n] = 1ULL;
        enc.beta_inv_mod_t = {ModInvPrime((2ULL * root_b) % pp.plain_modulus, pp.plain_modulus)};
    }
    else
    {
        enc.t_poly = {1ULL};
        for (const auto &slot_modulus : enc.slot_moduli)
        {
            enc.t_poly = MulPolyMod(enc.t_poly, slot_modulus, pp.plain_modulus);
        }
    }

    enc.phi_poly = PowerOfTwoCyclotomicPoly(pp.poly_modulus_degree);
    if (!pp.gbfv_binomial_t)
    {
        // General GBFV conversion factor beta = Phi_m / t. Multiplying a BFV
        // ciphertext by beta moves it into the simulated GBFV ideal.
        auto [beta, beta_rem] = DivModMonic(enc.phi_poly, enc.t_poly, pp.plain_modulus);
        beta_rem = TrimPoly(std::move(beta_rem));
        if (!(beta_rem.size() == 1 && beta_rem[0] == 0))
        {
            throw std::runtime_error("GBFV t(x) is not a factor of the BFV cyclotomic polynomial");
        }
        enc.beta_poly = TrimPoly(std::move(beta));
        enc.beta_inv_mod_t = InversePolyMod(enc.beta_poly, enc.t_poly, pp.plain_modulus);
    }
    enc.beta_plain = PlainFromPoly(enc.beta_poly);
    enc.beta_inv_plain = PlainFromPoly(enc.beta_inv_mod_t);
    enc.beta_is_one = IsOnePoly(enc.beta_poly);
    enc.beta_inv_is_one = IsOnePoly(enc.beta_inv_mod_t);

    enc.basis_polys.clear();
    enc.basis_polys.reserve(total_slots);
    for (std::size_t i = 0; i < total_slots; ++i)
    {
        // CRT basis: basis_i is 1 modulo slot_i and 0 modulo all other slots.
        auto div = DivModMonic(enc.t_poly, enc.slot_moduli[i], pp.plain_modulus);
        Poly Mi = std::move(div.first);
        Poly ri = ModPoly(Mi, enc.slot_moduli[i], pp.plain_modulus);
        Poly ri_inv = InversePolyMod(ri, enc.slot_moduli[i], pp.plain_modulus);
        Poly basis = ModPoly(MulPolyMod(Mi, ri_inv, pp.plain_modulus), enc.t_poly, pp.plain_modulus);
        enc.basis_polys.push_back(std::move(basis));
    }

    return enc;
}

// Multiplies a ciphertext polynomial by x^index in the negacyclic BFV ring.
// SEAL exposes the coefficient-level shift primitive used here.
Ciphertext MultiplyPowerOfX(const Ciphertext &src,
                            std::size_t index,
                            const SEALContext &context)
{
    auto context_data = context.get_context_data(src.parms_id());
    if (!context_data) throw std::runtime_error("missing context data for ciphertext");

    const auto &parms = context_data->parms();
    const std::size_t coeff_count = parms.poly_modulus_degree();
    const auto &coeff_modulus = parms.coeff_modulus();

    Ciphertext dst = src;
    for (std::size_t poly_idx = 0; poly_idx < src.size(); ++poly_idx)
    {
        for (std::size_t mod_idx = 0; mod_idx < coeff_modulus.size(); ++mod_idx)
        {
            auto *dst_ptr = dst.data(poly_idx) + mod_idx * coeff_count;
            const auto *src_ptr = src.data(poly_idx) + mod_idx * coeff_count;
            seal::util::negacyclic_shift_poly_coeffmod(
                src_ptr, coeff_count, index, coeff_modulus[mod_idx], dst_ptr);
        }
    }
    return dst;
}

// Plaintext reference version of negacyclic multiplication by x^shift.
std::vector<u64> NegacyclicShiftPlain(const std::vector<u64> &src,
                                      std::size_t shift,
                                      std::size_t degree,
                                      u64 mod)
{
    std::vector<u64> dst(degree, 0ULL);
    for (std::size_t i = 0; i < degree; ++i)
    {
        if (src[i] == 0) continue;
        std::size_t raw = i + shift;
        std::size_t pos = raw % degree;
        bool neg = ((raw / degree) & 1U) != 0;
        dst[pos] = neg ? (dst[pos] + mod - src[i]) % mod : (dst[pos] + src[i]) % mod;
    }
    return dst;
}

// Plaintext reference version of the BFV Galois automorphism x -> x^gal_elt.
std::vector<u64> ApplyAutomorphismPlain(const std::vector<u64> &src,
                                        std::size_t gal_elt,
                                        std::size_t degree,
                                        u64 mod)
{
    const std::size_t order = degree << 1U;
    std::vector<u64> dst(degree, 0ULL);
    for (std::size_t i = 0; i < degree; ++i)
    {
        if (src[i] == 0) continue;
        std::size_t raw = (i * gal_elt) % order;
        bool neg = raw >= degree;
        std::size_t pos = neg ? (raw - degree) : raw;
        dst[pos] = neg ? (dst[pos] + mod - src[i]) % mod : (dst[pos] + src[i]) % mod;
    }
    return dst;
}

// Adds two degree-N plaintext polynomials coefficient-wise.
std::vector<u64> AddPlainPoly(const std::vector<u64> &a,
                              const std::vector<u64> &b,
                              u64 mod)
{
    std::vector<u64> out(a.size(), 0ULL);
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        out[i] = (a[i] + b[i]) % mod;
    }
    return out;
}

// Computes the scalar correction introduced by the recursive expansion tree for
// one selector index. This exact routine is slow but useful for validation.
u64 PrecomputeSelectorWeight(const SmallPirParams &pp, std::size_t target)
{
    const std::size_t degree = pp.poly_modulus_degree;
    const std::size_t logical_n = degree;
    const std::size_t chunk_count = pp.c;
    std::size_t logm = 0;
    while ((1ULL << logm) < chunk_count) ++logm;

    std::vector<std::size_t> galois_elts(logm, 0);
    for (std::size_t i = 0; i < logm; ++i)
    {
        galois_elts[i] = (logical_n + (1ULL << i)) / (1ULL << i);
    }

    std::vector<u64> mono(degree, 0ULL);
    mono[target] = 1ULL;
    std::vector<std::vector<u64>> current{mono};

    for (std::size_t i = 0; i < logm; ++i)
    {
        // Mirror ExpandToPow2ThenTrim on plaintext monomials to learn the
        // diagonal scalar applied to the target selector.
        const bool is_last = (i == logm - 1);
        std::vector<std::vector<u64>> next(std::min<std::size_t>(2 * current.size(), chunk_count),
                                           std::vector<u64>(degree, 0ULL));
        const std::size_t tail_count = chunk_count - current.size();

        const std::size_t index_raw = (logical_n << 1U) - (1ULL << i);
        const std::size_t index = (index_raw * galois_elts[i]) % (logical_n << 1U);

        for (std::size_t j = 0; j < current.size(); ++j)
        {
            const auto &c0 = current[j];
            if (is_last && j >= tail_count)
            {
                next[j] = c0;
                continue;
            }

            const auto c1 = NegacyclicShiftPlain(c0, index, degree, pp.plain_modulus);
            const auto sub0 = ApplyAutomorphismPlain(c0, galois_elts[i], degree, pp.plain_modulus);
            next[j] = AddPlainPoly(sub0, c0, pp.plain_modulus);

            if (is_last)
            {
                next[j + current.size()] = c1;
                continue;
            }

            const auto sub1 = ApplyAutomorphismPlain(c1, galois_elts[i], degree, pp.plain_modulus);
            next[j + current.size()] = AddPlainPoly(sub1, c1, pp.plain_modulus);
        }
        current = std::move(next);
    }

    const auto &diag = current[target];
    for (std::size_t d = 1; d < degree; ++d)
    {
        if (diag[d] != 0)
        {
            throw std::runtime_error("selector weight precomputation yielded a non-constant polynomial");
        }
    }
    return diag[0] % pp.plain_modulus;
}

// Computes exact selector weights for all chunks by simulating expansion.
std::vector<u64> PrecomputeSelectorWeights(const SmallPirParams &pp)
{
    std::vector<u64> weights(pp.c, 0ULL);
    for (std::size_t target = 0; target < pp.c; ++target)
    {
        weights[target] = PrecomputeSelectorWeight(pp, target);
    }
    return weights;
}

// Closed-form selector weight used in normal runs. It captures the expansion
// tree's power-of-two scale and sign without replaying all automorphisms.
u64 SelectorWeightFormula(const SmallPirParams &pp, std::size_t target)
{
    std::size_t expand_len = 1ULL;
    while (expand_len < pp.c) expand_len <<= 1ULL;
    std::size_t scale = expand_len;
    if (pp.c < expand_len)
    {
        const std::size_t lower_half = expand_len >> 1U;
        const std::size_t tail_count = pp.c - lower_half;
        if (target >= tail_count) scale = lower_half;
    }

    bool odd_parity = false;
    for (std::size_t x = target; x != 0; x >>= 1ULL)
    {
        odd_parity = odd_parity != ((x & 1ULL) != 0);
    }

    const u64 scale_mod = static_cast<u64>(scale % pp.plain_modulus);
    return odd_parity ? ((pp.plain_modulus + pp.plain_modulus - scale_mod) % pp.plain_modulus) : scale_mod;
}

// Builds all selector weights using the closed-form formula.
std::vector<u64> BuildSelectorWeightsFormula(const SmallPirParams &pp)
{
    std::vector<u64> weights(pp.c, 0ULL);
    for (std::size_t target = 0; target < pp.c; ++target)
    {
        weights[target] = SelectorWeightFormula(pp, target);
    }
    return weights;
}

// Cache path for selector weights. It is keyed by N, c, expanded length, and p.
std::string SelectorWeightsCachePath(const SmallPirParams &pp)
{
    std::size_t expand_len = 1ULL;
    while (expand_len < pp.c) expand_len <<= 1ULL;
    return ".smallpir_cache/selector_weights_v3_N" + std::to_string(pp.poly_modulus_degree) +
           "_c" + std::to_string(pp.c) +
           "_expand" + std::to_string(expand_len) +
           "_p" + std::to_string(pp.plain_modulus) + ".bin";
}

// Loads cached selector weights and validates the parameters stored in the file.
bool LoadSelectorWeightsCache(const SmallPirParams &pp, std::vector<u64> &weights, std::size_t &completed)
{
    const std::string path = SelectorWeightsCachePath(pp);
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;

    char magic[8]{};
    std::uint64_t version = 0;
    std::uint64_t degree = 0;
    std::uint64_t chunk_count = 0;
    std::uint64_t plain_modulus = 0;
    std::uint64_t completed_count = 0;
    in.read(magic, sizeof(magic));
    in.read(reinterpret_cast<char *>(&version), sizeof(version));
    in.read(reinterpret_cast<char *>(&degree), sizeof(degree));
    in.read(reinterpret_cast<char *>(&chunk_count), sizeof(chunk_count));
    in.read(reinterpret_cast<char *>(&plain_modulus), sizeof(plain_modulus));
    in.read(reinterpret_cast<char *>(&completed_count), sizeof(completed_count));

    const std::string expected_magic = "SPIRSW3";
    if (!in || std::string(magic, magic + 7) != expected_magic || version != 3 ||
        degree != pp.poly_modulus_degree || chunk_count != pp.c ||
        plain_modulus != pp.plain_modulus || completed_count > pp.c)
    {
        return false;
    }

    std::vector<u64> cached(pp.c, 0ULL);
    in.read(reinterpret_cast<char *>(cached.data()),
            static_cast<std::streamsize>(cached.size() * sizeof(u64)));
    if (!in) return false;

    weights = std::move(cached);
    completed = static_cast<std::size_t>(completed_count);
    return true;
}

// Writes selector weights so future full-prep runs can skip recomputing them.
void SaveSelectorWeightsCache(const SmallPirParams &pp, const std::vector<u64> &weights, std::size_t completed)
{
    if (weights.size() != pp.c) return;
    if (completed > pp.c) completed = pp.c;

    const std::string path = SelectorWeightsCachePath(pp);
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) return;

    char magic[8]{'S', 'P', 'I', 'R', 'S', 'W', '3', '\0'};
    const std::uint64_t version = 3;
    const std::uint64_t degree = static_cast<std::uint64_t>(pp.poly_modulus_degree);
    const std::uint64_t chunk_count = static_cast<std::uint64_t>(pp.c);
    const std::uint64_t plain_modulus = static_cast<std::uint64_t>(pp.plain_modulus);
    const std::uint64_t completed_count = static_cast<std::uint64_t>(completed);

    out.write(magic, sizeof(magic));
    out.write(reinterpret_cast<const char *>(&version), sizeof(version));
    out.write(reinterpret_cast<const char *>(&degree), sizeof(degree));
    out.write(reinterpret_cast<const char *>(&chunk_count), sizeof(chunk_count));
    out.write(reinterpret_cast<const char *>(&plain_modulus), sizeof(plain_modulus));
    out.write(reinterpret_cast<const char *>(&completed_count), sizeof(completed_count));
    out.write(reinterpret_cast<const char *>(weights.data()),
              static_cast<std::streamsize>(weights.size() * sizeof(u64)));
}

// Loads cached selector weights if possible; otherwise builds and caches them.
// SMALLPIR_EXACT_SELECTOR_WEIGHTS forces the slow exact validation path.
std::vector<u64> LoadOrPrecomputeSelectorWeights(const SmallPirParams &pp)
{
    if (std::getenv("SMALLPIR_EXACT_SELECTOR_WEIGHTS") != nullptr)
    {
        if (SmallPirVerbose()) std::cerr << "[dbg] selector weights exact build\n";
        return PrecomputeSelectorWeights(pp);
    }

    std::vector<u64> weights;
    std::size_t completed = 0;
    const std::string path = SelectorWeightsCachePath(pp);
    if (LoadSelectorWeightsCache(pp, weights, completed))
    {
        if (completed == pp.c)
        {
            if (SmallPirVerbose()) std::cerr << "[dbg] selector weights cache hit: " << path << '\n';
            return weights;
        }
        if (SmallPirVerbose())
        {
            std::cerr << "[dbg] selector weights cache resume: " << path
                      << " completed=" << completed << " / " << pp.c << '\n';
        }
    }
    else
    {
        if (SmallPirVerbose()) std::cerr << "[dbg] selector weights cache miss: " << path << '\n';
    }

    if (SmallPirVerbose()) std::cerr << "[dbg] selector weights formula build\n";
    weights = BuildSelectorWeightsFormula(pp);
    SaveSelectorWeightsCache(pp, weights, pp.c);
    if (SmallPirVerbose()) std::cerr << "[dbg] selector weights cache saved: " << path << '\n';
    return weights;
}

// Returns the selector correction weight for each global chunk.
// In sliced mode, the effective weight is local_chunk_weight * slice_weight.
std::vector<u64> BuildEffectiveSelectorWeights(const SmallPirParams &pp,
                                               bool sliced,
                                               bool encrypted_slice)
{
    if (!sliced)
    {
        return LoadOrPrecomputeSelectorWeights(pp);
    }

    SmallPirParams local_pp = pp;
    local_pp.c = ChooseSliceLen(pp.c);
    const std::vector<u64> local_weights = LoadOrPrecomputeSelectorWeights(local_pp);

    const std::size_t slice_count = CeilDiv(pp.c, local_pp.c);
    std::vector<u64> slice_weights(slice_count, 1ULL);
    if (encrypted_slice)
    {
        SmallPirParams slice_pp = pp;
        slice_pp.c = slice_count;
        slice_weights = LoadOrPrecomputeSelectorWeights(slice_pp);
    }

    std::vector<u64> weights(pp.c, 0ULL);
    for (std::size_t chunk_id = 0; chunk_id < pp.c; ++chunk_id)
    {
        const std::size_t slice_id = chunk_id / local_pp.c;
        const std::size_t local_id = chunk_id % local_pp.c;
        weights[chunk_id] = static_cast<u64>(
            (static_cast<unsigned __int128>(local_weights[local_id]) *
             slice_weights[slice_id]) % pp.plain_modulus);
    }
    return weights;
}

// Packs a fingerprint and one value limb into a single plaintext slot payload.
u64 EncodeItem(u64 fp, u64 value, const SmallPirParams &pp)
{
    u64 packed = (fp << pp.value_bits) | value;
    return packed;
}

// Unpacks a slot payload into fingerprint and value limb.
std::pair<u64, u64> DecodeItem(u64 packed, const SmallPirParams &pp)
{
    u64 value_mask = (1ULL << pp.value_bits) - 1ULL;
    u64 value = packed & value_mask;
    u64 fp = packed >> pp.value_bits;
    return {fp, value};
}

// Generates deterministic synthetic database values for benchmarks.
// Large values are represented as multiple 32-bit limbs across replica databases.
u64 SyntheticSmallValueLimb(u64 key, std::size_t limb, const SmallPirParams &pp)
{
    const u64 word = Mix64(key ^ Mix64(static_cast<u64>(limb) + 0x51A11FACEULL));
    if (pp.value_bits >= 64) return word;
    return word & ((1ULL << pp.value_bits) - 1ULL);
}

// Computes the nonzero fingerprint used by the client to recognize the correct
// entry among the two candidate buckets returned by the server.
u64 Fingerprint(u64 key, const SmallPirParams &pp)
{
    u64 fp = HashWithSeed(0xABCDEF1234567890ULL, key) & ((1ULL << pp.fp_bits) - 1ULL);
    if (fp == 0) fp = 1;
    return fp;
}

// Computes a stable routing tag used to derive the alternate Vacuum Filter bucket.
u64 RoutingTag(u64 key)
{
    return HashWithSeed(0x726f7574652d7461ULL, key);
}

// Derives a nonzero XOR delta for the alternate bucket. Since b is a power of
// two, current_bucket ^ delta stays within the same chunk segment.
std::size_t BucketDelta(u64 route, const SmallPirParams &pp)
{
    std::size_t delta = static_cast<std::size_t>(route & (pp.b - 1U));
    if (delta == 0) delta = 1;
    return delta;
}

// Total number of buckets across all chunks.
std::size_t GlobalBucketCount(const SmallPirParams &pp)
{
    return pp.c * pp.b;
}

// Maps a key to its primary global bucket under the current VF seed.
std::size_t PrimaryGlobalBucket(u64 key, const SmallPirParams &pp)
{
    return static_cast<std::size_t>(HashWithSeed(pp.vf_primary_seed, key) % GlobalBucketCount(pp));
}

// Computes the alternate global bucket for the same key route.
std::size_t AltGlobalBucket(std::size_t current_global_bucket, u64 route, const SmallPirParams &pp)
{
    return current_global_bucket ^ BucketDelta(route, pp);
}

// Converts a global bucket id into (chunk id, bucket id inside chunk).
std::pair<std::size_t, std::size_t> GlobalToChunkBucket(std::size_t global_bucket, const SmallPirParams &pp)
{
    return {global_bucket / pp.b, global_bucket % pp.b};
}

// Mutable access to a Vacuum Filter bucket by global bucket id.
Bucket &AccessBucket(VacuumFilter &vf, std::size_t global_bucket, const SmallPirParams &pp)
{
    auto [chunk_id, bucket_id] = GlobalToChunkBucket(global_bucket, pp);
    return vf.chunks[chunk_id][bucket_id];
}

// Const access to a Vacuum Filter bucket by global bucket id.
const Bucket &AccessBucket(const VacuumFilter &vf, std::size_t global_bucket, const SmallPirParams &pp)
{
    auto [chunk_id, bucket_id] = GlobalToChunkBucket(global_bucket, pp);
    return vf.chunks[chunk_id][bucket_id];
}

// Returns the primary chunk and bucket for a key.
std::pair<std::size_t, std::size_t> PrimaryChunkBucket(u64 key, const SmallPirParams &pp)
{
    return GlobalToChunkBucket(PrimaryGlobalBucket(key, pp), pp);
}

// Inserts into the first empty slot of a bucket.
bool TryInsertIntoBucket(Bucket &bucket, const VfEntry &entry)
{
    for (auto &slot : bucket.slots)
    {
        if (!slot.has_value())
        {
            slot = entry;
            return true;
        }
    }
    return false;
}

// Counts occupied slots in a bucket.
std::size_t BucketOccupancy(const Bucket &bucket)
{
    std::size_t count = 0;
    for (const auto &slot : bucket.slots)
    {
        count += slot.has_value() ? 1ULL : 0ULL;
    }
    return count;
}

// One-step relocation helper: evict a resident entry to its alternate bucket
// and place the incoming entry in the vacated slot if possible.
bool TryRelocateFromBucket(VacuumFilter &vf,
                           std::size_t global_bucket,
                           const SmallPirParams &pp,
                           const VfEntry &incoming)
{
    Bucket &bucket = AccessBucket(vf, global_bucket, pp);
    for (std::size_t slot_idx = 0; slot_idx < bucket.slots.size(); ++slot_idx)
    {
        if (!bucket.slots[slot_idx].has_value()) continue;
        const VfEntry evicted = *bucket.slots[slot_idx];
        std::size_t alt = AltGlobalBucket(global_bucket, evicted.route, pp);
        if (alt == global_bucket) continue;
        if (TryInsertIntoBucket(AccessBucket(vf, alt, pp), evicted))
        {
            bucket.slots[slot_idx] = incoming;
            return true;
        }
    }
    return false;
}

// Builds the Vacuum Filter for the key-value database.
// Key steps:
// 1. Try several primary hash seeds and reject seeds whose per-chunk load
//    already exceeds b*s.
// 2. For a feasible seed, insert items in randomized order.
// 3. If both candidate buckets are full, use cuckoo-style kicks and local
//    relocations until the item is placed or the attempt fails.
VacuumFilter BuildVacuumFilter(const std::vector<std::pair<u64, u64>> &db, SmallPirParams &pp)
{
    std::vector<std::size_t> order(db.size(), 0ULL);
    for (std::size_t i = 0; i < db.size(); ++i) order[i] = i;

    const std::size_t chunk_capacity = pp.b * pp.s;
    std::size_t feasible_seed_count = 0;

    for (std::size_t seed_trial = 0; seed_trial < pp.vf_seed_trials; ++seed_trial)
    {
        // Precheck is cheap and avoids expensive insertion attempts for seeds
        // that overload at least one chunk beyond its b*s capacity.
        const u64 candidate_seed = static_cast<u64>(seed_trial);
        std::vector<std::size_t> chunk_counts(pp.c, 0ULL);
        for (const auto &[key, value] : db)
        {
            (void)value;
            std::size_t global_bucket = static_cast<std::size_t>(HashWithSeed(candidate_seed, key) % GlobalBucketCount(pp));
            chunk_counts[global_bucket / pp.b]++;
        }

        bool chunk_feasible = true;
        std::size_t max_chunk_load = 0;
        for (std::size_t count : chunk_counts)
        {
            max_chunk_load = std::max(max_chunk_load, count);
            if (count > chunk_capacity)
            {
                chunk_feasible = false;
                break;
            }
        }
        if (!chunk_feasible)
        {
            if (SmallPirVerbose() && (seed_trial < 4 || ((seed_trial + 1) % 8 == 0)))
            {
                std::cerr << "[vf] seed=" << candidate_seed
                          << " rejected by chunk-capacity precheck"
                          << " max_chunk_load=" << max_chunk_load
                          << " capacity=" << chunk_capacity << '\n';
            }
            continue;
        }

        ++feasible_seed_count;
        if (SmallPirVerbose())
        {
            std::cerr << "[vf] seed=" << candidate_seed
                      << " passed precheck"
                      << " max_chunk_load=" << max_chunk_load
                      << " attempt_budget=" << pp.vf_build_retries << '\n';
        }

        for (std::size_t attempt = 0; attempt < pp.vf_build_retries; ++attempt)
        {
            // Each attempt keeps the same hash seed but shuffles insertion order.
            pp.vf_primary_seed = candidate_seed;

            VacuumFilter vf;
            vf.c = pp.c;
            vf.b = pp.b;
            vf.s = pp.s;
            vf.chunks.assign(pp.c, std::vector<Bucket>(pp.b));
            for (std::size_t c = 0; c < pp.c; ++c)
            {
                for (std::size_t b = 0; b < pp.b; ++b)
                {
                    vf.chunks[c][b].slots.assign(pp.s, std::nullopt);
                }
            }

            std::mt19937_64 rng(candidate_seed ^ (0x9e3779b97f4a7c15ULL * (attempt + 1)));
            std::shuffle(order.begin(), order.end(), rng);

            bool build_ok = true;
            std::size_t inserted_count = 0;
            for (std::size_t pos = 0; pos < order.size(); ++pos)
            {
                // Create the stored payload. The server stores fp||value; the
                // client later checks fp after decrypting the two candidate buckets.
                const auto &[key, value] = db[order[pos]];
                const u64 fp = Fingerprint(key, pp);
                const u64 route = RoutingTag(key);
                const u64 packed = EncodeItem(fp, value, pp);
                VfEntry cur{fp, route, value, packed};

                const std::size_t global_bucket1 = PrimaryGlobalBucket(key, pp);
                const std::size_t global_bucket2 = AltGlobalBucket(global_bucket1, route, pp);
                const auto [chunk_id, bucket1] = GlobalToChunkBucket(global_bucket1, pp);
                const auto [chunk_id2, bucket2] = GlobalToChunkBucket(global_bucket2, pp);
                if (chunk_id != chunk_id2)
                {
                    throw std::runtime_error("VF alternate bucket escaped its bound segment");
                }

                const std::size_t occ1 = BucketOccupancy(AccessBucket(vf, global_bucket1, pp));
                const std::size_t occ2 = BucketOccupancy(AccessBucket(vf, global_bucket2, pp));
                const bool prefer_first = (occ1 <= occ2);

                if ((prefer_first &&
                     (TryInsertIntoBucket(AccessBucket(vf, global_bucket1, pp), cur) ||
                      TryInsertIntoBucket(AccessBucket(vf, global_bucket2, pp), cur))) ||
                    (!prefer_first &&
                     (TryInsertIntoBucket(AccessBucket(vf, global_bucket2, pp), cur) ||
                      TryInsertIntoBucket(AccessBucket(vf, global_bucket1, pp), cur))))
                {
                    ++inserted_count;
                    continue;
                }

                std::size_t cur_bucket = (rng() & 1ULL) ? global_bucket1 : global_bucket2;
                Bucket &start_bucket = AccessBucket(vf, cur_bucket, pp);
                std::size_t slot_idx = static_cast<std::size_t>(rng() % pp.s);
                std::swap(cur, *start_bucket.slots[slot_idx]);

                std::size_t alt_bucket = AltGlobalBucket(cur_bucket, cur.route, pp);
                bool inserted = false;
                for (std::size_t kick = 0; kick < pp.vf_max_kicks; ++kick)
                {
                    // First try to place the evicted item in its alternate bucket.
                    // If that bucket is full, look for a resident that can move
                    // one step to an empty alternate bucket before random kicking.
                    Bucket &alt = AccessBucket(vf, alt_bucket, pp);
                    if (BucketOccupancy(alt) < pp.s)
                    {
                        if (!TryInsertIntoBucket(alt, cur))
                        {
                            throw std::runtime_error("vacuum insert logic hit inconsistent non-full bucket");
                        }
                        inserted = true;
                        break;
                    }

                    bool moved_from_alt = false;
                    for (std::size_t j = 0; j < pp.s; ++j)
                    {
                        if (!alt.slots[j].has_value()) continue;
                        const VfEntry resident = *alt.slots[j];
                        const std::size_t next_bucket = AltGlobalBucket(alt_bucket, resident.route, pp);
                        Bucket &next = AccessBucket(vf, next_bucket, pp);
                        if (BucketOccupancy(next) < pp.s)
                        {
                            if (!TryInsertIntoBucket(next, resident))
                            {
                                throw std::runtime_error("vacuum relocate found non-full bucket but insertion failed");
                            }
                            alt.slots[j] = cur;
                            inserted = true;
                            moved_from_alt = true;
                            break;
                        }
                    }
                    if (moved_from_alt) break;

                    std::size_t kick_slot = static_cast<std::size_t>(rng() % pp.s);
                    std::swap(cur, *alt.slots[kick_slot]);
                    alt_bucket = AltGlobalBucket(alt_bucket, cur.route, pp);
                }
                if (!inserted)
                {
                    if (SmallPirVerbose())
                    {
                        std::cerr << "[vf] seed=" << candidate_seed
                                  << " attempt=" << attempt
                                  << " failed after inserting " << inserted_count
                                  << " / " << db.size()
                                  << " items; fp_class=" << (fp & 3ULL)
                                  << " gb1=" << global_bucket1
                                  << " gb2=" << global_bucket2 << '\n';
                    }
                    build_ok = false;
                    break;
                }
                ++inserted_count;
            }

            if (build_ok)
            {
                if (SmallPirVerbose())
                {
                    std::cerr << "[vf] build succeeded with seed=" << candidate_seed
                              << " attempt=" << attempt
                              << " feasible_seeds_seen=" << feasible_seed_count << '\n';
                }
                return vf;
            }
        }
    }

    throw std::runtime_error("vacuum filter insertion failed after retries; increase c/b/s or lower load");
}

// Packs every bucket in every chunk into a GBFV plaintext and divides by the
// selector weight for that chunk. The later homomorphic selector expansion
// multiplies by this weight, so this inverse keeps the selected chunk unchanged.
std::vector<Plaintext> BuildWeightedChunkPolynomials(
    const VacuumFilter &vf,
    const std::vector<u64> &selector_weights,
    const SmallPirParams &pp,
    const GbfvEncoding &enc)
{
    if (selector_weights.size() != pp.c)
    {
        throw std::invalid_argument("weighted chunk polynomial dimensions mismatch");
    }

    std::vector<Plaintext> weighted_chunks(pp.c);
    for (std::size_t chunk_id = 0; chunk_id < pp.c; ++chunk_id)
    {
        const u64 inv = ModInvPrime(selector_weights[chunk_id], pp.plain_modulus);
        Poly cp{0};
        for (std::size_t bucket_id = 0; bucket_id < pp.b; ++bucket_id)
        {
            if (pp.gbfv_fp4_slots)
            {
                // F_{p^4} slot mode: one bucket is one degree-4 slot, so the
                // s bucket entries become coefficients inside that slot.
                Poly bucket_poly(pp.s, 0ULL);
                for (std::size_t slot_id = 0; slot_id < pp.s; ++slot_id)
                {
                    const auto &entry = vf.chunks[chunk_id][bucket_id].slots[slot_id];
                    bucket_poly[slot_id] = entry.has_value() ? entry->packed : 0ULL;
                }
                Poly term = MulPolyMod(bucket_poly, enc.basis_polys[bucket_id], pp.plain_modulus);
                cp = AddPolyMod(cp, term, pp.plain_modulus);
            }
            else
            {
                // Linear slot mode: each bucket entry gets its own BFV batching slot.
                for (std::size_t slot_id = 0; slot_id < pp.s; ++slot_id)
                {
                    const auto &entry = vf.chunks[chunk_id][bucket_id].slots[slot_id];
                    const u64 value = entry.has_value() ? entry->packed : 0ULL;
                    const std::size_t flat_slot = bucket_id * pp.s + slot_id;
                    Poly term = MulPolyMod(Poly{value}, enc.basis_polys[flat_slot], pp.plain_modulus);
                    cp = AddPolyMod(cp, term, pp.plain_modulus);
                }
            }
        }
        cp = ModPoly(cp, enc.t_poly, pp.plain_modulus);
        for (u64 &coeff : cp)
        {
            coeff = static_cast<u64>((static_cast<unsigned __int128>(coeff) * inv) % pp.plain_modulus);
        }
        weighted_chunks[chunk_id] = PlainFromPoly(cp);
    }
    return weighted_chunks;
}

// Builds one representative packed chunk plaintext for online-only benchmarks.
// It avoids constructing the full database while still exercising multiplication
// by realistic nonzero plaintext polynomials.
Plaintext BuildRepresentativePackedChunkPlaintext(const SmallPirParams &pp,
                                                  const GbfvEncoding &enc,
                                                  double slot_load,
                                                  std::size_t limb)
{
    const std::size_t total_slots = pp.b * pp.s;
    Poly cp{0};
    for (std::size_t flat_slot = 0; flat_slot < total_slots; ++flat_slot)
    {
        const u64 keep_score = HashWithSeed(0x5041434b45444f4eULL, flat_slot + 1ULL) % 1000000ULL;
        if (static_cast<double>(keep_score) >= slot_load * 1000000.0) continue;

        const u64 key = flat_slot + 1ULL;
        const u64 fp = Fingerprint(key, pp);
        const u64 value = SyntheticSmallValueLimb(key, limb, pp);
        const u64 packed = EncodeItem(fp, value, pp);
        Poly term;
        if (pp.gbfv_fp4_slots)
        {
            const std::size_t bucket_id = flat_slot / pp.s;
            const std::size_t slot_id = flat_slot % pp.s;
            Poly bucket_poly(pp.s, 0ULL);
            bucket_poly[slot_id] = packed;
            term = MulPolyMod(bucket_poly, enc.basis_polys[bucket_id], pp.plain_modulus);
        }
        else
        {
            term = MulPolyMod(Poly{packed}, enc.basis_polys[flat_slot], pp.plain_modulus);
        }
        cp = AddPolyMod(cp, term, pp.plain_modulus);
    }
    cp = ModPoly(cp, enc.t_poly, pp.plain_modulus);
    return PlainFromPoly(cp);
}

// Converts plaintext chunks to NTT form so multiply_plain in the answer phase is
// measured in the same representation as real server evaluation.
std::vector<Plaintext> TransformPlaintextsToNtt(
    const std::vector<Plaintext> &plain_grid,
    const SEALContext &context,
    Evaluator &evaluator)
{
    const auto parms_id = context.first_parms_id();
    std::vector<Plaintext> ntt_grid = plain_grid;
 #pragma omp parallel num_threads(kThreadCount)
    {
        Evaluator local_evaluator(context);
#pragma omp for schedule(static)
        for (std::size_t i = 0; i < ntt_grid.size(); ++i)
        {
            local_evaluator.transform_to_ntt_inplace(ntt_grid[i], parms_id);
        }
    }
    return ntt_grid;
}

// Builds a compressed one-hot selector as an encrypted monomial x^target_index.
// The server expands this ciphertext into a full selector vector with rotations.
Ciphertext BuildCompactIndexQuery(const SmallPirParams &pp,
                                  Encryptor &encryptor,
                                  std::size_t target_index,
                                  std::size_t expand_count)
{
    const std::size_t expand_len = NextPow2(expand_count);
    if (expand_len > pp.poly_modulus_degree)
    {
        throw std::runtime_error("compact query expand length exceeds polynomial degree");
    }
    if (target_index >= expand_count)
    {
        throw std::runtime_error("compact query target index is out of range");
    }

    Plaintext pt = MakeMonomialPlain(target_index);
    Ciphertext ct;
    encryptor.encrypt(pt, ct);
    return ct;
}

// Builds the compressed selector for the global target chunk.
Ciphertext BuildCompactChunkQuery(const SmallPirParams &pp,
                                  Encryptor &encryptor,
                                  std::size_t target_chunk)
{
    return BuildCompactIndexQuery(pp, encryptor, target_chunk, pp.c);
}

// Builds the encrypted bucket mask selecting the two Vacuum Filter buckets that
// may contain the target key. In F_{p^4} mode each bucket corresponds to one slot.
Ciphertext BuildBucketMaskQuery(const SmallPirParams &pp,
                                const GbfvEncoding &enc,
                                Encryptor &encryptor,
                                std::size_t bucket1,
                                std::size_t bucket2)
{
    Poly mask{0ULL};
    if (pp.gbfv_fp4_slots)
    {
        // Add the two bucket basis polynomials; multiplying by this mask keeps
        // exactly the two candidate bucket slots in the selected chunk.
        mask = AddPolyMod(mask, enc.basis_polys[bucket1], pp.plain_modulus);
        mask = AddPolyMod(mask, enc.basis_polys[bucket2], pp.plain_modulus);
    }
    else
    {
        for (std::size_t slot_id = 0; slot_id < pp.s; ++slot_id)
        {
            const std::size_t flat1 = bucket1 * pp.s + slot_id;
            const std::size_t flat2 = bucket2 * pp.s + slot_id;
            mask = AddPolyMod(mask, enc.basis_polys[flat1], pp.plain_modulus);
            mask = AddPolyMod(mask, enc.basis_polys[flat2], pp.plain_modulus);
        }
    }
    mask = ModPoly(mask, enc.t_poly, pp.plain_modulus);
    Poly gbfv_mask = enc.beta_is_one ?
        mask :
        MulPolyModModulus(mask, enc.beta_poly, enc.phi_poly, pp.plain_modulus);
    Plaintext pt = PlainFromPoly(gbfv_mask);
    Ciphertext ct;
    encryptor.encrypt(pt, ct);
    return ct;
}

// Expands a compressed selector to chunk_count ciphertexts using the standard
// power-of-two PIR expansion tree, then trims the unused tail.
std::vector<Ciphertext> ExpandToPow2ThenTrim(const Ciphertext &compact_query,
                                             std::size_t chunk_count,
                                             const SEALContext &context,
                                             Evaluator &evaluator,
                                             const GaloisKeys &gk)
{
    const std::size_t logical_n = context.key_context_data()->parms().poly_modulus_degree();
    if (chunk_count > logical_n)
    {
        throw std::runtime_error("expand length exceeds polynomial degree");
    }

    std::vector<Ciphertext> current{compact_query};
    std::size_t logm = 0;
    while ((1ULL << logm) < chunk_count) ++logm;

    std::vector<std::uint32_t> galois_elts(logm, 0);
    for (std::size_t i = 0; i < logm; ++i)
    {
        galois_elts[i] = static_cast<std::uint32_t>((logical_n + (1ULL << i)) / (1ULL << i));
    }

    for (std::size_t i = 0; i < logm; ++i)
    {
        // Each level doubles the number of selector ciphertexts by combining a
        // negacyclic shift and a Galois automorphism. The last level may be
        // asymmetric when chunk_count is not a power of two.
        const bool is_last = (i == logm - 1);
        std::vector<Ciphertext> next(std::min<std::size_t>(2 * current.size(), chunk_count));
        const std::size_t tail_count = chunk_count - current.size();

        const std::size_t index_raw = (logical_n << 1U) - (1ULL << i);
        const std::size_t index = (index_raw * galois_elts[i]) % (logical_n << 1U);

        for (std::size_t j = 0; j < current.size(); ++j)
        {
            Ciphertext c0 = current[j];

            if (is_last && j >= tail_count)
            {
                next[j] = std::move(c0);
                continue;
            }

            Ciphertext c1 = MultiplyPowerOfX(c0, index, context);
            Ciphertext sub0;
            evaluator.apply_galois(c0, galois_elts[i], gk, sub0);
            evaluator.add(sub0, c0, next[j]);

            if (is_last)
            {
                next[j + current.size()] = std::move(c1);
                continue;
            }

            Ciphertext sub1;
            evaluator.apply_galois(c1, galois_elts[i], gk, sub1);
            evaluator.add(sub1, c1, next[j + current.size()]);
        }
        current = std::move(next);
    }
    return current;
}

// Converts a BFV selector ciphertext into the simulated GBFV ideal by applying
// beta when beta != 1.
Ciphertext ConvertBFVToGBFV(const Ciphertext &ct_bfv,
                            const GbfvEncoding &enc,
                            Evaluator &evaluator)
{
    Ciphertext out = ct_bfv;
    if (!enc.beta_is_one)
    {
        evaluator.multiply_plain_inplace(out, enc.beta_plain);
    }
    return out;
}

// Constructs the client query for a key.
// It derives the fingerprint, the two candidate buckets, and either one global
// chunk selector or two sliced selectors, then encrypts the bucket mask.
Query BuildQuery(u64 key,
                 const SmallPirParams &pp,
                 const GbfvEncoding &enc,
                 Encryptor &encryptor,
                 bool sliced,
                 bool encrypted_slice)
{
    Query q;
    q.fp = Fingerprint(key, pp);
    const u64 route = RoutingTag(key);
    const std::size_t global_bucket1 = PrimaryGlobalBucket(key, pp);
    const std::size_t global_bucket2 = AltGlobalBucket(global_bucket1, route, pp);
    auto [chunk_id, bucket1] = GlobalToChunkBucket(global_bucket1, pp);
    auto [chunk_id2, bucket2] = GlobalToChunkBucket(global_bucket2, pp);
    if (chunk_id != chunk_id2)
    {
        throw std::runtime_error("query buckets escape their bound segment");
    }
    q.target_chunk = chunk_id;
    q.bucket1 = bucket1;
    q.bucket2 = bucket2;
    if (sliced)
    {
        // Sliced mode separates "which slice" from "which chunk within slice".
        // If encrypted_slice is false, target_slice is used in the clear.
        q.slice_len = ChooseSliceLen(pp.c);
        q.slice_count = CeilDiv(pp.c, q.slice_len);
        q.target_slice = q.target_chunk / q.slice_len;
        q.local_chunk = q.target_chunk % q.slice_len;
        if (encrypted_slice)
        {
            q.q_sl_bfv = BuildCompactIndexQuery(pp, encryptor, q.target_slice, q.slice_count);
        }
        q.q_ch_bfv = BuildCompactIndexQuery(pp, encryptor, q.local_chunk, q.slice_len);
    }
    else
    {
        q.q_ch_bfv = BuildCompactChunkQuery(pp, encryptor, q.target_chunk);
    }
    q.q_bk_gbfv = BuildBucketMaskQuery(pp, enc, encryptor, q.bucket1, q.bucket2);
    return q;
}

// Expands a compressed BFV selector and converts every expanded ciphertext to
// GBFV form. transform_to_ntt prepares the result for multiply_plain.
std::vector<Ciphertext> ExpandCompactQueryToGBFV(const Ciphertext &compact_query,
                                                 std::size_t expand_count,
                                                 const SEALContext &context,
                                                 const GbfvEncoding &enc,
                                                 Evaluator &evaluator,
                                                 const GaloisKeys &gk,
                                                 bool transform_to_ntt)
{
    const auto expanded_bfv = ExpandToPow2ThenTrim(
        compact_query, expand_count, context, evaluator, gk);

    std::vector<Ciphertext> expanded_gbfv;
    expanded_gbfv.resize(expand_count);
#pragma omp parallel num_threads(kThreadCount)
    {
        Evaluator local_evaluator(context);
#pragma omp for schedule(static)
    for (std::size_t i = 0; i < expand_count; ++i)
    {
            Ciphertext ct = ConvertBFVToGBFV(expanded_bfv[i], enc, local_evaluator);
        if (transform_to_ntt)
        {
                local_evaluator.transform_to_ntt_inplace(ct);
        }
            expanded_gbfv[i] = std::move(ct);
        }
    }
    return expanded_gbfv;
}

// Expands the global chunk selector for the unsliced query variant.
std::vector<Ciphertext> ExpandChunkQueryToGBFV(const Query &query,
                                               const SmallPirParams &pp,
                                               const SEALContext &context,
                                               const GbfvEncoding &enc,
                                               Evaluator &evaluator,
                                               const GaloisKeys &gk)
{
    return ExpandCompactQueryToGBFV(
        query.q_ch_bfv, pp.c, context, enc, evaluator, gk, true);
}

// Homomorphically selects one chunk from all weighted chunk plaintexts:
// sum_i expanded_selector[i] * weighted_chunk[i].
Ciphertext SelectChunkCipher(const std::vector<Ciphertext> &expanded_gbfv,
                             const SmallPirParams &pp,
                             const std::vector<Plaintext> &weighted_chunk_polys_ntt,
                             const SEALContext &context,
                             Evaluator &evaluator)
{
    std::vector<Ciphertext> partials(kThreadCount);
    std::vector<unsigned char> partial_init(kThreadCount, 0);
#pragma omp parallel num_threads(kThreadCount)
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        Evaluator local_evaluator(context);
        Ciphertext local_sum;
        bool initialized = false;
#pragma omp for schedule(static)
    for (std::size_t i = 0; i < pp.c; ++i)
    {
        Ciphertext term;
            local_evaluator.multiply_plain(expanded_gbfv[i], weighted_chunk_polys_ntt[i], term);
        if (!initialized)
        {
                local_sum = std::move(term);
            initialized = true;
        }
        else
        {
                local_evaluator.add_inplace(local_sum, term);
            }
        }
        if (initialized)
        {
            partials[tid] = std::move(local_sum);
            partial_init[tid] = 1;
        }
    }

    Ciphertext selected_chunk_ct;
    bool initialized = false;
    for (int tid = 0; tid < kThreadCount; ++tid)
    {
        if (!partial_init[tid]) continue;
        if (!initialized)
        {
            selected_chunk_ct = std::move(partials[tid]);
            initialized = true;
        }
        else
        {
            evaluator.add_inplace(selected_chunk_ct, partials[tid]);
        }
    }
    if (!initialized)
    {
        throw std::runtime_error("empty chunk selection");
    }
    evaluator.transform_from_ntt_inplace(selected_chunk_ct);
    return selected_chunk_ct;
}

// Same selection as SelectChunkCipher, but only over one slice of global chunks.
Ciphertext SelectChunkCipherRange(const std::vector<Ciphertext> &expanded_local_gbfv,
                                  std::size_t global_begin,
                                  std::size_t global_end,
                                  const std::vector<Plaintext> &weighted_chunk_polys_ntt,
                                  const SEALContext &context,
                                  Evaluator &evaluator)
{
    std::vector<Ciphertext> partials(kThreadCount);
    std::vector<unsigned char> partial_init(kThreadCount, 0);
#pragma omp parallel num_threads(kThreadCount)
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        Evaluator local_evaluator(context);
        Ciphertext local_sum;
        bool initialized = false;
#pragma omp for schedule(static)
    for (std::size_t global_i = global_begin; global_i < global_end; ++global_i)
    {
        const std::size_t local_i = global_i - global_begin;
        Ciphertext term;
            local_evaluator.multiply_plain(
            expanded_local_gbfv[local_i], weighted_chunk_polys_ntt[global_i], term);
        if (!initialized)
        {
                local_sum = std::move(term);
            initialized = true;
        }
        else
        {
                local_evaluator.add_inplace(local_sum, term);
            }
        }
        if (initialized)
        {
            partials[tid] = std::move(local_sum);
            partial_init[tid] = 1;
        }
    }

    Ciphertext selected_chunk_ct;
    bool initialized = false;
    for (int tid = 0; tid < kThreadCount; ++tid)
    {
        if (!partial_init[tid]) continue;
        if (!initialized)
        {
            selected_chunk_ct = std::move(partials[tid]);
            initialized = true;
        }
        else
        {
            evaluator.add_inplace(selected_chunk_ct, partials[tid]);
        }
    }
    if (!initialized)
    {
        throw std::runtime_error("empty slice selection");
    }
    evaluator.transform_from_ntt_inplace(selected_chunk_ct);
    return selected_chunk_ct;
}

// Applies beta^{-1} after GBFV ct-ct multiplication so the result is back in the
// representation expected by recovery.
void NormalizeGBFVCiphertextInplace(Ciphertext &ct,
                                    const GbfvEncoding &enc,
                                    Evaluator &evaluator)
{
    if (!enc.beta_inv_is_one)
    {
        evaluator.multiply_plain_inplace(ct, enc.beta_inv_plain);
    }
}

// Server answer for the unsliced protocol.
// Steps: expand chunk selector, select the packed chunk, multiply by encrypted
// bucket mask, relinearize, and normalize the GBFV representation.
Response Answer(const Query &query,
                const SmallPirParams &pp,
                const std::vector<Plaintext> &weighted_chunk_polys_ntt,
                const SEALContext &context,
                const GbfvEncoding &enc,
                Evaluator &evaluator,
                const GaloisKeys &gk,
                const RelinKeys &relin_keys)
{
    using Clock = std::chrono::steady_clock;
    using Ms = std::chrono::duration<double, std::milli>;

    const auto t_expand_s = Clock::now();
    const auto expanded_gbfv = ExpandChunkQueryToGBFV(query, pp, context, enc, evaluator, gk);
    const auto t_expand_e = Clock::now();

    const auto t_select_s = Clock::now();
    Ciphertext selected_chunk_ct = SelectChunkCipher(
        expanded_gbfv, pp, weighted_chunk_polys_ntt, context, evaluator);
    const auto t_select_e = Clock::now();

    const auto t_ctct_s = Clock::now();
    Ciphertext ans_ct;
    // ct-ct multiplication intersects the selected chunk with the encrypted
    // two-bucket mask, leaving only candidate bucket contents.
    evaluator.multiply(selected_chunk_ct, query.q_bk_gbfv, ans_ct);
    evaluator.relinearize_inplace(ans_ct, relin_keys);
    NormalizeGBFVCiphertextInplace(ans_ct, enc, evaluator);
    const auto t_ctct_e = Clock::now();

    Response resp;
    resp.ans = std::move(ans_ct);
    resp.expand_ms = Ms(t_expand_e - t_expand_s).count();
    resp.select_ms = Ms(t_select_e - t_select_s).count();
    resp.ctct_ms = Ms(t_ctct_e - t_ctct_s).count();
    return resp;
}

// Server answer for fully encrypted slicing.
// E_chunk is expanded once per slice length and reused across all slices;
// E_slice selects the final slice result before the bucket mask is applied.
Response AnswerSliced(const Query &query,
                      const SmallPirParams &pp,
                      const std::vector<Plaintext> &weighted_chunk_polys_ntt,
                      const SEALContext &context,
                      const GbfvEncoding &enc,
                      Evaluator &evaluator,
                      const GaloisKeys &gk,
                      const RelinKeys &relin_keys)
{
    using Clock = std::chrono::steady_clock;
    using Ms = std::chrono::duration<double, std::milli>;

    if (query.slice_len == 0 || query.slice_count == 0)
    {
        throw std::runtime_error("sliced query was not initialized");
    }

    const auto t_expand_s = Clock::now();
    const auto expanded_chunk_gbfv = ExpandCompactQueryToGBFV(
        query.q_ch_bfv, query.slice_len, context, enc, evaluator, gk, true);
    const auto expanded_slice_gbfv = ExpandCompactQueryToGBFV(
        query.q_sl_bfv, query.slice_count, context, enc, evaluator, gk, false);
    const auto t_expand_e = Clock::now();

    const auto t_select_s = Clock::now();
    std::vector<Ciphertext> selected_slices;
    selected_slices.reserve(query.slice_count);
    for (std::size_t slice_id = 0; slice_id < query.slice_count; ++slice_id)
    {
        // Produce one encrypted partial answer per slice using the same
        // intra-slice expanded chunk selector.
        const std::size_t begin = slice_id * query.slice_len;
        const std::size_t end = std::min<std::size_t>(begin + query.slice_len, pp.c);
        selected_slices.push_back(SelectChunkCipherRange(
            expanded_chunk_gbfv, begin, end, weighted_chunk_polys_ntt, context, evaluator));
    }
    const auto t_select_e = Clock::now();

    const auto t_ctct_s = Clock::now();
    Ciphertext selected_chunk_ct;
    bool initialized = false;
    for (std::size_t slice_id = 0; slice_id < query.slice_count; ++slice_id)
    {
        // Multiply each slice partial by its encrypted slice selector bit and add.
        Ciphertext term;
        evaluator.multiply(selected_slices[slice_id], expanded_slice_gbfv[slice_id], term);
        evaluator.relinearize_inplace(term, relin_keys);
        NormalizeGBFVCiphertextInplace(term, enc, evaluator);
        if (!initialized)
        {
            selected_chunk_ct = std::move(term);
            initialized = true;
        }
        else
        {
            evaluator.add_inplace(selected_chunk_ct, term);
        }
    }
    if (!initialized)
    {
        throw std::runtime_error("empty slice aggregation");
    }

    Ciphertext ans_ct;
    evaluator.multiply(selected_chunk_ct, query.q_bk_gbfv, ans_ct);
    evaluator.relinearize_inplace(ans_ct, relin_keys);
    NormalizeGBFVCiphertextInplace(ans_ct, enc, evaluator);
    const auto t_ctct_e = Clock::now();

    Response resp;
    resp.ans = std::move(ans_ct);
    resp.expand_ms = Ms(t_expand_e - t_expand_s).count();
    resp.select_ms = Ms(t_select_e - t_select_s).count();
    resp.ctct_ms = Ms(t_ctct_e - t_ctct_s).count();
    return resp;
}

// Server answer for plaintext-slice mode. The target slice index is public, so
// the server expands only the intra-slice selector and evaluates one slice.
Response AnswerPlainSlice(const Query &query,
                          const SmallPirParams &pp,
                          const std::vector<Plaintext> &weighted_chunk_polys_ntt,
                          const SEALContext &context,
                          const GbfvEncoding &enc,
                          Evaluator &evaluator,
                          const GaloisKeys &gk,
                          const RelinKeys &relin_keys)
{
    using Clock = std::chrono::steady_clock;
    using Ms = std::chrono::duration<double, std::milli>;

    if (query.slice_len == 0 || query.slice_count == 0)
    {
        throw std::runtime_error("plain-slice query was not initialized");
    }

    const std::size_t begin = query.target_slice * query.slice_len;
    const std::size_t end = std::min<std::size_t>(begin + query.slice_len, pp.c);
    if (begin >= end)
    {
        throw std::runtime_error("plain-slice target range is empty");
    }

    const auto t_expand_s = Clock::now();
    const auto expanded_chunk_gbfv = ExpandCompactQueryToGBFV(
        query.q_ch_bfv, query.slice_len, context, enc, evaluator, gk, true);
    const auto t_expand_e = Clock::now();

    const auto t_select_s = Clock::now();
    Ciphertext selected_chunk_ct = SelectChunkCipherRange(
        expanded_chunk_gbfv, begin, end, weighted_chunk_polys_ntt, context, evaluator);
    const auto t_select_e = Clock::now();

    const auto t_ctct_s = Clock::now();
    Ciphertext ans_ct;
    evaluator.multiply(selected_chunk_ct, query.q_bk_gbfv, ans_ct);
    evaluator.relinearize_inplace(ans_ct, relin_keys);
    NormalizeGBFVCiphertextInplace(ans_ct, enc, evaluator);
    const auto t_ctct_e = Clock::now();

    Response resp;
    resp.ans = std::move(ans_ct);
    resp.expand_ms = Ms(t_expand_e - t_expand_s).count();
    resp.select_ms = Ms(t_select_e - t_select_s).count();
    resp.ctct_ms = Ms(t_ctct_e - t_ctct_s).count();
    return resp;
}

// Client recovery. It decrypts the returned GBFV ciphertext, projects it to the
// two candidate bucket slots, unpacks fp||value, and returns the value whose
// fingerprint matches the queried key.
std::optional<u64> RecoverValue(const Query &query,
                                const Response &resp,
                                const SmallPirParams &pp,
                                const GbfvEncoding &enc,
                                Decryptor &decryptor)
{
    std::array<Poly, 2> bucket_slots{Poly(pp.s, 0ULL), Poly(pp.s, 0ULL)};
    const std::array<std::size_t, 2> buckets{query.bucket1, query.bucket2};
    Poly poly = PolyFromPlain(DecryptPlain(decryptor, resp.ans), pp.plain_modulus);
    poly = MulPolyModModulus(poly, enc.beta_inv_mod_t, enc.t_poly, pp.plain_modulus);
    if (pp.gbfv_fp4_slots)
    {
        // In F_{p^4} mode one bucket is recovered by reducing modulo its
        // degree-4 slot factor.
        for (std::size_t which = 0; which < buckets.size(); ++which)
        {
            Poly theta = ModPoly(poly, enc.slot_moduli[buckets[which]], pp.plain_modulus);
            theta.resize(pp.s, 0ULL);
            for (std::size_t slot_id = 0; slot_id < pp.s; ++slot_id)
            {
                bucket_slots[which][slot_id] = theta[slot_id] % pp.plain_modulus;
            }
        }
    }
    else
    {
        // In linear slot mode each slot is recovered independently.
        for (std::size_t which = 0; which < buckets.size(); ++which)
        {
            for (std::size_t slot_id = 0; slot_id < pp.s; ++slot_id)
            {
                const std::size_t flat_slot = buckets[which] * pp.s + slot_id;
                Poly theta = ModPoly(poly, enc.slot_moduli[flat_slot], pp.plain_modulus);
                theta.resize(1, 0ULL);
                bucket_slots[which][slot_id] = theta[0] % pp.plain_modulus;
            }
        }
    }

    for (const Poly &bucket_data : bucket_slots)
    {
        // The server returns two full candidate buckets. The client performs the
        // final private-side fingerprint comparison locally.
        for (std::size_t i = 0; i < pp.s; ++i)
        {
            u64 digit = bucket_data[i] % pp.plain_modulus;
            if (std::getenv("SMALLPIR_DEBUG_RECOVER") != nullptr)
            {
                std::cerr << "[dbg] recover bucket_slot=" << i
                          << " digit=" << digit;
                if (digit != 0)
                {
                    auto [dbg_fp, dbg_value] = DecodeItem(digit, pp);
                    std::cerr << " fp=" << dbg_fp << " value=" << dbg_value;
                }
                std::cerr << '\n';
            }
            if (digit == 0) continue;
            auto [fp, value] = DecodeItem(digit, pp);
            if (fp == query.fp)
            {
                return value;
            }
        }
    }
    return std::nullopt;
}

// Debug/helper routine: project a decrypted polynomial to one bucket slot factor.
Poly DecodeBucketPoly(const Poly &poly,
                      const GbfvEncoding &enc,
                      std::size_t bucket_id,
                      const SmallPirParams &pp)
{
    Poly theta = ModPoly(poly, enc.slot_moduli[bucket_id], pp.plain_modulus);
    theta.resize(pp.s, 0ULL);
    return theta;
}

// End-to-end benchmark driver. Command-line parameters set n,N,b,s,c, load,
// plaintext modulus bits, fingerprint bits, and value size. Environment flags
// choose full prep vs online-only prep and sliced vs unsliced query modes.
int main(int argc, char **argv)
{
    try
    {
#ifdef _OPENMP
        omp_set_num_threads(kThreadCount);
#endif
        using Clock = std::chrono::steady_clock;
        using Ms = std::chrono::duration<double, std::milli>;

        SmallPirParams pp;
        std::size_t n_log2 = 20;
        std::size_t total_value_bits = 32;
        double target_load = 0.9343;
        const bool online_const_prep = std::getenv("SMALLPIR_ONLINE_ONLY") != nullptr;
        const bool online_packed_prep = std::getenv("SMALLPIR_ONLINE_PACKED_PREP") != nullptr;
        const bool sliced_query = std::getenv("SMALLPIR_SLICED") != nullptr;
        const bool encrypted_slice = sliced_query && std::getenv("SMALLPIR_ENCRYPTED_SLICE") != nullptr;
        const bool online_only = online_const_prep || online_packed_prep;
        pp.gbfv_fp4_slots = std::getenv("SMALLPIR_GBFV_FP4") != nullptr;
        pp.gbfv_binomial_t = std::getenv("SMALLPIR_GBFV_BINOMIAL") != nullptr;

        // Parse benchmark geometry and payload settings. If c is omitted, choose
        // it from the requested load factor.
        if (argc >= 2) n_log2 = static_cast<std::size_t>(std::stoull(argv[1]));
        if (argc >= 3) pp.poly_modulus_degree = static_cast<std::size_t>(std::stoull(argv[2]));
        if (argc >= 4) pp.b = static_cast<std::size_t>(std::stoull(argv[3]));
        if (argc >= 5) pp.s = static_cast<std::size_t>(std::stoull(argv[4]));
        if (argc >= 6) pp.c = static_cast<std::size_t>(std::stoull(argv[5]));
        else pp.c = AutoChunkCount(1ULL << n_log2, pp.b, pp.s, target_load);

        if (argc >= 7) target_load = std::stod(argv[6]);
        if (argc >= 8) pp.plain_modulus_bits = std::stoi(argv[7]);
        if (argc >= 9) pp.fp_bits = static_cast<std::size_t>(std::stoull(argv[8]));
        if (argc >= 10) total_value_bits = static_cast<std::size_t>(std::stoull(argv[9]));
        if (argc < 6) pp.c = AutoChunkCount(1ULL << n_log2, pp.b, pp.s, target_load);
        if (total_value_bits == 0) throw std::invalid_argument("value_bits must be positive");
        const std::size_t value_limb_count = CeilDiv(total_value_bits, 32);
        pp.value_bits = std::min<std::size_t>(total_value_bits, 32U);

        pp.slot_degree = pp.s;
        pp.vf_alt_masks = MakeAltMasks(pp.b);

        // Encrypted-slice mode consumes more noise because it performs an
        // additional ciphertext-ciphertext slice selection. Let the caller
        // choose a longer coeff modulus chain for those experiments.
        if (encrypted_slice)
        {
            std::size_t coeff_prime_count = 9;
            if (const char *env_prime_count = std::getenv("SMALLPIR_COEFF_PRIMES"))
            {
                coeff_prime_count = static_cast<std::size_t>(std::stoull(env_prime_count));
            }
            coeff_prime_count = std::max<std::size_t>(coeff_prime_count, 5U);
            pp.coeff_modulus_bits.assign(coeff_prime_count, 55);
        }

        if (n_log2 >= 22 && pp.vf_max_kicks < 65536) pp.vf_max_kicks = 65536;
        if (n_log2 >= 22 && pp.vf_build_retries < 32) pp.vf_build_retries = 32;
        if (n_log2 >= 22 && pp.vf_seed_trials < 128) pp.vf_seed_trials = 128;
        if (n_log2 >= 26 && pp.vf_max_kicks < 131072) pp.vf_max_kicks = 131072;
        if (n_log2 >= 26 && pp.vf_build_retries < 48) pp.vf_build_retries = 48;
        if (n_log2 >= 28 && pp.vf_max_kicks < 262144) pp.vf_max_kicks = 262144;
        if (n_log2 >= 28 && pp.vf_build_retries < 64) pp.vf_build_retries = 64;
        if (n_log2 >= 28 && pp.vf_seed_trials < 256) pp.vf_seed_trials = 256;

        // Create SEAL BFV parameters. We intentionally use sec_level_type::none
        // because these experiments include nonstandard research parameters.
        EncryptionParameters parms(scheme_type::bfv);
        parms.set_poly_modulus_degree(pp.poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(pp.poly_modulus_degree, pp.coeff_modulus_bits));
        if (pp.gbfv_fp4_slots)
        {
            parms.set_plain_modulus(FindFp4PlainModulus(pp.poly_modulus_degree, pp.plain_modulus_bits));
        }
        else
        {
            parms.set_plain_modulus(PlainModulus::Batching(pp.poly_modulus_degree, pp.plain_modulus_bits));
        }

        SEALContext context(parms, true, sec_level_type::none);
        if (!context.parameters_set()) throw std::runtime_error("invalid SEAL parameters");

        pp.plain_modulus = parms.plain_modulus().value();
        const u64 max_item_code =
            (((1ULL << pp.fp_bits) - 1ULL) << pp.value_bits) |
            ((1ULL << pp.value_bits) - 1ULL);
        if (pp.plain_modulus <= max_item_code)
        {
            throw std::runtime_error("plain modulus too small for packed slot payloads");
        }

        // Generate keys. Galois keys are created exactly for the rotations used
        // by compact selector expansion.
        KeyGenerator keygen(context);
        SecretKey sk = keygen.secret_key();
        PublicKey pk;
        keygen.create_public_key(pk);
        GaloisKeys gk;
        std::vector<std::uint32_t> galois_elts;
        const std::size_t expand_len = NextPow2(pp.c);
        const std::size_t logm = static_cast<std::size_t>(std::log2(static_cast<double>(expand_len)));
        for (std::size_t j = 0; j < logm; ++j)
        {
            galois_elts.push_back(static_cast<std::uint32_t>((pp.poly_modulus_degree >> j) + 1));
        }
        keygen.create_galois_keys(galois_elts, gk);
        RelinKeys relin_keys;
        keygen.create_relin_keys(relin_keys);

        Encryptor encryptor(context, pk);
        Evaluator evaluator(context);
        Decryptor decryptor(context, sk);
        std::vector<u64> selector_weights_public;
        // Full-prep needs selector weights to cancel expansion scaling in each
        // stored chunk. Online-only modes skip them because they use synthetic
        // representative plaintexts.
        if (online_only)
        {
            if (SmallPirVerbose()) std::cerr << "[dbg] online-only: skipping selector weights\n";
        }
        else
        {
            selector_weights_public = BuildEffectiveSelectorWeights(pp, sliced_query, encrypted_slice);
            if (SmallPirVerbose()) std::cerr << "[dbg] selector weights ready\n";
        }
        const GbfvEncoding gbfv = BuildToyGbfvEncoding(pp);
        if (SmallPirVerbose()) std::cerr << "[dbg] gbfv encoding ready\n";

        // Prep phase. Full mode builds a synthetic DB, inserts it into the
        // Vacuum Filter, and packs chunks. Online-only modes build placeholder
        // plaintexts so the online homomorphic path can be timed at large n.
        const std::size_t db_size = 1ULL << n_log2;
        const u64 target_key = std::min<u64>(523123ULL, static_cast<u64>(db_size));
        const auto t_prep_s = Clock::now();
        std::vector<std::vector<Plaintext>> weighted_chunk_polys_by_limb;
        weighted_chunk_polys_by_limb.reserve(value_limb_count);
        if (online_const_prep)
        {
            // Cheapest online-only placeholder: every chunk plaintext is 1.
            Plaintext one(1);
            one[0] = 1;
            for (std::size_t limb = 0; limb < value_limb_count; ++limb)
            {
                weighted_chunk_polys_by_limb.emplace_back(pp.c, one);
            }
            if (SmallPirVerbose()) std::cerr << "[dbg] online-only: using constant-one chunk plaintexts\n";
        }
        else if (online_packed_prep)
        {
            // More realistic online-only placeholder: one representative packed
            // chunk is reused for every chunk and every value limb.
            for (std::size_t limb = 0; limb < value_limb_count; ++limb)
            {
                Plaintext representative = BuildRepresentativePackedChunkPlaintext(
                    pp, gbfv, target_load, limb);
                weighted_chunk_polys_by_limb.emplace_back(pp.c, representative);
            }
            if (SmallPirVerbose()) std::cerr << "[dbg] online-only: using representative packed chunk plaintexts\n";
        }
        else
        {
            // Full prep path: build the actual synthetic database and construct
            // the Vacuum Filter before packing every chunk into GBFV plaintexts.
            if (value_limb_count != 1)
            {
                throw std::runtime_error("full prep currently supports one value limb; use online packed prep for larger values");
            }
            std::vector<std::pair<u64, u64>> db;
            db.reserve(db_size);
            for (u64 key = 1; key <= db_size; ++key)
            {
                u64 value = SyntheticSmallValueLimb(key, 0, pp);
                db.emplace_back(key, value);
            }
            if (SmallPirVerbose()) std::cerr << "[dbg] db ready size=" << db.size() << "\n";

            VacuumFilter vf = BuildVacuumFilter(db, pp);
            if (SmallPirVerbose()) std::cerr << "[dbg] vf ready\n";
            if (std::getenv("SMALLPIR_DEBUG_RECOVER") != nullptr)
            {
                const u64 route = RoutingTag(target_key);
                const std::size_t global_bucket1 = PrimaryGlobalBucket(target_key, pp);
                const std::size_t global_bucket2 = AltGlobalBucket(global_bucket1, route, pp);
                const auto [chunk_id, bucket1] = GlobalToChunkBucket(global_bucket1, pp);
                const auto [chunk_id2, bucket2] = GlobalToChunkBucket(global_bucket2, pp);
                std::cerr << "[dbg] target vf chunk=" << chunk_id
                          << " alt_chunk=" << chunk_id2
                          << " buckets=(" << bucket1 << "," << bucket2 << ")"
                          << " fp=" << Fingerprint(target_key, pp)
                          << " value=" << SyntheticSmallValueLimb(target_key, 0, pp)
                          << " packed=" << EncodeItem(Fingerprint(target_key, pp),
                                                      SyntheticSmallValueLimb(target_key, 0, pp),
                                                      pp)
                          << '\n';
                const std::array<std::size_t, 2> globals{global_bucket1, global_bucket2};
                for (std::size_t which = 0; which < globals.size(); ++which)
                {
                    const Bucket &bucket = AccessBucket(vf, globals[which], pp);
                    for (std::size_t slot_id = 0; slot_id < pp.s; ++slot_id)
                    {
                        std::cerr << "[dbg] vf bucket=" << which
                                  << " slot=" << slot_id;
                        if (bucket.slots[slot_id].has_value())
                        {
                            const VfEntry &entry = *bucket.slots[slot_id];
                            std::cerr << " fp=" << entry.fp
                                      << " value=" << entry.value
                                      << " packed=" << entry.packed
                                      << " target=" << (entry.fp == Fingerprint(target_key, pp) &&
                                                        entry.value == SyntheticSmallValueLimb(target_key, 0, pp));
                        }
                        else
                        {
                            std::cerr << " empty";
                        }
                        std::cerr << '\n';
                    }
                }
            }
            weighted_chunk_polys_by_limb.push_back(
                BuildWeightedChunkPolynomials(vf, selector_weights_public, pp, gbfv));
        }
        std::vector<std::vector<Plaintext>> weighted_chunk_polys_ntt_by_limb;
        weighted_chunk_polys_ntt_by_limb.reserve(value_limb_count);
        // Store server plaintexts in NTT form to match the online multiply path.
        for (const auto &weighted_chunk_polys : weighted_chunk_polys_by_limb)
        {
            weighted_chunk_polys_ntt_by_limb.push_back(
                TransformPlaintextsToNtt(weighted_chunk_polys, context, evaluator));
        }
        if (SmallPirVerbose()) std::cerr << "[dbg] chunk polys ready\n";
        const auto t_prep_e = Clock::now();

        // Query phase: the client encrypts chunk/slice selector(s) and bucket mask.
        const auto t_query_s = Clock::now();
        Query query = BuildQuery(target_key, pp, gbfv, encryptor, sliced_query, encrypted_slice);
        const auto t_query_e = Clock::now();

        // Answer phase: the server evaluates one response per 32-bit value limb.
        const auto t_answer_s = Clock::now();
        std::vector<Response> responses;
        responses.reserve(value_limb_count);
        Response total_resp;
        int min_noise_ans = std::numeric_limits<int>::max();
        for (std::size_t limb = 0; limb < value_limb_count; ++limb)
        {
            Response limb_resp = sliced_query ?
                (encrypted_slice ?
                     AnswerSliced(query, pp, weighted_chunk_polys_ntt_by_limb[limb], context, gbfv, evaluator, gk, relin_keys) :
                     AnswerPlainSlice(query, pp, weighted_chunk_polys_ntt_by_limb[limb], context, gbfv, evaluator, gk, relin_keys)) :
                Answer(query, pp, weighted_chunk_polys_ntt_by_limb[limb], context, gbfv, evaluator, gk, relin_keys);
            min_noise_ans = std::min(min_noise_ans, decryptor.invariant_noise_budget(limb_resp.ans));
            total_resp.expand_ms += limb_resp.expand_ms;
            total_resp.select_ms += limb_resp.select_ms;
            total_resp.ctct_ms += limb_resp.ctct_ms;
            responses.push_back(std::move(limb_resp));
        }
        const auto t_answer_e = Clock::now();

        // Decrypt/recover phase: client checks fingerprints inside the two
        // candidate buckets and reconstructs value limbs.
        const auto t_dec_s = Clock::now();
        std::vector<std::optional<u64>> recovered_limbs;
        recovered_limbs.reserve(value_limb_count);
        for (const auto &resp : responses)
        {
            recovered_limbs.push_back(RecoverValue(query, resp, pp, gbfv, decryptor));
        }
        const auto t_dec_e = Clock::now();

        bool ok = online_only;
        if (!online_only)
        {
            // Full mode verifies correctness against the synthetic value generator.
            ok = true;
            for (std::size_t limb = 0; limb < value_limb_count; ++limb)
            {
                const u64 expected_limb = SyntheticSmallValueLimb(target_key, limb, pp);
                ok = ok && recovered_limbs[limb].has_value() &&
                     recovered_limbs[limb].value() == expected_limb;
            }
        }

        std::string mode = online_packed_prep ? "ONLINE_ONLY_PACKED_PREP" :
                           (online_const_prep ? "ONLINE_ONLY_CONST_PREP" : "FULL");
        if (sliced_query) mode += encrypted_slice ? "_SLICED_ENC_SLICE" : "_SLICED_PLAIN_SLICE";

        const double prep_ms = Ms(t_prep_e - t_prep_s).count();
        const double query_ms = Ms(t_query_e - t_query_s).count();
        const double answer_ms = Ms(t_answer_e - t_answer_s).count();
        const double decrypt_ms = Ms(t_dec_e - t_dec_s).count();
        const double online_total_ms = query_ms + answer_ms + decrypt_ms;

        // Print a compact machine-readable timing line followed by readable labels.
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "timing_ms prep=" << prep_ms
                  << ", request=" << query_ms
                  << ", expand=" << total_resp.expand_ms
                  << ", select=" << total_resp.select_ms
                  << ", ctct=" << total_resp.ctct_ms
                  << ", response=" << answer_ms
                  << ", answer=" << decrypt_ms
                  << ", online_total=" << online_total_ms << "\n";

        std::cout << "Prep time        : " << prep_ms << " ms\n";
        std::cout << "Query time       : " << query_ms << " ms\n";
        std::cout << "Answer time      : " << answer_ms << " ms\n";
        std::cout << "  Expand time    : " << total_resp.expand_ms << " ms\n";
        std::cout << "  Select time    : " << total_resp.select_ms << " ms\n";
        std::cout << "  ct-ct time     : " << total_resp.ctct_ms << " ms\n";
        std::cout << "Decrypt time     : " << decrypt_ms << " ms\n";
        std::cout << "Online total     : " << online_total_ms << " ms\n";

        return ok ? 0 : 2;
    }
    catch (const std::exception &ex)
    {
        std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }
}
