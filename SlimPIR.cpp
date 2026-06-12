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

bool SmallPirVerbose()
{
    return std::getenv("SMALLPIR_VERBOSE") != nullptr;
}

struct VfEntry
{
    u64 fp{};
    u64 route{};
    u64 value{};
    u64 packed{};
};

struct Bucket
{
    std::vector<std::optional<VfEntry>> slots;
};

struct VacuumFilter
{
    std::size_t c{};
    std::size_t b{};
    std::size_t s{};
    std::vector<std::vector<Bucket>> chunks;
};

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

struct Response
{
    Ciphertext ans;
    double expand_ms{0.0};
    double select_ms{0.0};
    double ctct_ms{0.0};
};

u64 Mix64(u64 x)
{
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

u64 HashWithSeed(u64 seed, u64 x)
{
    return Mix64(seed ^ (x + 0x9e3779b97f4a7c15ULL));
}

std::size_t NextPow2(std::size_t x)
{
    std::size_t p = 1;
    while (p < x) p <<= 1U;
    return p;
}

std::size_t CeilDiv(std::size_t a, std::size_t b)
{
    if (b == 0) throw std::invalid_argument("division by zero");
    return (a + b - 1U) / b;
}

std::size_t ChooseSliceLen(std::size_t chunk_count)
{
    const auto root = static_cast<std::size_t>(
        std::ceil(std::sqrt(static_cast<long double>(chunk_count))));
    return NextPow2(std::max<std::size_t>(root, 1U));
}

std::size_t AutoChunkCount(std::size_t n, std::size_t b, std::size_t s, double target_load)
{
    const long double denom = static_cast<long double>(b) * static_cast<long double>(s) * target_load;
    return static_cast<std::size_t>(std::ceil(static_cast<long double>(n) / denom));
}

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

u64 MulMod128(u64 a, u64 b, u64 mod)
{
    return static_cast<u64>((static_cast<unsigned __int128>(a) * b) % mod);
}

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

u64 ModInvPrime(u64 x, u64 mod)
{
    if (x % mod == 0) throw std::runtime_error("inverse of zero does not exist");
    return ModPow(x, mod - 2, mod);
}

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

std::vector<u64> DecodePlaintextSlots(const BatchEncoder &encoder, const Plaintext &pt)
{
    std::vector<u64> slots;
    encoder.decode(pt, slots);
    return slots;
}

Plaintext EncodeAllSlots(const BatchEncoder &encoder, u64 value)
{
    std::vector<u64> slots(encoder.slot_count(), value);
    Plaintext pt;
    encoder.encode(slots, pt);
    return pt;
}

Ciphertext EncryptVector(const BatchEncoder &encoder,
                         Encryptor &encryptor,
                         const std::vector<u64> &values)
{
    Plaintext pt = EncodeVector(encoder, values);
    Ciphertext ct;
    encryptor.encrypt(pt, ct);
    return ct;
}

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

Plaintext DecryptPlain(Decryptor &decryptor, const Ciphertext &ct)
{
    Plaintext pt;
    decryptor.decrypt(ct, pt);
    return pt;
}

Plaintext MakeMonomialPlain(std::size_t exponent)
{
    Plaintext pt(exponent + 1);
    pt.set_zero();
    pt[exponent] = 1;
    return pt;
}

Plaintext MakeConstantPlain(u64 value)
{
    Plaintext pt(1);
    pt[0] = value;
    return pt;
}

Poly TrimPoly(Poly a)
{
    while (!a.empty() && a.back() == 0) a.pop_back();
    if (a.empty()) a.push_back(0);
    return a;
}

bool IsOnePoly(const Poly &poly)
{
    Poly trimmed = TrimPoly(poly);
    return trimmed.size() == 1 && trimmed[0] == 1ULL;
}

u64 ModSub(u64 a, u64 b, u64 mod)
{
    return (a >= b) ? (a - b) : (a + mod - b);
}

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

Poly PowerOfTwoCyclotomicPoly(std::size_t degree)
{
    Poly phi(degree + 1, 0ULL);
    phi[0] = 1ULL;
    phi[degree] = 1ULL;
    return phi;
}

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

Poly ModPoly(const Poly &a, const Poly &modulus, u64 mod)
{
    return DivModMonic(a, modulus, mod).second;
}

Poly MulPolyModModulus(const Poly &a, const Poly &b, const Poly &modulus, u64 mod)
{
    return ModPoly(MulPolyMod(a, b, mod), modulus, mod);
}

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

Poly XPowPowersMod(std::size_t power, const Poly &modulus, u64 mod)
{
    Poly x{0, 1};
    unsigned __int128 exp = 1;
    for (std::size_t i = 0; i < power; ++i) exp *= mod;
    return PolyPowMod(x, exp, modulus, mod);
}

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

Plaintext PlainFromPoly(const Poly &poly)
{
    Poly trimmed = TrimPoly(poly);
    Plaintext pt(trimmed.size());
    pt.set_zero();
    for (std::size_t i = 0; i < trimmed.size(); ++i) pt[i] = trimmed[i];
    return pt;
}

Poly PolyFromPlain(const Plaintext &pt, u64 mod)
{
    std::size_t deg = std::max<std::size_t>(1, pt.significant_coeff_count());
    Poly poly(deg, 0ULL);
    for (std::size_t i = 0; i < deg; ++i) poly[i] = pt[i] % mod;
    return TrimPoly(std::move(poly));
}

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
        auto div = DivModMonic(enc.t_poly, enc.slot_moduli[i], pp.plain_modulus);
        Poly Mi = std::move(div.first);
        Poly ri = ModPoly(Mi, enc.slot_moduli[i], pp.plain_modulus);
        Poly ri_inv = InversePolyMod(ri, enc.slot_moduli[i], pp.plain_modulus);
        Poly basis = ModPoly(MulPolyMod(Mi, ri_inv, pp.plain_modulus), enc.t_poly, pp.plain_modulus);
        enc.basis_polys.push_back(std::move(basis));
    }

    return enc;
}

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

std::vector<u64> PrecomputeSelectorWeights(const SmallPirParams &pp)
{
    std::vector<u64> weights(pp.c, 0ULL);
    for (std::size_t target = 0; target < pp.c; ++target)
    {
        weights[target] = PrecomputeSelectorWeight(pp, target);
    }
    return weights;
}

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

std::vector<u64> BuildSelectorWeightsFormula(const SmallPirParams &pp)
{
    std::vector<u64> weights(pp.c, 0ULL);
    for (std::size_t target = 0; target < pp.c; ++target)
    {
        weights[target] = SelectorWeightFormula(pp, target);
    }
    return weights;
}

std::string SelectorWeightsCachePath(const SmallPirParams &pp)
{
    std::size_t expand_len = 1ULL;
    while (expand_len < pp.c) expand_len <<= 1ULL;
    return ".smallpir_cache/selector_weights_v3_N" + std::to_string(pp.poly_modulus_degree) +
           "_c" + std::to_string(pp.c) +
           "_expand" + std::to_string(expand_len) +
           "_p" + std::to_string(pp.plain_modulus) + ".bin";
}

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

u64 EncodeItem(u64 fp, u64 value, const SmallPirParams &pp)
{
    u64 packed = (fp << pp.value_bits) | value;
    return packed;
}

std::pair<u64, u64> DecodeItem(u64 packed, const SmallPirParams &pp)
{
    u64 value_mask = (1ULL << pp.value_bits) - 1ULL;
    u64 value = packed & value_mask;
    u64 fp = packed >> pp.value_bits;
    return {fp, value};
}

u64 SyntheticSmallValueLimb(u64 key, std::size_t limb, const SmallPirParams &pp)
{
    const u64 word = Mix64(key ^ Mix64(static_cast<u64>(limb) + 0x51A11FACEULL));
    if (pp.value_bits >= 64) return word;
    return word & ((1ULL << pp.value_bits) - 1ULL);
}

u64 Fingerprint(u64 key, const SmallPirParams &pp)
{
    u64 fp = HashWithSeed(0xABCDEF1234567890ULL, key) & ((1ULL << pp.fp_bits) - 1ULL);
    if (fp == 0) fp = 1;
    return fp;
}

u64 RoutingTag(u64 key)
{
    return HashWithSeed(0x726f7574652d7461ULL, key);
}

std::size_t BucketDelta(u64 route, const SmallPirParams &pp)
{
    std::size_t delta = static_cast<std::size_t>(route & (pp.b - 1U));
    if (delta == 0) delta = 1;
    return delta;
}

std::size_t GlobalBucketCount(const SmallPirParams &pp)
{
    return pp.c * pp.b;
}

std::size_t PrimaryGlobalBucket(u64 key, const SmallPirParams &pp)
{
    return static_cast<std::size_t>(HashWithSeed(pp.vf_primary_seed, key) % GlobalBucketCount(pp));
}

std::size_t AltGlobalBucket(std::size_t current_global_bucket, u64 route, const SmallPirParams &pp)
{
    return current_global_bucket ^ BucketDelta(route, pp);
}

std::pair<std::size_t, std::size_t> GlobalToChunkBucket(std::size_t global_bucket, const SmallPirParams &pp)
{
    return {global_bucket / pp.b, global_bucket % pp.b};
}

Bucket &AccessBucket(VacuumFilter &vf, std::size_t global_bucket, const SmallPirParams &pp)
{
    auto [chunk_id, bucket_id] = GlobalToChunkBucket(global_bucket, pp);
    return vf.chunks[chunk_id][bucket_id];
}

const Bucket &AccessBucket(const VacuumFilter &vf, std::size_t global_bucket, const SmallPirParams &pp)
{
    auto [chunk_id, bucket_id] = GlobalToChunkBucket(global_bucket, pp);
    return vf.chunks[chunk_id][bucket_id];
}

std::pair<std::size_t, std::size_t> PrimaryChunkBucket(u64 key, const SmallPirParams &pp)
{
    return GlobalToChunkBucket(PrimaryGlobalBucket(key, pp), pp);
}

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

std::size_t BucketOccupancy(const Bucket &bucket)
{
    std::size_t count = 0;
    for (const auto &slot : bucket.slots)
    {
        count += slot.has_value() ? 1ULL : 0ULL;
    }
    return count;
}

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

VacuumFilter BuildVacuumFilter(const std::vector<std::pair<u64, u64>> &db, SmallPirParams &pp)
{
    std::vector<std::size_t> order(db.size(), 0ULL);
    for (std::size_t i = 0; i < db.size(); ++i) order[i] = i;

    const std::size_t chunk_capacity = pp.b * pp.s;
    std::size_t feasible_seed_count = 0;

    for (std::size_t seed_trial = 0; seed_trial < pp.vf_seed_trials; ++seed_trial)
    {
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

Ciphertext BuildCompactChunkQuery(const SmallPirParams &pp,
                                  Encryptor &encryptor,
                                  std::size_t target_chunk)
{
    return BuildCompactIndexQuery(pp, encryptor, target_chunk, pp.c);
}

Ciphertext BuildBucketMaskQuery(const SmallPirParams &pp,
                                const GbfvEncoding &enc,
                                Encryptor &encryptor,
                                std::size_t bucket1,
                                std::size_t bucket2)
{
    Poly mask{0ULL};
    if (pp.gbfv_fp4_slots)
    {
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

void NormalizeGBFVCiphertextInplace(Ciphertext &ct,
                                    const GbfvEncoding &enc,
                                    Evaluator &evaluator)
{
    if (!enc.beta_inv_is_one)
    {
        evaluator.multiply_plain_inplace(ct, enc.beta_inv_plain);
    }
}

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

Poly DecodeBucketPoly(const Poly &poly,
                      const GbfvEncoding &enc,
                      std::size_t bucket_id,
                      const SmallPirParams &pp)
{
    Poly theta = ModPoly(poly, enc.slot_moduli[bucket_id], pp.plain_modulus);
    theta.resize(pp.s, 0ULL);
    return theta;
}

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

        const std::size_t db_size = 1ULL << n_log2;
        const u64 target_key = std::min<u64>(523123ULL, static_cast<u64>(db_size));
        const auto t_prep_s = Clock::now();
        std::vector<std::vector<Plaintext>> weighted_chunk_polys_by_limb;
        weighted_chunk_polys_by_limb.reserve(value_limb_count);
        if (online_const_prep)
        {
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
        for (const auto &weighted_chunk_polys : weighted_chunk_polys_by_limb)
        {
            weighted_chunk_polys_ntt_by_limb.push_back(
                TransformPlaintextsToNtt(weighted_chunk_polys, context, evaluator));
        }
        if (SmallPirVerbose()) std::cerr << "[dbg] chunk polys ready\n";
        const auto t_prep_e = Clock::now();

        const auto t_query_s = Clock::now();
        Query query = BuildQuery(target_key, pp, gbfv, encryptor, sliced_query, encrypted_slice);
        const auto t_query_e = Clock::now();

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
