# 2026PIR

This is the c++ implementation of SlimPIR (submitted to ICDE2027) using the Microsoft SEAL library~\cite{sealcrypto}. We performed our experiments server running Ubuntu 20.04.6, equipped with an Intel Xeon Gold CPU @ 2.10 GHz and 192 GB of RAM. All experiments are performed with single-threaded execution.


# SlimPIR README

`SlimPIR.cpp` is a single-file implementation of the SlimPIR/SmallPIR prototype. It benchmarks a keyword PIR design based on a Vacuum Filter layout and BFV/GBFV-style encrypted selectors.

This README explains how to build the file, run it, and configure the main parameters, especially the database size and value size.

## Dependencies

- A C++17 compiler, such as `g++`
- Microsoft SEAL headers and static library
- Optional: OpenMP for multi-threaded builds

The current workspace can reuse the SEAL build under `Pantheon/SEAL_parallel`.

## Build

Single-thread build:

```bash
g++ -O2 -std=c++17 SlimPIR.cpp -o SlimPIR \
  -IPantheon/SEAL_parallel/native/src \
  -IPantheon/SEAL_parallel/build/native/src \
  Pantheon/SEAL_parallel/build/lib/libseal-0.0.a \
  -pthread
```

Example 8-thread build:

```bash
g++ -O2 -std=c++17 -fopenmp -DSMALLPIR_THREAD_COUNT=8 SlimPIR.cpp -o SlimPIR_t8 \
  -IPantheon/SEAL_parallel/native/src \
  -IPantheon/SEAL_parallel/build/native/src \
  Pantheon/SEAL_parallel/build/lib/libseal-0.0.a \
  -pthread
```

To build other thread counts, change `SMALLPIR_THREAD_COUNT` and the output name, for example `SlimPIR_t16` or `SlimPIR_t32`.

## Command-Line Parameters

Run format:

```bash
./SlimPIR [n_log2] [N] [b] [s] [c] [load] [plain_modulus_bits] [fp_bits] [value_bits]
```

Parameter meaning:

| Position | Parameter | Meaning | Default |
|---:|---|---|---:|
| 1 | `n_log2` | Number of records is `2^n_log2` | `20` |
| 2 | `N` | BFV polynomial modulus degree | `8192` |
| 3 | `b` | Number of buckets per chunk | `1024` |
| 4 | `s` | Number of slots per bucket | `4` |
| 5 | `c` | Number of chunks | computed from `load` if omitted |
| 6 | `load` | Target Vacuum Filter load factor | `0.9343` |
| 7 | `plain_modulus_bits` | Bit length of the plaintext modulus | `37` |
| 8 | `fp_bits` | Fingerprint bit length | `4` |
| 9 | `value_bits` | Value size in bits | `32` |

The two most important parameters are:

- Database size: set `n_log2`. For example, `28` means `2^28` records.
- Value size: set `value_bits`. For example, `32` means each value is 32 bits.

The current full-preprocessing path supports one 32-bit value limb. Therefore, use `value_bits <= 32` for full end-to-end correctness tests unless the code is extended to support full preprocessing for multiple limbs.

## Recommended Parameters

The following parameter set is used for the current 32-bit value experiments.

| Records | `n_log2` | `N` | `c` | `b` | `s` | Load |
|---:|---:|---:|---:|---:|---:|---:|
| `2^20` | 20 | 4096 | 284 | 1024 | 4 | about 0.9000 |
| `2^22` | 22 | 8192 | 568 | 2048 | 4 | about 0.9000 |
| `2^24` | 24 | 16384 | 1137 | 4096 | 4 | about 0.9000 |
| `2^26` | 26 | 16384 | 4551 | 4096 | 4 | about 0.9001 |
| `2^28` | 28 | 32768 | 9102 | 8192 | 4 | about 0.9000 |

When `c` is explicitly provided, the program uses that value directly. The `load` argument is only used to compute `c` when `c` is omitted.

## Run Examples

Small correctness test with `2^20` records and 32-bit values:

```bash
SMALLPIR_GBFV_FP4=1 \
./SlimPIR 20 4096 1024 4 284 0.900022 50 16 32
```

Larger run with `2^24` records and 32-bit values:

```bash
SMALLPIR_GBFV_FP4=1 \
./SlimPIR 24 16384 4096 4 1137 0.900000 50 16 32
```

The last argument controls the value size. For example, this command sets 16-bit values:

```bash
SMALLPIR_GBFV_FP4=1 \
./SlimPIR 20 4096 1024 4 284 0.900022 50 16 16
```

## Environment Variables

| Environment variable | Meaning |
|---|---|
| `SMALLPIR_GBFV_FP4=1` | Use simulated `F_{p^4}` slots; each bucket is packed into one degree-4 slot |
| `SMALLPIR_GBFV_BINOMIAL=1` | Use the binomial `t(x)` GBFV simulation path |
| `SMALLPIR_SLICED=1` | Use sliced chunk selection |
| `SMALLPIR_ENCRYPTED_SLICE=1` | Encrypt the slice selector; requires `SMALLPIR_SLICED=1` |
| `SMALLPIR_COEFF_PRIMES=6` | In encrypted-slice mode, set the coefficient modulus chain to `6 x 55-bit` |
| `SMALLPIR_VERBOSE=1` | Print debug messages |
| `SMALLPIR_DEBUG_RECOVER=1` | Print bucket and slot details during recovery |
| `SMALLPIR_EXACT_SELECTOR_WEIGHTS=1` | Use the slow exact selector-weight computation path for validation |

## Output Fields

All timing values are reported in milliseconds.

| Field | Meaning |
|---|---|
| `prep` | Database construction, Vacuum Filter insertion, and chunk packing time |
| `request` / `Query time` | Client query generation time |
| `expand` | Server selector expansion time |
| `select` | Server chunk or slice selection time |
| `ctct` | Server ciphertext-ciphertext multiplication time |
| `response` / `Answer time` | Total server answer time |
| `answer` / `Decrypt time` | Client decryption and fingerprint-checking time |
| `online_total` | `Query + Answer + Decrypt` |

## Notes

- The code creates the SEAL context with `sec_level_type::none` to allow non-standard research parameters. This does not mean that the parameters automatically satisfy SEAL's default 128-bit security checks.
- `SMALLPIR_GBFV_FP4=1` expects `s=4`, since one bucket is represented as one simulated `F_{p^4}` slot.
- Encrypted-slice mode consumes more noise than plaintext-slice mode. Use `SMALLPIR_COEFF_PRIMES=6` or a larger chain if the noise budget is insufficient.
- Large full-preprocessing runs can be slow and memory-intensive because the program builds the synthetic database, inserts all records into the Vacuum Filter, and packs every chunk.
