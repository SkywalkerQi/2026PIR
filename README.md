# 2026PIR

This is the c++ implementation of SlimPIR (submitted to ICDE2027) using the Microsoft SEAL library~\cite{sealcrypto}. We performed our experiments server running Ubuntu 20.04.6, equipped with an Intel Xeon Gold CPU @ 2.10 GHz and 192 GB of RAM. All experiments are performed with single-threaded execution.


# SlimPIR 使用说明

`SlimPIR.cpp` 是 SmallPIR/SlimPIR 的单文件实验程序，用于测试 Vacuum Filter + GBFV/BFV selector 的在线查询流程。程序支持 full-prep 和 online-only 两种测试方式；大规模实验建议先用 online-only packed prep。

## 依赖

- C++17 编译器，例如 `g++`
- Microsoft SEAL 头文件和静态库
- 可选：OpenMP，用于多线程版本

本仓库当前可直接复用 `Pantheon/SEAL_parallel` 里的 SEAL：

```bash
g++ -O2 -std=c++17 SlimPIR.cpp -o SlimPIR \
  -IPantheon/SEAL_parallel/native/src \
  -IPantheon/SEAL_parallel/build/native/src \
  Pantheon/SEAL_parallel/build/lib/libseal-0.0.a \
  -pthread
```

如果要编译多线程版本，例如 8 线程：

```bash
g++ -O2 -std=c++17 -fopenmp -DSMALLPIR_THREAD_COUNT=8 SlimPIR.cpp -o SlimPIR_t8 \
  -IPantheon/SEAL_parallel/native/src \
  -IPantheon/SEAL_parallel/build/native/src \
  Pantheon/SEAL_parallel/build/lib/libseal-0.0.a \
  -pthread
```

## 命令行参数

运行格式：

```bash
./SlimPIR [n_log2] [N] [b] [s] [c] [load] [plain_modulus_bits] [fp_bits] [value_bits]
```

参数含义：

| 位置 | 参数 | 含义 | 默认值 |
|---:|---|---|---:|
| 1 | `n_log2` | 数据量为 `2^n_log2` | `20` |
| 2 | `N` | BFV 多项式阶，即 `poly_modulus_degree` | `8192` |
| 3 | `b` | 每个 chunk 中的 bucket 数 | `1024` |
| 4 | `s` | 每个 bucket 中的 slot 数 | `4` |
| 5 | `c` | chunk 数 | 自动按 load 计算 |
| 6 | `load` | Vacuum Filter 目标负载率 | `0.9343` |
| 7 | `plain_modulus_bits` | plaintext modulus 的 bit 数 | `37` |
| 8 | `fp_bits` | fingerprint bit 数 | `4` |
| 9 | `value_bits` | 每条数据 value 的 bit 数 | `32` |

最常改的是：

- 数据量：改第 1 个参数 `n_log2`，例如 `28` 表示 `2^28` 条数据。
- 数据大小：改第 9 个参数 `value_bits`，例如 `32`、`64`、`128`、`256`。

注意：`value_bits > 32` 时，程序会把 value 拆成多个 32-bit limb，并对每个 limb 跑一次响应流程。当前 full-prep 路径只支持一个 limb；测试 64/128/256-bit value 时请使用 `SMALLPIR_ONLINE_PACKED_PREP=1`。

## 常用环境变量

| 环境变量 | 作用 |
|---|---|
| `SMALLPIR_ONLINE_ONLY=1` | online-only，使用常数 plaintext 占位，最快但不够真实 |
| `SMALLPIR_ONLINE_PACKED_PREP=1` | online-only，构造代表性的 packed chunk plaintext，推荐用于大规模在线测试 |
| `SMALLPIR_SLICED=1` | 开启 sliced query，将 chunk selector 拆成 slice selector + intra-slice selector |
| `SMALLPIR_ENCRYPTED_SLICE=1` | slice selector 也加密；需要同时设置 `SMALLPIR_SLICED=1` |
| `SMALLPIR_GBFV_FP4=1` | 使用模拟的 `F_{p^4}` slot，每个 bucket 打包到一个 degree-4 slot |
| `SMALLPIR_GBFV_BINOMIAL=1` | 使用 binomial `t(x)` 的 GBFV 模拟路径 |
| `SMALLPIR_COEFF_PRIMES=6` | encrypted-slice 模式下设置 coeff modulus 为 `6 x 55-bit` |
| `SMALLPIR_VERBOSE=1` | 输出调试信息 |
| `SMALLPIR_DEBUG_RECOVER=1` | 输出恢复阶段的 bucket/slot 调试信息 |
| `SMALLPIR_EXACT_SELECTOR_WEIGHTS=1` | 使用慢速精确 selector weight 预计算，通常只用于验证 |

## 推荐运行方式

### 1. 小规模 full-prep 正确性测试

full-prep 会真实构建 synthetic database、插入 Vacuum Filter，并打包所有 chunk。建议先用 `2^20` 以内测试：

```bash
SMALLPIR_GBFV_FP4=1 \
./SlimPIR 20 4096 1024 4 284 0.900022 50 16 32
```

如果 full-prep 成功，程序会输出：

```text
timing_ms prep=..., request=..., expand=..., select=..., ctct=..., response=..., answer=..., online_total=...
```

### 2. 大规模 online-only packed prep 测试

大规模测试建议使用 packed prep，占位 plaintext 更接近真实 packed chunk，但不构建完整数据库：

```bash
SMALLPIR_GBFV_FP4=1 \
SMALLPIR_ONLINE_PACKED_PREP=1 \
SMALLPIR_SLICED=1 \
SMALLPIR_ENCRYPTED_SLICE=1 \
SMALLPIR_COEFF_PRIMES=6 \
./SlimPIR 28 32768 8192 4 9102 0.900022 50 16 32
```

其中最后的 `32` 是 value bit 数。如果要测试 128-bit value：

```bash
SMALLPIR_GBFV_FP4=1 \
SMALLPIR_ONLINE_PACKED_PREP=1 \
SMALLPIR_SLICED=1 \
SMALLPIR_ENCRYPTED_SLICE=1 \
SMALLPIR_COEFF_PRIMES=6 \
./SlimPIR 28 32768 8192 4 9102 0.900022 50 16 128
```

## 推荐参数表

下面是当前实验中使用的一组 32-bit value 参数：

| 数据量 | `n_log2` | `N` | `c` | `b` | `s` | load 约值 |
|---:|---:|---:|---:|---:|---:|---:|
| `2^20` | 20 | 4096 | 284 | 1024 | 4 | 0.9000 |
| `2^22` | 22 | 8192 | 568 | 2048 | 4 | 0.9000 |
| `2^24` | 24 | 16384 | 1137 | 4096 | 4 | 0.9000 |
| `2^26` | 26 | 16384 | 4551 | 4096 | 4 | 0.9001 |
| `2^28` | 28 | 32768 | 9102 | 8192 | 4 | 0.9000 |

运行时命令中的 `load` 参数只用于 online-only representative plaintext 的填充率，或者在省略 `c` 时自动计算 chunk 数。若已经显式给出 `c`，程序不会用 load 重新覆盖 `c`。

## 输出字段

程序输出的核心时间字段如下，单位都是 ms：

| 字段 | 含义 |
|---|---|
| `prep` | full-prep 或 online-only plaintext 准备时间 |
| `request` / `Query time` | 客户端生成加密查询的时间 |
| `expand` | 服务端扩展 compact selector 的时间 |
| `select` | 服务端用 expanded selector 选择 chunk/slice 的时间 |
| `ctct` | 服务端执行 ciphertext-ciphertext bucket/slice 选择的时间 |
| `response` / `Answer time` | 服务端完整 Answer 时间 |
| `answer` / `Decrypt time` | 客户端解密并比较 fingerprint 的时间 |
| `online_total` | `Query + Answer + Decrypt` |

## 注意事项

- `SEALContext` 使用 `sec_level_type::none`，这是为了允许研究实验中的非标准参数通过 SEAL 检查，不代表默认 128-bit security。
- encrypted-slice 模式默认使用 `9 x 55-bit` coeff modulus；如果设置 `SMALLPIR_COEFF_PRIMES=6`，则使用 `6 x 55-bit`。
- `SMALLPIR_GBFV_FP4=1` 要求 `s=4`，因为每个 bucket 被模拟为一个 `F_{p^4}` slot。
- full-prep 对大规模数据会非常慢且占用大量内存；大规模画图数据建议使用 `SMALLPIR_ONLINE_PACKED_PREP=1`。
