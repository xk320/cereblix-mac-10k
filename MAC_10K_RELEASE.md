# Cereblix Mac 10K Client v1.2.12

This build is based on upstream `Cerebra-CBR/cereblix` commit
`1f19999e71574643cb3e23482f99e648e481990e` and continues the Apple Silicon
optimization track on the latest miner v1.2 codebase.

## What Changed Since v1.2.11

- Added a darwin/arm64 C VM execution fast path for the NeuroMorph instruction
  loop, including arm64 AES instructions for the VM `AESR` opcode.
- Added a darwin/arm64 C program-generation fast path, reducing isolated
  `BenchmarkGenProgram` on `macmini87` from about `4.13 us/op` to `3.02 us/op`.
- Added an arm64 slow-vs-fast differential test covering pre-dataset and
  post-dataset heights across multiple epoch parameters.
- Retuned Apple Silicon default threads for the C VM path. `macmini87` now peaks
  at 10 threads in pool accepted-share testing.

## What Is Included

- Upstream miner v1.2 connection resilience and macOS-aware update support.
- Latest upstream consensus changes including the gated soft fee floor.
- Mac darwin/arm64 ARM AES scratch-fill fast path.
- Mac darwin/arm64 NEON scratch-fold fast path.
- Mac darwin/arm64 C VM execution and program-generation fast paths.
- Apple Silicon worker QoS and 10-thread recommendation for 10-core M4.
- Offline macOS arm64 tar.gz and zip packages.
- Standalone `cereblix-miner-darwin-arm64` for the miner self-update path.

## macmini87 Pool Result

Machine:

- Mac model: `Mac16,10`
- CPU: Apple M4
- CPU topology: 4 performance cores + 6 efficiency cores
- Node: `https://cereblix.com/pool/api`
- Mode: pool

Comparison on the same M4 Mac mini with real accepted pool shares:

| Build | Threads | Avg hashrate | Min | Max | Accepted shares |
| --- | ---: | ---: | ---: | ---: | ---: |
| Upstream v1.2 original | 12 | 4,571.3 H/s | 4,529.7 H/s | 4,615.9 H/s | 10 |
| Mac 10K v1.2.10 | 12 | 12,204.4 H/s | 12,140.8 H/s | 12,255.7 H/s | 19 |
| Mac optimized v1.2.11 | 11 | 14,771.0 H/s | 14,675.9 H/s | 14,903.3 H/s | 14 |
| Mac optimized v1.2.12 | 10 | 15,282.0 H/s | 15,125.5 H/s | 15,411.3 H/s | 26 |

Thread scan for this v1.2.12 candidate:

| Threads | Avg hashrate | Min | Max | Accepted shares |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 15,282.0 H/s | 15,125.5 H/s | 15,411.3 H/s | 26 |
| 11 | 15,058.0 H/s | 14,770.8 H/s | 15,235.1 H/s | 18 |
| 12 | 15,133.3 H/s | 14,965.9 H/s | 15,211.8 H/s | 15 |

Improvement versus upstream v1.2: `+234%`, about `3.34x` the original v1.2
hashrate. Improvement versus v1.2.11: `+3.5%` in the best pool-verified
thread setting.

## Run Pool

```sh
./cereblix-miner \
  -node https://cereblix.com/pool/api \
  -addr crb1_your_wallet_address_here \
  -threads 10
```

If you omit `-threads` on a 10-core Apple Silicon Mac, the miner recommends
10 threads automatically.

## Verify

```sh
shasum -a 256 -c SHA256SUMS
```

If macOS quarantine blocks execution after downloading the archive, remove the
quarantine attribute from the extracted directory:

```sh
xattr -dr com.apple.quarantine cereblix-mac-10k-darwin-arm64-offline
```

## Build From Source

```sh
CGO_ENABLED=1 GOOS=darwin GOARCH=arm64 CC='clang -arch arm64' \
  go build -o cereblix-miner ./cmd/cereblix-miner
```

PGO can be supplied with `-pgo=/path/to/cereblix-nm-arm64-pgo.pprof` when
building on a machine that has the profile file.
