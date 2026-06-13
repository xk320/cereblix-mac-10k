# Cereblix Mac 10K Client v1.2.11

This build is based on upstream `Cerebra-CBR/cereblix` commit
`1f19999e71574643cb3e23482f99e648e481990e` and ports the Apple Silicon
optimization onto the latest miner v1.2 codebase.

## What Changed Since v1.2.10

- Moved float finite repair into the floating-point opcode handlers instead of
  checking every VM instruction.
- Added a darwin/arm64 NEON scratch-fold fast path.
- Tuned Apple Silicon default threads for macmini87 from 12 to 11 based on pool
  accepted-share testing.
- Kept consensus hash compatibility verified by `TestCrossPlatformHash` on
  macmini87 arm64.

## What Is Included

- Upstream miner v1.2 connection resilience and macOS-aware update support.
- Latest upstream consensus changes including the gated soft fee floor.
- Mac darwin/arm64 ARM AES scratch-fill fast path.
- Apple Silicon worker QoS and 11-thread recommendation for 10-core M4.
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

Improvement versus upstream v1.2: `+223%`, about `3.23x` the original v1.2
hashrate. Improvement versus v1.2.10: `+21%`.

## Run Pool

```sh
./cereblix-miner \
  -node https://cereblix.com/pool/api \
  -addr crb1_your_wallet_address_here \
  -threads 11
```

If you omit `-threads` on a 10-core Apple Silicon Mac, the miner recommends
11 threads automatically.

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
