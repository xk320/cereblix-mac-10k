# Cereblix Mac 10K Client

This build is an Apple Silicon optimized Cereblix client package focused on
public solo mining performance on macOS arm64.

## What Changed

- Added a darwin/arm64/cgo NeuroMorph scratch-fill fast path using ARM AES
  instructions.
- Kept the consensus hash byte-for-byte compatible with the original
  NeuroMorph implementation.
- Updated Apple Silicon miner thread recommendation to 1.2x logical CPUs.
- Added target comparison, thread recommendation, arm64 worker QoS, and
  NeuroMorph stage benchmarks.
- Added Metal experiments under `experiments/metal/` for future GPU research.

## macmini87 Result

Machine:

- Mac model: `Mac16,10`
- CPU: Apple M4
- CPU topology: 4 performance cores + 6 efficiency cores
- Node: `https://cereblix.com/api`
- Mode: public solo

Final public solo measurements:

| Threads | Hashrate |
| ---: | ---: |
| 4 | ~8.8-9.1 kH/s |
| 8 | ~11.4-11.7 kH/s |
| 10 | ~12.0-12.3 kH/s |
| 12 | ~12.1-12.4 kH/s |
| 14 | ~12.1-12.3 kH/s |
| 16 | ~12.0-12.2 kH/s |

The best tested setting on macmini87 was 12 threads.

## Run

```sh
./cereblix-miner \
  -node https://cereblix.com/api \
  -addr crb1_your_wallet_address_here \
  -threads 12
```

If you omit `-threads` on a 10-core Apple Silicon Mac, the miner recommends
12 threads automatically.

## Verify

```sh
shasum -a 256 -c SHA256SUMS
```

If macOS quarantine blocks execution after downloading the archive, remove the
quarantine attribute from the extracted directory:

```sh
xattr -dr com.apple.quarantine cereblix-darwin-arm64
```

## Build From Source

```sh
CGO_ENABLED=1 GOOS=darwin GOARCH=arm64 CC='clang -arch arm64' \
  go build -o cereblix-miner ./cmd/cereblix-miner
```

PGO can be supplied with `-pgo=/path/to/cereblix-nm-arm64-pgo.pprof` when
building on a machine that has the profile file.
