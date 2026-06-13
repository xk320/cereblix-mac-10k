# Cereblix Mac 10K Client v1.2.10

This build is based on upstream `Cerebra-CBR/cereblix` commit
`1f19999e71574643cb3e23482f99e648e481990e` and ports the Apple Silicon 10K
optimization onto the latest miner v1.2 codebase.

## What Changed

- Rebased on upstream miner v1.2 and latest consensus changes.
- Added a darwin/arm64/cgo NeuroMorph scratch-fill fast path using ARM AES
  instructions.
- Kept the consensus hash byte-for-byte compatible with the original
  NeuroMorph implementation.
- Preserved upstream v1.2 connection resilience and self-update behavior, but
  pointed Mac 10K updates at this repository.
- Updated Apple Silicon miner thread recommendation to 1.2x logical CPUs.
- Added target comparison, thread recommendation, arm64 worker QoS, and
  NeuroMorph stage benchmarks.
- Added Metal experiments under `experiments/metal/` for future GPU research.

## macmini87 Pool Result

Machine:

- Mac model: `Mac16,10`
- CPU: Apple M4
- CPU topology: 4 performance cores + 6 efficiency cores
- Node: `https://cereblix.com/pool/api`
- Mode: pool
- Threads: 12

Comparison against upstream original v1.2 at the same upstream commit:

| Build | Avg hashrate | Min | Max | Accepted shares |
| --- | ---: | ---: | ---: | ---: |
| Upstream v1.2 original | 4,571.3 H/s | 4,529.7 H/s | 4,615.9 H/s | 10 |
| Mac 10K optimized v1.2.10 | 12,204.4 H/s | 12,140.8 H/s | 12,255.7 H/s | 19 |

Improvement: `+167%`, about `2.67x` the original v1.2 hashrate.

A public solo sanity run of the same optimized binary measured 12,335.5 H/s
average across 8 samples, but pool mode is the recommended practical mode for
this device.

## Run Pool

```sh
./cereblix-miner \
  -node https://cereblix.com/pool/api \
  -addr crb1_your_wallet_address_here \
  -threads 12
```

## Run Solo

```sh
./cereblix-miner \
  -node https://cereblix.com/api \
  -addr crb1_your_wallet_address_here \
  -threads 12
```

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
