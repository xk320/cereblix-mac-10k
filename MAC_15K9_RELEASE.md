# Cereblix Mac 15.9K Client v1.3.0

This Apple Silicon build is based on upstream `Cereblix/cereblix` commit
`b874b02` (`node v2.1.1`, miner `v1.3`) plus the Mac darwin/arm64 miner
optimization stack.

## What Is Included

- Latest upstream node v2.1.1 changes through `b874b02`.
- Darwin/arm64 C fast path for NeuroMorph scratch fill, program generation,
  VM execution, and incremental scratch folding.
- Arm64 AES instructions for scratch fill and VM `AESR` execution.
- Branch-counter loop-stamp optimization in the arm64 VM fast path.
- Apple Silicon worker QoS and 10-thread recommendation for 10-core M4.
- Offline macOS arm64 tar.gz and zip packages.
- Standalone `cereblix-miner-darwin-arm64` for the miner self-update path.

## macmini87 Pool Result

Machine:

- CPU: Apple M4
- CPU topology: 4 performance cores + 6 efficiency cores
- Node: `https://cereblix.com/pool/api`
- Mode: pool
- Wallet used for validation: `crb1f97cb89127790f17c2cd991302b7318fdbf57d0b`

Best accepted-share validation window:

| Build | Upstream base | Threads | Avg hashrate | Min | Max | Accepted shares | Rejects |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mac optimized v1.3.0 branch-stamp | `907810f` | 10 | 15,903.9 H/s | 15,445.3 H/s | 16,349.9 H/s | 23 | 0 |

Latest-upstream compatibility validation:

| Build | Upstream base | Threads | Avg hashrate | Min | Max | Accepted shares | Rejects |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mac optimized v1.3.0 latest baseline | `b874b02` | 10 | 15,772.2 H/s | 15,615.7 H/s | 15,955.5 H/s | 27 | 0 |

The highest verified hashrate record remains `15,903.9 H/s`. The release source
is kept on the latest upstream-compatible `b874b02` base.

## One-Command Pool Mining

```sh
./cereblix-miner \
  -node https://cereblix.com/pool/api \
  -addr crb1f97cb89127790f17c2cd991302b7318fdbf57d0b \
  -threads 10
```

## One-Command Solo Mining

Solo mining is valid but not recommended at this hashrate because finding a full
block is a lottery.

```sh
./cereblix-miner \
  -node https://cereblix.com/api \
  -addr crb1f97cb89127790f17c2cd991302b7318fdbf57d0b \
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
xattr -dr com.apple.quarantine cereblix-mac-15k9-darwin-arm64-offline
```

## Build From Source

```sh
CGO_ENABLED=1 GOOS=darwin GOARCH=arm64 CC='clang -arch arm64' \
  go build -trimpath -ldflags='-s -w' -o cereblix-miner ./cmd/cereblix-miner
```
