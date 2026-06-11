# Cerebra (CRB)

**A CPU-only cryptocurrency built from scratch on the self-mutating NeuroMorph
proof-of-work algorithm.** No GPU, no ASIC — ever. One CPU, one vote.

- 🌐 Site & explorer: http://188.34.181.191/cerebra/
- 💼 Web wallet: http://188.34.181.191/cerebra/wallet/
- 📖 Full design: [ARCHITECTURE.md](ARCHITECTURE.md)

> ⚗️ Experimental software, launched in a single day with **zero premine, zero
> fund, zero promises**. NeuroMorph is a new algorithm without years of external
> audit. The coin has no price — it only gets value from demand that may never
> appear. Don't invest what you can't lose. DYOR.

---

## Why Cerebra

- **🧬 Self-mutating algorithm.** Every 4096 blocks (~2.8 days) NeuroMorph
  rebuilds its own VM semantics from chain entropy — opcode weights, program
  length, constants, AES keys all change. Fixed-function hardware for an
  algorithm that doesn't exist yet is impossible. That is lifelong ASIC
  resistance by construction, not by promise.
- **⚖️ 1 CPU = 1 vote.** Random programs with data-dependent branches starve
  GPUs (warp divergence) — any laptop competes. No farms.
- **🤝 Fair launch.** Empty genesis block, coins exist only from mining.
- **📡 Lightweight node.** One dependency-free Go binary; the chain is
  human-readable JSONL.

## Coin parameters

| | |
|---|---|
| Ticker | CRB (1 CRB = 10⁸ synapses) |
| Algorithm | NeuroMorph v1 — self-mutating PoW VM, CPU-only |
| Block time | 60 s, retarget every 20 blocks |
| Reward | 50 CRB, halving every 1,051,200 blocks (~2 years) |
| Max supply | ~105,120,000 CRB |
| VM mutation epoch | 4096 blocks |
| Premine | **0** |
| Signatures / addresses | ed25519 · `crb1` + SHA-256(pubkey)[:20] |

## Build

Requires Go 1.21+. Zero external dependencies (standard library only).

```sh
git clone https://github.com/Cerebra-CBR/cerebra.git
cd cerebra
go build ./...

# or build each tool:
go build -o cerebrad        ./cmd/cerebrad
go build -o cerebra-miner   ./cmd/cerebra-miner
go build -o cerebra-wallet  ./cmd/cerebra-wallet
```

Cross-compile (e.g. Windows from Linux):

```sh
GOOS=windows GOARCH=amd64 go build -o cerebra-miner.exe ./cmd/cerebra-miner
```

## Mine

```sh
# 1. create a wallet address
cerebra-wallet new main

# 2. point the miner at any node (the public seed by default)
cerebra-miner -addr crb1YOURADDRESS            # uses all cores
cerebra-miner -addr crb1YOURADDRESS -threads 4 # limit cores
```

> Antivirus software often flags unsigned CPU miners as PUA — add an exclusion
> for the miner file rather than disabling protection.

## Run a full node

```sh
cerebrad -datadir ./data                       # follow the chain
cerebrad -datadir ./data -mine -threads 2 -coinbase crb1YOURADDRESS  # node + miner
```

Your own node's RPC is at `http://127.0.0.1:18751/api`. Point the wallet/miner
at it with `-node http://127.0.0.1:18751/api`.

## Standalone CLI wallet

A local key store + RPC client + block explorer, independent of the website
(like `bitcoin-cli`). Keys live only on your machine in `~/.cerebra/wallet.json`
(optionally passphrase-encrypted with PBKDF2 + AES-GCM).

```sh
cerebra-wallet                      # interactive shell
cerebra-wallet new main             # create address
cerebra-wallet list                 # addresses + balances
cerebra-wallet send crb1... 12.5    # sign locally, broadcast
cerebra-wallet encrypt              # passphrase-protect the wallet
cerebra-wallet tx <txid>            # explorer: look up a transaction
cerebra-wallet block 42             # explorer: show a block
cerebra-wallet richlist             # top addresses
```

## Repository layout

```
neuromorph/   NeuroMorph PoW virtual machine
core/         chain, state, mempool, consensus rules
node/         P2P sync, JSON RPC, getwork/submitwork, built-in miner
cmd/          cerebrad, cerebra-miner, cerebra-wallet
web/          project site + block explorer + web wallet
deploy/       systemd unit
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the complete technical specification.

## License

[MIT](LICENSE). Mine it, fork it, mirror it.
