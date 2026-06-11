// Package core implements Cereblix chain data structures and consensus rules.
// Note: protocol literals below ("cerebra-tx-v1", the genesis message, etc.)
// are consensus-critical and intentionally keep their original spelling - the
// live chain's hashes and signatures depend on them. Only the brand changed.
package core

import (
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"math/big"
	"strings"
)

const (
	CoinUnit           = 100_000_000 // synapses per 1 CRB
	BlockTargetSpacing = 60          // seconds
	RetargetWindow     = 20          // blocks
	HalvingInterval    = 1_051_200   // blocks (~2 years at 60 s)
	InitialReward      = 50 * CoinUnit
	EpochLength        = 4096 // blocks per NeuroMorph semantic epoch
	MaxBlockTxs        = 200
	MaxFutureDrift     = 300 // seconds
	AddrPrefix         = "crb1"

	// CoinbaseMaturity: a block reward becomes spendable only once it is this
	// many blocks deep, so a reward reversed by a reorg cannot have been spent.
	// Set to match the reorg-depth cap. Enforced from MaturityHeight onward
	// (earlier blocks are grandfathered so the existing chain stays valid).
	CoinbaseMaturity = 100
	MaturityHeight   = 500
)

// MaxTarget is the easiest allowed target (difficulty floor).
var MaxTarget = new(big.Int).Lsh(big.NewInt(1), 244)

// GenesisTarget sets initial difficulty (~45 s on 3 cores at ~240 H/s/core).
var GenesisTarget = new(big.Int).Lsh(big.NewInt(1), 241)

func BlockSubsidy(height uint64) uint64 {
	halvings := height / HalvingInterval
	if halvings >= 33 {
		return 0
	}
	return InitialReward >> halvings
}

// ---------------------------------------------------------------- addresses

func AddrFromPub(pub []byte) string {
	h := sha256.Sum256(pub)
	return AddrPrefix + hex.EncodeToString(h[:20])
}

func ValidAddr(a string) bool {
	if !strings.HasPrefix(a, AddrPrefix) || len(a) != len(AddrPrefix)+40 {
		return false
	}
	_, err := hex.DecodeString(a[len(AddrPrefix):])
	return err == nil
}

// ------------------------------------------------------------- transactions

type Tx struct {
	FromPub string `json:"from_pub"` // hex ed25519 pubkey; empty => coinbase
	To      string `json:"to"`
	Amount  uint64 `json:"amount"`
	Fee     uint64 `json:"fee"`
	Nonce   uint64 `json:"nonce"` // account nonce; block height for coinbase
	Sig     string `json:"sig"`   // hex ed25519 signature
}

func (t *Tx) SigningPayload() []byte {
	return []byte(fmt.Sprintf("cerebra-tx-v1|%s|%s|%d|%d|%d",
		t.FromPub, t.To, t.Amount, t.Fee, t.Nonce))
}

func (t *Tx) ID() string {
	h := sha256.Sum256([]byte(string(t.SigningPayload()) + "|" + t.Sig))
	return hex.EncodeToString(h[:])
}

func (t *Tx) IsCoinbase() bool { return t.FromPub == "" }

func (t *Tx) FromAddr() (string, error) {
	if t.IsCoinbase() {
		return "", errors.New("coinbase has no sender")
	}
	pub, err := hex.DecodeString(t.FromPub)
	if err != nil || len(pub) != ed25519.PublicKeySize {
		return "", errors.New("bad pubkey")
	}
	return AddrFromPub(pub), nil
}

// CheckSig validates structure and signature of a normal (non-coinbase) tx.
func (t *Tx) CheckSig() error {
	if t.IsCoinbase() {
		return errors.New("coinbase tx not allowed here")
	}
	if !ValidAddr(t.To) {
		return errors.New("bad destination address")
	}
	if t.Amount == 0 {
		return errors.New("zero amount")
	}
	pub, err := hex.DecodeString(t.FromPub)
	if err != nil || len(pub) != ed25519.PublicKeySize {
		return errors.New("bad pubkey")
	}
	sig, err := hex.DecodeString(t.Sig)
	if err != nil || len(sig) != ed25519.SignatureSize {
		return errors.New("bad signature encoding")
	}
	if !ed25519.Verify(ed25519.PublicKey(pub), t.SigningPayload(), sig) {
		return errors.New("invalid signature")
	}
	return nil
}

func SignTx(t *Tx, priv ed25519.PrivateKey) {
	t.FromPub = hex.EncodeToString(priv.Public().(ed25519.PublicKey))
	t.Sig = hex.EncodeToString(ed25519.Sign(priv, t.SigningPayload()))
}

// ------------------------------------------------------------------- blocks

type Block struct {
	Version  uint32 `json:"v"`
	Height   uint64 `json:"height"`
	Time     uint64 `json:"time"`
	PrevHash string `json:"prev"`
	TxRoot   string `json:"txroot"`
	Target   string `json:"target"` // 64-hex big-endian target
	Nonce    uint64 `json:"nonce"`
	Txs      []*Tx  `json:"txs"`
}

const HeaderLen = 4 + 8 + 8 + 32 + 32 + 32 + 8
const NonceOffset = HeaderLen - 8

func (b *Block) HeaderBytes() []byte {
	out := make([]byte, HeaderLen)
	binary.LittleEndian.PutUint32(out[0:4], b.Version)
	binary.LittleEndian.PutUint64(out[4:12], b.Height)
	binary.LittleEndian.PutUint64(out[12:20], b.Time)
	prev, _ := hex.DecodeString(b.PrevHash)
	copy(out[20:52], prev)
	root, _ := hex.DecodeString(b.TxRoot)
	copy(out[52:84], root)
	tgt, _ := hex.DecodeString(b.Target)
	copy(out[84:116], tgt)
	binary.LittleEndian.PutUint64(out[116:124], b.Nonce)
	return out
}

// Hash is the block id: sha256 of the serialized header.
func (b *Block) Hash() string {
	h := sha256.Sum256(b.HeaderBytes())
	return hex.EncodeToString(h[:])
}

func ComputeTxRoot(txs []*Tx) string {
	h := sha256.New()
	h.Write([]byte("cerebra-txroot-v1"))
	for _, t := range txs {
		id, _ := hex.DecodeString(t.ID())
		h.Write(id)
	}
	return hex.EncodeToString(h.Sum(nil))
}

func (b *Block) TotalFees() uint64 {
	var fees uint64
	for _, t := range b.Txs {
		if !t.IsCoinbase() {
			fees += t.Fee
		}
	}
	return fees
}

// TargetInt parses the header target as a big.Int.
func (b *Block) TargetInt() (*big.Int, error) {
	raw, err := hex.DecodeString(b.Target)
	if err != nil || len(raw) != 32 {
		return nil, errors.New("bad target encoding")
	}
	return new(big.Int).SetBytes(raw), nil
}

func TargetToHex(t *big.Int) string {
	raw := t.Bytes()
	out := make([]byte, 32)
	copy(out[32-len(raw):], raw)
	return hex.EncodeToString(out)
}

// WorkOf returns the expected hash count for a target: 2^256 / (target+1).
func WorkOf(target *big.Int) *big.Int {
	num := new(big.Int).Lsh(big.NewInt(1), 256)
	den := new(big.Int).Add(target, big.NewInt(1))
	return num.Div(num, den)
}

func HashMeetsTarget(powHash [32]byte, target *big.Int) bool {
	return new(big.Int).SetBytes(powHash[:]).Cmp(target) <= 0
}

// ------------------------------------------------------------------ genesis

const GenesisMessage = "Cerebra genesis | 2026-06-11 | one CPU, one vote — silicon shall not rule"

func GenesisBlock() *Block {
	cb := &Tx{
		To:     "crb1" + strings.Repeat("0", 40), // unspendable
		Amount: 0,
		Nonce:  0,
		Sig:    hex.EncodeToString([]byte(GenesisMessage)),
	}
	b := &Block{
		Version:  1,
		Height:   0,
		Time:     1781136000, // 2026-06-11 00:00:00 UTC
		PrevHash: strings.Repeat("0", 64),
		Target:   TargetToHex(GenesisTarget),
		Nonce:    0,
		Txs:      []*Tx{cb},
	}
	b.TxRoot = ComputeTxRoot(b.Txs)
	return b
}
