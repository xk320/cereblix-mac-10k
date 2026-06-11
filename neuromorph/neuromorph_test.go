package neuromorph

import (
	"encoding/hex"
	"testing"
)

func TestDeterminism(t *testing.T) {
	p := DeriveParams(EpochSeed0())
	vm1 := NewVM(p)
	vm2 := NewVM(p)
	header := make([]byte, 124)
	for i := range header {
		header[i] = byte(i * 7)
	}
	h1 := vm1.Hash(header)
	h2 := vm2.Hash(header)
	if h1 != h2 {
		t.Fatalf("non-deterministic: %x vs %x", h1, h2)
	}
	// Same VM reused must give the same answer (buffer reset correctness).
	h3 := vm1.Hash(header)
	if h1 != h3 {
		t.Fatalf("vm reuse changes result: %x vs %x", h1, h3)
	}
	header[5] ^= 1
	h4 := vm1.Hash(header)
	if h4 == h1 {
		t.Fatal("hash ignores input changes")
	}
	t.Logf("nm hash: %s", hex.EncodeToString(h1[:]))
}

func TestEpochsDiffer(t *testing.T) {
	p0 := DeriveParams(EpochSeed0())
	p1 := DeriveParams([]byte("some other epoch boundary hash..32b"))
	if p0.ProgSize == p1.ProgSize && p0.Loops == p1.Loops && p0.RotSalt == p1.RotSalt {
		t.Fatal("epoch params do not vary")
	}
}

func BenchmarkHash(b *testing.B) {
	p := DeriveParams(EpochSeed0())
	vm := NewVM(p)
	header := make([]byte, 124)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		header[120] = byte(i)
		vm.Hash(header)
	}
}
