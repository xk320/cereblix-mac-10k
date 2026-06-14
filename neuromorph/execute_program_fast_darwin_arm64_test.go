//go:build darwin && arm64 && cgo

package neuromorph

import (
	"crypto/sha256"
	"encoding/binary"
	"math"
	"testing"
)

func TestExecuteProgramFastMatchesSlowPath(t *testing.T) {
	params := []*Params{
		DeriveParams(EpochSeed0()),
		DeriveParams([]byte("cereblix-mac-fast-vm-second-epoch-seed")),
	}
	heights := []uint64{0, DatasetHeight, DatasetHeight + 100}

	for pi, p := range params {
		for hi, height := range heights {
			for n := 0; n < 16; n++ {
				header := make([]byte, 124)
				for i := range header {
					header[i] = byte(i*17 + n*31 + pi*7 + hi*13)
				}

				forceSlowProgram = true
				slow := NewVM(p).Hash(header, height)
				forceSlowProgram = false
				fast := NewVM(p).Hash(header, height)

				if fast != slow {
					t.Fatalf("fast VM mismatch params=%d height=%d nonce=%d\nfast %x\nslow %x", pi, height, n, fast, slow)
				}
			}
		}
	}
}

func BenchmarkFillScratchFoldFast(b *testing.B) {
	p := DeriveParams(EpochSeed0())
	vm := NewVM(p)
	seed := benchmarkSeed()
	var keyInput [33]byte
	copy(keyInput[:32], seed[:])
	keyInput[32] = 0x53
	scratchKey := sha256.Sum256(keyInput[:])
	var fold [8]uint64

	b.SetBytes(int64(len(vm.scratch) * 8))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if !fillScratchFoldFast(scratchKey, vm.scratch, &fold) {
			b.Fatal("fillScratchFoldFast unavailable")
		}
	}
}

func BenchmarkExecuteProgramFoldFastDataset(b *testing.B) {
	p := DeriveParams(EpochSeed0())
	vm := NewVM(p)
	seed := benchmarkSeed()
	var keyInput [33]byte
	copy(keyInput[:32], seed[:])
	keyInput[32] = 0x53
	scratchKey := sha256.Sum256(keyInput[:])
	var fold [8]uint64
	if !fillScratchFoldFast(scratchKey, vm.scratch, &fold) {
		b.Fatal("fillScratchFoldFast unavailable")
	}
	vm.genProgram(seed)
	dataset := getDataset(p.DatasetKey)

	var r [16]uint64
	var f [8]float64
	for i := 0; i < 4; i++ {
		r[i] = binary.LittleEndian.Uint64(seed[i*8 : i*8+8])
	}
	for i := 4; i < 16; i++ {
		r[i] = vm.scratch[i] ^ p.RotSalt
	}
	for i := 0; i < 8; i++ {
		f[i] = math.Float64frombits((uint64(1023) << 52) | (vm.scratch[16+i] & 0x000FFFFFFFFFFFFF))
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if !executeProgramFoldFast(p, vm.prog, vm.taken, vm.scratch, dataset, &r, &f, &fold, true) {
			b.Fatal("executeProgramFoldFast unavailable")
		}
	}
}
