//go:build darwin && arm64 && cgo

package neuromorph

import "testing"

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
