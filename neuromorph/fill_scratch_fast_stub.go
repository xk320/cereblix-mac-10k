//go:build !darwin || !arm64 || !cgo

package neuromorph

func fillScratchFast(key [32]byte, scratch []uint64) bool {
	return false
}

