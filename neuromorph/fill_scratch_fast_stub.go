//go:build !darwin || !arm64 || !cgo

package neuromorph

func fillScratchFast(key [32]byte, scratch []uint64) bool {
	return false
}

func fillScratchFoldFast(key [32]byte, scratch []uint64, fold *[8]uint64) bool {
	return false
}

func genProgramFast(key [32]byte, opTable *[256]uint8, prog []instr) bool {
	return false
}

func executeProgramFast(p *Params, prog []instr, taken []uint8, scratch []uint64, dataset []uint64, r *[16]uint64, f *[8]float64, useDataset bool) bool {
	return false
}

func executeProgramFoldFast(p *Params, prog []instr, taken []uint8, scratch []uint64, dataset []uint64, r *[16]uint64, f *[8]float64, fold *[8]uint64, useDataset bool) bool {
	return false
}

func foldScratchFast(scratch []uint64, fold *[8]uint64) bool {
	return false
}
