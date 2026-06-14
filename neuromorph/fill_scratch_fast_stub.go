//go:build !darwin || !arm64 || !cgo

package neuromorph

import "unsafe"

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

func expandAES128RoundKeys(key *[16]byte, rk *[176]byte) {
	_ = key
	_ = rk
}

func newJITResidentBuffer(progSize int) unsafe.Pointer {
	_ = progSize
	return nil
}

func freeJITBuffer(buf unsafe.Pointer) {
	_ = buf
}

func jitRealExecuteProbeResidentReuse(buf unsafe.Pointer, p *Params, prog []instr, r *[16]uint64, scratch []uint64, fold *[8]uint64, f *[8]float64, dataset []uint64, useDataset bool, rk *[176]byte, taken []uint8) bool {
	return false
}
