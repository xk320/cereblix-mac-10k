//go:build darwin && arm64 && cgo

package neuromorph

import (
	"crypto/sha256"
	"encoding/binary"
	"math"
	"math/bits"
	"testing"
)

var jitHashProbeSink [32]byte

func finalizeProbeHash(seed [32]byte, r *[16]uint64, f *[8]float64, fold *[8]uint64) [32]byte {
	var out [4 + 32 + 16*8 + 8*8 + 8*8]byte
	pos := 0
	pos += copy(out[pos:], "NMv1")
	pos += copy(out[pos:], seed[:])
	for i := 0; i < 16; i++ {
		binary.LittleEndian.PutUint64(out[pos:pos+8], r[i])
		pos += 8
	}
	for i := 0; i < 8; i++ {
		binary.LittleEndian.PutUint64(out[pos:pos+8], math.Float64bits(f[i]))
		pos += 8
	}
	for i := 0; i < 8; i++ {
		binary.LittleEndian.PutUint64(out[pos:pos+8], fold[i])
		pos += 8
	}
	return sha256.Sum256(out[:])
}

func jitProbeSupportedOp(op uint8) bool {
	switch op {
	case opIADD, opIMUL, opIMULH, opIXOR, opIROTR, opINEG, opFADD, opFMUL, opFDIV, opFSQRT, opLOAD, opSTORE, opCBRANCH, opAESR, opXDOM:
		return true
	default:
		return false
	}
}

func TestJITSupportedSpanCoverage(t *testing.T) {
	params := []*Params{
		DeriveParams(EpochSeed0()),
		DeriveParams([]byte("cereblix-mac-fast-vm-second-epoch-seed")),
	}
	for pi, p := range params {
		totalInstr := 0
		supportedInstr := 0
		totalSpans := 0
		supportedSpans := 0
		maxSpan := 0
		shortSpans := 0
		vm := NewVM(p)
		for n := 0; n < 256; n++ {
			header := make([]byte, 124)
			for i := range header {
				header[i] = byte(i*17 + n*31 + pi*7)
			}
			var seedInput [8 + 256]byte
			copy(seedInput[:8], "nm-seed|")
			copy(seedInput[8:], header)
			seed := sha256.Sum256(seedInput[:8+len(header)])
			vm.genProgram(seed)

			span := 0
			for _, ins := range vm.prog {
				totalInstr++
				if jitProbeSupportedOp(ins.op) {
					supportedInstr++
					span++
					continue
				}
				totalSpans++
				if span > 0 {
					supportedSpans++
					if span > maxSpan {
						maxSpan = span
					}
					if span < 4 {
						shortSpans++
					}
					span = 0
				}
			}
			if span > 0 {
				totalSpans++
				supportedSpans++
				if span > maxSpan {
					maxSpan = span
				}
				if span < 4 {
					shortSpans++
				}
			}
		}
		coverage := float64(supportedInstr) * 100 / float64(totalInstr)
		avgSupportedSpan := float64(supportedInstr) / float64(supportedSpans)
		t.Logf("params=%d supported_instr=%d/%d coverage=%.1f%% supported_spans=%d total_spans=%d avg_supported_span=%.2f max_span=%d short_supported_spans=%d",
			pi, supportedInstr, totalInstr, coverage, supportedSpans, totalSpans, avgSupportedSpan, maxSpan, shortSpans)
	}
}

func TestJITDynamicOpCoverage(t *testing.T) {
	params := []*Params{
		DeriveParams(EpochSeed0()),
		DeriveParams([]byte("cereblix-mac-fast-vm-second-epoch-seed")),
	}
	names := []string{"IADD", "IMUL", "IMULH", "IXOR", "IROTR", "INEG", "FADD", "FMUL", "FDIV", "FSQRT", "LOAD", "STORE", "CBRANCH", "AESR", "XDOM"}
	for pi, p := range params {
		var counts [numOps]int64
		var supported int64
		var total int64
		var branchesTaken int64
		var branchesSeen int64
		vm := NewVM(p)
		for n := 0; n < 64; n++ {
			header := make([]byte, 124)
			for i := range header {
				header[i] = byte(i*17 + n*31 + pi*7)
			}
			var seedInput [8 + 256]byte
			copy(seedInput[:8], "nm-seed|")
			copy(seedInput[8:], header)
			seed := sha256.Sum256(seedInput[:8+len(header)])
			vm.fillScratch(seed)
			vm.genProgram(seed)

			var r [16]uint64
			var f [8]float64
			for i := 0; i < 4; i++ {
				r[i] = binary.LittleEndian.Uint64(seed[i*8 : i*8+8])
			}
			for i := 4; i < 16; i++ {
				r[i] = vm.scratch[i] ^ p.RotSalt
			}
			for i := 0; i < 8; i++ {
				f[i] = normFloat(vm.scratch[16+i])
			}

			for loop := 0; loop < p.Loops; loop++ {
				for i := range vm.taken {
					vm.taken[i] = 0
				}
				pc := 0
				for pc < p.ProgSize {
					ins := vm.prog[pc]
					counts[ins.op]++
					total++
					if jitProbeSupportedOp(ins.op) {
						supported++
					}
					d := ins.dst
					s := ins.src
					imm := uint64(ins.imm)
					switch ins.op {
					case opIADD:
						r[d] += r[s] + imm
					case opIMUL:
						r[d] *= r[s] | 1
					case opIMULH:
						hi, _ := bits.Mul64(r[d], r[s])
						r[d] = hi ^ imm
					case opIXOR:
						r[d] ^= r[s] + p.RotSalt
					case opIROTR:
						r[d] = bits.RotateLeft64(r[d], -int((r[s]^imm)&63))
					case opINEG:
						r[d] = ^r[d] + imm
					case opFADD:
						f[d&7] = f[d&7] + f[s&7]
						if b := math.Float64bits(f[d&7]); b&0x7FF0000000000000 == 0x7FF0000000000000 || b<<1 == 0 {
							f[d&7] = normFloat(r[d] | 1)
						}
					case opFMUL:
						f[d&7] = f[d&7] * f[s&7]
						if b := math.Float64bits(f[d&7]); b&0x7FF0000000000000 == 0x7FF0000000000000 || b<<1 == 0 {
							f[d&7] = normFloat(r[d] | 1)
						}
					case opFDIV:
						f[d&7] = f[d&7] / normFloat(math.Float64bits(f[s&7]))
						if b := math.Float64bits(f[d&7]); b&0x7FF0000000000000 == 0x7FF0000000000000 || b<<1 == 0 {
							f[d&7] = normFloat(r[d] | 1)
						}
					case opFSQRT:
						f[d&7] = math.Sqrt(math.Abs(f[d&7]))
						if b := math.Float64bits(f[d&7]); b&0x7FF0000000000000 == 0x7FF0000000000000 || b<<1 == 0 {
							f[d&7] = normFloat(r[d] | 1)
						}
					case opLOAD:
						addr := (r[s] + imm) & scratchMask
						r[d] ^= vm.scratch[addr>>3]
					case opSTORE:
						addr := (r[d] + imm) & scratchMask
						vm.scratch[addr>>3] ^= r[s] + uint64(loop)
					case opCBRANCH:
						branchesSeen++
						if (r[d]+imm)&p.BranchMask == 0 && vm.taken[pc] < 8 {
							branchesTaken++
							vm.taken[pc]++
							back := int(ins.imm%31) + 1
							pc -= back
							if pc < 0 {
								pc = 0
							}
							continue
						}
					case opAESR:
						addr := (r[s] + imm) & scratchMask & ^uint64(15)
						w := addr >> 3
						var aesIn, aesOut [16]byte
						binary.LittleEndian.PutUint64(aesIn[0:8], vm.scratch[w])
						binary.LittleEndian.PutUint64(aesIn[8:16], vm.scratch[w+1])
						vm.aes.Encrypt(aesOut[:], aesIn[:])
						vm.scratch[w] = binary.LittleEndian.Uint64(aesOut[0:8])
						vm.scratch[w+1] = binary.LittleEndian.Uint64(aesOut[8:16])
						r[d] ^= vm.scratch[w]
					case opXDOM:
						if ins.imm&1 == 0 {
							r[d] ^= math.Float64bits(f[s&7])
						} else {
							f[d&7] = f[d&7] * normFloat(r[s])
						}
					}
					pc++
				}
			}
		}
		t.Logf("params=%d dynamic_supported=%d/%d coverage=%.1f%% branches_taken=%d/%d", pi, supported, total, float64(supported)*100/float64(total), branchesTaken, branchesSeen)
		for op, name := range names {
			t.Logf("params=%d op=%s count=%d share=%.2f%%", pi, name, counts[op], float64(counts[op])*100/float64(total))
		}
	}
}

func TestJITAddChainMatchesInterpreter(t *testing.T) {
	for _, ninst := range []int{1, 8, 64, 640} {
		if !jitAddChainMatches(ninst) {
			t.Fatalf("jit add chain mismatch or compile failure for ninst=%d", ninst)
		}
	}
}

func TestJITIntMixMatchesInterpreter(t *testing.T) {
	for _, ninst := range []int{1, 5, 32, 128, 640} {
		if !jitIntMixMatches(ninst) {
			t.Fatalf("jit int mix mismatch or compile failure for ninst=%d", ninst)
		}
		if !jitIntMixResidentMatches(ninst) {
			t.Fatalf("resident jit int mix mismatch or compile failure for ninst=%d", ninst)
		}
	}
}

func TestJITMemMixMatchesInterpreter(t *testing.T) {
	for _, ninst := range []int{1, 7, 32, 128, 640} {
		if !jitMemMixMatches(ninst) {
			t.Fatalf("jit mem mix mismatch or compile failure for ninst=%d", ninst)
		}
		if !jitResMemMixMatches(ninst) {
			t.Fatalf("resident jit mem mix mismatch or compile failure for ninst=%d", ninst)
		}
	}
}

func TestJITResFloatMixMatchesInterpreter(t *testing.T) {
	for _, ninst := range []int{1, 7, 32, 128, 640} {
		if !jitResFloatMixMatches(ninst) {
			which, ref, jit, _ := debugResFloatMix(ninst)
			t.Fatalf("resident jit float mix mismatch or compile failure for ninst=%d field=%d ref=%016x jit=%016x", ninst, which, ref, jit)
		}
	}
}

func TestJITResFloatMixFirstMismatch(t *testing.T) {
	for ninst := 1; ninst <= 640; ninst++ {
		which, ref, jit, ok := debugResFloatMix(ninst)
		if !ok {
			t.Fatalf("first resident float mismatch ninst=%d op=%d field=%d ref=%016x jit=%016x", ninst, nmResFloatMixOpForTest(ninst-1), which, ref, jit)
		}
	}
}

func nmResFloatMixOpForTest(i int) int {
	return i % 10
}

func TestJITRealProgramOneLoopMatchesReference(t *testing.T) {
	params := []*Params{
		DeriveParams(EpochSeed0()),
		DeriveParams([]byte("cereblix-mac-fast-vm-second-epoch-seed")),
	}
	var rk [176]byte
	initProbeRoundKeys(&rk)
	for pi, p := range params {
		for n := 0; n < 16; n++ {
			vm := NewVM(p)
			header := make([]byte, 124)
			for i := range header {
				header[i] = byte(i*17 + n*31 + pi*7)
			}
			var seedInput [8 + 256]byte
			copy(seedInput[:8], "nm-seed|")
			copy(seedInput[8:], header)
			seed := sha256.Sum256(seedInput[:8+len(header)])
			vm.fillScratch(seed)
			vm.genProgram(seed)

			scratchRef := append([]uint64(nil), vm.scratch...)
			scratchJIT := append([]uint64(nil), vm.scratch...)
			scratchResident := append([]uint64(nil), vm.scratch...)
			var rRef, rJIT, rResident [16]uint64
			var fRef, fJIT, fResident [8]float64
			for i := 0; i < 4; i++ {
				rRef[i] = binary.LittleEndian.Uint64(seed[i*8 : i*8+8])
			}
			for i := 4; i < 16; i++ {
				rRef[i] = vm.scratch[i] ^ p.RotSalt
			}
			for i := 0; i < 8; i++ {
				fRef[i] = normFloat(vm.scratch[16+i])
			}
			rJIT = rRef
			rResident = rRef
			fJIT = fRef
			fResident = fRef

			var foldRef, foldJIT, foldResident [8]uint64
			takenRef := make([]uint8, len(vm.prog))
			takenJIT := make([]uint8, len(vm.prog))
			takenResident := make([]uint8, len(vm.prog))
			jitRealProgramReference(p, vm.prog, takenRef, scratchRef, &rRef, &fRef, &foldRef, &rk, 0)
			if !jitRealProgramOnce(vm.prog, p.BranchMask, &rJIT, scratchJIT, &foldJIT, &fJIT, p.RotSalt, 0, &rk, takenJIT) {
				t.Fatalf("jit real program compile/execute failed params=%d nonce=%d", pi, n)
			}
			if !jitRealProgramResidentOnce(vm.prog, p.BranchMask, &rResident, scratchResident, &foldResident, &fResident, p.RotSalt, 0, &rk, takenResident) {
				t.Fatalf("resident jit real program compile/execute failed params=%d nonce=%d", pi, n)
			}
			if rRef != rJIT || fRef != fJIT || foldRef != foldJIT {
				t.Fatalf("jit real program state mismatch params=%d nonce=%d\nrRef=%x\nrJIT=%x\nfRef=%v\nfJIT=%v\nfoldRef=%x\nfoldJIT=%x", pi, n, rRef, rJIT, fRef, fJIT, foldRef, foldJIT)
			}
			if rRef != rResident || fRef != fResident || foldRef != foldResident {
				t.Fatalf("resident jit real program state mismatch params=%d nonce=%d\nrRef=%x\nrResident=%x\nfRef=%v\nfResident=%v\nfoldRef=%x\nfoldResident=%x", pi, n, rRef, rResident, fRef, fResident, foldRef, foldResident)
			}
			for i := range takenRef {
				if takenRef[i] != takenJIT[i] {
					t.Fatalf("jit taken mismatch params=%d nonce=%d pc=%d ref=%d jit=%d", pi, n, i, takenRef[i], takenJIT[i])
				}
				if takenRef[i] != takenResident[i] {
					t.Fatalf("resident jit taken mismatch params=%d nonce=%d pc=%d ref=%d jit=%d", pi, n, i, takenRef[i], takenResident[i])
				}
			}
			for i := range scratchRef {
				if scratchRef[i] != scratchJIT[i] {
					t.Fatalf("jit scratch mismatch params=%d nonce=%d idx=%d ref=%x jit=%x", pi, n, i, scratchRef[i], scratchJIT[i])
				}
				if scratchRef[i] != scratchResident[i] {
					t.Fatalf("resident jit scratch mismatch params=%d nonce=%d idx=%d ref=%x jit=%x", pi, n, i, scratchRef[i], scratchResident[i])
				}
			}
		}
	}
}

func TestJITRealExecuteProbeMatchesReference(t *testing.T) {
	params := []*Params{
		DeriveParams(EpochSeed0()),
		DeriveParams([]byte("cereblix-mac-fast-vm-second-epoch-seed")),
	}
	var rk [176]byte
	initProbeRoundKeys(&rk)
	for pi, p := range params {
		for _, useDataset := range []bool{false, true} {
			for n := 0; n < 4; n++ {
				vm := NewVM(p)
				seed := benchmarkSeed()
				seed[0] ^= byte(pi*17 + n*31)
				vm.fillScratch(seed)
				vm.genProgram(seed)
				dataset := []uint64(nil)
				if useDataset {
					dataset = getDataset(p.DatasetKey)
				}

				scratchRef := append([]uint64(nil), vm.scratch...)
				scratchJIT := append([]uint64(nil), vm.scratch...)
				scratchResident := append([]uint64(nil), vm.scratch...)
				scratchResidentReuse := append([]uint64(nil), vm.scratch...)
				var rRef, rJIT, rResident, rResidentReuse [16]uint64
				var fRef, fJIT, fResident, fResidentReuse [8]float64
				initProbeState(seed, p, vm.scratch, &rRef, &fRef)
				rJIT = rRef
				rResident = rRef
				rResidentReuse = rRef
				fJIT = fRef
				fResident = fRef
				fResidentReuse = fRef

				var foldRef, foldJIT, foldResident, foldResidentReuse [8]uint64
				foldScratchGo(scratchRef, &foldRef)
				foldJIT = foldRef
				foldResident = foldRef
				foldResidentReuse = foldRef
				takenJIT := make([]uint8, len(vm.prog))
				takenResident := make([]uint8, len(vm.prog))
				takenResidentReuse := make([]uint8, len(vm.prog))
				residentBuf := newJITResidentBuffer(len(vm.prog))
				if residentBuf == nil {
					t.Fatal("failed to allocate resident jit buffer")
				}
				jitRealExecuteReference(p, vm.prog, scratchRef, dataset, &rRef, &fRef, &foldRef, &rk, useDataset)
				if !jitRealExecuteProbe(p, vm.prog, &rJIT, scratchJIT, &foldJIT, &fJIT, dataset, useDataset, &rk, takenJIT) {
					freeJITBuffer(residentBuf)
					t.Fatalf("jit real execute probe failed params=%d dataset=%v nonce=%d", pi, useDataset, n)
				}
				if !jitRealExecuteProbeResident(p, vm.prog, &rResident, scratchResident, &foldResident, &fResident, dataset, useDataset, &rk, takenResident) {
					freeJITBuffer(residentBuf)
					t.Fatalf("resident jit real execute probe failed params=%d dataset=%v nonce=%d", pi, useDataset, n)
				}
				if !jitRealExecuteProbeResidentReuse(residentBuf, p, vm.prog, &rResidentReuse, scratchResidentReuse, &foldResidentReuse, &fResidentReuse, dataset, useDataset, &rk, takenResidentReuse) {
					freeJITBuffer(residentBuf)
					t.Fatalf("resident jit real execute probe reuse failed params=%d dataset=%v nonce=%d", pi, useDataset, n)
				}
				freeJITBuffer(residentBuf)
				if rRef != rJIT || fRef != fJIT || foldRef != foldJIT {
					t.Fatalf("jit real execute state mismatch params=%d dataset=%v nonce=%d\nrRef=%x\nrJIT=%x\nfRef=%v\nfJIT=%v\nfoldRef=%x\nfoldJIT=%x", pi, useDataset, n, rRef, rJIT, fRef, fJIT, foldRef, foldJIT)
				}
				if rRef != rResident || fRef != fResident || foldRef != foldResident {
					t.Fatalf("resident jit real execute state mismatch params=%d dataset=%v nonce=%d\nrRef=%x\nrResident=%x\nfRef=%v\nfResident=%v\nfoldRef=%x\nfoldResident=%x", pi, useDataset, n, rRef, rResident, fRef, fResident, foldRef, foldResident)
				}
				if rRef != rResidentReuse || fRef != fResidentReuse || foldRef != foldResidentReuse {
					t.Fatalf("resident reuse jit real execute state mismatch params=%d dataset=%v nonce=%d\nrRef=%x\nrResident=%x\nfRef=%v\nfResident=%v\nfoldRef=%x\nfoldResident=%x", pi, useDataset, n, rRef, rResidentReuse, fRef, fResidentReuse, foldRef, foldResidentReuse)
				}
				for i := range scratchRef {
					if scratchRef[i] != scratchJIT[i] {
						t.Fatalf("jit real execute scratch mismatch params=%d dataset=%v nonce=%d idx=%d ref=%x jit=%x", pi, useDataset, n, i, scratchRef[i], scratchJIT[i])
					}
					if scratchRef[i] != scratchResident[i] {
						t.Fatalf("resident jit real execute scratch mismatch params=%d dataset=%v nonce=%d idx=%d ref=%x jit=%x", pi, useDataset, n, i, scratchRef[i], scratchResident[i])
					}
					if scratchRef[i] != scratchResidentReuse[i] {
						t.Fatalf("resident reuse jit real execute scratch mismatch params=%d dataset=%v nonce=%d idx=%d ref=%x jit=%x", pi, useDataset, n, i, scratchRef[i], scratchResidentReuse[i])
					}
				}
			}
		}
	}
}

func TestJITRealExecuteProbeMatchesFastPathWithRealAES(t *testing.T) {
	params := []*Params{
		DeriveParams(EpochSeed0()),
		DeriveParams([]byte("cereblix-mac-fast-vm-second-epoch-seed")),
	}
	for pi, p := range params {
		var rk [176]byte
		expandAES128RoundKeys(&p.AesKey, &rk)
		for _, useDataset := range []bool{false, true} {
			for n := 0; n < 8; n++ {
				vm := NewVM(p)
				seed := benchmarkSeed()
				seed[0] ^= byte(pi*17 + n*31)
				vm.fillScratch(seed)
				vm.genProgram(seed)
				dataset := []uint64(nil)
				if useDataset {
					dataset = getDataset(p.DatasetKey)
				}

				scratchFast := append([]uint64(nil), vm.scratch...)
				scratchJIT := append([]uint64(nil), vm.scratch...)
				var rFast, rJIT [16]uint64
				var fFast, fJIT [8]float64
				initProbeState(seed, p, vm.scratch, &rFast, &fFast)
				rJIT = rFast
				fJIT = fFast

				var foldFast, foldJIT [8]uint64
				foldScratchGo(scratchFast, &foldFast)
				foldScratchGo(scratchJIT, &foldJIT)
				takenFast := make([]uint8, len(vm.prog))
				takenJIT := make([]uint8, len(vm.prog))
				if !executeProgramFoldFast(p, vm.prog, takenFast, scratchFast, dataset, &rFast, &fFast, &foldFast, useDataset) {
					t.Fatalf("fast path unavailable params=%d dataset=%v nonce=%d", pi, useDataset, n)
				}
				if !jitRealExecuteProbe(p, vm.prog, &rJIT, scratchJIT, &foldJIT, &fJIT, dataset, useDataset, &rk, takenJIT) {
					t.Fatalf("jit real execute probe failed params=%d dataset=%v nonce=%d", pi, useDataset, n)
				}
				if rFast != rJIT || fFast != fJIT || foldFast != foldJIT {
					t.Fatalf("jit real-aes execute state mismatch params=%d dataset=%v nonce=%d\nrFast=%x\nrJIT=%x\nfFast=%v\nfJIT=%v\nfoldFast=%x\nfoldJIT=%x", pi, useDataset, n, rFast, rJIT, fFast, fJIT, foldFast, foldJIT)
				}
				for i := range scratchFast {
					if scratchFast[i] != scratchJIT[i] {
						t.Fatalf("jit real-aes scratch mismatch params=%d dataset=%v nonce=%d idx=%d fast=%x jit=%x", pi, useDataset, n, i, scratchFast[i], scratchJIT[i])
					}
				}
			}
		}
	}
}

func initProbeState(seed [32]byte, p *Params, scratch []uint64, r *[16]uint64, f *[8]float64) {
	for i := 0; i < 4; i++ {
		r[i] = binary.LittleEndian.Uint64(seed[i*8 : i*8+8])
	}
	for i := 4; i < 16; i++ {
		r[i] = scratch[i] ^ p.RotSalt
	}
	for i := 0; i < 8; i++ {
		f[i] = normFloat(scratch[16+i])
	}
}

func foldScratchGo(scratch []uint64, fold *[8]uint64) {
	for i := range fold {
		fold[i] = 0
	}
	for i, v := range scratch {
		fold[i&7] ^= v
	}
}

func jitRealExecuteReference(p *Params, prog []instr, scratch []uint64, dataset []uint64, r *[16]uint64, f *[8]float64, fold *[8]uint64, rk *[176]byte, useDataset bool) {
	taken := make([]uint8, len(prog))
	for loop := 0; loop < p.Loops; loop++ {
		for i := range taken {
			taken[i] = 0
		}
		jitRealProgramReference(p, prog, taken, scratch, r, f, fold, rk, uint64(loop))
		if useDataset {
			addr := (r[1] ^ p.RotSalt) & datasetMask
			for k := 0; k < datasetReadsPerLoop; k++ {
				v := dataset[addr>>3]
				r[k&15] ^= v
				addr = (v + r[(k+1)&15] + uint64(loop)) & datasetMask
			}
		}
		base := ((r[0] ^ uint64(loop)*0x9E3779B97F4A7C15) & scratchMask) >> 3
		for i := 0; i < 16; i++ {
			idx := (base + uint64(i)) & scratchWordMask
			scratch[idx] ^= r[i]
			fold[idx&7] ^= r[i]
		}
		for i := 0; i < 8; i++ {
			r[i+8] ^= math.Float64bits(f[i])
		}
	}
}

func jitRealProgramReference(p *Params, prog []instr, taken []uint8, scratch []uint64, r *[16]uint64, f *[8]float64, fold *[8]uint64, rk *[176]byte, loop uint64) {
	pc := 0
	for pc < len(prog) {
		ins := prog[pc]
		d := ins.dst
		s := ins.src
		imm := uint64(ins.imm)
		switch ins.op {
		case opIADD:
			r[d] += r[s] + imm
		case opIMUL:
			r[d] *= r[s] | 1
		case opIMULH:
			hi, _ := bits.Mul64(r[d], r[s])
			r[d] = hi ^ imm
		case opIXOR:
			r[d] ^= r[s] + p.RotSalt
		case opIROTR:
			r[d] = bits.RotateLeft64(r[d], -int((r[s]^imm)&63))
		case opINEG:
			r[d] = ^r[d] + imm
		case opFADD:
			f[d&7] = f[d&7] + f[s&7]
			if b := math.Float64bits(f[d&7]); b&0x7FF0000000000000 == 0x7FF0000000000000 || b<<1 == 0 {
				f[d&7] = normFloat(r[d] | 1)
			}
		case opFMUL:
			f[d&7] = f[d&7] * f[s&7]
			if b := math.Float64bits(f[d&7]); b&0x7FF0000000000000 == 0x7FF0000000000000 || b<<1 == 0 {
				f[d&7] = normFloat(r[d] | 1)
			}
		case opFDIV:
			f[d&7] = f[d&7] / normFloat(math.Float64bits(f[s&7]))
			if b := math.Float64bits(f[d&7]); b&0x7FF0000000000000 == 0x7FF0000000000000 || b<<1 == 0 {
				f[d&7] = normFloat(r[d] | 1)
			}
		case opFSQRT:
			f[d&7] = math.Sqrt(math.Abs(f[d&7]))
			if b := math.Float64bits(f[d&7]); b&0x7FF0000000000000 == 0x7FF0000000000000 || b<<1 == 0 {
				f[d&7] = normFloat(r[d] | 1)
			}
		case opLOAD:
			addr := (r[s] + imm) & scratchMask
			r[d] ^= scratch[addr>>3]
		case opSTORE:
			addr := (r[d] + imm) & scratchMask
			idx := addr >> 3
			delta := r[s] + loop
			scratch[idx] ^= delta
			fold[idx&7] ^= delta
		case opCBRANCH:
			if (r[d]+imm)&p.BranchMask == 0 && taken[pc] < 8 {
				taken[pc]++
				back := int(ins.imm%31) + 1
				pc -= back
				if pc < 0 {
					pc = 0
				}
				continue
			}
		case opAESR:
			addr := ((r[s] + imm) & scratchMask) & ^uint64(15)
			w := addr >> 3
			old0 := scratch[w]
			old1 := scratch[w+1]
			var block [16]byte
			binary.LittleEndian.PutUint64(block[0:8], old0)
			binary.LittleEndian.PutUint64(block[8:16], old1)
			probeEncryptBlock(&block, rk)
			scratch[w] = binary.LittleEndian.Uint64(block[0:8])
			scratch[w+1] = binary.LittleEndian.Uint64(block[8:16])
			fold[w&7] ^= old0 ^ scratch[w]
			fold[(w+1)&7] ^= old1 ^ scratch[w+1]
			r[d] ^= scratch[w]
		case opXDOM:
			if ins.imm&1 == 0 {
				r[d] ^= math.Float64bits(f[s&7])
			} else {
				f[d&7] = f[d&7] * normFloat(r[s])
			}
		}
		pc++
	}
}

func BenchmarkInterpAddChain640(b *testing.B) {
	var sink uint64
	for i := 0; i < b.N; i++ {
		sink ^= runInterpAddChain(1, 640)
	}
	_ = sink
}

func BenchmarkJITAddChain640(b *testing.B) {
	var sink uint64
	for i := 0; i < b.N; i++ {
		sink ^= runJITAddChain(1, 640)
	}
	_ = sink
}

func BenchmarkJITAddChain640ExecuteOnly(b *testing.B) {
	var sink uint64
	sink = runJITAddChain(b.N, 640)
	_ = sink
}

func BenchmarkJITAddChain640ReuseMprotect(b *testing.B) {
	var sink uint64
	sink = runJITAddChainReuseMprotect(b.N, 640)
	_ = sink
}

func BenchmarkInterpIntMix640(b *testing.B) {
	var sink uint64
	for i := 0; i < b.N; i++ {
		sink ^= runInterpIntMix(1, 640)
	}
	_ = sink
}

func BenchmarkJITIntMix640(b *testing.B) {
	var sink uint64
	for i := 0; i < b.N; i++ {
		sink ^= runJITIntMix(1, 640)
	}
	_ = sink
}

func BenchmarkJITIntMix640ExecuteOnly(b *testing.B) {
	var sink uint64
	sink = runJITIntMix(b.N, 640)
	_ = sink
}

func BenchmarkJITIntMix640ResidentExecuteOnly(b *testing.B) {
	var sink uint64
	sink = runJITIntMixResident(b.N, 640)
	_ = sink
}

func BenchmarkJITIntMix640ReuseMprotect(b *testing.B) {
	var sink uint64
	sink = runJITIntMixReuseMprotect(b.N, 640)
	_ = sink
}

func BenchmarkInterpMemMix640(b *testing.B) {
	var sink uint64
	sink = runInterpMemMix(b.N, 640)
	_ = sink
}

func BenchmarkInterpResMemMix640(b *testing.B) {
	var sink uint64
	sink = runInterpResMemMix(b.N, 640)
	_ = sink
}

func BenchmarkInterpResFloatMix640(b *testing.B) {
	var sink uint64
	sink = runInterpResFloatMix(b.N, 640)
	_ = sink
}

func BenchmarkJITMemMix640ExecuteOnly(b *testing.B) {
	var sink uint64
	sink = runJITMemMix(b.N, 640)
	_ = sink
}

func BenchmarkJITResMemMix640ExecuteOnly(b *testing.B) {
	var sink uint64
	sink = runJITResMemMix(b.N, 640)
	_ = sink
}

func BenchmarkJITResFloatMix640ExecuteOnly(b *testing.B) {
	var sink uint64
	sink = runJITResFloatMix(b.N, 640)
	_ = sink
}

func BenchmarkJITMemMix640ReuseMprotect(b *testing.B) {
	var sink uint64
	sink = runJITMemMixReuseMprotect(b.N, 640)
	_ = sink
}

func BenchmarkJITRealProgramOneLoopCompileExecute(b *testing.B) {
	p := DeriveParams(EpochSeed0())
	vm := NewVM(p)
	seed := benchmarkSeed()
	vm.fillScratch(seed)
	vm.genProgram(seed)
	var rk [176]byte
	initProbeRoundKeys(&rk)
	var r [16]uint64
	var f [8]float64
	for i := 0; i < 4; i++ {
		r[i] = binary.LittleEndian.Uint64(seed[i*8 : i*8+8])
	}
	for i := 4; i < 16; i++ {
		r[i] = vm.scratch[i] ^ p.RotSalt
	}
	for i := 0; i < 8; i++ {
		f[i] = normFloat(vm.scratch[16+i])
	}
	var fold [8]uint64
	taken := make([]uint8, len(vm.prog))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := range taken {
			taken[j] = 0
		}
		if !jitRealProgramOnce(vm.prog, p.BranchMask, &r, vm.scratch, &fold, &f, p.RotSalt, uint64(i), &rk, taken) {
			b.Fatal("jit real program failed")
		}
	}
}

func BenchmarkJITRealProgramOneLoopExecuteOnly(b *testing.B) {
	p := DeriveParams(EpochSeed0())
	vm := NewVM(p)
	seed := benchmarkSeed()
	vm.fillScratch(seed)
	vm.genProgram(seed)
	var rk [176]byte
	initProbeRoundKeys(&rk)
	var r [16]uint64
	var f [8]float64
	for i := 0; i < 4; i++ {
		r[i] = binary.LittleEndian.Uint64(seed[i*8 : i*8+8])
	}
	for i := 4; i < 16; i++ {
		r[i] = vm.scratch[i] ^ p.RotSalt
	}
	for i := 0; i < 8; i++ {
		f[i] = normFloat(vm.scratch[16+i])
	}
	var fold [8]uint64
	taken := make([]uint8, len(vm.prog))

	b.ResetTimer()
	if !jitRealProgramLoop(vm.prog, p.BranchMask, &r, vm.scratch, &fold, &f, p.RotSalt, 0, &rk, taken, b.N) {
		b.Fatal("jit real program loop failed")
	}
}

func BenchmarkJITRealProgramOneLoopResidentExecuteOnly(b *testing.B) {
	p := DeriveParams(EpochSeed0())
	vm := NewVM(p)
	seed := benchmarkSeed()
	vm.fillScratch(seed)
	vm.genProgram(seed)
	var rk [176]byte
	initProbeRoundKeys(&rk)
	var r [16]uint64
	var f [8]float64
	for i := 0; i < 4; i++ {
		r[i] = binary.LittleEndian.Uint64(seed[i*8 : i*8+8])
	}
	for i := 4; i < 16; i++ {
		r[i] = vm.scratch[i] ^ p.RotSalt
	}
	for i := 0; i < 8; i++ {
		f[i] = normFloat(vm.scratch[16+i])
	}
	var fold [8]uint64
	taken := make([]uint8, len(vm.prog))

	b.ResetTimer()
	if !jitRealProgramResidentLoop(vm.prog, p.BranchMask, &r, vm.scratch, &fold, &f, p.RotSalt, 0, &rk, taken, b.N) {
		b.Fatal("resident jit real program loop failed")
	}
}

func BenchmarkJITRealExecuteProbeDataset(b *testing.B) {
	p := DeriveParams(EpochSeed0())
	vm := NewVM(p)
	seed := benchmarkSeed()
	vm.fillScratch(seed)
	vm.genProgram(seed)
	dataset := getDataset(p.DatasetKey)
	var rk [176]byte
	expandAES128RoundKeys(&p.AesKey, &rk)
	var r [16]uint64
	var f [8]float64
	initProbeState(seed, p, vm.scratch, &r, &f)
	var fold [8]uint64
	foldScratchGo(vm.scratch, &fold)
	taken := make([]uint8, len(vm.prog))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if !jitRealExecuteProbe(p, vm.prog, &r, vm.scratch, &fold, &f, dataset, true, &rk, taken) {
			b.Fatal("jit real execute probe failed")
		}
	}
}

func BenchmarkJITRealExecuteProbeDatasetResident(b *testing.B) {
	p := DeriveParams(EpochSeed0())
	vm := NewVM(p)
	seed := benchmarkSeed()
	vm.fillScratch(seed)
	vm.genProgram(seed)
	dataset := getDataset(p.DatasetKey)
	var rk [176]byte
	expandAES128RoundKeys(&p.AesKey, &rk)
	var r [16]uint64
	var f [8]float64
	initProbeState(seed, p, vm.scratch, &r, &f)
	var fold [8]uint64
	foldScratchGo(vm.scratch, &fold)
	taken := make([]uint8, len(vm.prog))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if !jitRealExecuteProbeResident(p, vm.prog, &r, vm.scratch, &fold, &f, dataset, true, &rk, taken) {
			b.Fatal("resident jit real execute probe failed")
		}
	}
}

func BenchmarkJITHashDatasetProbe(b *testing.B) {
	p := DeriveParams(EpochSeed0())
	vm := NewVM(p)
	header := make([]byte, 124)
	for i := range header {
		header[i] = byte(i*17 + 11)
	}
	dataset := getDataset(p.DatasetKey)
	var rk [176]byte
	expandAES128RoundKeys(&p.AesKey, &rk)
	var fold [8]uint64
	taken := make([]uint8, p.ProgSize)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		binary.LittleEndian.PutUint64(header[len(header)-8:], uint64(i))
		var seedInput [8 + 256]byte
		copy(seedInput[:8], "nm-seed|")
		copy(seedInput[8:], header)
		seed := sha256.Sum256(seedInput[:8+len(header)])
		var keyInput [33]byte
		copy(keyInput[:32], seed[:])
		keyInput[32] = 0x53
		scratchKey := sha256.Sum256(keyInput[:])
		if !fillScratchFoldFast(scratchKey, vm.scratch, &fold) {
			b.Fatal("fillScratchFoldFast unavailable")
		}
		vm.genProgram(seed)
		var r [16]uint64
		var f [8]float64
		initProbeState(seed, p, vm.scratch, &r, &f)
		if !jitRealExecuteProbe(p, vm.prog, &r, vm.scratch, &fold, &f, dataset, true, &rk, taken) {
			b.Fatal("jit real execute probe failed")
		}
	}
}

func BenchmarkJITHashDatasetProbeResident(b *testing.B) {
	p := DeriveParams(EpochSeed0())
	vm := NewVM(p)
	header := make([]byte, 124)
	for i := range header {
		header[i] = byte(i*17 + 11)
	}
	dataset := getDataset(p.DatasetKey)
	var rk [176]byte
	expandAES128RoundKeys(&p.AesKey, &rk)
	var fold [8]uint64
	taken := make([]uint8, p.ProgSize)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		binary.LittleEndian.PutUint64(header[len(header)-8:], uint64(i))
		var seedInput [8 + 256]byte
		copy(seedInput[:8], "nm-seed|")
		copy(seedInput[8:], header)
		seed := sha256.Sum256(seedInput[:8+len(header)])
		var keyInput [33]byte
		copy(keyInput[:32], seed[:])
		keyInput[32] = 0x53
		scratchKey := sha256.Sum256(keyInput[:])
		if !fillScratchFoldFast(scratchKey, vm.scratch, &fold) {
			b.Fatal("fillScratchFoldFast unavailable")
		}
		vm.genProgram(seed)
		var r [16]uint64
		var f [8]float64
		initProbeState(seed, p, vm.scratch, &r, &f)
		if !jitRealExecuteProbeResident(p, vm.prog, &r, vm.scratch, &fold, &f, dataset, true, &rk, taken) {
			b.Fatal("resident jit real execute probe failed")
		}
		jitHashProbeSink = finalizeProbeHash(seed, &r, &f, &fold)
	}
}

func BenchmarkJITHashDatasetProbeReuseBuffer(b *testing.B) {
	p := DeriveParams(EpochSeed0())
	vm := NewVM(p)
	buf := newJITBuffer(p.ProgSize)
	if buf == nil {
		b.Fatal("failed to allocate jit buffer")
	}
	defer freeJITBuffer(buf)
	header := make([]byte, 124)
	for i := range header {
		header[i] = byte(i*17 + 11)
	}
	dataset := getDataset(p.DatasetKey)
	var rk [176]byte
	expandAES128RoundKeys(&p.AesKey, &rk)
	var fold [8]uint64
	taken := make([]uint8, p.ProgSize)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		binary.LittleEndian.PutUint64(header[len(header)-8:], uint64(i))
		var seedInput [8 + 256]byte
		copy(seedInput[:8], "nm-seed|")
		copy(seedInput[8:], header)
		seed := sha256.Sum256(seedInput[:8+len(header)])
		var keyInput [33]byte
		copy(keyInput[:32], seed[:])
		keyInput[32] = 0x53
		scratchKey := sha256.Sum256(keyInput[:])
		if !fillScratchFoldFast(scratchKey, vm.scratch, &fold) {
			b.Fatal("fillScratchFoldFast unavailable")
		}
		vm.genProgram(seed)
		var r [16]uint64
		var f [8]float64
		initProbeState(seed, p, vm.scratch, &r, &f)
		if !jitRealExecuteProbeReuse(buf, p, vm.prog, &r, vm.scratch, &fold, &f, dataset, true, &rk, taken) {
			b.Fatal("jit real execute probe reuse failed")
		}
		jitHashProbeSink = finalizeProbeHash(seed, &r, &f, &fold)
	}
}

func BenchmarkJITHashDatasetProbeResidentReuseBuffer(b *testing.B) {
	p := DeriveParams(EpochSeed0())
	vm := NewVM(p)
	buf := newJITResidentBuffer(p.ProgSize)
	if buf == nil {
		b.Fatal("failed to allocate resident jit buffer")
	}
	defer freeJITBuffer(buf)
	header := make([]byte, 124)
	for i := range header {
		header[i] = byte(i*17 + 11)
	}
	dataset := getDataset(p.DatasetKey)
	var rk [176]byte
	expandAES128RoundKeys(&p.AesKey, &rk)
	var fold [8]uint64
	taken := make([]uint8, p.ProgSize)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		binary.LittleEndian.PutUint64(header[len(header)-8:], uint64(i))
		var seedInput [8 + 256]byte
		copy(seedInput[:8], "nm-seed|")
		copy(seedInput[8:], header)
		seed := sha256.Sum256(seedInput[:8+len(header)])
		var keyInput [33]byte
		copy(keyInput[:32], seed[:])
		keyInput[32] = 0x53
		scratchKey := sha256.Sum256(keyInput[:])
		if !fillScratchFoldFast(scratchKey, vm.scratch, &fold) {
			b.Fatal("fillScratchFoldFast unavailable")
		}
		vm.genProgram(seed)
		var r [16]uint64
		var f [8]float64
		initProbeState(seed, p, vm.scratch, &r, &f)
		if !jitRealExecuteProbeResidentReuse(buf, p, vm.prog, &r, vm.scratch, &fold, &f, dataset, true, &rk, taken) {
			b.Fatal("resident jit real execute probe reuse failed")
		}
		jitHashProbeSink = finalizeProbeHash(seed, &r, &f, &fold)
	}
}

func BenchmarkJITHashDatasetProbeReuseBufferParallel(b *testing.B) {
	p := DeriveParams(EpochSeed0())
	dataset := getDataset(p.DatasetKey)
	var rk [176]byte
	expandAES128RoundKeys(&p.AesKey, &rk)
	b.RunParallel(func(pb *testing.PB) {
		vm := NewVM(p)
		buf := newJITBuffer(p.ProgSize)
		if buf == nil {
			b.Fatal("failed to allocate jit buffer")
		}
		defer freeJITBuffer(buf)
		header := make([]byte, 124)
		for i := range header {
			header[i] = byte(i*17 + 11)
		}
		var fold [8]uint64
		taken := make([]uint8, p.ProgSize)
		nonce := uint64(0)
		for pb.Next() {
			binary.LittleEndian.PutUint64(header[len(header)-8:], nonce)
			nonce++
			var seedInput [8 + 256]byte
			copy(seedInput[:8], "nm-seed|")
			copy(seedInput[8:], header)
			seed := sha256.Sum256(seedInput[:8+len(header)])
			var keyInput [33]byte
			copy(keyInput[:32], seed[:])
			keyInput[32] = 0x53
			scratchKey := sha256.Sum256(keyInput[:])
			if !fillScratchFoldFast(scratchKey, vm.scratch, &fold) {
				b.Fatal("fillScratchFoldFast unavailable")
			}
			vm.genProgram(seed)
			var r [16]uint64
			var f [8]float64
			initProbeState(seed, p, vm.scratch, &r, &f)
			if !jitRealExecuteProbeReuse(buf, p, vm.prog, &r, vm.scratch, &fold, &f, dataset, true, &rk, taken) {
				b.Fatal("jit real execute probe reuse failed")
			}
			jitHashProbeSink = finalizeProbeHash(seed, &r, &f, &fold)
		}
	})
}

func BenchmarkJITHashDatasetProbeResidentReuseBufferParallel(b *testing.B) {
	p := DeriveParams(EpochSeed0())
	dataset := getDataset(p.DatasetKey)
	var rk [176]byte
	expandAES128RoundKeys(&p.AesKey, &rk)
	b.RunParallel(func(pb *testing.PB) {
		vm := NewVM(p)
		buf := newJITResidentBuffer(p.ProgSize)
		if buf == nil {
			b.Fatal("failed to allocate resident jit buffer")
		}
		defer freeJITBuffer(buf)
		header := make([]byte, 124)
		for i := range header {
			header[i] = byte(i*17 + 11)
		}
		var fold [8]uint64
		taken := make([]uint8, p.ProgSize)
		nonce := uint64(0)
		for pb.Next() {
			binary.LittleEndian.PutUint64(header[len(header)-8:], nonce)
			nonce++
			var seedInput [8 + 256]byte
			copy(seedInput[:8], "nm-seed|")
			copy(seedInput[8:], header)
			seed := sha256.Sum256(seedInput[:8+len(header)])
			var keyInput [33]byte
			copy(keyInput[:32], seed[:])
			keyInput[32] = 0x53
			scratchKey := sha256.Sum256(keyInput[:])
			if !fillScratchFoldFast(scratchKey, vm.scratch, &fold) {
				b.Fatal("fillScratchFoldFast unavailable")
			}
			vm.genProgram(seed)
			var r [16]uint64
			var f [8]float64
			initProbeState(seed, p, vm.scratch, &r, &f)
			if !jitRealExecuteProbeResidentReuse(buf, p, vm.prog, &r, vm.scratch, &fold, &f, dataset, true, &rk, taken) {
				b.Fatal("resident jit real execute probe reuse failed")
			}
			jitHashProbeSink = finalizeProbeHash(seed, &r, &f, &fold)
		}
	})
}
