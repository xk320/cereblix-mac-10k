//go:build darwin && arm64 && cgo

package neuromorph

/*
#cgo CFLAGS: -O3 -march=armv8-a+crypto
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <arm_neon.h>

static const uint8_t nm_sbox[256] = {
	0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
	0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
	0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
	0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
	0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
	0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
	0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
	0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
	0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
	0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
	0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
	0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
	0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
	0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
	0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
	0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
};

static void nm_expand_aes128(const uint8_t key[16], uint8_t rk[176]) {
	static const uint8_t rcon[11] = {0x00,0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1b,0x36};
	memcpy(rk, key, 16);
	uint8_t temp[4];
	int bytes = 16;
	int round = 1;
	while (bytes < 176) {
		temp[0] = rk[bytes - 4];
		temp[1] = rk[bytes - 3];
		temp[2] = rk[bytes - 2];
		temp[3] = rk[bytes - 1];
		if ((bytes & 15) == 0) {
			uint8_t first = temp[0];
			temp[0] = nm_sbox[temp[1]] ^ rcon[round++];
			temp[1] = nm_sbox[temp[2]];
			temp[2] = nm_sbox[temp[3]];
			temp[3] = nm_sbox[first];
		}
		for (int i = 0; i < 4; i++) {
			rk[bytes] = rk[bytes - 16] ^ temp[i];
			bytes++;
		}
	}
}

static inline uint8x16_t nm_aes128_encrypt(uint8x16_t block, const uint8x16_t rk[11]) {
	block = vaesmcq_u8(vaeseq_u8(block, rk[0]));
	block = vaesmcq_u8(vaeseq_u8(block, rk[1]));
	block = vaesmcq_u8(vaeseq_u8(block, rk[2]));
	block = vaesmcq_u8(vaeseq_u8(block, rk[3]));
	block = vaesmcq_u8(vaeseq_u8(block, rk[4]));
	block = vaesmcq_u8(vaeseq_u8(block, rk[5]));
	block = vaesmcq_u8(vaeseq_u8(block, rk[6]));
	block = vaesmcq_u8(vaeseq_u8(block, rk[7]));
	block = vaesmcq_u8(vaeseq_u8(block, rk[8]));
	block = vaeseq_u8(block, rk[9]);
	return veorq_u8(block, rk[10]);
}

static inline void nm_aes128_encrypt4(uint8x16_t *b0, uint8x16_t *b1, uint8x16_t *b2, uint8x16_t *b3, const uint8x16_t rk[11]) {
	*b0 = vaesmcq_u8(vaeseq_u8(*b0, rk[0]));
	*b1 = vaesmcq_u8(vaeseq_u8(*b1, rk[0]));
	*b2 = vaesmcq_u8(vaeseq_u8(*b2, rk[0]));
	*b3 = vaesmcq_u8(vaeseq_u8(*b3, rk[0]));
	*b0 = vaesmcq_u8(vaeseq_u8(*b0, rk[1]));
	*b1 = vaesmcq_u8(vaeseq_u8(*b1, rk[1]));
	*b2 = vaesmcq_u8(vaeseq_u8(*b2, rk[1]));
	*b3 = vaesmcq_u8(vaeseq_u8(*b3, rk[1]));
	*b0 = vaesmcq_u8(vaeseq_u8(*b0, rk[2]));
	*b1 = vaesmcq_u8(vaeseq_u8(*b1, rk[2]));
	*b2 = vaesmcq_u8(vaeseq_u8(*b2, rk[2]));
	*b3 = vaesmcq_u8(vaeseq_u8(*b3, rk[2]));
	*b0 = vaesmcq_u8(vaeseq_u8(*b0, rk[3]));
	*b1 = vaesmcq_u8(vaeseq_u8(*b1, rk[3]));
	*b2 = vaesmcq_u8(vaeseq_u8(*b2, rk[3]));
	*b3 = vaesmcq_u8(vaeseq_u8(*b3, rk[3]));
	*b0 = vaesmcq_u8(vaeseq_u8(*b0, rk[4]));
	*b1 = vaesmcq_u8(vaeseq_u8(*b1, rk[4]));
	*b2 = vaesmcq_u8(vaeseq_u8(*b2, rk[4]));
	*b3 = vaesmcq_u8(vaeseq_u8(*b3, rk[4]));
	*b0 = vaesmcq_u8(vaeseq_u8(*b0, rk[5]));
	*b1 = vaesmcq_u8(vaeseq_u8(*b1, rk[5]));
	*b2 = vaesmcq_u8(vaeseq_u8(*b2, rk[5]));
	*b3 = vaesmcq_u8(vaeseq_u8(*b3, rk[5]));
	*b0 = vaesmcq_u8(vaeseq_u8(*b0, rk[6]));
	*b1 = vaesmcq_u8(vaeseq_u8(*b1, rk[6]));
	*b2 = vaesmcq_u8(vaeseq_u8(*b2, rk[6]));
	*b3 = vaesmcq_u8(vaeseq_u8(*b3, rk[6]));
	*b0 = vaesmcq_u8(vaeseq_u8(*b0, rk[7]));
	*b1 = vaesmcq_u8(vaeseq_u8(*b1, rk[7]));
	*b2 = vaesmcq_u8(vaeseq_u8(*b2, rk[7]));
	*b3 = vaesmcq_u8(vaeseq_u8(*b3, rk[7]));
	*b0 = vaesmcq_u8(vaeseq_u8(*b0, rk[8]));
	*b1 = vaesmcq_u8(vaeseq_u8(*b1, rk[8]));
	*b2 = vaesmcq_u8(vaeseq_u8(*b2, rk[8]));
	*b3 = vaesmcq_u8(vaeseq_u8(*b3, rk[8]));
	*b0 = veorq_u8(vaeseq_u8(*b0, rk[9]), rk[10]);
	*b1 = veorq_u8(vaeseq_u8(*b1, rk[9]), rk[10]);
	*b2 = veorq_u8(vaeseq_u8(*b2, rk[9]), rk[10]);
	*b3 = veorq_u8(vaeseq_u8(*b3, rk[9]), rk[10]);
}

static void nm_fill_scratch_arm64(const uint8_t key32[32], uint64_t *dst, size_t words) {
	uint8_t rkBytes[176];
	uint8x16_t rk[11];
	uint64_t tail;
	nm_expand_aes128(key32, rkBytes);
	for (int i = 0; i < 11; i++) {
		rk[i] = vld1q_u8(rkBytes + i * 16);
	}
	memcpy(&tail, key32 + 24, sizeof(tail));
	for (uint64_t i = 0; i < (uint64_t)words; i += 8) {
		uint64_t in0[2] = {i, tail};
		uint64_t in1[2] = {i + 2, tail};
		uint64_t in2[2] = {i + 4, tail};
		uint64_t in3[2] = {i + 6, tail};
		uint8x16_t b0 = vld1q_u8((const uint8_t *)in0);
		uint8x16_t b1 = vld1q_u8((const uint8_t *)in1);
		uint8x16_t b2 = vld1q_u8((const uint8_t *)in2);
		uint8x16_t b3 = vld1q_u8((const uint8_t *)in3);
		nm_aes128_encrypt4(&b0, &b1, &b2, &b3, rk);
		vst1q_u8((uint8_t *)(dst + i), b0);
		vst1q_u8((uint8_t *)(dst + i + 2), b1);
		vst1q_u8((uint8_t *)(dst + i + 4), b2);
		vst1q_u8((uint8_t *)(dst + i + 6), b3);
	}
}

typedef struct {
	uint8_t op;
	uint8_t dst;
	uint8_t src;
	uint8_t pad;
	uint32_t imm;
} nm_instr;

static inline uint32_t nm_le32(const uint8_t *p) {
	return ((uint32_t)p[0]) | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static void nm_gen_program_arm64(const uint8_t key32[32], const uint8_t op_table[256], nm_instr *prog, size_t prog_size) {
	uint8_t rkBytes[176];
	uint8x16_t rk[11];
	uint8_t inBytes[16];
	nm_expand_aes128(key32, rkBytes);
	for (int i = 0; i < 11; i++) {
		rk[i] = vld1q_u8(rkBytes + i * 16);
	}
	memcpy(inBytes, key32 + 16, 16);
	size_t i = 0;
	while (i < prog_size) {
		uint8x16_t block = vld1q_u8(inBytes);
		block = nm_aes128_encrypt(block, rk);
		uint8_t out[16];
		vst1q_u8(out, block);
		memcpy(inBytes, out, 16);
		uint64_t ctr;
		memcpy(&ctr, inBytes, sizeof(ctr));
		ctr++;
		memcpy(inBytes, &ctr, sizeof(ctr));
		for (int off = 0; off < 16 && i < prog_size; off += 8) {
			const uint8_t *b = out + off;
			prog[i].op = op_table[b[0]];
			prog[i].dst = b[1] & 15;
			prog[i].src = b[2] & 15;
			prog[i].pad = 0;
			prog[i].imm = nm_le32(b + 4);
			i++;
		}
	}
}

static inline double nm_norm_float(uint64_t x) {
	uint64_t bits = (UINT64_C(1023) << 52) | (x & UINT64_C(0x000FFFFFFFFFFFFF));
	double out;
	memcpy(&out, &bits, sizeof(out));
	return out;
}

static inline uint64_t nm_float_bits(double x) {
	uint64_t out;
	memcpy(&out, &x, sizeof(out));
	return out;
}

static inline int nm_bad_float_bits(uint64_t bits) {
	uint64_t abs_bits = bits & UINT64_C(0x7FFFFFFFFFFFFFFF);
	return abs_bits == 0 || abs_bits >= UINT64_C(0x7FF0000000000000);
}

static inline int nm_bad_float(double x) {
	return nm_bad_float_bits(nm_float_bits(x));
}

static inline uint64_t nm_rotr64(uint64_t x, uint64_t n) {
	n &= 63;
	if (n == 0) {
		return x;
	}
	return (x >> n) | (x << (64 - n));
}

static void nm_execute_program_arm64(
	const nm_instr *prog,
	size_t prog_size,
	uint8_t *taken,
	uint64_t *scratch,
	uint64_t r[16],
	double f[8],
	uint32_t loops,
	uint64_t branch_mask,
	uint64_t rot_salt,
	const uint64_t *dataset,
	uint8_t use_dataset,
	const uint8_t aes_key[16]
) {
	uint8_t rkBytes[176];
	uint8x16_t rk[11];
	nm_expand_aes128(aes_key, rkBytes);
	for (int i = 0; i < 11; i++) {
		rk[i] = vld1q_u8(rkBytes + i * 16);
	}
	for (uint32_t loop = 0; loop < loops; loop++) {
		memset(taken, 0, prog_size);
		int pc = 0;
		while (pc < (int)prog_size) {
			nm_instr ins = prog[pc];
			uint8_t d = ins.dst;
			uint8_t s = ins.src;
			uint64_t imm = ins.imm;
			switch (ins.op) {
			case 0:
				r[d] += r[s] + imm;
				break;
			case 1:
				r[d] *= r[s] | 1;
				break;
			case 2:
				r[d] = (uint64_t)(((__uint128_t)r[d] * (__uint128_t)r[s]) >> 64) ^ imm;
				break;
			case 3:
				r[d] ^= r[s] + rot_salt;
				break;
			case 4:
				r[d] = nm_rotr64(r[d], (r[s] ^ imm) & 63);
				break;
			case 5:
				r[d] = ~r[d] + imm;
				break;
			case 6:
				f[d & 7] = f[d & 7] + f[s & 7];
				if (nm_bad_float(f[d & 7])) {
					f[d & 7] = nm_norm_float(r[d] | 1);
				}
				break;
			case 7:
				f[d & 7] = f[d & 7] * f[s & 7];
				if (nm_bad_float(f[d & 7])) {
					f[d & 7] = nm_norm_float(r[d] | 1);
				}
				break;
			case 8:
				f[d & 7] = f[d & 7] / nm_norm_float(nm_float_bits(f[s & 7]));
				if (nm_bad_float(f[d & 7])) {
					f[d & 7] = nm_norm_float(r[d] | 1);
				}
				break;
			case 9:
				f[d & 7] = sqrt(fabs(f[d & 7]));
				if (nm_bad_float(f[d & 7])) {
					f[d & 7] = nm_norm_float(r[d] | 1);
				}
				break;
			case 10: {
				uint64_t addr = (r[s] + imm) & UINT64_C(0x1FFFF8);
				r[d] ^= scratch[addr >> 3];
				break;
			}
			case 11: {
				uint64_t addr = (r[d] + imm) & UINT64_C(0x1FFFF8);
				scratch[addr >> 3] ^= r[s] + (uint64_t)loop;
				break;
			}
			case 12:
				if (((r[d] + imm) & branch_mask) == 0 && taken[pc] < 8) {
					taken[pc]++;
					int back = (int)(ins.imm % 31) + 1;
					pc -= back;
					if (pc < 0) {
						pc = 0;
					}
					continue;
				}
				break;
			case 13: {
				uint64_t addr = ((r[s] + imm) & UINT64_C(0x1FFFF8)) & ~UINT64_C(15);
				uint64_t w = addr >> 3;
				uint8x16_t block = vld1q_u8((const uint8_t *)(scratch + w));
				block = nm_aes128_encrypt(block, rk);
				vst1q_u8((uint8_t *)(scratch + w), block);
				r[d] ^= scratch[w];
				break;
			}
			case 14:
				if ((ins.imm & 1) == 0) {
					r[d] ^= nm_float_bits(f[s & 7]);
				} else {
					f[d & 7] = f[d & 7] * nm_norm_float(r[s]);
				}
				break;
			}
			pc++;
		}
		if (use_dataset) {
			uint64_t addr = (r[1] ^ rot_salt) & UINT64_C(0x3FFFFF8);
			for (int k = 0; k < 64; k++) {
				uint64_t v = dataset[addr >> 3];
				r[k & 15] ^= v;
				addr = (v + r[(k + 1) & 15] + (uint64_t)loop) & UINT64_C(0x3FFFFF8);
			}
		}
		uint64_t base = ((r[0] ^ ((uint64_t)loop * UINT64_C(0x9E3779B97F4A7C15))) & UINT64_C(0x1FFFF8)) >> 3;
		for (int i = 0; i < 16; i++) {
			scratch[(base + (uint64_t)i) & UINT64_C(0x3FFFF)] ^= r[i];
		}
		for (int i = 0; i < 8; i++) {
			r[i + 8] ^= nm_float_bits(f[i]);
		}
	}
}

static void nm_fold_scratch_arm64(const uint64_t *src, size_t words, uint64_t out[8]) {
	uint64x2_t a0 = vdupq_n_u64(0), a1 = vdupq_n_u64(0), a2 = vdupq_n_u64(0), a3 = vdupq_n_u64(0);
	for (size_t i = 0; i < words; i += 8) {
		a0 = veorq_u64(a0, vld1q_u64(src + i));
		a1 = veorq_u64(a1, vld1q_u64(src + i + 2));
		a2 = veorq_u64(a2, vld1q_u64(src + i + 4));
		a3 = veorq_u64(a3, vld1q_u64(src + i + 6));
	}
	vst1q_u64(out, a0);
	vst1q_u64(out + 2, a1);
	vst1q_u64(out + 4, a2);
	vst1q_u64(out + 6, a3);
}
*/
import "C"
import "unsafe"

func fillScratchFast(key [32]byte, scratch []uint64) bool {
	if len(scratch) == 0 {
		return true
	}
	C.nm_fill_scratch_arm64(
		(*C.uint8_t)(unsafe.Pointer(&key[0])),
		(*C.uint64_t)(unsafe.Pointer(&scratch[0])),
		C.size_t(len(scratch)),
	)
	return true
}

func genProgramFast(key [32]byte, opTable *[256]uint8, prog []instr) bool {
	if len(prog) == 0 {
		return true
	}
	C.nm_gen_program_arm64(
		(*C.uint8_t)(unsafe.Pointer(&key[0])),
		(*C.uint8_t)(unsafe.Pointer(&opTable[0])),
		(*C.nm_instr)(unsafe.Pointer(&prog[0])),
		C.size_t(len(prog)),
	)
	return true
}

func executeProgramFast(p *Params, prog []instr, taken []uint8, scratch []uint64, dataset []uint64, r *[16]uint64, f *[8]float64, useDataset bool) bool {
	if len(prog) == 0 || len(scratch) == 0 {
		return true
	}
	var ds *C.uint64_t
	var useDS C.uint8_t
	if useDataset {
		if len(dataset) == 0 {
			return false
		}
		ds = (*C.uint64_t)(unsafe.Pointer(&dataset[0]))
		useDS = 1
	}
	C.nm_execute_program_arm64(
		(*C.nm_instr)(unsafe.Pointer(&prog[0])),
		C.size_t(len(prog)),
		(*C.uint8_t)(unsafe.Pointer(&taken[0])),
		(*C.uint64_t)(unsafe.Pointer(&scratch[0])),
		(*C.uint64_t)(unsafe.Pointer(&r[0])),
		(*C.double)(unsafe.Pointer(&f[0])),
		C.uint32_t(p.Loops),
		C.uint64_t(p.BranchMask),
		C.uint64_t(p.RotSalt),
		ds,
		useDS,
		(*C.uint8_t)(unsafe.Pointer(&p.AesKey[0])),
	)
	return true
}

func foldScratchFast(scratch []uint64, fold *[8]uint64) bool {
	if len(scratch) == 0 {
		return true
	}
	C.nm_fold_scratch_arm64(
		(*C.uint64_t)(unsafe.Pointer(&scratch[0])),
		C.size_t(len(scratch)),
		(*C.uint64_t)(unsafe.Pointer(&fold[0])),
	)
	return true
}
