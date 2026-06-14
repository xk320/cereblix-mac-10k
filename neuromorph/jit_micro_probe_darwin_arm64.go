//go:build darwin && arm64 && cgo

package neuromorph

/*
#include <stdint.h>
#include <math.h>
#include <arm_neon.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <pthread.h>
#include <unistd.h>

void nm_expand_aes128(const uint8_t key[16], uint8_t rk[176]);

typedef void (*nm_jit_fn)(uint64_t *r);
typedef void (*nm_jit_int_fn)(uint64_t *r, uint64_t rot_salt);
typedef void (*nm_jit_mem_fn)(uint64_t *r, uint64_t *scratch, uint64_t *fold, double *f, uint64_t rot_salt, uint64_t loop, const uint8_t *rk, uint8_t *taken);

typedef struct {
	uint8_t op;
	uint8_t dst;
	uint8_t src;
	uint8_t pad;
	uint32_t imm;
} nm_probe_instr;

typedef struct {
	void *mem;
	size_t alloc;
	int map_jit;
} nm_jit_buffer;

#ifndef MAP_JIT
#define MAP_JIT 0
#endif

static inline void nm_jit_write_protect(int enabled) {
#if defined(__APPLE__) && defined(__aarch64__)
	pthread_jit_write_protect_np(enabled);
#else
	(void)enabled;
#endif
}

static inline int nm_probe_bad_float_bits(uint64_t bits) {
	uint64_t abs_bits = bits & UINT64_C(0x7FFFFFFFFFFFFFFF);
	return abs_bits == 0 || abs_bits >= UINT64_C(0x7FF0000000000000);
}

static inline void nm_emit32(uint32_t **p, uint32_t v) {
	*(*p)++ = v;
}

static inline void nm_emit_mov64(uint32_t **p, int rd, uint64_t imm);
static uint64_t nm_bits_from_double(double x);
static double nm_norm_from_u64(uint64_t x);

static inline void nm_emit_ldr(uint32_t **p, int rt, int rn, int word_off) {
	nm_emit32(p, 0xF9400000u | ((uint32_t)word_off << 10) | ((uint32_t)rn << 5) | (uint32_t)rt);
}

static inline void nm_emit_str(uint32_t **p, int rt, int rn, int word_off) {
	nm_emit32(p, 0xF9000000u | ((uint32_t)word_off << 10) | ((uint32_t)rn << 5) | (uint32_t)rt);
}

static inline void nm_emit_add_reg(uint32_t **p, int rd, int rn, int rm) {
	nm_emit32(p, 0x8B000000u | ((uint32_t)rm << 16) | ((uint32_t)rn << 5) | (uint32_t)rd);
}

static inline void nm_emit_sub_reg(uint32_t **p, int rd, int rn, int rm) {
	nm_emit32(p, 0xCB000000u | ((uint32_t)rm << 16) | ((uint32_t)rn << 5) | (uint32_t)rd);
}

static inline void nm_emit_eor_reg(uint32_t **p, int rd, int rn, int rm) {
	nm_emit32(p, 0xCA000000u | ((uint32_t)rm << 16) | ((uint32_t)rn << 5) | (uint32_t)rd);
}

static inline void nm_emit_orr_reg(uint32_t **p, int rd, int rn, int rm) {
	nm_emit32(p, 0xAA000000u | ((uint32_t)rm << 16) | ((uint32_t)rn << 5) | (uint32_t)rd);
}

static inline void nm_emit_mul_reg(uint32_t **p, int rd, int rn, int rm) {
	nm_emit32(p, 0x9B007C00u | ((uint32_t)rm << 16) | ((uint32_t)rn << 5) | (uint32_t)rd);
}

static inline void nm_emit_umulh_reg(uint32_t **p, int rd, int rn, int rm) {
	nm_emit32(p, 0x9BC07C00u | ((uint32_t)rm << 16) | ((uint32_t)rn << 5) | (uint32_t)rd);
}

static inline void nm_emit_and_reg(uint32_t **p, int rd, int rn, int rm) {
	nm_emit32(p, 0x8A000000u | ((uint32_t)rm << 16) | ((uint32_t)rn << 5) | (uint32_t)rd);
}

static inline void nm_emit_and_scratch_mask_x10(uint32_t **p) {
	nm_emit32(p, 0x927D454Au);
}

static inline void nm_emit_and63_x10(uint32_t **p) {
	nm_emit32(p, 0x9240154Au);
}

static inline void nm_emit_and63_x20(uint32_t **p) {
	nm_emit32(p, 0x92401694u);
}

static inline void nm_emit_and_scratch_mask_x20(uint32_t **p) {
	nm_emit32(p, 0x927D4694u);
}

static inline void nm_emit_and_scratch_block_mask_x20(uint32_t **p) {
	nm_emit32(p, 0x927C4294u);
}

static inline void nm_emit_lsr_x20_x20_3(uint32_t **p) {
	nm_emit32(p, 0xD343FE94u);
}

static inline void nm_emit_and7_x20(uint32_t **p) {
	nm_emit32(p, 0x92400A94u);
}

static inline void nm_emit_and_scratch_block_mask_x11_x10(uint32_t **p) {
	nm_emit32(p, 0x927C414Bu);
}

static inline void nm_emit_ror_reg(uint32_t **p, int rd, int rn, int rm) {
	nm_emit32(p, 0x9AC02C00u | ((uint32_t)rm << 16) | ((uint32_t)rn << 5) | (uint32_t)rd);
}

static inline void nm_emit_lsr_x11_x10_3(uint32_t **p) {
	nm_emit32(p, 0xD343FD4Bu);
}

static inline void nm_emit_and7_x11(uint32_t **p) {
	nm_emit32(p, 0x9240096Bu);
}

static inline void nm_emit_ldr_regoff(uint32_t **p, int rt, int rn, int rm) {
	nm_emit32(p, 0xF8606800u | ((uint32_t)rm << 16) | ((uint32_t)rn << 5) | (uint32_t)rt);
}

static inline void nm_emit_str_regoff(uint32_t **p, int rt, int rn, int rm) {
	nm_emit32(p, 0xF8206800u | ((uint32_t)rm << 16) | ((uint32_t)rn << 5) | (uint32_t)rt);
}

static inline void nm_emit_ldr_scaled_regoff(uint32_t **p, int rt, int rn, int rm) {
	nm_emit32(p, 0xF8607800u | ((uint32_t)rm << 16) | ((uint32_t)rn << 5) | (uint32_t)rt);
}

static inline void nm_emit_str_scaled_regoff(uint32_t **p, int rt, int rn, int rm) {
	nm_emit32(p, 0xF8207800u | ((uint32_t)rm << 16) | ((uint32_t)rn << 5) | (uint32_t)rt);
}

static inline void nm_emit_ldr_q_regoff(uint32_t **p, int qt, int rn, int rm) {
	nm_emit32(p, 0x3CE06800u | ((uint32_t)rm << 16) | ((uint32_t)rn << 5) | (uint32_t)qt);
}

static inline void nm_emit_str_q_regoff(uint32_t **p, int qt, int rn, int rm) {
	nm_emit32(p, 0x3CA06800u | ((uint32_t)rm << 16) | ((uint32_t)rn << 5) | (uint32_t)qt);
}

static inline void nm_emit_ldr_q_imm(uint32_t **p, int qt, int rn, int q_off) {
	nm_emit32(p, 0x3DC00000u | ((uint32_t)q_off << 10) | ((uint32_t)rn << 5) | (uint32_t)qt);
}

static inline void nm_emit_aese(uint32_t **p, int qd, int qm) {
	nm_emit32(p, 0x4E284800u | ((uint32_t)qm << 5) | (uint32_t)qd);
}

static inline void nm_emit_aesmc(uint32_t **p, int qd, int qn) {
	nm_emit32(p, 0x4E286800u | ((uint32_t)qn << 5) | (uint32_t)qd);
}

static inline void nm_emit_eor_v(uint32_t **p, int qd, int qn, int qm) {
	nm_emit32(p, 0x6E201C00u | ((uint32_t)qm << 16) | ((uint32_t)qn << 5) | (uint32_t)qd);
}

static inline void nm_emit_ldr_d(uint32_t **p, int dt, int rn, int double_off) {
	nm_emit32(p, 0xFD400000u | ((uint32_t)double_off << 10) | ((uint32_t)rn << 5) | (uint32_t)dt);
}

static inline void nm_emit_str_d(uint32_t **p, int dt, int rn, int double_off) {
	nm_emit32(p, 0xFD000000u | ((uint32_t)double_off << 10) | ((uint32_t)rn << 5) | (uint32_t)dt);
}

static inline void nm_emit_fmul_d(uint32_t **p, int dd, int dn, int dm) {
	nm_emit32(p, 0x1E600800u | ((uint32_t)dm << 16) | ((uint32_t)dn << 5) | (uint32_t)dd);
}

static inline void nm_emit_fadd_d(uint32_t **p, int dd, int dn, int dm) {
	nm_emit32(p, 0x1E602800u | ((uint32_t)dm << 16) | ((uint32_t)dn << 5) | (uint32_t)dd);
}

static inline void nm_emit_fdiv_d(uint32_t **p, int dd, int dn, int dm) {
	nm_emit32(p, 0x1E601800u | ((uint32_t)dm << 16) | ((uint32_t)dn << 5) | (uint32_t)dd);
}

static inline void nm_emit_fsqrt_d(uint32_t **p, int dd, int dn) {
	nm_emit32(p, 0x1E61C000u | ((uint32_t)dn << 5) | (uint32_t)dd);
}

static inline void nm_emit_fabs_d(uint32_t **p, int dd, int dn) {
	nm_emit32(p, 0x1E60C000u | ((uint32_t)dn << 5) | (uint32_t)dd);
}

static inline void nm_emit_fmov_d_from_x(uint32_t **p, int dd, int rn) {
	nm_emit32(p, 0x9E670000u | ((uint32_t)rn << 5) | (uint32_t)dd);
}

static inline void nm_emit_fmov_x_from_d(uint32_t **p, int rd, int dn) {
	nm_emit32(p, 0x9E660000u | ((uint32_t)dn << 5) | (uint32_t)rd);
}

static inline void nm_emit_and_mantissa_x11(uint32_t **p) {
	nm_emit32(p, 0x9240CD6Bu);
}

static inline void nm_emit_cmp_x9_zero(uint32_t **p) {
	nm_emit32(p, 0xF100013Fu);
}

static inline void nm_emit_cmp_x20_zero(uint32_t **p) {
	nm_emit32(p, 0xF100029Fu);
}

static inline void nm_emit_and_abs_x13_x11(uint32_t **p) {
	nm_emit32(p, 0x9240F96Du);
}

static inline void nm_emit_cmp_x13_zero(uint32_t **p) {
	nm_emit32(p, 0xF10001BFu);
}

static inline void nm_emit_cmp_x13_x12(uint32_t **p) {
	nm_emit32(p, 0xEB0C01BFu);
}

static inline void nm_emit_orr_x11_imm1(uint32_t **p) {
	nm_emit32(p, 0xB240016Bu);
}

static inline void nm_emit_b_eq_4(uint32_t **p) {
	nm_emit32(p, 0x54000080u);
}

static inline void nm_emit_b_lo_8(uint32_t **p) {
	nm_emit32(p, 0x54000103u);
}

static inline void nm_emit_b_ne_7(uint32_t **p) {
	nm_emit32(p, 0x540000E1u);
}

static inline void nm_emit_b_hs_4(uint32_t **p) {
	nm_emit32(p, 0x54000082u);
}

static inline void nm_emit_ldrb_w12_x7(uint32_t **p, int byte_off) {
	nm_emit32(p, 0x39400000u | ((uint32_t)byte_off << 10) | (7u << 5) | 12u);
}

static inline void nm_emit_strb_w12_x7(uint32_t **p, int byte_off) {
	nm_emit32(p, 0x39000000u | ((uint32_t)byte_off << 10) | (7u << 5) | 12u);
}

static inline void nm_emit_ldrb_w22_x7(uint32_t **p, int byte_off) {
	nm_emit32(p, 0x39400000u | ((uint32_t)byte_off << 10) | (7u << 5) | 22u);
}

static inline void nm_emit_strb_w22_x7(uint32_t **p, int byte_off) {
	nm_emit32(p, 0x39000000u | ((uint32_t)byte_off << 10) | (7u << 5) | 22u);
}

static inline void nm_emit_ldrb_w22_x27(uint32_t **p, int byte_off) {
	nm_emit32(p, 0x39400000u | ((uint32_t)byte_off << 10) | (27u << 5) | 22u);
}

static inline void nm_emit_strb_w22_x27(uint32_t **p, int byte_off) {
	nm_emit32(p, 0x39000000u | ((uint32_t)byte_off << 10) | (27u << 5) | 22u);
}

static inline void nm_emit_cmp_w12_8(uint32_t **p) {
	nm_emit32(p, 0x7100219Fu);
}

static inline void nm_emit_cmp_w22_8(uint32_t **p) {
	nm_emit32(p, 0x710022DFu);
}

static inline void nm_emit_add_w12_1(uint32_t **p) {
	nm_emit32(p, 0x1100058Cu);
}

static inline void nm_emit_add_w22_1(uint32_t **p) {
	nm_emit32(p, 0x110006D6u);
}

static inline void nm_emit_b_patch(uint32_t **p) {
	nm_emit32(p, 0x14000000u);
}

static inline void nm_patch_b(uint32_t *branch, uint32_t *target) {
	int64_t delta = (int64_t)(target - branch);
	*branch = 0x14000000u | ((uint32_t)delta & 0x03FFFFFFu);
}

static inline void nm_emit_bcond_patch(uint32_t **p, int cond) {
	nm_emit32(p, 0x54000000u | (uint32_t)(cond & 15));
}

static inline void nm_patch_bcond(uint32_t *branch, uint32_t *target, int cond) {
	int64_t delta = (int64_t)(target - branch);
	*branch = 0x54000000u | (((uint32_t)delta & 0x7FFFFu) << 5) | (uint32_t)(cond & 15);
}

static inline void nm_emit_and_abs_x21_x20(uint32_t **p) {
	nm_emit32(p, 0x9240FA95u);
}

static inline void nm_emit_cmp_x21_zero(uint32_t **p) {
	nm_emit32(p, 0xF10002BFu);
}

static inline void nm_emit_cmp_reg(uint32_t **p, int rn, int rm) {
	nm_emit32(p, 0xEB00001Fu | ((uint32_t)rm << 16) | ((uint32_t)rn << 5));
}

static inline void nm_emit_orr_imm1(uint32_t **p, int rd, int rn) {
	nm_emit32(p, 0xB2400000u | ((uint32_t)rn << 5) | (uint32_t)rd);
}

static inline void nm_emit_and_mantissa_x20(uint32_t **p) {
	nm_emit32(p, 0x9240CE94u);
}

static inline void nm_emit_float_repair_resident(uint32_t **p, int f_dst, int r_fallback) {
	nm_emit_fmov_x_from_d(p, 20, f_dst);
	nm_emit_and_abs_x21_x20(p);
	nm_emit_cmp_x21_zero(p);
	uint32_t *b_zero = *p;
	nm_emit_bcond_patch(p, 0); // eq -> repair
	nm_emit_mov64(p, 22, UINT64_C(0x7FF0000000000000));
	nm_emit_cmp_reg(p, 21, 22);
	uint32_t *b_good = *p;
	nm_emit_bcond_patch(p, 3); // lo -> done
	uint32_t *repair = *p;
	nm_emit_orr_imm1(p, 20, r_fallback);
	nm_emit_and_mantissa_x20(p);
	nm_emit_mov64(p, 22, UINT64_C(0x3FF0000000000000));
	nm_emit_orr_reg(p, 20, 20, 22);
	nm_emit_fmov_d_from_x(p, f_dst, 20);
	uint32_t *done = *p;
	nm_patch_bcond(b_zero, repair, 0);
	nm_patch_bcond(b_good, done, 3);
}

static inline void nm_emit_float_repair_d9(uint32_t **p, int r_fallback, int f_dst) {
	nm_emit_fmov_x_from_d(p, 11, 9);
	nm_emit_and_abs_x13_x11(p);
	nm_emit_cmp_x13_zero(p);
	nm_emit_b_eq_4(p);
	nm_emit_mov64(p, 12, UINT64_C(0x7FF0000000000000));
	nm_emit_cmp_x13_x12(p);
	nm_emit_b_lo_8(p);
	nm_emit_ldr(p, 11, 0, r_fallback);
	nm_emit_orr_x11_imm1(p);
	nm_emit_and_mantissa_x11(p);
	nm_emit_mov64(p, 12, UINT64_C(0x3FF0000000000000));
	nm_emit_orr_reg(p, 11, 11, 12);
	nm_emit_fmov_d_from_x(p, 9, 11);
	nm_emit_str_d(p, 9, 3, f_dst & 7);
}

static inline void nm_emit_mov64(uint32_t **p, int rd, uint64_t imm) {
	nm_emit32(p, 0xD2800000u | (((uint32_t)(imm & 0xFFFFu)) << 5) | (uint32_t)rd);
	if ((imm >> 16) & 0xFFFFu) {
		nm_emit32(p, 0xF2A00000u | ((uint32_t)((imm >> 16) & 0xFFFFu) << 5) | (uint32_t)rd);
	}
	if ((imm >> 32) & 0xFFFFu) {
		nm_emit32(p, 0xF2C00000u | ((uint32_t)((imm >> 32) & 0xFFFFu) << 5) | (uint32_t)rd);
	}
	if ((imm >> 48) & 0xFFFFu) {
		nm_emit32(p, 0xF2E00000u | ((uint32_t)((imm >> 48) & 0xFFFFu) << 5) | (uint32_t)rd);
	}
}

static nm_jit_fn nm_compile_add_chain(int ninst, void **mapping, size_t *mapping_len) {
	size_t bytes = (size_t)ninst * 7 * 4 + 4;
	long page = sysconf(_SC_PAGESIZE);
	size_t alloc = (bytes + (size_t)page - 1) & ~((size_t)page - 1);
	void *mem = mmap(NULL, alloc, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
	if (mem == MAP_FAILED) {
		return NULL;
	}
	uint32_t *p = (uint32_t *)mem;
	for (int i = 0; i < ninst; i++) {
		int d = (i * 7) & 15;
		int s = (i * 11 + 1) & 15;
		uint64_t imm = (uint64_t)(0x9E37u + (uint32_t)i * 0x101u);
		nm_emit_ldr(&p, 9, 0, d);
		nm_emit_ldr(&p, 10, 0, s);
		nm_emit_mov64(&p, 11, imm);
		nm_emit_add_reg(&p, 9, 9, 10);
		nm_emit_add_reg(&p, 9, 9, 11);
		nm_emit_str(&p, 9, 0, d);
	}
	nm_emit32(&p, 0xD65F03C0u);
	__builtin___clear_cache((char *)mem, (char *)p);
	if (mprotect(mem, alloc, PROT_READ | PROT_EXEC) != 0) {
		munmap(mem, alloc);
		return NULL;
	}
	*mapping = mem;
	*mapping_len = alloc;
	return (nm_jit_fn)mem;
}

static void nm_emit_add_chain_into(void *mem, int ninst, uint64_t salt) {
	uint32_t *p = (uint32_t *)mem;
	for (int i = 0; i < ninst; i++) {
		int d = (i * 7) & 15;
		int s = (i * 11 + 1) & 15;
		uint64_t imm = (uint64_t)(0x9E37u + (uint32_t)i * 0x101u) ^ salt;
		nm_emit_ldr(&p, 9, 0, d);
		nm_emit_ldr(&p, 10, 0, s);
		nm_emit_mov64(&p, 11, imm);
		nm_emit_add_reg(&p, 9, 9, 10);
		nm_emit_add_reg(&p, 9, 9, 11);
		nm_emit_str(&p, 9, 0, d);
	}
	nm_emit32(&p, 0xD65F03C0u);
	__builtin___clear_cache((char *)mem, (char *)p);
}

static void nm_interp_add_chain(uint64_t *r, int ninst) {
	for (int i = 0; i < ninst; i++) {
		int d = (i * 7) & 15;
		int s = (i * 11 + 1) & 15;
		uint64_t imm = (uint64_t)(0x9E37u + (uint32_t)i * 0x101u);
		r[d] += r[s] + imm;
	}
}

static int nm_jit_add_chain_matches(int ninst) {
	uint64_t a[16], b[16];
	for (int i = 0; i < 16; i++) {
		a[i] = b[i] = 0x1234567800000000ull + (uint64_t)i * 0x11111111ull;
	}
	void *mapping = NULL;
	size_t mapping_len = 0;
	nm_jit_fn fn = nm_compile_add_chain(ninst, &mapping, &mapping_len);
	if (fn == NULL) {
		return 0;
	}
	nm_interp_add_chain(a, ninst);
	fn(b);
	int ok = memcmp(a, b, sizeof(a)) == 0;
	munmap(mapping, mapping_len);
	return ok;
}

static uint64_t nm_run_interp_add_chain(int iters, int ninst) {
	uint64_t r[16];
	for (int i = 0; i < 16; i++) {
		r[i] = 0x1234567800000000ull + (uint64_t)i * 0x11111111ull;
	}
	for (int i = 0; i < iters; i++) {
		nm_interp_add_chain(r, ninst);
	}
	uint64_t out = 0;
	for (int i = 0; i < 16; i++) {
		out ^= r[i];
	}
	return out;
}

static uint64_t nm_run_jit_add_chain(int iters, int ninst) {
	uint64_t r[16];
	for (int i = 0; i < 16; i++) {
		r[i] = 0x1234567800000000ull + (uint64_t)i * 0x11111111ull;
	}
	void *mapping = NULL;
	size_t mapping_len = 0;
	nm_jit_fn fn = nm_compile_add_chain(ninst, &mapping, &mapping_len);
	if (fn == NULL) {
		return 0;
	}
	for (int i = 0; i < iters; i++) {
		fn(r);
	}
	uint64_t out = 0;
	for (int i = 0; i < 16; i++) {
		out ^= r[i];
	}
	munmap(mapping, mapping_len);
	return out;
}

static uint64_t nm_run_jit_add_chain_reuse_mprotect(int iters, int ninst) {
	uint64_t r[16];
	for (int i = 0; i < 16; i++) {
		r[i] = 0x1234567800000000ull + (uint64_t)i * 0x11111111ull;
	}
	size_t bytes = (size_t)ninst * 7 * 4 + 4;
	long page = sysconf(_SC_PAGESIZE);
	size_t alloc = (bytes + (size_t)page - 1) & ~((size_t)page - 1);
	void *mem = mmap(NULL, alloc, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
	if (mem == MAP_FAILED) {
		return 0;
	}
	for (int i = 0; i < iters; i++) {
		if (mprotect(mem, alloc, PROT_READ | PROT_WRITE) != 0) {
			munmap(mem, alloc);
			return 0;
		}
		nm_emit_add_chain_into(mem, ninst, (uint64_t)i);
		if (mprotect(mem, alloc, PROT_READ | PROT_EXEC) != 0) {
			munmap(mem, alloc);
			return 0;
		}
		((nm_jit_fn)mem)(r);
	}
	uint64_t out = 0;
	for (int i = 0; i < 16; i++) {
		out ^= r[i];
	}
	munmap(mem, alloc);
	return out;
}

static inline int nm_intmix_dst(int i) {
	return (i * 7 + 3) & 15;
}

static inline int nm_intmix_src(int i) {
	return (i * 11 + 1) & 15;
}

static inline uint64_t nm_intmix_imm(int i) {
	return (uint64_t)(0x9E3779B9u + (uint32_t)i * 0x45D9F3Bu);
}

static inline int nm_intmix_op(int i) {
	return i % 5;
}

static void nm_emit_intmix_into(void *mem, int ninst) {
	uint32_t *p = (uint32_t *)mem;
	for (int i = 0; i < ninst; i++) {
		int d = nm_intmix_dst(i);
		int s = nm_intmix_src(i);
		uint64_t imm = nm_intmix_imm(i);
		switch (nm_intmix_op(i)) {
		case 0:
			nm_emit_ldr(&p, 9, 0, d);
			nm_emit_ldr(&p, 10, 0, s);
			nm_emit_mov64(&p, 11, imm);
			nm_emit_add_reg(&p, 9, 9, 10);
			nm_emit_add_reg(&p, 9, 9, 11);
			nm_emit_str(&p, 9, 0, d);
			break;
		case 1:
			nm_emit_ldr(&p, 9, 0, d);
			nm_emit_ldr(&p, 10, 0, s);
			nm_emit_mov64(&p, 11, 1);
			nm_emit_orr_reg(&p, 10, 10, 11);
			nm_emit_mul_reg(&p, 9, 9, 10);
			nm_emit_str(&p, 9, 0, d);
			break;
		case 2:
			nm_emit_ldr(&p, 9, 0, d);
			nm_emit_ldr(&p, 10, 0, s);
			nm_emit_umulh_reg(&p, 9, 9, 10);
			nm_emit_mov64(&p, 11, imm);
			nm_emit_eor_reg(&p, 9, 9, 11);
			nm_emit_str(&p, 9, 0, d);
			break;
		case 3:
			nm_emit_ldr(&p, 9, 0, d);
			nm_emit_ldr(&p, 10, 0, s);
			nm_emit_add_reg(&p, 10, 10, 1);
			nm_emit_eor_reg(&p, 9, 9, 10);
			nm_emit_str(&p, 9, 0, d);
			break;
		default:
			nm_emit_ldr(&p, 9, 0, d);
			nm_emit_mov64(&p, 11, imm - 1);
			nm_emit_sub_reg(&p, 9, 11, 9);
			nm_emit_str(&p, 9, 0, d);
			break;
		}
	}
	nm_emit32(&p, 0xD65F03C0u);
	__builtin___clear_cache((char *)mem, (char *)p);
}

static nm_jit_int_fn nm_compile_intmix(int ninst, void **mapping, size_t *mapping_len) {
	size_t bytes = (size_t)ninst * 8 * 4 + 4;
	long page = sysconf(_SC_PAGESIZE);
	size_t alloc = (bytes + (size_t)page - 1) & ~((size_t)page - 1);
	void *mem = mmap(NULL, alloc, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
	if (mem == MAP_FAILED) {
		return NULL;
	}
	nm_emit_intmix_into(mem, ninst);
	if (mprotect(mem, alloc, PROT_READ | PROT_EXEC) != 0) {
		munmap(mem, alloc);
		return NULL;
	}
	*mapping = mem;
	*mapping_len = alloc;
	return (nm_jit_int_fn)mem;
}

static void nm_interp_intmix(uint64_t *r, int ninst, uint64_t rot_salt) {
	for (int i = 0; i < ninst; i++) {
		int d = nm_intmix_dst(i);
		int s = nm_intmix_src(i);
		uint64_t imm = nm_intmix_imm(i);
		switch (nm_intmix_op(i)) {
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
		default:
			r[d] = ~r[d] + imm;
			break;
		}
	}
}

static int nm_jit_intmix_matches(int ninst) {
	uint64_t a[16], b[16];
	for (int i = 0; i < 16; i++) {
		a[i] = b[i] = 0x1234567800000000ull + (uint64_t)i * 0x11111111ull;
	}
	void *mapping = NULL;
	size_t mapping_len = 0;
	nm_jit_int_fn fn = nm_compile_intmix(ninst, &mapping, &mapping_len);
	if (fn == NULL) {
		return 0;
	}
	nm_interp_intmix(a, ninst, 0xD00DFEED12345678ull);
	fn(b, 0xD00DFEED12345678ull);
	int ok = memcmp(a, b, sizeof(a)) == 0;
	munmap(mapping, mapping_len);
	return ok;
}

static inline int nm_resident_rreg(int idx) {
	return 1 + idx;
}

static void nm_emit_intmix_resident_into(void *mem, int ninst) {
	uint32_t *p = (uint32_t *)mem;
	for (int i = 0; i < 16; i++) {
		nm_emit_ldr(&p, nm_resident_rreg(i), 0, i);
	}
	for (int i = 0; i < ninst; i++) {
		int d = nm_resident_rreg(nm_intmix_dst(i));
		int s = nm_resident_rreg(nm_intmix_src(i));
		uint64_t imm = nm_intmix_imm(i);
		switch (nm_intmix_op(i)) {
		case 0:
			nm_emit_mov64(&p, 17, imm);
			nm_emit_add_reg(&p, d, d, s);
			nm_emit_add_reg(&p, d, d, 17);
			break;
		case 1:
			nm_emit_mov64(&p, 17, 1);
			nm_emit_orr_reg(&p, 17, s, 17);
			nm_emit_mul_reg(&p, d, d, 17);
			break;
		case 2:
			nm_emit_umulh_reg(&p, d, d, s);
			nm_emit_mov64(&p, 17, imm);
			nm_emit_eor_reg(&p, d, d, 17);
			break;
		case 3:
			nm_emit_mov64(&p, 17, 0xD00DFEED12345678ull);
			nm_emit_add_reg(&p, 17, s, 17);
			nm_emit_eor_reg(&p, d, d, 17);
			break;
		default:
			nm_emit_mov64(&p, 17, imm - 1);
			nm_emit_sub_reg(&p, d, 17, d);
			break;
		}
	}
	for (int i = 0; i < 16; i++) {
		nm_emit_str(&p, nm_resident_rreg(i), 0, i);
	}
	nm_emit32(&p, 0xD65F03C0u);
	__builtin___clear_cache((char *)mem, (char *)p);
}

static nm_jit_fn nm_compile_intmix_resident(int ninst, void **mapping, size_t *mapping_len) {
	size_t bytes = (size_t)ninst * 6 * 4 + 16 * 2 * 4 + 4;
	long page = sysconf(_SC_PAGESIZE);
	size_t alloc = (bytes + (size_t)page - 1) & ~((size_t)page - 1);
	void *mem = mmap(NULL, alloc, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
	if (mem == MAP_FAILED) {
		return NULL;
	}
	nm_emit_intmix_resident_into(mem, ninst);
	if (mprotect(mem, alloc, PROT_READ | PROT_EXEC) != 0) {
		munmap(mem, alloc);
		return NULL;
	}
	*mapping = mem;
	*mapping_len = alloc;
	return (nm_jit_fn)mem;
}

static int nm_jit_intmix_resident_matches(int ninst) {
	uint64_t a[16], b[16];
	for (int i = 0; i < 16; i++) {
		a[i] = b[i] = 0x1234567800000000ull + (uint64_t)i * 0x11111111ull;
	}
	void *mapping = NULL;
	size_t mapping_len = 0;
	nm_jit_fn fn = nm_compile_intmix_resident(ninst, &mapping, &mapping_len);
	if (fn == NULL) {
		return 0;
	}
	nm_interp_intmix(a, ninst, 0xD00DFEED12345678ull);
	fn(b);
	int ok = memcmp(a, b, sizeof(a)) == 0;
	munmap(mapping, mapping_len);
	return ok;
}

static uint64_t nm_run_jit_intmix_resident(int iters, int ninst) {
	uint64_t r[16];
	for (int i = 0; i < 16; i++) {
		r[i] = 0x1234567800000000ull + (uint64_t)i * 0x11111111ull;
	}
	void *mapping = NULL;
	size_t mapping_len = 0;
	nm_jit_fn fn = nm_compile_intmix_resident(ninst, &mapping, &mapping_len);
	if (fn == NULL) {
		return 0;
	}
	for (int i = 0; i < iters; i++) {
		fn(r);
	}
	uint64_t out = 0;
	for (int i = 0; i < 16; i++) {
		out ^= r[i];
	}
	munmap(mapping, mapping_len);
	return out;
}

static uint64_t nm_run_interp_intmix(int iters, int ninst) {
	uint64_t r[16];
	for (int i = 0; i < 16; i++) {
		r[i] = 0x1234567800000000ull + (uint64_t)i * 0x11111111ull;
	}
	for (int i = 0; i < iters; i++) {
		nm_interp_intmix(r, ninst, 0xD00DFEED12345678ull);
	}
	uint64_t out = 0;
	for (int i = 0; i < 16; i++) {
		out ^= r[i];
	}
	return out;
}

static uint64_t nm_run_jit_intmix(int iters, int ninst) {
	uint64_t r[16];
	for (int i = 0; i < 16; i++) {
		r[i] = 0x1234567800000000ull + (uint64_t)i * 0x11111111ull;
	}
	void *mapping = NULL;
	size_t mapping_len = 0;
	nm_jit_int_fn fn = nm_compile_intmix(ninst, &mapping, &mapping_len);
	if (fn == NULL) {
		return 0;
	}
	for (int i = 0; i < iters; i++) {
		fn(r, 0xD00DFEED12345678ull);
	}
	uint64_t out = 0;
	for (int i = 0; i < 16; i++) {
		out ^= r[i];
	}
	munmap(mapping, mapping_len);
	return out;
}

static uint64_t nm_run_jit_intmix_reuse_mprotect(int iters, int ninst) {
	uint64_t r[16];
	for (int i = 0; i < 16; i++) {
		r[i] = 0x1234567800000000ull + (uint64_t)i * 0x11111111ull;
	}
	size_t bytes = (size_t)ninst * 8 * 4 + 4;
	long page = sysconf(_SC_PAGESIZE);
	size_t alloc = (bytes + (size_t)page - 1) & ~((size_t)page - 1);
	void *mem = mmap(NULL, alloc, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
	if (mem == MAP_FAILED) {
		return 0;
	}
	for (int i = 0; i < iters; i++) {
		if (mprotect(mem, alloc, PROT_READ | PROT_WRITE) != 0) {
			munmap(mem, alloc);
			return 0;
		}
		nm_emit_intmix_into(mem, ninst);
		if (mprotect(mem, alloc, PROT_READ | PROT_EXEC) != 0) {
			munmap(mem, alloc);
			return 0;
		}
		((nm_jit_int_fn)mem)(r, 0xD00DFEED12345678ull);
	}
	uint64_t out = 0;
	for (int i = 0; i < 16; i++) {
		out ^= r[i];
	}
	munmap(mem, alloc);
	return out;
}

static inline int nm_memmix_dst(int i) {
	return (i * 5 + 2) & 15;
}

static inline int nm_memmix_src(int i) {
	return (i * 9 + 1) & 15;
}

static inline uint64_t nm_memmix_imm(int i) {
	return (uint64_t)(0xA5A5u + (uint32_t)i * 0x1F123BB5u);
}

static inline int nm_memmix_op(int i) {
	return i % 15;
}

static inline int nm_resmemmix_op(int i) {
	return i & 7;
}

static void nm_interp_resmemmix(uint64_t *r, uint64_t *scratch, uint64_t fold[8], int ninst, uint64_t rot_salt, uint64_t loop) {
	for (int i = 0; i < ninst; i++) {
		int d = nm_memmix_dst(i);
		int s = nm_memmix_src(i);
		uint64_t imm = nm_memmix_imm(i);
		switch (nm_resmemmix_op(i)) {
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
			r[d] = ~r[d] + imm;
			break;
		case 5:
			{
				uint64_t n = (r[s] ^ imm) & 63;
				r[d] = n == 0 ? r[d] : ((r[d] >> n) | (r[d] << (64 - n)));
			}
			break;
		case 6: {
			uint64_t addr = (r[s] + imm) & UINT64_C(0x1FFFF8);
			r[d] ^= scratch[addr >> 3];
			break;
		}
		default: {
			uint64_t addr = (r[d] + imm) & UINT64_C(0x1FFFF8);
			uint64_t idx = addr >> 3;
			uint64_t delta = r[s] + loop;
			scratch[idx] ^= delta;
			fold[idx & 7] ^= delta;
			break;
		}
		}
	}
}

static inline int nm_resmem_rreg(int idx) {
	if (idx < 14) {
		return 4 + idx;
	}
	return idx == 14 ? 19 : 23;
}

static void nm_emit_resmemmix_into(void *mem, int ninst) {
	uint32_t *p = (uint32_t *)mem;
	nm_emit32(&p, 0xA9BD53F3u); // stp x19, x20, [sp, #-48]!
	nm_emit32(&p, 0xA9015BF5u); // stp x21, x22, [sp, #16]
	nm_emit32(&p, 0xA90263F7u); // stp x23, x24, [sp, #32]
	for (int i = 0; i < 16; i++) {
		nm_emit_ldr(&p, nm_resmem_rreg(i), 0, i);
	}
	for (int i = 0; i < ninst; i++) {
		int d = nm_resmem_rreg(nm_memmix_dst(i));
		int s = nm_resmem_rreg(nm_memmix_src(i));
		uint64_t imm = nm_memmix_imm(i);
		switch (nm_resmemmix_op(i)) {
		case 0:
			nm_emit_mov64(&p, 20, imm);
			nm_emit_add_reg(&p, d, d, s);
			nm_emit_add_reg(&p, d, d, 20);
			break;
		case 1:
			nm_emit_mov64(&p, 20, 1);
			nm_emit_orr_reg(&p, 20, s, 20);
			nm_emit_mul_reg(&p, d, d, 20);
			break;
		case 2:
			nm_emit_umulh_reg(&p, d, d, s);
			nm_emit_mov64(&p, 20, imm);
			nm_emit_eor_reg(&p, d, d, 20);
			break;
		case 3:
			nm_emit_mov64(&p, 20, 0xD00DFEED12345678ull);
			nm_emit_add_reg(&p, 20, s, 20);
			nm_emit_eor_reg(&p, d, d, 20);
			break;
		case 4:
			nm_emit_mov64(&p, 20, imm - 1);
			nm_emit_sub_reg(&p, d, 20, d);
			break;
		case 5:
			nm_emit_mov64(&p, 20, imm);
			nm_emit_eor_reg(&p, 20, s, 20);
			nm_emit_and63_x20(&p);
			nm_emit_ror_reg(&p, d, d, 20);
			break;
		case 6:
			nm_emit_mov64(&p, 20, imm);
			nm_emit_add_reg(&p, 20, s, 20);
			nm_emit_and_scratch_mask_x20(&p);
			nm_emit_ldr_regoff(&p, 21, 1, 20);
			nm_emit_eor_reg(&p, d, d, 21);
			break;
		default:
			nm_emit_mov64(&p, 20, imm);
			nm_emit_add_reg(&p, 20, d, 20);
			nm_emit_and_scratch_mask_x20(&p);
			nm_emit_add_reg(&p, 21, s, 3);
			nm_emit_ldr_regoff(&p, 22, 1, 20);
			nm_emit_eor_reg(&p, 22, 22, 21);
			nm_emit_str_regoff(&p, 22, 1, 20);
			nm_emit_lsr_x20_x20_3(&p);
			nm_emit_and7_x20(&p);
			nm_emit_ldr_scaled_regoff(&p, 22, 2, 20);
			nm_emit_eor_reg(&p, 22, 22, 21);
			nm_emit_str_scaled_regoff(&p, 22, 2, 20);
			break;
		}
	}
	for (int i = 0; i < 16; i++) {
		nm_emit_str(&p, nm_resmem_rreg(i), 0, i);
	}
	nm_emit32(&p, 0xA94263F7u); // ldp x23, x24, [sp, #32]
	nm_emit32(&p, 0xA9415BF5u); // ldp x21, x22, [sp, #16]
	nm_emit32(&p, 0xA8C353F3u); // ldp x19, x20, [sp], #48
	nm_emit32(&p, 0xD65F03C0u);
	__builtin___clear_cache((char *)mem, (char *)p);
}

typedef void (*nm_jit_resmem_fn)(uint64_t *r, uint64_t *scratch, uint64_t *fold, uint64_t loop);

static nm_jit_resmem_fn nm_compile_resmemmix(int ninst, void **mapping, size_t *mapping_len) {
	size_t bytes = (size_t)ninst * 10 * 4 + 16 * 2 * 4 + 7 * 4;
	long page = sysconf(_SC_PAGESIZE);
	size_t alloc = (bytes + (size_t)page - 1) & ~((size_t)page - 1);
	void *mem = mmap(NULL, alloc, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
	if (mem == MAP_FAILED) {
		return NULL;
	}
	nm_emit_resmemmix_into(mem, ninst);
	if (mprotect(mem, alloc, PROT_READ | PROT_EXEC) != 0) {
		munmap(mem, alloc);
		return NULL;
	}
	*mapping = mem;
	*mapping_len = alloc;
	return (nm_jit_resmem_fn)mem;
}

static inline int nm_resfloatmix_op(int i) {
	return i % 10;
}

static void nm_interp_resfloatmix(uint64_t *r, double *f, int ninst, uint64_t rot_salt) {
	for (int i = 0; i < ninst; i++) {
		int d = nm_memmix_dst(i);
		int s = nm_memmix_src(i);
		uint64_t imm = nm_memmix_imm(i);
		switch (nm_resfloatmix_op(i)) {
		case 0:
			r[d] += r[s] + imm;
			break;
		case 1:
			r[d] ^= r[s] + rot_salt;
			break;
		case 2:
			r[d] ^= nm_bits_from_double(f[s & 7]);
			break;
		case 3:
			f[d & 7] = f[d & 7] * nm_norm_from_u64(r[s]);
			break;
		case 4:
			f[d & 7] = f[d & 7] + f[s & 7];
			if (nm_probe_bad_float_bits(nm_bits_from_double(f[d & 7]))) {
				f[d & 7] = nm_norm_from_u64(r[d] | 1);
			}
			break;
		case 5:
			f[d & 7] = f[d & 7] * f[s & 7];
			if (nm_probe_bad_float_bits(nm_bits_from_double(f[d & 7]))) {
				f[d & 7] = nm_norm_from_u64(r[d] | 1);
			}
			break;
		case 6:
			f[d & 7] = f[d & 7] / nm_norm_from_u64(nm_bits_from_double(f[s & 7]));
			if (nm_probe_bad_float_bits(nm_bits_from_double(f[d & 7]))) {
				f[d & 7] = nm_norm_from_u64(r[d] | 1);
			}
			break;
		case 7:
			f[d & 7] = sqrt(fabs(f[d & 7]));
			if (nm_probe_bad_float_bits(nm_bits_from_double(f[d & 7]))) {
				f[d & 7] = nm_norm_from_u64(r[d] | 1);
			}
			break;
		case 8: {
			uint64_t n = (r[s] ^ imm) & 63;
			r[d] = n == 0 ? r[d] : ((r[d] >> n) | (r[d] << (64 - n)));
			break;
		}
		default:
			r[d] = ~r[d] + imm;
			break;
		}
	}
}

typedef void (*nm_jit_resfloat_fn)(uint64_t *r, double *f);

static void nm_emit_resfloatmix_into(void *mem, int ninst) {
	uint32_t *p = (uint32_t *)mem;
	nm_emit32(&p, 0xA9BD53F3u); // stp x19, x20, [sp, #-48]!
	nm_emit32(&p, 0xA9015BF5u); // stp x21, x22, [sp, #16]
	nm_emit32(&p, 0xA90263F7u); // stp x23, x24, [sp, #32]
	for (int i = 0; i < 16; i++) {
		nm_emit_ldr(&p, nm_resmem_rreg(i), 0, i);
	}
	for (int i = 0; i < 8; i++) {
		nm_emit_ldr_d(&p, i, 1, i);
	}
	for (int i = 0; i < ninst; i++) {
		int d0 = nm_memmix_dst(i);
		int s0 = nm_memmix_src(i);
		int d = nm_resmem_rreg(d0);
		int s = nm_resmem_rreg(s0);
		int fd = d0 & 7;
		int fs = s0 & 7;
		uint64_t imm = nm_memmix_imm(i);
		switch (nm_resfloatmix_op(i)) {
		case 0:
			nm_emit_mov64(&p, 20, imm);
			nm_emit_add_reg(&p, d, d, s);
			nm_emit_add_reg(&p, d, d, 20);
			break;
		case 1:
			nm_emit_mov64(&p, 20, 0xD00DFEED12345678ull);
			nm_emit_add_reg(&p, 20, s, 20);
			nm_emit_eor_reg(&p, d, d, 20);
			break;
		case 2:
			nm_emit_fmov_x_from_d(&p, 20, fs);
			nm_emit_eor_reg(&p, d, d, 20);
			break;
		case 3:
			nm_emit_orr_reg(&p, 20, s, 31);
			nm_emit_and_mantissa_x20(&p);
			nm_emit_mov64(&p, 22, UINT64_C(0x3FF0000000000000));
			nm_emit_orr_reg(&p, 20, 20, 22);
			nm_emit_fmov_d_from_x(&p, 8, 20);
			nm_emit_fmul_d(&p, fd, fd, 8);
			break;
		case 4:
			nm_emit_fadd_d(&p, fd, fd, fs);
			nm_emit_float_repair_resident(&p, fd, d);
			break;
		case 5:
			nm_emit_fmul_d(&p, fd, fd, fs);
			nm_emit_float_repair_resident(&p, fd, d);
			break;
		case 6:
			nm_emit_fmov_x_from_d(&p, 20, fs);
			nm_emit_and_mantissa_x20(&p);
			nm_emit_mov64(&p, 22, UINT64_C(0x3FF0000000000000));
			nm_emit_orr_reg(&p, 20, 20, 22);
			nm_emit_fmov_d_from_x(&p, 8, 20);
			nm_emit_fdiv_d(&p, fd, fd, 8);
			nm_emit_float_repair_resident(&p, fd, d);
			break;
		case 7:
			nm_emit_fabs_d(&p, fd, fd);
			nm_emit_fsqrt_d(&p, fd, fd);
			nm_emit_float_repair_resident(&p, fd, d);
			break;
		case 8:
			nm_emit_mov64(&p, 20, imm);
			nm_emit_eor_reg(&p, 20, s, 20);
			nm_emit_and63_x20(&p);
			nm_emit_ror_reg(&p, d, d, 20);
			break;
		default:
			nm_emit_mov64(&p, 20, imm - 1);
			nm_emit_sub_reg(&p, d, 20, d);
			break;
		}
	}
	for (int i = 0; i < 16; i++) {
		nm_emit_str(&p, nm_resmem_rreg(i), 0, i);
	}
	for (int i = 0; i < 8; i++) {
		nm_emit_str_d(&p, i, 1, i);
	}
	nm_emit32(&p, 0xA94263F7u); // ldp x23, x24, [sp, #32]
	nm_emit32(&p, 0xA9415BF5u); // ldp x21, x22, [sp, #16]
	nm_emit32(&p, 0xA8C353F3u); // ldp x19, x20, [sp], #48
	nm_emit32(&p, 0xD65F03C0u);
	__builtin___clear_cache((char *)mem, (char *)p);
}

static nm_jit_resfloat_fn nm_compile_resfloatmix(int ninst, void **mapping, size_t *mapping_len) {
	size_t bytes = (size_t)ninst * 40 * 4 + (16 * 2 + 8 * 2 + 7) * 4;
	long page = sysconf(_SC_PAGESIZE);
	size_t alloc = (bytes + (size_t)page - 1) & ~((size_t)page - 1);
	void *mem = mmap(NULL, alloc, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
	if (mem == MAP_FAILED) {
		return NULL;
	}
	nm_emit_resfloatmix_into(mem, ninst);
	if (mprotect(mem, alloc, PROT_READ | PROT_EXEC) != 0) {
		munmap(mem, alloc);
		return NULL;
	}
	*mapping = mem;
	*mapping_len = alloc;
	return (nm_jit_resfloat_fn)mem;
}

static inline uint8x16_t nm_probe_aes_encrypt(uint8x16_t block, const uint8x16_t *rk) {
	for (int i = 0; i < 9; i++) {
		block = vaesmcq_u8(vaeseq_u8(block, rk[i]));
	}
	block = vaeseq_u8(block, rk[9]);
	return veorq_u8(block, rk[10]);
}

static void nm_init_probe_round_keys(uint8_t *rk_bytes) {
	for (int i = 0; i < 176; i++) {
		rk_bytes[i] = (uint8_t)(i * 37 + 19);
	}
}

static void nm_probe_encrypt_block(uint8_t *block_bytes, const uint8_t *rk_bytes) {
	uint8x16_t block = vld1q_u8(block_bytes);
	block = nm_probe_aes_encrypt(block, (const uint8x16_t *)rk_bytes);
	vst1q_u8(block_bytes, block);
}

static void nm_emit_memmix_into(void *mem, int ninst) {
	uint32_t *p = (uint32_t *)mem;
	uint32_t **offsets = (uint32_t **)malloc((size_t)ninst * sizeof(uint32_t *));
	uint32_t **branch_sites = (uint32_t **)malloc((size_t)ninst * sizeof(uint32_t *));
	int *branch_targets = (int *)malloc((size_t)ninst * sizeof(int));
	int branch_count = 0;
	if (offsets == NULL || branch_sites == NULL || branch_targets == NULL) {
		free(offsets);
		free(branch_sites);
		free(branch_targets);
		return;
	}
	for (int i = 0; i < ninst; i++) {
		offsets[i] = p;
		int d = nm_memmix_dst(i);
		int s = nm_memmix_src(i);
		uint64_t imm = nm_memmix_imm(i);
		switch (nm_memmix_op(i)) {
		case 0:
			nm_emit_ldr(&p, 9, 0, d);
			nm_emit_ldr(&p, 10, 0, s);
			nm_emit_mov64(&p, 11, imm);
			nm_emit_add_reg(&p, 9, 9, 10);
			nm_emit_add_reg(&p, 9, 9, 11);
			nm_emit_str(&p, 9, 0, d);
			break;
		case 1:
			nm_emit_ldr(&p, 9, 0, d);
			nm_emit_ldr(&p, 10, 0, s);
			nm_emit_mov64(&p, 11, 1);
			nm_emit_orr_reg(&p, 10, 10, 11);
			nm_emit_mul_reg(&p, 9, 9, 10);
			nm_emit_str(&p, 9, 0, d);
			break;
		case 2:
			nm_emit_ldr(&p, 9, 0, d);
			nm_emit_ldr(&p, 10, 0, s);
			nm_emit_umulh_reg(&p, 9, 9, 10);
			nm_emit_mov64(&p, 11, imm);
			nm_emit_eor_reg(&p, 9, 9, 11);
			nm_emit_str(&p, 9, 0, d);
			break;
		case 3:
			nm_emit_ldr(&p, 9, 0, d);
			nm_emit_ldr(&p, 10, 0, s);
			nm_emit_add_reg(&p, 10, 10, 4);
			nm_emit_eor_reg(&p, 9, 9, 10);
			nm_emit_str(&p, 9, 0, d);
			break;
		case 4:
			nm_emit_ldr(&p, 9, 0, d);
			nm_emit_mov64(&p, 11, imm - 1);
			nm_emit_sub_reg(&p, 9, 11, 9);
			nm_emit_str(&p, 9, 0, d);
			break;
		case 5:
			nm_emit_ldr(&p, 9, 0, d);
			nm_emit_ldr(&p, 10, 0, s);
			nm_emit_mov64(&p, 11, imm);
			nm_emit_eor_reg(&p, 10, 10, 11);
			nm_emit_and63_x10(&p);
			nm_emit_ror_reg(&p, 9, 9, 10);
			nm_emit_str(&p, 9, 0, d);
			break;
		case 6:
			nm_emit_ldr(&p, 9, 0, d);
			nm_emit_ldr(&p, 10, 0, s);
			nm_emit_mov64(&p, 11, imm);
			nm_emit_add_reg(&p, 10, 10, 11);
			nm_emit_and_scratch_mask_x10(&p);
			nm_emit_ldr_regoff(&p, 11, 1, 10);
			nm_emit_eor_reg(&p, 9, 9, 11);
			nm_emit_str(&p, 9, 0, d);
			break;
		case 7:
			nm_emit_ldr(&p, 10, 0, d);
			nm_emit_mov64(&p, 11, imm);
			nm_emit_add_reg(&p, 10, 10, 11);
			nm_emit_and_scratch_mask_x10(&p);
			nm_emit_ldr(&p, 12, 0, s);
			nm_emit_add_reg(&p, 12, 12, 5);
			nm_emit_ldr_regoff(&p, 13, 1, 10);
			nm_emit_eor_reg(&p, 13, 13, 12);
			nm_emit_str_regoff(&p, 13, 1, 10);
			nm_emit_lsr_x11_x10_3(&p);
			nm_emit_and7_x11(&p);
			nm_emit_ldr_scaled_regoff(&p, 13, 2, 11);
			nm_emit_eor_reg(&p, 13, 13, 12);
			nm_emit_str_scaled_regoff(&p, 13, 2, 11);
			break;
		case 8:
			if ((i & 1) == 0) {
				nm_emit_ldr_d(&p, 9, 3, s & 7);
				nm_emit_fmov_x_from_d(&p, 11, 9);
				nm_emit_ldr(&p, 10, 0, d);
				nm_emit_eor_reg(&p, 10, 10, 11);
				nm_emit_str(&p, 10, 0, d);
			} else {
				nm_emit_ldr(&p, 11, 0, s);
				nm_emit_and_mantissa_x11(&p);
				nm_emit_mov64(&p, 12, UINT64_C(0x3FF0000000000000));
				nm_emit_orr_reg(&p, 11, 11, 12);
				nm_emit_fmov_d_from_x(&p, 10, 11);
				nm_emit_ldr_d(&p, 9, 3, d & 7);
				nm_emit_fmul_d(&p, 9, 9, 10);
				nm_emit_str_d(&p, 9, 3, d & 7);
			}
			break;
		case 9:
			nm_emit_ldr_d(&p, 9, 3, d & 7);
			nm_emit_ldr_d(&p, 10, 3, s & 7);
			nm_emit_fadd_d(&p, 9, 9, 10);
			nm_emit_float_repair_d9(&p, d, d);
			break;
		case 10:
			nm_emit_ldr_d(&p, 9, 3, d & 7);
			nm_emit_ldr_d(&p, 10, 3, s & 7);
			nm_emit_fmul_d(&p, 9, 9, 10);
			nm_emit_float_repair_d9(&p, d, d);
			break;
		case 11:
			nm_emit_ldr_d(&p, 10, 3, s & 7);
			nm_emit_fmov_x_from_d(&p, 11, 10);
			nm_emit_and_mantissa_x11(&p);
			nm_emit_mov64(&p, 12, UINT64_C(0x3FF0000000000000));
			nm_emit_orr_reg(&p, 11, 11, 12);
			nm_emit_fmov_d_from_x(&p, 10, 11);
			nm_emit_ldr_d(&p, 9, 3, d & 7);
			nm_emit_fdiv_d(&p, 9, 9, 10);
			nm_emit_float_repair_d9(&p, d, d);
			break;
		case 12:
			nm_emit_ldr(&p, 10, 0, s);
			nm_emit_mov64(&p, 11, imm);
			nm_emit_add_reg(&p, 10, 10, 11);
			nm_emit_and_scratch_mask_x10(&p);
			nm_emit_and_scratch_block_mask_x11_x10(&p);
			nm_emit_ldr_q_regoff(&p, 0, 1, 11);
			for (int round = 0; round < 9; round++) {
				nm_emit_ldr_q_imm(&p, 1, 6, round);
				nm_emit_aese(&p, 0, 1);
				nm_emit_aesmc(&p, 0, 0);
			}
			nm_emit_ldr_q_imm(&p, 1, 6, 9);
			nm_emit_aese(&p, 0, 1);
			nm_emit_ldr_q_imm(&p, 1, 6, 10);
			nm_emit_eor_v(&p, 0, 0, 1);
			nm_emit_str_q_regoff(&p, 0, 1, 11);
			nm_emit_ldr_regoff(&p, 12, 1, 11);
			nm_emit_ldr(&p, 9, 0, d);
			nm_emit_eor_reg(&p, 9, 9, 12);
			nm_emit_str(&p, 9, 0, d);
			break;
		case 13: {
			int back = (int)(imm % 31) + 1;
			int target = i - back;
			if (target < 0) {
				target = 0;
			}
			nm_emit_ldr(&p, 9, 0, d);
			nm_emit_mov64(&p, 11, imm);
			nm_emit_add_reg(&p, 9, 9, 11);
			nm_emit_mov64(&p, 11, UINT64_C(0x7F8));
			nm_emit_and_reg(&p, 9, 9, 11);
			nm_emit_cmp_x9_zero(&p);
			nm_emit_b_ne_7(&p);
			nm_emit_ldrb_w12_x7(&p, i);
			nm_emit_cmp_w12_8(&p);
			nm_emit_b_hs_4(&p);
			nm_emit_add_w12_1(&p);
			nm_emit_strb_w12_x7(&p, i);
			branch_sites[branch_count] = p;
			branch_targets[branch_count] = target;
			branch_count++;
			nm_emit_b_patch(&p);
			break;
		}
		default:
			nm_emit_ldr_d(&p, 9, 3, d & 7);
			nm_emit_fabs_d(&p, 9, 9);
			nm_emit_fsqrt_d(&p, 9, 9);
			nm_emit_float_repair_d9(&p, d, d);
			break;
		}
	}
	nm_emit32(&p, 0xD65F03C0u);
	for (int i = 0; i < branch_count; i++) {
		nm_patch_b(branch_sites[i], offsets[branch_targets[i]]);
	}
	__builtin___clear_cache((char *)mem, (char *)p);
	free(offsets);
	free(branch_sites);
	free(branch_targets);
}

static void nm_emit_real_program_into(void *mem, const nm_probe_instr *prog, int ninst, uint64_t branch_mask) {
	uint32_t *p = (uint32_t *)mem;
	uint32_t **offsets = (uint32_t **)malloc((size_t)ninst * sizeof(uint32_t *));
	uint32_t **branch_sites = (uint32_t **)malloc((size_t)ninst * sizeof(uint32_t *));
	int *branch_targets = (int *)malloc((size_t)ninst * sizeof(int));
	int branch_count = 0;
	if (offsets == NULL || branch_sites == NULL || branch_targets == NULL) {
		free(offsets);
		free(branch_sites);
		free(branch_targets);
		return;
	}
	for (int i = 0; i < ninst; i++) {
		offsets[i] = p;
		int d = prog[i].dst & 15;
		int s = prog[i].src & 15;
		uint64_t imm = prog[i].imm;
		switch (prog[i].op) {
		case 0:
			nm_emit_ldr(&p, 9, 0, d);
			nm_emit_ldr(&p, 10, 0, s);
			nm_emit_mov64(&p, 11, imm);
			nm_emit_add_reg(&p, 9, 9, 10);
			nm_emit_add_reg(&p, 9, 9, 11);
			nm_emit_str(&p, 9, 0, d);
			break;
		case 1:
			nm_emit_ldr(&p, 9, 0, d);
			nm_emit_ldr(&p, 10, 0, s);
			nm_emit_mov64(&p, 11, 1);
			nm_emit_orr_reg(&p, 10, 10, 11);
			nm_emit_mul_reg(&p, 9, 9, 10);
			nm_emit_str(&p, 9, 0, d);
			break;
		case 2:
			nm_emit_ldr(&p, 9, 0, d);
			nm_emit_ldr(&p, 10, 0, s);
			nm_emit_umulh_reg(&p, 9, 9, 10);
			nm_emit_mov64(&p, 11, imm);
			nm_emit_eor_reg(&p, 9, 9, 11);
			nm_emit_str(&p, 9, 0, d);
			break;
		case 3:
			nm_emit_ldr(&p, 9, 0, d);
			nm_emit_ldr(&p, 10, 0, s);
			nm_emit_add_reg(&p, 10, 10, 4);
			nm_emit_eor_reg(&p, 9, 9, 10);
			nm_emit_str(&p, 9, 0, d);
			break;
		case 4:
			nm_emit_ldr(&p, 9, 0, d);
			nm_emit_ldr(&p, 10, 0, s);
			nm_emit_mov64(&p, 11, imm);
			nm_emit_eor_reg(&p, 10, 10, 11);
			nm_emit_and63_x10(&p);
			nm_emit_ror_reg(&p, 9, 9, 10);
			nm_emit_str(&p, 9, 0, d);
			break;
		case 5:
			nm_emit_ldr(&p, 9, 0, d);
			nm_emit_mov64(&p, 11, imm - 1);
			nm_emit_sub_reg(&p, 9, 11, 9);
			nm_emit_str(&p, 9, 0, d);
			break;
		case 6:
			nm_emit_ldr_d(&p, 9, 3, d & 7);
			nm_emit_ldr_d(&p, 10, 3, s & 7);
			nm_emit_fadd_d(&p, 9, 9, 10);
			nm_emit_float_repair_d9(&p, d, d);
			break;
		case 7:
			nm_emit_ldr_d(&p, 9, 3, d & 7);
			nm_emit_ldr_d(&p, 10, 3, s & 7);
			nm_emit_fmul_d(&p, 9, 9, 10);
			nm_emit_float_repair_d9(&p, d, d);
			break;
		case 8:
			nm_emit_ldr_d(&p, 10, 3, s & 7);
			nm_emit_fmov_x_from_d(&p, 11, 10);
			nm_emit_and_mantissa_x11(&p);
			nm_emit_mov64(&p, 12, UINT64_C(0x3FF0000000000000));
			nm_emit_orr_reg(&p, 11, 11, 12);
			nm_emit_fmov_d_from_x(&p, 10, 11);
			nm_emit_ldr_d(&p, 9, 3, d & 7);
			nm_emit_fdiv_d(&p, 9, 9, 10);
			nm_emit_float_repair_d9(&p, d, d);
			break;
		case 9:
			nm_emit_ldr_d(&p, 9, 3, d & 7);
			nm_emit_fabs_d(&p, 9, 9);
			nm_emit_fsqrt_d(&p, 9, 9);
			nm_emit_float_repair_d9(&p, d, d);
			break;
		case 10:
			nm_emit_ldr(&p, 9, 0, d);
			nm_emit_ldr(&p, 10, 0, s);
			nm_emit_mov64(&p, 11, imm);
			nm_emit_add_reg(&p, 10, 10, 11);
			nm_emit_and_scratch_mask_x10(&p);
			nm_emit_ldr_regoff(&p, 11, 1, 10);
			nm_emit_eor_reg(&p, 9, 9, 11);
			nm_emit_str(&p, 9, 0, d);
			break;
		case 11:
			nm_emit_ldr(&p, 10, 0, d);
			nm_emit_mov64(&p, 11, imm);
			nm_emit_add_reg(&p, 10, 10, 11);
			nm_emit_and_scratch_mask_x10(&p);
			nm_emit_ldr(&p, 12, 0, s);
			nm_emit_add_reg(&p, 12, 12, 5);
			nm_emit_ldr_regoff(&p, 13, 1, 10);
			nm_emit_eor_reg(&p, 13, 13, 12);
			nm_emit_str_regoff(&p, 13, 1, 10);
			nm_emit_lsr_x11_x10_3(&p);
			nm_emit_and7_x11(&p);
			nm_emit_ldr_scaled_regoff(&p, 13, 2, 11);
			nm_emit_eor_reg(&p, 13, 13, 12);
			nm_emit_str_scaled_regoff(&p, 13, 2, 11);
			break;
		case 12: {
			int back = (int)(imm % 31) + 1;
			int target = i - back;
			if (target < 0) {
				target = 0;
			}
			nm_emit_ldr(&p, 9, 0, d);
			nm_emit_mov64(&p, 11, imm);
			nm_emit_add_reg(&p, 9, 9, 11);
			nm_emit_mov64(&p, 11, branch_mask);
			nm_emit_and_reg(&p, 9, 9, 11);
			nm_emit_cmp_x9_zero(&p);
			nm_emit_b_ne_7(&p);
			nm_emit_ldrb_w12_x7(&p, i);
			nm_emit_cmp_w12_8(&p);
			nm_emit_b_hs_4(&p);
			nm_emit_add_w12_1(&p);
			nm_emit_strb_w12_x7(&p, i);
			branch_sites[branch_count] = p;
			branch_targets[branch_count] = target;
			branch_count++;
			nm_emit_b_patch(&p);
			break;
		}
		case 13:
			nm_emit_ldr(&p, 10, 0, s);
			nm_emit_mov64(&p, 11, imm);
			nm_emit_add_reg(&p, 10, 10, 11);
			nm_emit_and_scratch_mask_x10(&p);
			nm_emit_and_scratch_block_mask_x11_x10(&p);
			nm_emit_ldr_q_regoff(&p, 0, 1, 11);
			nm_emit_ldr_regoff(&p, 12, 1, 11);
			nm_emit_mov64(&p, 14, 8);
			nm_emit_add_reg(&p, 14, 11, 14);
			nm_emit_ldr_regoff(&p, 13, 1, 14);
			for (int round = 0; round < 9; round++) {
				nm_emit_ldr_q_imm(&p, 1, 6, round);
				nm_emit_aese(&p, 0, 1);
				nm_emit_aesmc(&p, 0, 0);
			}
			nm_emit_ldr_q_imm(&p, 1, 6, 9);
			nm_emit_aese(&p, 0, 1);
			nm_emit_ldr_q_imm(&p, 1, 6, 10);
			nm_emit_eor_v(&p, 0, 0, 1);
			nm_emit_str_q_regoff(&p, 0, 1, 11);
			nm_emit_ldr_regoff(&p, 14, 1, 11);
			nm_emit_mov64(&p, 15, 8);
			nm_emit_add_reg(&p, 15, 11, 15);
			nm_emit_ldr_regoff(&p, 10, 1, 15);
			nm_emit_eor_reg(&p, 12, 12, 14);
			nm_emit_eor_reg(&p, 13, 13, 10);
			nm_emit_orr_reg(&p, 10, 31, 11);
			nm_emit_lsr_x11_x10_3(&p);
			nm_emit_and7_x11(&p);
			nm_emit_ldr_scaled_regoff(&p, 10, 2, 11);
			nm_emit_eor_reg(&p, 10, 10, 12);
			nm_emit_str_scaled_regoff(&p, 10, 2, 11);
			nm_emit_mov64(&p, 12, 1);
			nm_emit_add_reg(&p, 11, 11, 12);
			nm_emit_and7_x11(&p);
			nm_emit_ldr_scaled_regoff(&p, 12, 2, 11);
			nm_emit_eor_reg(&p, 12, 12, 13);
			nm_emit_str_scaled_regoff(&p, 12, 2, 11);
			nm_emit_ldr(&p, 9, 0, d);
			nm_emit_eor_reg(&p, 9, 9, 14);
			nm_emit_str(&p, 9, 0, d);
			break;
		case 14:
			if ((imm & 1) == 0) {
				nm_emit_ldr_d(&p, 9, 3, s & 7);
				nm_emit_fmov_x_from_d(&p, 11, 9);
				nm_emit_ldr(&p, 10, 0, d);
				nm_emit_eor_reg(&p, 10, 10, 11);
				nm_emit_str(&p, 10, 0, d);
			} else {
				nm_emit_ldr(&p, 11, 0, s);
				nm_emit_and_mantissa_x11(&p);
				nm_emit_mov64(&p, 12, UINT64_C(0x3FF0000000000000));
				nm_emit_orr_reg(&p, 11, 11, 12);
				nm_emit_fmov_d_from_x(&p, 10, 11);
				nm_emit_ldr_d(&p, 9, 3, d & 7);
				nm_emit_fmul_d(&p, 9, 9, 10);
				nm_emit_str_d(&p, 9, 3, d & 7);
			}
			break;
		}
	}
	nm_emit32(&p, 0xD65F03C0u);
	for (int i = 0; i < branch_count; i++) {
		nm_patch_b(branch_sites[i], offsets[branch_targets[i]]);
	}
	__builtin___clear_cache((char *)mem, (char *)p);
	free(offsets);
	free(branch_sites);
	free(branch_targets);
}

static void nm_emit_real_program_resident_into(void *mem, const nm_probe_instr *prog, int ninst, uint64_t branch_mask) {
	uint32_t *p = (uint32_t *)mem;
	uint32_t **offsets = (uint32_t **)malloc((size_t)ninst * sizeof(uint32_t *));
	uint32_t **branch_sites = (uint32_t **)malloc((size_t)ninst * sizeof(uint32_t *));
	int *branch_targets = (int *)malloc((size_t)ninst * sizeof(int));
	int branch_count = 0;
	if (offsets == NULL || branch_sites == NULL || branch_targets == NULL) {
		free(offsets);
		free(branch_sites);
		free(branch_targets);
		return;
	}
	nm_emit32(&p, 0xA9BA53F3u); // stp x19, x20, [sp, #-96]!
	nm_emit32(&p, 0xA9015BF5u); // stp x21, x22, [sp, #16]
	nm_emit32(&p, 0xA90263F7u); // stp x23, x24, [sp, #32]
	nm_emit32(&p, 0xA9036BF9u); // stp x25, x26, [sp, #48]
	nm_emit32(&p, 0xA90473FBu); // stp x27, x28, [sp, #64]
	nm_emit_orr_reg(&p, 28, 3, 31);
	nm_emit_orr_reg(&p, 24, 4, 31);
	nm_emit_orr_reg(&p, 25, 5, 31);
	nm_emit_orr_reg(&p, 26, 6, 31);
	nm_emit_orr_reg(&p, 27, 7, 31);
	for (int i = 0; i < 16; i++) {
		nm_emit_ldr(&p, nm_resmem_rreg(i), 0, i);
	}
	for (int i = 0; i < 8; i++) {
		nm_emit_ldr_d(&p, i, 3, i);
	}
	for (int i = 0; i < ninst; i++) {
		offsets[i] = p;
		int d0 = prog[i].dst & 15;
		int s0 = prog[i].src & 15;
		int d = nm_resmem_rreg(d0);
		int s = nm_resmem_rreg(s0);
		int fd = d0 & 7;
		int fs = s0 & 7;
		uint64_t imm = prog[i].imm;
		switch (prog[i].op) {
		case 0:
			nm_emit_mov64(&p, 20, imm);
			nm_emit_add_reg(&p, d, d, s);
			nm_emit_add_reg(&p, d, d, 20);
			break;
		case 1:
			nm_emit_mov64(&p, 20, 1);
			nm_emit_orr_reg(&p, 20, s, 20);
			nm_emit_mul_reg(&p, d, d, 20);
			break;
		case 2:
			nm_emit_umulh_reg(&p, d, d, s);
			nm_emit_mov64(&p, 20, imm);
			nm_emit_eor_reg(&p, d, d, 20);
			break;
		case 3:
			nm_emit_add_reg(&p, 20, s, 24);
			nm_emit_eor_reg(&p, d, d, 20);
			break;
		case 4:
			nm_emit_mov64(&p, 20, imm);
			nm_emit_eor_reg(&p, 20, s, 20);
			nm_emit_and63_x20(&p);
			nm_emit_ror_reg(&p, d, d, 20);
			break;
		case 5:
			nm_emit_mov64(&p, 20, imm - 1);
			nm_emit_sub_reg(&p, d, 20, d);
			break;
		case 6:
			nm_emit_fadd_d(&p, fd, fd, fs);
			nm_emit_float_repair_resident(&p, fd, d);
			break;
		case 7:
			nm_emit_fmul_d(&p, fd, fd, fs);
			nm_emit_float_repair_resident(&p, fd, d);
			break;
		case 8:
			nm_emit_fmov_x_from_d(&p, 20, fs);
			nm_emit_and_mantissa_x20(&p);
			nm_emit_mov64(&p, 22, UINT64_C(0x3FF0000000000000));
			nm_emit_orr_reg(&p, 20, 20, 22);
			nm_emit_fmov_d_from_x(&p, 16, 20);
			nm_emit_fdiv_d(&p, fd, fd, 16);
			nm_emit_float_repair_resident(&p, fd, d);
			break;
		case 9:
			nm_emit_fabs_d(&p, fd, fd);
			nm_emit_fsqrt_d(&p, fd, fd);
			nm_emit_float_repair_resident(&p, fd, d);
			break;
		case 10:
			nm_emit_mov64(&p, 20, imm);
			nm_emit_add_reg(&p, 20, s, 20);
			nm_emit_and_scratch_mask_x20(&p);
			nm_emit_ldr_regoff(&p, 21, 1, 20);
			nm_emit_eor_reg(&p, d, d, 21);
			break;
		case 11:
			nm_emit_mov64(&p, 20, imm);
			nm_emit_add_reg(&p, 20, d, 20);
			nm_emit_and_scratch_mask_x20(&p);
			nm_emit_add_reg(&p, 21, s, 25);
			nm_emit_ldr_regoff(&p, 22, 1, 20);
			nm_emit_eor_reg(&p, 22, 22, 21);
			nm_emit_str_regoff(&p, 22, 1, 20);
			nm_emit_lsr_x20_x20_3(&p);
			nm_emit_and7_x20(&p);
			nm_emit_ldr_scaled_regoff(&p, 22, 2, 20);
			nm_emit_eor_reg(&p, 22, 22, 21);
			nm_emit_str_scaled_regoff(&p, 22, 2, 20);
			break;
		case 12: {
			int back = (int)(imm % 31) + 1;
			int target = i - back;
			if (target < 0) {
				target = 0;
			}
			nm_emit_mov64(&p, 20, imm);
			nm_emit_add_reg(&p, 20, d, 20);
			nm_emit_mov64(&p, 21, branch_mask);
			nm_emit_and_reg(&p, 20, 20, 21);
			nm_emit_cmp_x20_zero(&p);
			uint32_t *b_ne = p;
			nm_emit_bcond_patch(&p, 1);
			nm_emit_ldrb_w22_x27(&p, i);
			nm_emit_cmp_w22_8(&p);
			uint32_t *b_hs = p;
			nm_emit_bcond_patch(&p, 2);
			nm_emit_add_w22_1(&p);
			nm_emit_strb_w22_x27(&p, i);
			branch_sites[branch_count] = p;
			branch_targets[branch_count] = target;
			branch_count++;
			nm_emit_b_patch(&p);
			uint32_t *done = p;
			nm_patch_bcond(b_ne, done, 1);
			nm_patch_bcond(b_hs, done, 2);
			break;
		}
		case 13:
			nm_emit_mov64(&p, 20, imm);
			nm_emit_add_reg(&p, 20, s, 20);
			nm_emit_and_scratch_block_mask_x20(&p);
			nm_emit_ldr_q_regoff(&p, 16, 1, 20);
			nm_emit_ldr_regoff(&p, 21, 1, 20);
			nm_emit_mov64(&p, 22, 8);
			nm_emit_add_reg(&p, 22, 20, 22);
			nm_emit_ldr_regoff(&p, 22, 1, 22);
			for (int round = 0; round < 9; round++) {
				nm_emit_ldr_q_imm(&p, 17, 26, round);
				nm_emit_aese(&p, 16, 17);
				nm_emit_aesmc(&p, 16, 16);
			}
			nm_emit_ldr_q_imm(&p, 17, 26, 9);
			nm_emit_aese(&p, 16, 17);
			nm_emit_ldr_q_imm(&p, 17, 26, 10);
			nm_emit_eor_v(&p, 16, 16, 17);
			nm_emit_str_q_regoff(&p, 16, 1, 20);
			nm_emit_ldr_regoff(&p, 3, 1, 20);
			nm_emit_eor_reg(&p, 21, 21, 3);
			nm_emit_eor_reg(&p, d, d, 3);
			nm_emit_mov64(&p, 3, 8);
			nm_emit_add_reg(&p, 3, 20, 3);
			nm_emit_ldr_regoff(&p, 3, 1, 3);
			nm_emit_eor_reg(&p, 22, 22, 3);
			nm_emit_lsr_x20_x20_3(&p);
			nm_emit_and7_x20(&p);
			nm_emit_ldr_scaled_regoff(&p, 3, 2, 20);
			nm_emit_eor_reg(&p, 3, 3, 21);
			nm_emit_str_scaled_regoff(&p, 3, 2, 20);
			nm_emit_mov64(&p, 3, 1);
			nm_emit_add_reg(&p, 20, 20, 3);
			nm_emit_and7_x20(&p);
			nm_emit_ldr_scaled_regoff(&p, 3, 2, 20);
			nm_emit_eor_reg(&p, 3, 3, 22);
			nm_emit_str_scaled_regoff(&p, 3, 2, 20);
			break;
		case 14:
			if ((imm & 1) == 0) {
				nm_emit_fmov_x_from_d(&p, 20, fs);
				nm_emit_eor_reg(&p, d, d, 20);
			} else {
				nm_emit_orr_reg(&p, 20, s, 31);
				nm_emit_and_mantissa_x20(&p);
				nm_emit_mov64(&p, 22, UINT64_C(0x3FF0000000000000));
				nm_emit_orr_reg(&p, 20, 20, 22);
				nm_emit_fmov_d_from_x(&p, 16, 20);
				nm_emit_fmul_d(&p, fd, fd, 16);
			}
			break;
		}
	}
	for (int i = 0; i < branch_count; i++) {
		nm_patch_b(branch_sites[i], offsets[branch_targets[i]]);
	}
	for (int i = 0; i < 16; i++) {
		nm_emit_str(&p, nm_resmem_rreg(i), 0, i);
	}
	for (int i = 0; i < 8; i++) {
		nm_emit_str_d(&p, i, 28, i);
	}
	nm_emit32(&p, 0xA94473FBu); // ldp x27, x28, [sp, #64]
	nm_emit32(&p, 0xA9436BF9u); // ldp x25, x26, [sp, #48]
	nm_emit32(&p, 0xA94263F7u); // ldp x23, x24, [sp, #32]
	nm_emit32(&p, 0xA9415BF5u); // ldp x21, x22, [sp, #16]
	nm_emit32(&p, 0xA8C653F3u); // ldp x19, x20, [sp], #96
	nm_emit32(&p, 0xD65F03C0u);
	__builtin___clear_cache((char *)mem, (char *)p);
	free(offsets);
	free(branch_sites);
	free(branch_targets);
}

static nm_jit_mem_fn nm_compile_memmix(int ninst, void **mapping, size_t *mapping_len) {
	size_t bytes = (size_t)ninst * 80 * 4 + 4;
	long page = sysconf(_SC_PAGESIZE);
	size_t alloc = (bytes + (size_t)page - 1) & ~((size_t)page - 1);
	void *mem = mmap(NULL, alloc, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
	if (mem == MAP_FAILED) {
		return NULL;
	}
	nm_emit_memmix_into(mem, ninst);
	if (mprotect(mem, alloc, PROT_READ | PROT_EXEC) != 0) {
		munmap(mem, alloc);
		return NULL;
	}
	*mapping = mem;
	*mapping_len = alloc;
	return (nm_jit_mem_fn)mem;
}

static int nm_jit_real_program_once(
	const nm_probe_instr *prog,
	int ninst,
	uint64_t branch_mask,
	uint64_t *r,
	uint64_t *scratch,
	uint64_t *fold,
	double *f,
	uint64_t rot_salt,
	uint64_t loop,
	const uint8_t *rk,
	uint8_t *taken
) {
	size_t bytes = (size_t)ninst * 80 * 4 + 4;
	long page = sysconf(_SC_PAGESIZE);
	size_t alloc = (bytes + (size_t)page - 1) & ~((size_t)page - 1);
	void *mem = mmap(NULL, alloc, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
	if (mem == MAP_FAILED) {
		return 0;
	}
	nm_emit_real_program_into(mem, prog, ninst, branch_mask);
	if (mprotect(mem, alloc, PROT_READ | PROT_EXEC) != 0) {
		munmap(mem, alloc);
		return 0;
	}
	((nm_jit_mem_fn)mem)(r, scratch, fold, f, rot_salt, loop, rk, taken);
	munmap(mem, alloc);
	return 1;
}

static int nm_jit_real_program_resident_once(
	const nm_probe_instr *prog,
	int ninst,
	uint64_t branch_mask,
	uint64_t *r,
	uint64_t *scratch,
	uint64_t *fold,
	double *f,
	uint64_t rot_salt,
	uint64_t loop,
	const uint8_t *rk,
	uint8_t *taken
) {
	(void)fold;
	size_t bytes = (size_t)ninst * 120 * 4 + 512;
	long page = sysconf(_SC_PAGESIZE);
	size_t alloc = (bytes + (size_t)page - 1) & ~((size_t)page - 1);
	void *mem = mmap(NULL, alloc, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
	if (mem == MAP_FAILED) {
		return 0;
	}
	nm_emit_real_program_resident_into(mem, prog, ninst, branch_mask);
	if (mprotect(mem, alloc, PROT_READ | PROT_EXEC) != 0) {
		munmap(mem, alloc);
		return 0;
	}
	((nm_jit_mem_fn)mem)(r, scratch, fold, f, rot_salt, loop, rk, taken);
	munmap(mem, alloc);
	return 1;
}

static int nm_jit_real_program_loop(
	const nm_probe_instr *prog,
	int ninst,
	uint64_t branch_mask,
	uint64_t *r,
	uint64_t *scratch,
	uint64_t *fold,
	double *f,
	uint64_t rot_salt,
	uint64_t first_loop,
	const uint8_t *rk,
	uint8_t *taken,
	int iters
) {
	size_t bytes = (size_t)ninst * 80 * 4 + 4;
	long page = sysconf(_SC_PAGESIZE);
	size_t alloc = (bytes + (size_t)page - 1) & ~((size_t)page - 1);
	void *mem = mmap(NULL, alloc, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
	if (mem == MAP_FAILED) {
		return 0;
	}
	nm_emit_real_program_into(mem, prog, ninst, branch_mask);
	if (mprotect(mem, alloc, PROT_READ | PROT_EXEC) != 0) {
		munmap(mem, alloc);
		return 0;
	}
	for (int i = 0; i < iters; i++) {
		memset(taken, 0, (size_t)ninst);
		((nm_jit_mem_fn)mem)(r, scratch, fold, f, rot_salt, first_loop + (uint64_t)i, rk, taken);
	}
	munmap(mem, alloc);
	return 1;
}

static int nm_jit_real_program_resident_loop(
	const nm_probe_instr *prog,
	int ninst,
	uint64_t branch_mask,
	uint64_t *r,
	uint64_t *scratch,
	uint64_t *fold,
	double *f,
	uint64_t rot_salt,
	uint64_t first_loop,
	const uint8_t *rk,
	uint8_t *taken,
	int iters
) {
	size_t bytes = (size_t)ninst * 120 * 4 + 512;
	long page = sysconf(_SC_PAGESIZE);
	size_t alloc = (bytes + (size_t)page - 1) & ~((size_t)page - 1);
	void *mem = mmap(NULL, alloc, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
	if (mem == MAP_FAILED) {
		return 0;
	}
	nm_emit_real_program_resident_into(mem, prog, ninst, branch_mask);
	if (mprotect(mem, alloc, PROT_READ | PROT_EXEC) != 0) {
		munmap(mem, alloc);
		return 0;
	}
	for (int i = 0; i < iters; i++) {
		memset(taken, 0, (size_t)ninst);
		((nm_jit_mem_fn)mem)(r, scratch, fold, f, rot_salt, first_loop + (uint64_t)i, rk, taken);
	}
	munmap(mem, alloc);
	return 1;
}

static int nm_jit_real_execute_probe(
	const nm_probe_instr *prog,
	int ninst,
	uint32_t loops,
	uint64_t branch_mask,
	uint64_t rot_salt,
	uint64_t *r,
	uint64_t *scratch,
	uint64_t *fold,
	double *f,
	const uint64_t *dataset,
	uint8_t use_dataset,
	const uint8_t *rk,
	uint8_t *taken
) {
	size_t bytes = (size_t)ninst * 80 * 4 + 4;
	long page = sysconf(_SC_PAGESIZE);
	size_t alloc = (bytes + (size_t)page - 1) & ~((size_t)page - 1);
	void *mem = mmap(NULL, alloc, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
	if (mem == MAP_FAILED) {
		return 0;
	}
	nm_emit_real_program_into(mem, prog, ninst, branch_mask);
	if (mprotect(mem, alloc, PROT_READ | PROT_EXEC) != 0) {
		munmap(mem, alloc);
		return 0;
	}
	nm_jit_mem_fn fn = (nm_jit_mem_fn)mem;
	for (uint32_t loop = 0; loop < loops; loop++) {
		memset(taken, 0, (size_t)ninst);
		fn(r, scratch, fold, f, rot_salt, (uint64_t)loop, rk, taken);
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
			uint64_t idx = (base + (uint64_t)i) & UINT64_C(0x3FFFF);
			scratch[idx] ^= r[i];
			fold[idx & 7] ^= r[i];
		}
		for (int i = 0; i < 8; i++) {
			r[i + 8] ^= nm_bits_from_double(f[i]);
		}
	}
	munmap(mem, alloc);
	return 1;
}

static int nm_jit_real_execute_probe_resident(
	const nm_probe_instr *prog,
	int ninst,
	uint32_t loops,
	uint64_t branch_mask,
	uint64_t rot_salt,
	uint64_t *r,
	uint64_t *scratch,
	uint64_t *fold,
	double *f,
	const uint64_t *dataset,
	uint8_t use_dataset,
	const uint8_t *rk,
	uint8_t *taken
) {
	size_t bytes = (size_t)ninst * 120 * 4 + 512;
	long page = sysconf(_SC_PAGESIZE);
	size_t alloc = (bytes + (size_t)page - 1) & ~((size_t)page - 1);
	void *mem = mmap(NULL, alloc, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
	if (mem == MAP_FAILED) {
		return 0;
	}
	nm_emit_real_program_resident_into(mem, prog, ninst, branch_mask);
	if (mprotect(mem, alloc, PROT_READ | PROT_EXEC) != 0) {
		munmap(mem, alloc);
		return 0;
	}
	nm_jit_mem_fn fn = (nm_jit_mem_fn)mem;
	for (uint32_t loop = 0; loop < loops; loop++) {
		memset(taken, 0, (size_t)ninst);
		fn(r, scratch, fold, f, rot_salt, (uint64_t)loop, rk, taken);
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
			uint64_t idx = (base + (uint64_t)i) & UINT64_C(0x3FFFF);
			scratch[idx] ^= r[i];
			fold[idx & 7] ^= r[i];
		}
		for (int i = 0; i < 8; i++) {
			r[i + 8] ^= nm_bits_from_double(f[i]);
		}
	}
	munmap(mem, alloc);
	return 1;
}

static nm_jit_buffer *nm_jit_buffer_new_bytes(size_t bytes) {
	nm_jit_buffer *buf = (nm_jit_buffer *)calloc(1, sizeof(nm_jit_buffer));
	if (buf == NULL) {
		return NULL;
	}
	long page = sysconf(_SC_PAGESIZE);
	buf->alloc = (bytes + (size_t)page - 1) & ~((size_t)page - 1);
	buf->mem = mmap(NULL, buf->alloc, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANON | MAP_JIT, -1, 0);
	if (buf->mem != MAP_FAILED) {
		buf->map_jit = 1;
		nm_jit_write_protect(1);
		return buf;
	}
	buf->mem = mmap(NULL, buf->alloc, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
	if (buf->mem == MAP_FAILED) {
		free(buf);
		return NULL;
	}
	return buf;
}

static nm_jit_buffer *nm_jit_buffer_new(int ninst) {
	return nm_jit_buffer_new_bytes((size_t)ninst * 80 * 4 + 4);
}

static nm_jit_buffer *nm_jit_resident_buffer_new(int ninst) {
	return nm_jit_buffer_new_bytes((size_t)ninst * 120 * 4 + 512);
}

static void nm_jit_buffer_free(nm_jit_buffer *buf) {
	if (buf == NULL) {
		return;
	}
	if (buf->mem != NULL) {
		munmap(buf->mem, buf->alloc);
	}
	free(buf);
}

static int nm_jit_real_execute_probe_reuse(
	nm_jit_buffer *buf,
	const nm_probe_instr *prog,
	int ninst,
	uint32_t loops,
	uint64_t branch_mask,
	uint64_t rot_salt,
	uint64_t *r,
	uint64_t *scratch,
	uint64_t *fold,
	double *f,
	const uint64_t *dataset,
	uint8_t use_dataset,
	const uint8_t *rk,
	uint8_t *taken
) {
	if (buf == NULL || buf->mem == NULL) {
		return 0;
	}
	if (buf->map_jit) {
		nm_jit_write_protect(0);
		nm_emit_real_program_into(buf->mem, prog, ninst, branch_mask);
		nm_jit_write_protect(1);
	} else {
		if (mprotect(buf->mem, buf->alloc, PROT_READ | PROT_WRITE) != 0) {
			return 0;
		}
		nm_emit_real_program_into(buf->mem, prog, ninst, branch_mask);
		if (mprotect(buf->mem, buf->alloc, PROT_READ | PROT_EXEC) != 0) {
			return 0;
		}
	}
	nm_jit_mem_fn fn = (nm_jit_mem_fn)buf->mem;
	int branch_indices[768];
	int branch_count = 0;
	for (int i = 0; i < ninst && i < 768; i++) {
		if (prog[i].op == 12) {
			branch_indices[branch_count++] = i;
		}
	}
	for (uint32_t loop = 0; loop < loops; loop++) {
		for (int i = 0; i < branch_count; i++) {
			taken[branch_indices[i]] = 0;
		}
		fn(r, scratch, fold, f, rot_salt, (uint64_t)loop, rk, taken);
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
			uint64_t idx = (base + (uint64_t)i) & UINT64_C(0x3FFFF);
			scratch[idx] ^= r[i];
			fold[idx & 7] ^= r[i];
		}
		for (int i = 0; i < 8; i++) {
			r[i + 8] ^= nm_bits_from_double(f[i]);
		}
	}
	return 1;
}

static int nm_jit_real_execute_probe_resident_reuse(
	nm_jit_buffer *buf,
	const nm_probe_instr *prog,
	int ninst,
	uint32_t loops,
	uint64_t branch_mask,
	uint64_t rot_salt,
	uint64_t *r,
	uint64_t *scratch,
	uint64_t *fold,
	double *f,
	const uint64_t *dataset,
	uint8_t use_dataset,
	const uint8_t *rk,
	uint8_t *taken
) {
	if (buf == NULL || buf->mem == NULL) {
		return 0;
	}
	if (buf->map_jit) {
		nm_jit_write_protect(0);
		nm_emit_real_program_resident_into(buf->mem, prog, ninst, branch_mask);
		nm_jit_write_protect(1);
	} else {
		if (mprotect(buf->mem, buf->alloc, PROT_READ | PROT_WRITE) != 0) {
			return 0;
		}
		nm_emit_real_program_resident_into(buf->mem, prog, ninst, branch_mask);
		if (mprotect(buf->mem, buf->alloc, PROT_READ | PROT_EXEC) != 0) {
			return 0;
		}
	}
	nm_jit_mem_fn fn = (nm_jit_mem_fn)buf->mem;
	for (uint32_t loop = 0; loop < loops; loop++) {
		memset(taken, 0, (size_t)ninst);
		fn(r, scratch, fold, f, rot_salt, (uint64_t)loop, rk, taken);
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
			uint64_t idx = (base + (uint64_t)i) & UINT64_C(0x3FFFF);
			scratch[idx] ^= r[i];
			fold[idx & 7] ^= r[i];
		}
		for (int i = 0; i < 8; i++) {
			r[i + 8] ^= nm_bits_from_double(f[i]);
		}
	}
	return 1;
}

static double nm_norm_from_u64(uint64_t x) {
	uint64_t bits = (UINT64_C(1023) << 52) | (x & UINT64_C(0x000FFFFFFFFFFFFF));
	double out;
	memcpy(&out, &bits, sizeof(out));
	return out;
}

static uint64_t nm_bits_from_double(double x) {
	uint64_t out;
	memcpy(&out, &x, sizeof(out));
	return out;
}

static void nm_interp_memmix(uint64_t *r, uint64_t *scratch, uint64_t *fold, double *f, const uint8x16_t *rk, uint8_t *taken, int ninst, uint64_t rot_salt, uint64_t loop) {
	int pc = 0;
	while (pc < ninst) {
		int i = pc;
		int d = nm_memmix_dst(i);
		int s = nm_memmix_src(i);
		uint64_t imm = nm_memmix_imm(i);
		switch (nm_memmix_op(i)) {
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
			r[d] = ~r[d] + imm;
			break;
		case 5: {
			uint64_t n = (r[s] ^ imm) & 63;
			if (n != 0) {
				r[d] = (r[d] >> n) | (r[d] << (64 - n));
			}
			break;
		}
		case 6: {
			uint64_t addr = (r[s] + imm) & UINT64_C(0x1FFFF8);
			r[d] ^= scratch[addr >> 3];
			break;
		}
		case 7: {
			uint64_t addr = (r[d] + imm) & UINT64_C(0x1FFFF8);
			uint64_t idx = addr >> 3;
			uint64_t delta = r[s] + loop;
			scratch[idx] ^= delta;
			fold[idx & 7] ^= delta;
			break;
		}
		case 8:
			if ((i & 1) == 0) {
				r[d] ^= nm_bits_from_double(f[s & 7]);
			} else {
				f[d & 7] = f[d & 7] * nm_norm_from_u64(r[s]);
			}
			break;
		case 9:
			f[d & 7] = f[d & 7] + f[s & 7];
			if (nm_probe_bad_float_bits(nm_bits_from_double(f[d & 7]))) {
				f[d & 7] = nm_norm_from_u64(r[d] | 1);
			}
			break;
		case 10:
			f[d & 7] = f[d & 7] * f[s & 7];
			if (nm_probe_bad_float_bits(nm_bits_from_double(f[d & 7]))) {
				f[d & 7] = nm_norm_from_u64(r[d] | 1);
			}
			break;
		case 11:
			f[d & 7] = f[d & 7] / nm_norm_from_u64(nm_bits_from_double(f[s & 7]));
			if (nm_probe_bad_float_bits(nm_bits_from_double(f[d & 7]))) {
				f[d & 7] = nm_norm_from_u64(r[d] | 1);
			}
			break;
		case 12: {
			uint64_t addr = ((r[s] + imm) & UINT64_C(0x1FFFF8)) & ~UINT64_C(15);
			uint64_t w = addr >> 3;
			uint8x16_t block = vld1q_u8((const uint8_t *)(scratch + w));
			block = nm_probe_aes_encrypt(block, rk);
			vst1q_u8((uint8_t *)(scratch + w), block);
			r[d] ^= scratch[w];
			break;
		}
		case 13:
			if (((r[d] + imm) & UINT64_C(0x7F8)) == 0 && taken[i] < 8) {
				taken[i]++;
				int back = (int)(imm % 31) + 1;
				pc -= back;
				if (pc < 0) {
					pc = 0;
				}
				continue;
			}
			break;
		default:
			f[d & 7] = sqrt(fabs(f[d & 7]));
			if (nm_probe_bad_float_bits(nm_bits_from_double(f[d & 7]))) {
				f[d & 7] = nm_norm_from_u64(r[d] | 1);
			}
			break;
		}
		pc++;
	}
}

static void nm_init_memmix_state(uint64_t *r, uint64_t *scratch, uint64_t *fold, double *f) {
	for (int i = 0; i < 16; i++) {
		r[i] = 0x1234567800000000ull + (uint64_t)i * 0x11111111ull;
	}
	for (int i = 0; i < 8; i++) {
		fold[i] = 0xD00DFEED00000000ull + (uint64_t)i * 0x01010101ull;
		f[i] = nm_norm_from_u64(0xABCDEF1234567890ull + (uint64_t)i * 0x22222222ull);
	}
	for (uint64_t i = 0; i < UINT64_C(262144); i++) {
		scratch[i] = (i * UINT64_C(0x9E3779B97F4A7C15)) ^ UINT64_C(0xA5A5A5A5A5A5A5A5);
	}
}

static void nm_init_resfloatmix_state(uint64_t *r, double *f) {
	for (int i = 0; i < 16; i++) {
		r[i] = 0x1234567800000000ull + (uint64_t)i * 0x11111111ull;
	}
	for (int i = 0; i < 8; i++) {
		f[i] = nm_norm_from_u64(0xABCDEF1234567890ull + (uint64_t)i * 0x22222222ull);
	}
}

static int nm_jit_resfloatmix_matches(int ninst) {
	uint64_t a[16], b[16];
	double f_a[8], f_b[8];
	nm_init_resfloatmix_state(a, f_a);
	nm_init_resfloatmix_state(b, f_b);
	void *mapping = NULL;
	size_t mapping_len = 0;
	nm_jit_resfloat_fn fn = nm_compile_resfloatmix(ninst, &mapping, &mapping_len);
	if (fn == NULL) {
		return 0;
	}
	nm_interp_resfloatmix(a, f_a, ninst, 0xD00DFEED12345678ull);
	fn(b, f_b);
	int ok = memcmp(a, b, sizeof(a)) == 0 && memcmp(f_a, f_b, sizeof(f_a)) == 0;
	munmap(mapping, mapping_len);
	return ok;
}

static int nm_jit_resfloatmix_debug(int ninst, uint64_t out[3]) {
	uint64_t a[16], b[16];
	double f_a[8], f_b[8];
	out[0] = out[1] = out[2] = 0;
	nm_init_resfloatmix_state(a, f_a);
	nm_init_resfloatmix_state(b, f_b);
	void *mapping = NULL;
	size_t mapping_len = 0;
	nm_jit_resfloat_fn fn = nm_compile_resfloatmix(ninst, &mapping, &mapping_len);
	if (fn == NULL) {
		out[0] = UINT64_C(0xFFFFFFFFFFFFFFFF);
		return 0;
	}
	nm_interp_resfloatmix(a, f_a, ninst, 0xD00DFEED12345678ull);
	fn(b, f_b);
	for (int i = 0; i < 16; i++) {
		if (a[i] != b[i]) {
			out[0] = (uint64_t)i;
			out[1] = a[i];
			out[2] = b[i];
			munmap(mapping, mapping_len);
			return 0;
		}
	}
	for (int i = 0; i < 8; i++) {
		uint64_t aa = nm_bits_from_double(f_a[i]);
		uint64_t bb = nm_bits_from_double(f_b[i]);
		if (aa != bb) {
			out[0] = (uint64_t)(100 + i);
			out[1] = aa;
			out[2] = bb;
			munmap(mapping, mapping_len);
			return 0;
		}
	}
	munmap(mapping, mapping_len);
	return 1;
}

static uint64_t nm_finish_resfloatmix_state(uint64_t *r, double *f) {
	uint64_t out = 0;
	for (int i = 0; i < 16; i++) {
		out ^= r[i];
	}
	for (int i = 0; i < 8; i++) {
		out ^= nm_bits_from_double(f[i]);
	}
	return out;
}

static uint64_t nm_run_interp_resfloatmix(int iters, int ninst) {
	uint64_t r[16];
	double f[8];
	nm_init_resfloatmix_state(r, f);
	for (int i = 0; i < iters; i++) {
		nm_interp_resfloatmix(r, f, ninst, 0xD00DFEED12345678ull);
	}
	return nm_finish_resfloatmix_state(r, f);
}

static uint64_t nm_run_jit_resfloatmix(int iters, int ninst) {
	uint64_t r[16];
	double f[8];
	nm_init_resfloatmix_state(r, f);
	void *mapping = NULL;
	size_t mapping_len = 0;
	nm_jit_resfloat_fn fn = nm_compile_resfloatmix(ninst, &mapping, &mapping_len);
	if (fn == NULL) {
		return 0;
	}
	for (int i = 0; i < iters; i++) {
		fn(r, f);
	}
	uint64_t out = nm_finish_resfloatmix_state(r, f);
	munmap(mapping, mapping_len);
	return out;
}

static uint64_t nm_finish_memmix_state(uint64_t *r, uint64_t *scratch, uint64_t *fold, double *f) {
	uint64_t out = 0;
	for (int i = 0; i < 16; i++) {
		out ^= r[i];
	}
	for (int i = 0; i < 8; i++) {
		out ^= fold[i];
		out ^= nm_bits_from_double(f[i]);
	}
	for (int i = 0; i < 262144; i += 4096) {
		out ^= scratch[i];
	}
	return out;
}

static int nm_jit_memmix_matches(int ninst) {
	uint64_t a[16], b[16], fold_a[8], fold_b[8];
	double f_a[8], f_b[8];
	uint8_t *taken_a = (uint8_t *)calloc((size_t)ninst, sizeof(uint8_t));
	uint8_t *taken_b = (uint8_t *)calloc((size_t)ninst, sizeof(uint8_t));
	uint64_t *scratch_a = (uint64_t *)malloc(262144 * sizeof(uint64_t));
	uint64_t *scratch_b = (uint64_t *)malloc(262144 * sizeof(uint64_t));
	if (scratch_a == NULL || scratch_b == NULL || taken_a == NULL || taken_b == NULL) {
		free(scratch_a);
		free(scratch_b);
		free(taken_a);
		free(taken_b);
		return 0;
	}
	nm_init_memmix_state(a, scratch_a, fold_a, f_a);
	nm_init_memmix_state(b, scratch_b, fold_b, f_b);
	void *mapping = NULL;
	size_t mapping_len = 0;
	nm_jit_mem_fn fn = nm_compile_memmix(ninst, &mapping, &mapping_len);
	if (fn == NULL) {
		free(scratch_a);
		free(scratch_b);
		return 0;
	}
	uint8_t rk_bytes[176];
	nm_init_probe_round_keys(rk_bytes);
	nm_interp_memmix(a, scratch_a, fold_a, f_a, (const uint8x16_t *)rk_bytes, taken_a, ninst, 0xD00DFEED12345678ull, 7);
	fn(b, scratch_b, fold_b, f_b, 0xD00DFEED12345678ull, 7, rk_bytes, taken_b);
	int ok = memcmp(a, b, sizeof(a)) == 0 &&
		memcmp(f_a, f_b, sizeof(f_a)) == 0 &&
		memcmp(fold_a, fold_b, sizeof(fold_a)) == 0 &&
		memcmp(scratch_a, scratch_b, 262144 * sizeof(uint64_t)) == 0 &&
		memcmp(taken_a, taken_b, (size_t)ninst) == 0;
	munmap(mapping, mapping_len);
	free(scratch_a);
	free(scratch_b);
	free(taken_a);
	free(taken_b);
	return ok;
}

static int nm_jit_resmemmix_matches(int ninst) {
	uint64_t a[16], b[16], fold_a[8], fold_b[8];
	double f_a[8], f_b[8];
	uint64_t *scratch_a = (uint64_t *)malloc(262144 * sizeof(uint64_t));
	uint64_t *scratch_b = (uint64_t *)malloc(262144 * sizeof(uint64_t));
	if (scratch_a == NULL || scratch_b == NULL) {
		free(scratch_a);
		free(scratch_b);
		return 0;
	}
	nm_init_memmix_state(a, scratch_a, fold_a, f_a);
	nm_init_memmix_state(b, scratch_b, fold_b, f_b);
	void *mapping = NULL;
	size_t mapping_len = 0;
	nm_jit_resmem_fn fn = nm_compile_resmemmix(ninst, &mapping, &mapping_len);
	if (fn == NULL) {
		free(scratch_a);
		free(scratch_b);
		return 0;
	}
	nm_interp_resmemmix(a, scratch_a, fold_a, ninst, 0xD00DFEED12345678ull, 7);
	fn(b, scratch_b, fold_b, 7);
	int ok = memcmp(a, b, sizeof(a)) == 0 &&
		memcmp(fold_a, fold_b, sizeof(fold_a)) == 0 &&
		memcmp(scratch_a, scratch_b, 262144 * sizeof(uint64_t)) == 0;
	munmap(mapping, mapping_len);
	free(scratch_a);
	free(scratch_b);
	return ok;
}

static uint64_t nm_run_interp_memmix(int iters, int ninst) {
	uint64_t r[16], fold[8];
	double f[8];
	uint8_t rk_bytes[176];
	uint8_t *taken = (uint8_t *)calloc((size_t)ninst, sizeof(uint8_t));
	uint64_t *scratch = (uint64_t *)malloc(262144 * sizeof(uint64_t));
	if (scratch == NULL || taken == NULL) {
		free(taken);
		return 0;
	}
	nm_init_memmix_state(r, scratch, fold, f);
	nm_init_probe_round_keys(rk_bytes);
	for (int i = 0; i < iters; i++) {
		memset(taken, 0, (size_t)ninst);
		nm_interp_memmix(r, scratch, fold, f, (const uint8x16_t *)rk_bytes, taken, ninst, 0xD00DFEED12345678ull, (uint64_t)i);
	}
	uint64_t out = nm_finish_memmix_state(r, scratch, fold, f);
	free(scratch);
	free(taken);
	return out;
}

static uint64_t nm_run_interp_resmemmix(int iters, int ninst) {
	uint64_t r[16], fold[8];
	double f[8];
	uint64_t *scratch = (uint64_t *)malloc(262144 * sizeof(uint64_t));
	if (scratch == NULL) {
		return 0;
	}
	nm_init_memmix_state(r, scratch, fold, f);
	for (int i = 0; i < iters; i++) {
		nm_interp_resmemmix(r, scratch, fold, ninst, 0xD00DFEED12345678ull, (uint64_t)i);
	}
	uint64_t out = nm_finish_memmix_state(r, scratch, fold, f);
	free(scratch);
	return out;
}

static uint64_t nm_run_jit_resmemmix(int iters, int ninst) {
	uint64_t r[16], fold[8];
	double f[8];
	uint64_t *scratch = (uint64_t *)malloc(262144 * sizeof(uint64_t));
	if (scratch == NULL) {
		return 0;
	}
	nm_init_memmix_state(r, scratch, fold, f);
	void *mapping = NULL;
	size_t mapping_len = 0;
	nm_jit_resmem_fn fn = nm_compile_resmemmix(ninst, &mapping, &mapping_len);
	if (fn == NULL) {
		free(scratch);
		return 0;
	}
	for (int i = 0; i < iters; i++) {
		fn(r, scratch, fold, (uint64_t)i);
	}
	uint64_t out = nm_finish_memmix_state(r, scratch, fold, f);
	munmap(mapping, mapping_len);
	free(scratch);
	return out;
}

static uint64_t nm_run_jit_memmix(int iters, int ninst) {
	uint64_t r[16], fold[8];
	double f[8];
	uint8_t rk_bytes[176];
	uint8_t *taken = (uint8_t *)calloc((size_t)ninst, sizeof(uint8_t));
	uint64_t *scratch = (uint64_t *)malloc(262144 * sizeof(uint64_t));
	if (scratch == NULL || taken == NULL) {
		free(taken);
		return 0;
	}
	nm_init_memmix_state(r, scratch, fold, f);
	nm_init_probe_round_keys(rk_bytes);
	void *mapping = NULL;
	size_t mapping_len = 0;
	nm_jit_mem_fn fn = nm_compile_memmix(ninst, &mapping, &mapping_len);
	if (fn == NULL) {
		free(scratch);
		return 0;
	}
	for (int i = 0; i < iters; i++) {
		memset(taken, 0, (size_t)ninst);
		fn(r, scratch, fold, f, 0xD00DFEED12345678ull, (uint64_t)i, rk_bytes, taken);
	}
	uint64_t out = nm_finish_memmix_state(r, scratch, fold, f);
	munmap(mapping, mapping_len);
	free(scratch);
	free(taken);
	return out;
}

static uint64_t nm_run_jit_memmix_reuse_mprotect(int iters, int ninst) {
	uint64_t r[16], fold[8];
	double f[8];
	uint8_t rk_bytes[176];
	uint8_t *taken = (uint8_t *)calloc((size_t)ninst, sizeof(uint8_t));
	uint64_t *scratch = (uint64_t *)malloc(262144 * sizeof(uint64_t));
	if (scratch == NULL || taken == NULL) {
		free(taken);
		return 0;
	}
	nm_init_memmix_state(r, scratch, fold, f);
	nm_init_probe_round_keys(rk_bytes);
	size_t bytes = (size_t)ninst * 80 * 4 + 4;
	long page = sysconf(_SC_PAGESIZE);
	size_t alloc = (bytes + (size_t)page - 1) & ~((size_t)page - 1);
	void *mem = mmap(NULL, alloc, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
	if (mem == MAP_FAILED) {
		free(scratch);
		return 0;
	}
	for (int i = 0; i < iters; i++) {
		if (mprotect(mem, alloc, PROT_READ | PROT_WRITE) != 0) {
			munmap(mem, alloc);
			free(scratch);
			return 0;
		}
		nm_emit_memmix_into(mem, ninst);
		if (mprotect(mem, alloc, PROT_READ | PROT_EXEC) != 0) {
			munmap(mem, alloc);
			free(scratch);
			return 0;
		}
		memset(taken, 0, (size_t)ninst);
		((nm_jit_mem_fn)mem)(r, scratch, fold, f, 0xD00DFEED12345678ull, (uint64_t)i, rk_bytes, taken);
	}
	uint64_t out = nm_finish_memmix_state(r, scratch, fold, f);
	munmap(mem, alloc);
	free(scratch);
	free(taken);
	return out;
}
*/
import "C"
import "unsafe"

func jitAddChainMatches(ninst int) bool {
	return C.nm_jit_add_chain_matches(C.int(ninst)) != 0
}

func runInterpAddChain(iters, ninst int) uint64 {
	return uint64(C.nm_run_interp_add_chain(C.int(iters), C.int(ninst)))
}

func runJITAddChain(iters, ninst int) uint64 {
	return uint64(C.nm_run_jit_add_chain(C.int(iters), C.int(ninst)))
}

func runJITAddChainReuseMprotect(iters, ninst int) uint64 {
	return uint64(C.nm_run_jit_add_chain_reuse_mprotect(C.int(iters), C.int(ninst)))
}

func jitIntMixMatches(ninst int) bool {
	return C.nm_jit_intmix_matches(C.int(ninst)) != 0
}

func jitIntMixResidentMatches(ninst int) bool {
	return C.nm_jit_intmix_resident_matches(C.int(ninst)) != 0
}

func runInterpIntMix(iters, ninst int) uint64 {
	return uint64(C.nm_run_interp_intmix(C.int(iters), C.int(ninst)))
}

func runJITIntMix(iters, ninst int) uint64 {
	return uint64(C.nm_run_jit_intmix(C.int(iters), C.int(ninst)))
}

func runJITIntMixResident(iters, ninst int) uint64 {
	return uint64(C.nm_run_jit_intmix_resident(C.int(iters), C.int(ninst)))
}

func runJITIntMixReuseMprotect(iters, ninst int) uint64 {
	return uint64(C.nm_run_jit_intmix_reuse_mprotect(C.int(iters), C.int(ninst)))
}

func jitMemMixMatches(ninst int) bool {
	return C.nm_jit_memmix_matches(C.int(ninst)) != 0
}

func jitResMemMixMatches(ninst int) bool {
	return C.nm_jit_resmemmix_matches(C.int(ninst)) != 0
}

func jitResFloatMixMatches(ninst int) bool {
	return C.nm_jit_resfloatmix_matches(C.int(ninst)) != 0
}

func debugResFloatMix(ninst int) (uint64, uint64, uint64, bool) {
	var out [3]C.uint64_t
	ok := C.nm_jit_resfloatmix_debug(C.int(ninst), (*C.uint64_t)(unsafe.Pointer(&out[0]))) != 0
	return uint64(out[0]), uint64(out[1]), uint64(out[2]), ok
}

func runInterpMemMix(iters, ninst int) uint64 {
	return uint64(C.nm_run_interp_memmix(C.int(iters), C.int(ninst)))
}

func runInterpResFloatMix(iters, ninst int) uint64 {
	return uint64(C.nm_run_interp_resfloatmix(C.int(iters), C.int(ninst)))
}

func runInterpResMemMix(iters, ninst int) uint64 {
	return uint64(C.nm_run_interp_resmemmix(C.int(iters), C.int(ninst)))
}

func runJITMemMix(iters, ninst int) uint64 {
	return uint64(C.nm_run_jit_memmix(C.int(iters), C.int(ninst)))
}

func runJITResMemMix(iters, ninst int) uint64 {
	return uint64(C.nm_run_jit_resmemmix(C.int(iters), C.int(ninst)))
}

func runJITResFloatMix(iters, ninst int) uint64 {
	return uint64(C.nm_run_jit_resfloatmix(C.int(iters), C.int(ninst)))
}

func runJITMemMixReuseMprotect(iters, ninst int) uint64 {
	return uint64(C.nm_run_jit_memmix_reuse_mprotect(C.int(iters), C.int(ninst)))
}

func initProbeRoundKeys(rk *[176]byte) {
	C.nm_init_probe_round_keys((*C.uint8_t)(unsafe.Pointer(&rk[0])))
}

func expandAES128RoundKeys(key *[16]byte, rk *[176]byte) {
	C.nm_expand_aes128(
		(*C.uint8_t)(unsafe.Pointer(&key[0])),
		(*C.uint8_t)(unsafe.Pointer(&rk[0])),
	)
}

func probeEncryptBlock(block *[16]byte, rk *[176]byte) {
	C.nm_probe_encrypt_block(
		(*C.uint8_t)(unsafe.Pointer(&block[0])),
		(*C.uint8_t)(unsafe.Pointer(&rk[0])),
	)
}

func jitRealProgramOnce(prog []instr, branchMask uint64, r *[16]uint64, scratch []uint64, fold *[8]uint64, f *[8]float64, rotSalt uint64, loop uint64, rk *[176]byte, taken []uint8) bool {
	if len(prog) == 0 || len(scratch) == 0 || len(taken) < len(prog) {
		return false
	}
	return C.nm_jit_real_program_once(
		(*C.nm_probe_instr)(unsafe.Pointer(&prog[0])),
		C.int(len(prog)),
		C.uint64_t(branchMask),
		(*C.uint64_t)(unsafe.Pointer(&r[0])),
		(*C.uint64_t)(unsafe.Pointer(&scratch[0])),
		(*C.uint64_t)(unsafe.Pointer(&fold[0])),
		(*C.double)(unsafe.Pointer(&f[0])),
		C.uint64_t(rotSalt),
		C.uint64_t(loop),
		(*C.uint8_t)(unsafe.Pointer(&rk[0])),
		(*C.uint8_t)(unsafe.Pointer(&taken[0])),
	) != 0
}

func jitRealProgramResidentOnce(prog []instr, branchMask uint64, r *[16]uint64, scratch []uint64, fold *[8]uint64, f *[8]float64, rotSalt uint64, loop uint64, rk *[176]byte, taken []uint8) bool {
	if len(prog) == 0 || len(scratch) == 0 || len(taken) < len(prog) {
		return false
	}
	return C.nm_jit_real_program_resident_once(
		(*C.nm_probe_instr)(unsafe.Pointer(&prog[0])),
		C.int(len(prog)),
		C.uint64_t(branchMask),
		(*C.uint64_t)(unsafe.Pointer(&r[0])),
		(*C.uint64_t)(unsafe.Pointer(&scratch[0])),
		(*C.uint64_t)(unsafe.Pointer(&fold[0])),
		(*C.double)(unsafe.Pointer(&f[0])),
		C.uint64_t(rotSalt),
		C.uint64_t(loop),
		(*C.uint8_t)(unsafe.Pointer(&rk[0])),
		(*C.uint8_t)(unsafe.Pointer(&taken[0])),
	) != 0
}

func jitRealProgramLoop(prog []instr, branchMask uint64, r *[16]uint64, scratch []uint64, fold *[8]uint64, f *[8]float64, rotSalt uint64, firstLoop uint64, rk *[176]byte, taken []uint8, iters int) bool {
	if len(prog) == 0 || len(scratch) == 0 || len(taken) < len(prog) || iters < 0 {
		return false
	}
	return C.nm_jit_real_program_loop(
		(*C.nm_probe_instr)(unsafe.Pointer(&prog[0])),
		C.int(len(prog)),
		C.uint64_t(branchMask),
		(*C.uint64_t)(unsafe.Pointer(&r[0])),
		(*C.uint64_t)(unsafe.Pointer(&scratch[0])),
		(*C.uint64_t)(unsafe.Pointer(&fold[0])),
		(*C.double)(unsafe.Pointer(&f[0])),
		C.uint64_t(rotSalt),
		C.uint64_t(firstLoop),
		(*C.uint8_t)(unsafe.Pointer(&rk[0])),
		(*C.uint8_t)(unsafe.Pointer(&taken[0])),
		C.int(iters),
	) != 0
}

func jitRealProgramResidentLoop(prog []instr, branchMask uint64, r *[16]uint64, scratch []uint64, fold *[8]uint64, f *[8]float64, rotSalt uint64, firstLoop uint64, rk *[176]byte, taken []uint8, iters int) bool {
	if len(prog) == 0 || len(scratch) == 0 || len(taken) < len(prog) || iters < 0 {
		return false
	}
	return C.nm_jit_real_program_resident_loop(
		(*C.nm_probe_instr)(unsafe.Pointer(&prog[0])),
		C.int(len(prog)),
		C.uint64_t(branchMask),
		(*C.uint64_t)(unsafe.Pointer(&r[0])),
		(*C.uint64_t)(unsafe.Pointer(&scratch[0])),
		(*C.uint64_t)(unsafe.Pointer(&fold[0])),
		(*C.double)(unsafe.Pointer(&f[0])),
		C.uint64_t(rotSalt),
		C.uint64_t(firstLoop),
		(*C.uint8_t)(unsafe.Pointer(&rk[0])),
		(*C.uint8_t)(unsafe.Pointer(&taken[0])),
		C.int(iters),
	) != 0
}

func jitRealExecuteProbe(p *Params, prog []instr, r *[16]uint64, scratch []uint64, fold *[8]uint64, f *[8]float64, dataset []uint64, useDataset bool, rk *[176]byte, taken []uint8) bool {
	if len(prog) == 0 || len(scratch) == 0 || len(taken) < len(prog) {
		return false
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
	return C.nm_jit_real_execute_probe(
		(*C.nm_probe_instr)(unsafe.Pointer(&prog[0])),
		C.int(len(prog)),
		C.uint32_t(p.Loops),
		C.uint64_t(p.BranchMask),
		C.uint64_t(p.RotSalt),
		(*C.uint64_t)(unsafe.Pointer(&r[0])),
		(*C.uint64_t)(unsafe.Pointer(&scratch[0])),
		(*C.uint64_t)(unsafe.Pointer(&fold[0])),
		(*C.double)(unsafe.Pointer(&f[0])),
		ds,
		useDS,
		(*C.uint8_t)(unsafe.Pointer(&rk[0])),
		(*C.uint8_t)(unsafe.Pointer(&taken[0])),
	) != 0
}

func jitRealExecuteProbeResident(p *Params, prog []instr, r *[16]uint64, scratch []uint64, fold *[8]uint64, f *[8]float64, dataset []uint64, useDataset bool, rk *[176]byte, taken []uint8) bool {
	if len(prog) == 0 || len(scratch) == 0 || len(taken) < len(prog) {
		return false
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
	return C.nm_jit_real_execute_probe_resident(
		(*C.nm_probe_instr)(unsafe.Pointer(&prog[0])),
		C.int(len(prog)),
		C.uint32_t(p.Loops),
		C.uint64_t(p.BranchMask),
		C.uint64_t(p.RotSalt),
		(*C.uint64_t)(unsafe.Pointer(&r[0])),
		(*C.uint64_t)(unsafe.Pointer(&scratch[0])),
		(*C.uint64_t)(unsafe.Pointer(&fold[0])),
		(*C.double)(unsafe.Pointer(&f[0])),
		ds,
		useDS,
		(*C.uint8_t)(unsafe.Pointer(&rk[0])),
		(*C.uint8_t)(unsafe.Pointer(&taken[0])),
	) != 0
}

func newJITBuffer(progSize int) unsafe.Pointer {
	return unsafe.Pointer(C.nm_jit_buffer_new(C.int(progSize)))
}

func newJITResidentBuffer(progSize int) unsafe.Pointer {
	return unsafe.Pointer(C.nm_jit_resident_buffer_new(C.int(progSize)))
}

func freeJITBuffer(buf unsafe.Pointer) {
	C.nm_jit_buffer_free((*C.nm_jit_buffer)(buf))
}

func jitRealExecuteProbeReuse(buf unsafe.Pointer, p *Params, prog []instr, r *[16]uint64, scratch []uint64, fold *[8]uint64, f *[8]float64, dataset []uint64, useDataset bool, rk *[176]byte, taken []uint8) bool {
	if buf == nil || len(prog) == 0 || len(scratch) == 0 || len(taken) < len(prog) {
		return false
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
	return C.nm_jit_real_execute_probe_reuse(
		(*C.nm_jit_buffer)(buf),
		(*C.nm_probe_instr)(unsafe.Pointer(&prog[0])),
		C.int(len(prog)),
		C.uint32_t(p.Loops),
		C.uint64_t(p.BranchMask),
		C.uint64_t(p.RotSalt),
		(*C.uint64_t)(unsafe.Pointer(&r[0])),
		(*C.uint64_t)(unsafe.Pointer(&scratch[0])),
		(*C.uint64_t)(unsafe.Pointer(&fold[0])),
		(*C.double)(unsafe.Pointer(&f[0])),
		ds,
		useDS,
		(*C.uint8_t)(unsafe.Pointer(&rk[0])),
		(*C.uint8_t)(unsafe.Pointer(&taken[0])),
	) != 0
}

func jitRealExecuteProbeResidentReuse(buf unsafe.Pointer, p *Params, prog []instr, r *[16]uint64, scratch []uint64, fold *[8]uint64, f *[8]float64, dataset []uint64, useDataset bool, rk *[176]byte, taken []uint8) bool {
	if buf == nil || len(prog) == 0 || len(scratch) == 0 || len(taken) < len(prog) {
		return false
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
	return C.nm_jit_real_execute_probe_resident_reuse(
		(*C.nm_jit_buffer)(buf),
		(*C.nm_probe_instr)(unsafe.Pointer(&prog[0])),
		C.int(len(prog)),
		C.uint32_t(p.Loops),
		C.uint64_t(p.BranchMask),
		C.uint64_t(p.RotSalt),
		(*C.uint64_t)(unsafe.Pointer(&r[0])),
		(*C.uint64_t)(unsafe.Pointer(&scratch[0])),
		(*C.uint64_t)(unsafe.Pointer(&fold[0])),
		(*C.double)(unsafe.Pointer(&f[0])),
		ds,
		useDS,
		(*C.uint8_t)(unsafe.Pointer(&rk[0])),
		(*C.uint8_t)(unsafe.Pointer(&taken[0])),
	) != 0
}
