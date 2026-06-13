#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <mach/mach_time.h>

static const uint8_t kSbox[256] = {
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

static uint8_t xtime8(uint8_t x) {
    return (uint8_t)((x << 1) ^ (((x >> 7) & 1) * 0x1b));
}

static uint32_t be32(const uint8_t *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

static void expandAES128Key(const uint8_t key[16], uint8_t roundBytes[176], uint32_t roundWords[44]) {
    static const uint8_t rcon[11] = {
        0x00,0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1b,0x36
    };
    memcpy(roundBytes, key, 16);
    uint8_t temp[4];
    int bytes = 16;
    int round = 1;
    while (bytes < 176) {
        for (int i = 0; i < 4; i++) temp[i] = roundBytes[bytes - 4 + i];
        if ((bytes % 16) == 0) {
            uint8_t first = temp[0];
            temp[0] = kSbox[temp[1]] ^ rcon[round++];
            temp[1] = kSbox[temp[2]];
            temp[2] = kSbox[temp[3]];
            temp[3] = kSbox[first];
        }
        for (int i = 0; i < 4; i++) {
            roundBytes[bytes] = roundBytes[bytes - 16] ^ temp[i];
            bytes++;
        }
    }
    for (int i = 0; i < 44; i++) roundWords[i] = be32(roundBytes + i * 4);
}

static void initTables(uint32_t te0[256], uint32_t te1[256], uint32_t te2[256], uint32_t te3[256]) {
    for (int i = 0; i < 256; i++) {
        uint8_t s = kSbox[i];
        uint8_t s2 = xtime8(s);
        uint8_t s3 = s2 ^ s;
        te0[i] = ((uint32_t)s2 << 24) | ((uint32_t)s << 16) | ((uint32_t)s << 8) | s3;
        te1[i] = ((uint32_t)s3 << 24) | ((uint32_t)s2 << 16) | ((uint32_t)s << 8) | s;
        te2[i] = ((uint32_t)s << 24) | ((uint32_t)s3 << 16) | ((uint32_t)s2 << 8) | s;
        te3[i] = ((uint32_t)s << 24) | ((uint32_t)s << 16) | ((uint32_t)s3 << 8) | s2;
    }
}

static NSString *metalSource(void) {
    return @"#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "static inline uint get_be(device const uchar *p, uint off) {\n"
    "  return ((uint)p[off] << 24) | ((uint)p[off+1] << 16) | ((uint)p[off+2] << 8) | (uint)p[off+3];\n"
    "}\n"
    "static inline void put_be(device uchar *p, uint off, uint x) {\n"
    "  p[off]=(uchar)(x >> 24); p[off+1]=(uchar)(x >> 16); p[off+2]=(uchar)(x >> 8); p[off+3]=(uchar)x;\n"
    "}\n"
    "kernel void aes128_ttable(device uchar *out [[buffer(0)]], device const uchar *in [[buffer(1)]], device const uint *rk [[buffer(2)]], device const uint *te0 [[buffer(3)]], device const uint *te1 [[buffer(4)]], device const uint *te2 [[buffer(5)]], device const uint *te3 [[buffer(6)]], device const uchar *sb [[buffer(7)]], uint gid [[thread_position_in_grid]]) {\n"
    "  uint off = gid * 16;\n"
    "  uint s0 = get_be(in, off) ^ rk[0];\n"
    "  uint s1 = get_be(in, off + 4) ^ rk[1];\n"
    "  uint s2 = get_be(in, off + 8) ^ rk[2];\n"
    "  uint s3 = get_be(in, off + 12) ^ rk[3];\n"
    "  for (uint r = 1; r < 10; r++) {\n"
    "    uint t0 = te0[(s0 >> 24) & 255] ^ te1[(s1 >> 16) & 255] ^ te2[(s2 >> 8) & 255] ^ te3[s3 & 255] ^ rk[r*4];\n"
    "    uint t1 = te0[(s1 >> 24) & 255] ^ te1[(s2 >> 16) & 255] ^ te2[(s3 >> 8) & 255] ^ te3[s0 & 255] ^ rk[r*4+1];\n"
    "    uint t2 = te0[(s2 >> 24) & 255] ^ te1[(s3 >> 16) & 255] ^ te2[(s0 >> 8) & 255] ^ te3[s1 & 255] ^ rk[r*4+2];\n"
    "    uint t3 = te0[(s3 >> 24) & 255] ^ te1[(s0 >> 16) & 255] ^ te2[(s1 >> 8) & 255] ^ te3[s2 & 255] ^ rk[r*4+3];\n"
    "    s0=t0; s1=t1; s2=t2; s3=t3;\n"
    "  }\n"
    "  uint f0 = ((uint)sb[(s0 >> 24) & 255] << 24) ^ ((uint)sb[(s1 >> 16) & 255] << 16) ^ ((uint)sb[(s2 >> 8) & 255] << 8) ^ (uint)sb[s3 & 255] ^ rk[40];\n"
    "  uint f1 = ((uint)sb[(s1 >> 24) & 255] << 24) ^ ((uint)sb[(s2 >> 16) & 255] << 16) ^ ((uint)sb[(s3 >> 8) & 255] << 8) ^ (uint)sb[s0 & 255] ^ rk[41];\n"
    "  uint f2 = ((uint)sb[(s2 >> 24) & 255] << 24) ^ ((uint)sb[(s3 >> 16) & 255] << 16) ^ ((uint)sb[(s0 >> 8) & 255] << 8) ^ (uint)sb[s1 & 255] ^ rk[42];\n"
    "  uint f3 = ((uint)sb[(s3 >> 24) & 255] << 24) ^ ((uint)sb[(s0 >> 16) & 255] << 16) ^ ((uint)sb[(s1 >> 8) & 255] << 8) ^ (uint)sb[s2 & 255] ^ rk[43];\n"
    "  put_be(out, off, f0); put_be(out, off + 4, f1); put_be(out, off + 8, f2); put_be(out, off + 12, f3);\n"
    "}\n";
}

int main() {
    @autoreleasepool {
        const uint8_t key[16] = {
            0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
            0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f
        };
        const uint8_t plaintext[16] = {
            0x00,0x11,0x22,0x33,0x44,0x55,0x66,0x77,
            0x88,0x99,0xaa,0xbb,0xcc,0xdd,0xee,0xff
        };
        const uint8_t expected[16] = {
            0x69,0xc4,0xe0,0xd8,0x6a,0x7b,0x04,0x30,
            0xd8,0xcd,0xb7,0x80,0x70,0xb4,0xc5,0x5a
        };
        uint8_t roundBytes[176];
        uint32_t roundWords[44], te0[256], te1[256], te2[256], te3[256];
        expandAES128Key(key, roundBytes, roundWords);
        initTables(te0, te1, te2, te3);

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { fprintf(stderr, "no metal device\n"); return 1; }
        printf("device=%s\n", dev.name.UTF8String);
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:metalSource() options:nil error:&err];
        if (!lib) { fprintf(stderr, "library error: %s\n", err.localizedDescription.UTF8String); return 2; }
        id<MTLFunction> fn = [lib newFunctionWithName:@"aes128_ttable"];
        id<MTLComputePipelineState> pipe = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pipe) { fprintf(stderr, "pipeline error: %s\n", err.localizedDescription.UTF8String); return 3; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        id<MTLBuffer> rk = [dev newBufferWithBytes:roundWords length:sizeof(roundWords) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bte0 = [dev newBufferWithBytes:te0 length:sizeof(te0) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bte1 = [dev newBufferWithBytes:te1 length:sizeof(te1) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bte2 = [dev newBufferWithBytes:te2 length:sizeof(te2) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bte3 = [dev newBufferWithBytes:te3 length:sizeof(te3) options:MTLResourceStorageModeShared];
        id<MTLBuffer> sb = [dev newBufferWithBytes:kSbox length:sizeof(kSbox) options:MTLResourceStorageModeShared];
        id<MTLBuffer> in = [dev newBufferWithBytes:plaintext length:16 options:MTLResourceStorageModeShared];
        id<MTLBuffer> out = [dev newBufferWithLength:16 options:MTLResourceStorageModeShared];

        void (^encode)(id<MTLComputeCommandEncoder>, id<MTLBuffer>, id<MTLBuffer>) = ^(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> dst, id<MTLBuffer> src) {
            [enc setComputePipelineState:pipe];
            [enc setBuffer:dst offset:0 atIndex:0];
            [enc setBuffer:src offset:0 atIndex:1];
            [enc setBuffer:rk offset:0 atIndex:2];
            [enc setBuffer:bte0 offset:0 atIndex:3];
            [enc setBuffer:bte1 offset:0 atIndex:4];
            [enc setBuffer:bte2 offset:0 atIndex:5];
            [enc setBuffer:bte3 offset:0 atIndex:6];
            [enc setBuffer:sb offset:0 atIndex:7];
        };

        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        encode(enc, out, in);
        [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        uint8_t *got = (uint8_t *)out.contents;
        printf("ciphertext=");
        for (int i = 0; i < 16; i++) printf("%02x", got[i]);
        printf("\n");
        int ok = memcmp(got, expected, 16) == 0;
        printf("aes128_ttable_vector=%s\n", ok ? "pass" : "fail");
        if (!ok) return 4;

        const NSUInteger blocks = 1u << 20;
        const NSUInteger iters = 32;
        id<MTLBuffer> benchIn = [dev newBufferWithLength:blocks * 16 options:MTLResourceStorageModeShared];
        id<MTLBuffer> benchOut = [dev newBufferWithLength:blocks * 16 options:MTLResourceStorageModeShared];
        uint8_t *benchBytes = (uint8_t *)benchIn.contents;
        for (NSUInteger i = 0; i < blocks * 16; i++) benchBytes[i] = (uint8_t)(i * 131u + (i >> 7));
        MTLSize grid = MTLSizeMake(blocks, 1, 1);
        NSUInteger tg = pipe.maxTotalThreadsPerThreadgroup < 256 ? pipe.maxTotalThreadsPerThreadgroup : 256;
        MTLSize group = MTLSizeMake(tg, 1, 1);
        uint64_t start = mach_absolute_time();
        for (NSUInteger i = 0; i < iters; i++) {
            id<MTLCommandBuffer> bcb = [q commandBuffer];
            id<MTLComputeCommandEncoder> benc = [bcb computeCommandEncoder];
            encode(benc, benchOut, benchIn);
            [benc dispatchThreads:grid threadsPerThreadgroup:group];
            [benc endEncoding];
            [bcb commit];
            [bcb waitUntilCompleted];
        }
        uint64_t end = mach_absolute_time();
        mach_timebase_info_data_t tb; mach_timebase_info(&tb);
        double seconds = (double)(end - start) * (double)tb.numer / (double)tb.denom / 1e9;
        uint8_t *benchGot = (uint8_t *)benchOut.contents;
        uint64_t checksum = 0;
        for (NSUInteger i = 0; i < blocks * 16; i += (blocks * 16) / 1024) checksum += benchGot[i];
        double blocksPerSec = (double)blocks * (double)iters / seconds;
        printf("elapsed=%.3fs aes_blocks=%.1fM/s aes_only_2MiB_fill_cap=%.1f H/s checksum=%llu\n",
               seconds, blocksPerSec / 1e6, blocksPerSec / 131072.0, checksum);
        return 0;
    }
}
