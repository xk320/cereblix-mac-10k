#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <mach/mach_time.h>

int main() {
    @autoreleasepool {
        const NSUInteger lanesN = 256;
        const NSUInteger wordsPerLane = (2u << 20) / 4;
        const NSUInteger iters = 80;
        NSString *src = @"#include <metal_stdlib>\nusing namespace metal;\nkernel void scratch_hash(device uint *out [[buffer(0)]], device uint *scratch [[buffer(1)]], uint gid [[thread_position_in_grid]]) {\n uint base = gid * 524288u;\n uint x = gid * 747796405u + 2891336453u;\n for (uint i = 0; i < 524288u; i++) { x ^= x >> 16; x *= 2246822519u; x ^= x >> 13; scratch[base+i] = x; }\n uint acc = x;\n for (uint loop = 0; loop < 48; loop++) {\n   for (uint k = 0; k < 64; k++) { uint idx = (acc + k * 2654435761u + loop * 1013904223u) & 524287u; uint v = scratch[base+idx]; acc ^= v; scratch[base + ((idx + acc) & 524287u)] = acc + v; }\n }\n uint fold = 0;\n for (uint i = 0; i < 524288u; i += 16) fold ^= scratch[base+i];\n out[gid] = fold ^ acc;\n}\n";
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { fprintf(stderr, "no metal device\n"); return 1; }
        printf("device=%s\n", dev.name.UTF8String);
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
        if (!lib) { fprintf(stderr, "library error: %s\n", err.localizedDescription.UTF8String); return 2; }
        id<MTLFunction> fn = [lib newFunctionWithName:@"scratch_hash"];
        id<MTLComputePipelineState> pipe = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pipe) { fprintf(stderr, "pipeline error: %s\n", err.localizedDescription.UTF8String); return 3; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        id<MTLBuffer> out = [dev newBufferWithLength:lanesN * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> scratch = [dev newBufferWithLength:lanesN * wordsPerLane * sizeof(uint32_t) options:MTLResourceStorageModePrivate];
        MTLSize grid = MTLSizeMake(lanesN, 1, 1);
        MTLSize group = MTLSizeMake(64, 1, 1);
        uint64_t start = mach_absolute_time();
        for (NSUInteger i = 0; i < iters; i++) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pipe];
            [enc setBuffer:out offset:0 atIndex:0];
            [enc setBuffer:scratch offset:0 atIndex:1];
            [enc dispatchThreads:grid threadsPerThreadgroup:group];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }
        uint64_t end = mach_absolute_time();
        mach_timebase_info_data_t tb; mach_timebase_info(&tb);
        double seconds = (double)(end - start) * (double)tb.numer / (double)tb.denom / 1e9;
        uint32_t *p = (uint32_t *)out.contents;
        uint64_t checksum = 0;
        for (NSUInteger i = 0; i < lanesN; i++) checksum += p[i];
        double hashes = (double)lanesN * (double)iters / seconds;
        double scratchGB = (double)lanesN * (double)iters * 2.0 / seconds / 1024.0;
        printf("elapsed=%.3fs hashes=%.1f H/s scratch_write_equiv=%.1f GiB/s checksum=%llu\n", seconds, hashes, scratchGB, checksum);
    }
    return 0;
}

