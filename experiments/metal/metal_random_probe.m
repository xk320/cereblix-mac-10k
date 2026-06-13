#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <mach/mach_time.h>

int main() {
    @autoreleasepool {
        const NSUInteger lanesN = 1u << 22;
        const NSUInteger wordsN = 64u << 20 >> 2;
        const NSUInteger iters = 120;
        NSString *src = @"#include <metal_stdlib>\nusing namespace metal;\nkernel void random_walk(device uint *out [[buffer(0)]], device const uint *ds [[buffer(1)]], uint gid [[thread_position_in_grid]]) {\n uint x = gid * 747796405u + 2891336453u;\n uint acc = x;\n for (uint i = 0; i < 64; i++) { uint idx = (x + acc * 1664525u + i * 1013904223u) & 0x00ffffffu; uint v = ds[idx]; acc ^= v; x = (v + (acc << 5) + (acc >> 3)); }\n out[gid] = acc;\n}\n";
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { fprintf(stderr, "no metal device\n"); return 1; }
        printf("device=%s\n", dev.name.UTF8String);
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
        if (!lib) { fprintf(stderr, "library error: %s\n", err.localizedDescription.UTF8String); return 2; }
        id<MTLFunction> fn = [lib newFunctionWithName:@"random_walk"];
        id<MTLComputePipelineState> pipe = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pipe) { fprintf(stderr, "pipeline error: %s\n", err.localizedDescription.UTF8String); return 3; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        id<MTLBuffer> out = [dev newBufferWithLength:lanesN * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> ds = [dev newBufferWithLength:wordsN * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        uint32_t *dsp = (uint32_t *)ds.contents;
        for (NSUInteger i = 0; i < wordsN; i++) dsp[i] = (uint32_t)(i * 2654435761u) ^ (uint32_t)(i >> 7);
        MTLSize grid = MTLSizeMake(lanesN, 1, 1);
        NSUInteger tg = pipe.maxTotalThreadsPerThreadgroup < 256 ? pipe.maxTotalThreadsPerThreadgroup : 256;
        MTLSize group = MTLSizeMake(tg, 1, 1);
        uint64_t start = mach_absolute_time();
        for (NSUInteger i = 0; i < iters; i++) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pipe];
            [enc setBuffer:out offset:0 atIndex:0];
            [enc setBuffer:ds offset:0 atIndex:1];
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
        for (NSUInteger i = 0; i < lanesN; i += lanesN / 1024) checksum += p[i];
        double lanes = (double)lanesN * (double)iters / seconds / 1e6;
        double reads = (double)lanesN * (double)iters * 64.0 / seconds / 1e9;
        printf("elapsed=%.3fs lanes=%.1fM/s random_reads=%.1fG/s checksum=%llu\n", seconds, lanes, reads, checksum);
    }
    return 0;
}

