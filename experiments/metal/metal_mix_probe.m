#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <mach/mach_time.h>

int main() {
    @autoreleasepool {
        const NSUInteger n = 1u << 24;
        const NSUInteger iters = 200;
        NSString *src = @"#include <metal_stdlib>\nusing namespace metal;\nkernel void mix_kernel(device uint *out [[buffer(0)]], uint gid [[thread_position_in_grid]]) {\n uint x = gid * 747796405u + 2891336453u;\n for (uint i = 0; i < 256; i++) { x ^= x >> 16; x *= 2246822519u; x ^= x >> 13; x *= 3266489917u; x ^= x >> 16; }\n out[gid] = x;\n}\n";
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { fprintf(stderr, "no metal device\n"); return 1; }
        printf("device=%s\n", dev.name.UTF8String);
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
        if (!lib) { fprintf(stderr, "library error: %s\n", err.localizedDescription.UTF8String); return 2; }
        id<MTLFunction> fn = [lib newFunctionWithName:@"mix_kernel"];
        id<MTLComputePipelineState> pipe = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pipe) { fprintf(stderr, "pipeline error: %s\n", err.localizedDescription.UTF8String); return 3; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        id<MTLBuffer> out = [dev newBufferWithLength:n * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        MTLSize grid = MTLSizeMake(n, 1, 1);
        NSUInteger tg = pipe.maxTotalThreadsPerThreadgroup < 256 ? pipe.maxTotalThreadsPerThreadgroup : 256;
        MTLSize group = MTLSizeMake(tg, 1, 1);
        uint64_t start = mach_absolute_time();
        for (NSUInteger i = 0; i < iters; i++) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pipe];
            [enc setBuffer:out offset:0 atIndex:0];
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
        for (NSUInteger i = 0; i < n; i += n / 1024) checksum += p[i];
        double lanes = (double)n * (double)iters / seconds / 1e6;
        double ops = (double)n * (double)iters * 256.0 / seconds / 1e9;
        printf("elapsed=%.3fs lanes=%.1fM/s mix_ops=%.1fG/s checksum=%llu\n", seconds, lanes, ops, checksum);
    }
    return 0;
}

