#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "../solver.h"
#include "shader_source.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <openssl/sha.h>

static constexpr uint32_t kThreads   = 1 << 27;  // 128M threads per dispatch
static constexpr uint32_t kGroupSize = 256;       // threads per threadgroup
static constexpr int      kInFlight  = 4;         // quad-buffered command submission
static constexpr int      kSaltBytes = 48;
static constexpr uint64_t kMaxSalt   = (1ULL << 48) - 1;

// Must match Metal shader struct exactly
struct GPUParams {
    uint32_t mask0;
    uint32_t target0;
    uint32_t mask1;
    uint32_t target1;
    uint32_t prefix_len;
    uint32_t pre_state[5];
    uint32_t tail_w[4];        // block words 12-15 only
    uint32_t _pad;
    uint64_t base_salt;
};

// Result buffer layout: [found:u32][salt:u64][hash:20 bytes]
static constexpr int kResultSize = 4 + 8 + 20;

// ---------------------------------------------------------------------------
// Precompute GPU parameters from template + prefix
// ---------------------------------------------------------------------------

static GPUParams precompute(const ObjectTemplate& tpl, const std::string& prefix_hex) {
    GPUParams p{};
    uint32_t n = static_cast<uint32_t>(prefix_hex.size());
    p.prefix_len = n;

    // Build prefix mask/target
    auto prefix = hex_string_to_bytes(prefix_hex);
    uint32_t pw[5] = {};
    for (size_t i = 0; i < prefix.size(); ++i)
        pw[i/4] |= uint32_t(prefix[i]) << ((3-(i%4))*8);

    p.mask0   = (n >= 8) ? ~0u : (~0u << ((8 - n) * 4));
    p.target0 = pw[0] & p.mask0;
    if (n > 8) {
        uint32_t rem = n - 8;
        p.mask1   = (rem >= 8) ? ~0u : (~0u << ((8 - rem) * 4));
        p.target1 = pw[1] & p.mask1;
    }

    // SHA1 pre-state: compress all blocks before the salt
    uint32_t total = static_cast<uint32_t>(tpl.bytes.size());
    int bb = (tpl.salt_offset / 64) * 64;

    SHA_CTX ctx;
    SHA1_Init(&ctx);
    if (bb > 0) SHA1_Update(&ctx, tpl.bytes.data(), bb);
    p.pre_state[0]=ctx.h0; p.pre_state[1]=ctx.h1; p.pre_state[2]=ctx.h2;
    p.pre_state[3]=ctx.h3; p.pre_state[4]=ctx.h4;

    // Build final 64-byte block: zero salt region, add SHA1 padding
    uint8_t block[64] = {};
    uint32_t avail = total - bb;
    memcpy(block, tpl.bytes.data() + bb, avail);
    memset(block + (tpl.salt_offset - bb), 0, kSaltBytes);
    block[avail] = 0x80;
    uint64_t bits = uint64_t(total) * 8;
    for (int i = 0; i < 8; ++i) block[63-i] = uint8_t(bits >> (i*8));

    // Only extract words 12-15 (the tail after the 48-byte salt region)
    for (int i = 0; i < 4; ++i) {
        int o = (12 + i) * 4;
        p.tail_w[i] = (uint32_t(block[o])<<24)|(uint32_t(block[o+1])<<16)|
                       (uint32_t(block[o+2])<<8)|uint32_t(block[o+3]);
    }
    return p;
}

// ---------------------------------------------------------------------------
// Pipeline creation: precompiled metallib with source fallback
// ---------------------------------------------------------------------------

static id<MTLComputePipelineState> create_pipeline(id<MTLDevice> dev) {
    NSError* err = nil;

    // Try precompiled metallib first (sha1.metallib next to the binary)
    NSString* exe = [[NSBundle mainBundle] executablePath];
    NSString* dir = [exe stringByDeletingLastPathComponent];
    NSString* libPath = [dir stringByAppendingPathComponent:@"sha1.metallib"];

    if ([[NSFileManager defaultManager] fileExistsAtPath:libPath]) {
        // Skip zero-byte placeholder (created when Metal compiler is unavailable)
        NSDictionary* attrs = [[NSFileManager defaultManager] attributesOfItemAtPath:libPath error:nil];
        if ([attrs fileSize] > 0) {
            NSURL* url = [NSURL fileURLWithPath:libPath];
            id<MTLLibrary> lib = [dev newLibraryWithURL:url error:&err];
            if (lib) {
                id<MTLFunction> fn = [lib newFunctionWithName:@"bruteforce_sha1"];
                if (fn) {
                    auto pipe = [dev newComputePipelineStateWithFunction:fn error:&err];
                    if (pipe) return pipe;
                }
            }
            fprintf(stderr, "Note: metallib load failed, falling back to source compilation\n");
        }
    }

    // Fallback: compile from embedded source (shader_source.h included at file scope)
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    opts.fastMathEnabled = YES;
    id<MTLLibrary> lib = [dev newLibraryWithSource:
        [NSString stringWithUTF8String:kMetalShaderSource] options:opts error:&err];
    if (!lib) {
        fprintf(stderr, "Shader Error: %s\n", err.localizedDescription.UTF8String);
        return nil;
    }

    auto pipe = [dev newComputePipelineStateWithFunction:
        [lib newFunctionWithName:@"bruteforce_sha1"] error:&err];
    if (!pipe)
        fprintf(stderr, "Pipeline Error: %s\n", err.localizedDescription.UTF8String);
    return pipe;
}

// ---------------------------------------------------------------------------
// Result extraction helper
// ---------------------------------------------------------------------------

struct GPUHit {
    uint64_t salt;
    uint8_t  hash_raw[20];
};

static bool read_result(const id<MTLBuffer> buf, GPUHit& out) {
    auto* r = static_cast<const uint8_t*>(buf.contents);
    uint32_t hit;
    memcpy(&hit, r, 4);
    if (!hit) return false;
    memcpy(&out.salt, r + 4, 8);
    memcpy(out.hash_raw, r + 12, 20);
    return true;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

std::optional<SolveResult> solve(const ObjectTemplate& tpl,
                                 const std::string& prefix_hex,
                                 const std::string& backend_override) {
    @autoreleasepool {
        if (!backend_override.empty() && backend_override != "metal") {
            fprintf(stderr, "Error: backend '%s' not available (macOS Metal build)\n",
                    backend_override.c_str());
            return std::nullopt;
        }

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { fprintf(stderr, "Error: No Metal device\n"); return std::nullopt; }

        fprintf(stderr, "Device         %s\nDispatch       %uM x %d in-flight\n\n",
                dev.name.UTF8String, kThreads >> 20, kInFlight);

        auto pipe = create_pipeline(dev);
        if (!pipe) return std::nullopt;

        id<MTLCommandQueue> q = [dev newCommandQueue];
        GPUParams base_params = precompute(tpl, prefix_hex);

        // Params buffer: CPU writes once per dispatch, GPU reads only.
        // WriteCombined avoids polluting CPU cache on the memcpy.
        auto par_opts = MTLResourceStorageModeShared
                      | MTLResourceCPUCacheModeWriteCombined
                      | MTLResourceHazardTrackingModeUntracked;
        // Result buffer: GPU writes, CPU reads — use default cache mode.
        auto res_opts = MTLResourceStorageModeShared
                      | MTLResourceHazardTrackingModeUntracked;

        id<MTLBuffer> par[kInFlight], res[kInFlight];
        for (int i = 0; i < kInFlight; ++i) {
            par[i] = [dev newBufferWithLength:sizeof(GPUParams) options:par_opts];
            res[i] = [dev newBufferWithLength:kResultSize options:res_opts];
        }

        auto t0 = std::chrono::steady_clock::now();
        int next_sec = 1;
        bool progress = false;
        uint64_t base = 0;

        // Fire a dispatch: stamp params, encode, submit
        auto fire = [&](int i, uint64_t salt) -> id<MTLCommandBuffer> {
            memset(res[i].contents, 0, kResultSize);
            auto* p = static_cast<GPUParams*>(par[i].contents);
            *p = base_params;
            p->base_salt = salt;

            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pipe];
            [enc setBuffer:par[i] offset:0 atIndex:0];
            [enc setBuffer:res[i] offset:0 atIndex:1];
            [enc dispatchThreads:MTLSizeMake(kThreads,1,1)
                threadsPerThreadgroup:MTLSizeMake(kGroupSize,1,1)];
            [enc endEncoding];
            [cb commit];
            return cb;
        };

        // Prime the pipeline
        id<MTLCommandBuffer> fl[kInFlight];
        for (int i = 0; i < kInFlight; ++i) {
            fl[i] = fire(i, base);
            base += kThreads;
        }

        int idx = 0;
        for (;;) {
            [fl[idx] waitUntilCompleted];

            GPUHit hit;
            if (read_result(res[idx], hit)) {
                GPUHit best = hit;
                for (int i = 0; i < kInFlight; ++i) {
                    if (i == idx) continue;
                    [fl[i] waitUntilCompleted];
                    GPUHit other;
                    if (read_result(res[i], other) && other.salt < best.salt)
                        best = other;
                }

                auto out = tpl;
                out.set_salt(best.salt);

                SolveResult sr;
                sr.payload = out.payload();
                static constexpr char hx[] = "0123456789abcdef";
                for (int i = 0; i < 20; ++i) {
                    sr.hash[i*2]   = hx[best.hash_raw[i] >> 4];
                    sr.hash[i*2+1] = hx[best.hash_raw[i] & 0xF];
                }

                double secs = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t0).count();
                if (progress) fprintf(stderr, "\n");
                fprintf(stderr,
                    "Found          %s\n"
                    "Time           %.2fs\n"
                    "Throughput     %.2f GH/s\n"
                    "Hashes Tried   %.2fG\n",
                    std::string(sr.hash.data(), 40).c_str(),
                    secs, double(base)/secs/1e9, double(base)/1e9);
                return sr;
            }

            fl[idx] = fire(idx, base);
            base += kThreads;

            auto now = std::chrono::steady_clock::now();
            int secs = int(std::chrono::duration<double>(now - t0).count());
            if (secs >= next_sec) {
                next_sec = secs + 1;
                progress = true;
                fprintf(stderr, "%.2fG hashes | %.2f GH/s | %ds elapsed\n",
                    double(base)/1e9, double(base)/secs/1e9, secs);
            }

            idx = (idx + 1) % kInFlight;

            if (base > kMaxSalt - kThreads) {
                for (int i = 0; i < kInFlight; ++i) [fl[i] waitUntilCompleted];
                return std::nullopt;
            }
        }
    }
}
