#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "solver.h"
#include "shader_source.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <openssl/sha.h>

static constexpr uint32_t kThreads = 1 << 27;
static constexpr uint32_t kGroupSize = 256;
static constexpr int kInFlight = 3;
static constexpr int kSaltBytes = 48;

// Must match Metal shader struct
struct GPUParams {
    uint32_t prefix_len;
    uint32_t _pad[3];
    uint64_t base_salt;
    uint32_t pre_state[5];
    uint32_t block_w[16];
    uint32_t prefix_words[5];
};

static GPUParams precompute(const ObjectTemplate& tpl, const std::string& prefix_hex) {
    GPUParams p{};
    p.prefix_len = static_cast<uint32_t>(prefix_hex.size());

    auto prefix = hex_string_to_bytes(prefix_hex);
    for (size_t i = 0; i < prefix.size(); ++i)
        p.prefix_words[i/4] |= uint32_t(prefix[i]) << ((3-(i%4))*8);

    uint32_t total = static_cast<uint32_t>(tpl.bytes.size());
    int bb = (tpl.salt_offset / 64) * 64;

    SHA_CTX ctx;
    SHA1_Init(&ctx);
    if (bb > 0) SHA1_Update(&ctx, tpl.bytes.data(), bb);
    p.pre_state[0]=ctx.h0; p.pre_state[1]=ctx.h1; p.pre_state[2]=ctx.h2;
    p.pre_state[3]=ctx.h3; p.pre_state[4]=ctx.h4;

    // Build the final block: copy data, zero salt, add SHA1 padding
    uint8_t block[64] = {};
    uint32_t avail = total - bb;
    memcpy(block, tpl.bytes.data() + bb, avail);
    memset(block + (tpl.salt_offset - bb), 0, kSaltBytes);
    block[avail] = 0x80;
    uint64_t bits = uint64_t(total) * 8;
    for (int i = 0; i < 8; ++i) block[63-i] = uint8_t(bits >> (i*8));

    for (int i = 0; i < 16; ++i) {
        int o = i * 4;
        p.block_w[i] = (uint32_t(block[o])<<24)|(uint32_t(block[o+1])<<16)|
                        (uint32_t(block[o+2])<<8)|uint32_t(block[o+3]);
    }
    return p;
}

std::optional<SolveResult> solve(const ObjectTemplate& tpl,
                                     const std::string& prefix_hex) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { fprintf(stderr, "Error: No Metal device\n"); return std::nullopt; }

        fprintf(stderr, "Device         %s\nDispatch       %uM × %d in-flight\n\n",
                dev.name.UTF8String, kThreads >> 20, kInFlight);

        NSError* err = nil;
        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        opts.fastMathEnabled = YES;
        id<MTLLibrary> lib = [dev newLibraryWithSource:
            [NSString stringWithUTF8String:kMetalShaderSource] options:opts error:&err];
        if (!lib) { fprintf(stderr, "Shader Error: %s\n", err.localizedDescription.UTF8String); return std::nullopt; }

        id<MTLComputePipelineState> pipe =
            [dev newComputePipelineStateWithFunction:
                [lib newFunctionWithName:@"bruteforce_sha1"] error:&err];
        if (!pipe) { fprintf(stderr, "Pipeline Error: %s\n", err.localizedDescription.UTF8String); return std::nullopt; }

        id<MTLCommandQueue> q = [dev newCommandQueue];
        GPUParams base_params = precompute(tpl, prefix_hex);

        auto buf_opts = MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked;
        id<MTLBuffer> par[kInFlight], res[kInFlight];
        for (int i = 0; i < kInFlight; ++i) {
            par[i] = [dev newBufferWithLength:sizeof(GPUParams) options:buf_opts];
            res[i] = [dev newBufferWithLength:64 options:buf_opts];
        }

        auto t0 = std::chrono::steady_clock::now();
        int next_sec = 1;
        bool progress = false;
        uint64_t base = 0;

        auto fire = [&](int i, uint64_t salt) -> id<MTLCommandBuffer> {
            memset(res[i].contents, 0, 64);
            auto* p = static_cast<GPUParams*>(par[i].contents);
            *p = base_params;
            p->base_salt = salt;

            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:pipe];
            [e setBuffer:par[i] offset:0 atIndex:0];
            [e setBuffer:res[i] offset:0 atIndex:1];
            [e dispatchThreads:MTLSizeMake(kThreads,1,1)
                threadsPerThreadgroup:MTLSizeMake(kGroupSize,1,1)];
            [e endEncoding];
            [cb commit];
            return cb;
        };

        id<MTLCommandBuffer> fl[kInFlight];
        for (int i = 0; i < kInFlight; ++i) { fl[i] = fire(i, base); base += kThreads; }

        int idx = 0;
        while (true) {
            [fl[idx] waitUntilCompleted];

            auto* r = static_cast<uint32_t*>(res[idx].contents);
            if (r[0]) {
                for (int i = 0; i < kInFlight; ++i)
                    if (i != idx) [fl[i] waitUntilCompleted];

                uint64_t salt; memcpy(&salt, (uint8_t*)r+4, 8);
                auto out = tpl;
                out.set_salt(salt);

                SolveResult sr;
                sr.payload = out.payload();
                auto* raw = (uint8_t*)r + 12;
                static constexpr char hx[] = "0123456789abcdef";
                for (int i = 0; i < 20; ++i) {
                    sr.hash[i*2]   = hx[raw[i] >> 4];
                    sr.hash[i*2+1] = hx[raw[i] & 0xF];
                }

                double secs = std::chrono::duration<double>(
                    std::chrono::steady_clock::now()-t0).count();
                if (progress) fprintf(stderr, "\n");
                fprintf(stderr,
                    "✓ Found        %s\n"
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
                fprintf(stderr, "⏳ %.2fG hashes | %.2f GH/s | %ds elapsed\n",
                    double(base)/1e9, double(base)/secs/1e9, secs);
            }

            idx = (idx + 1) % kInFlight;
            if (base > (uint64_t(1) << 48) - kThreads) {
                for (int i = 0; i < kInFlight; ++i) [fl[i] waitUntilCompleted];
                return std::nullopt;
            }
        }
    }
}
