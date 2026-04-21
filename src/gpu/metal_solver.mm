// Metal backend: drive the GPU SHA-1 kernel and report the winning salt.
//
// We dispatch kThreads at a time, keeping kInFlight command buffers
// concurrent so the GPU is never idle. When any flight hits, we drain all
// flights and return the smallest-salt hit — results then don't depend on
// scheduling order.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "../solver.h"
#include "shader_source.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <openssl/sha.h>

namespace {

// kGroupSize must match the shader's per-TG precompute (256 threads, 16 W[10] nibbles).
constexpr uint32_t kThreads    = 1 << 27;
constexpr uint32_t kGroupSize  = 256;
constexpr int      kInFlight   = 4;
constexpr int      kSaltBytes  = 48;
constexpr uint64_t kMaxSalt    = (1ULL << 48) - 1;
constexpr int      kResultSize = 4 + 8 + 20;   // [found:u32][salt:u64][hash:20]

// Must match the shader's GPUParams layout.
struct GPUParams {
    uint32_t mask0, target0, mask1, target1;
    uint32_t prefix_len;
    uint32_t pre_state[5];
    uint32_t tail_w[4];
    uint32_t _pad;
    uint64_t base_salt;
};

// Big-endian pack: up to 4 bytes into the high end of a word; mask covers the
// top `nibbles` nibbles.
static void pack_prefix(const uint8_t* bytes, size_t nbytes, size_t nibbles,
                        uint32_t& mask, uint32_t& target) {
    uint32_t w = 0;
    for (size_t i = 0; i < nbytes; ++i) w |= uint32_t(bytes[i]) << ((3 - i) * 8);
    mask   = (nibbles >= 8) ? ~0u : (~0u << ((8 - nibbles) * 4));
    target = w & mask;
}

static GPUParams build_params(const ObjectTemplate& tpl, const std::string& prefix_hex) {
    GPUParams p{};
    p.prefix_len = uint32_t(prefix_hex.size());
    auto prefix = hex_string_to_bytes(prefix_hex);
    pack_prefix(prefix.data(), std::min<size_t>(prefix.size(), 4),
                std::min<size_t>(p.prefix_len, 8), p.mask0, p.target0);
    if (p.prefix_len > 8)
        pack_prefix(prefix.data() + 4, prefix.size() - 4,
                    p.prefix_len - 8, p.mask1, p.target1);

    // SHA-1 state after compressing all full blocks before the salt region.
    int salt_block = (tpl.salt_offset / 64) * 64;
    SHA_CTX ctx;
    SHA1_Init(&ctx);
    if (salt_block > 0) SHA1_Update(&ctx, tpl.bytes.data(), salt_block);
    p.pre_state[0] = ctx.h0; p.pre_state[1] = ctx.h1; p.pre_state[2] = ctx.h2;
    p.pre_state[3] = ctx.h3; p.pre_state[4] = ctx.h4;

    // Final block W[12..15]: salt zeroed, 0x80 terminator, big-endian bit length.
    uint32_t total = tpl.bytes.size(), avail = total - salt_block;
    uint8_t block[64] = {};
    memcpy(block, tpl.bytes.data() + salt_block, avail);
    memset(block + (tpl.salt_offset - salt_block), 0, kSaltBytes);
    block[avail] = 0x80;
    uint64_t bits = uint64_t(total) * 8;
    for (int i = 0; i < 8; ++i) block[63 - i] = uint8_t(bits >> (i * 8));
    for (int i = 0; i < 4; ++i) {
        const uint8_t* b = block + (12 + i) * 4;
        p.tail_w[i] = (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16)
                    | (uint32_t(b[2]) <<  8) |  uint32_t(b[3]);
    }
    return p;
}

// Prefer the precompiled metallib next to the binary; fall back to source.
static id<MTLComputePipelineState> create_pipeline(id<MTLDevice> dev) {
    NSError* err = nil;
    NSString* libp = [[[[NSBundle mainBundle] executablePath] stringByDeletingLastPathComponent]
                      stringByAppendingPathComponent:@"sha1.metallib"];
    NSFileManager* fm = [NSFileManager defaultManager];
    id<MTLLibrary> lib = nil;
    if ([fm fileExistsAtPath:libp] && [[fm attributesOfItemAtPath:libp error:nil] fileSize] > 0)
        lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:libp] error:&err];
    if (!lib) {
        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        opts.fastMathEnabled = YES;
        lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kMetalShaderSource]
                                options:opts error:&err];
    }
    if (!lib) { fprintf(stderr, "Shader Error: %s\n", err.localizedDescription.UTF8String); return nil; }
    auto pipe = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"bruteforce_sha1"]
                                                   error:&err];
    if (!pipe) fprintf(stderr, "Pipeline Error: %s\n", err.localizedDescription.UTF8String);
    return pipe;
}

// Result buffer layout: [found:u32][salt:u64][hash:20]. Returns true on hit.
static bool read_hit(id<MTLBuffer> buf, uint64_t& salt, HexDigest& hex) {
    auto* r = static_cast<const uint8_t*>(buf.contents);
    uint32_t found; memcpy(&found, r, 4);
    if (!found) return false;
    memcpy(&salt, r + 4, 8);
    static constexpr char hx[] = "0123456789abcdef";
    for (int i = 0; i < 20; ++i) {
        hex[i*2    ] = hx[r[12 + i] >> 4];
        hex[i*2 + 1] = hx[r[12 + i] & 0xF];
    }
    return true;
}

} // namespace

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
        GPUParams base = build_params(tpl, prefix_hex);

        auto par_opts = MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined
                      | MTLResourceHazardTrackingModeUntracked;
        auto res_opts = MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked;
        id<MTLBuffer>        par[kInFlight], res[kInFlight];
        id<MTLCommandBuffer> cb [kInFlight];
        for (int i = 0; i < kInFlight; ++i) {
            par[i] = [dev newBufferWithLength:sizeof(GPUParams) options:par_opts];
            res[i] = [dev newBufferWithLength:kResultSize       options:res_opts];
        }

        auto fire = [&](int i, uint64_t salt) {
            memset(res[i].contents, 0, kResultSize);
            auto* p = static_cast<GPUParams*>(par[i].contents);
            *p = base; p->base_salt = salt;
            cb[i] = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb[i] computeCommandEncoder];
            [enc setComputePipelineState:pipe];
            [enc setBuffer:par[i] offset:0 atIndex:0];
            [enc setBuffer:res[i] offset:0 atIndex:1];
            [enc dispatchThreads:MTLSizeMake(kThreads, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(kGroupSize, 1, 1)];
            [enc endEncoding];
            [cb[i] commit];
        };

        uint64_t next_salt = 0;
        for (int i = 0; i < kInFlight; ++i) { fire(i, next_salt); next_salt += kThreads; }

        auto t0 = std::chrono::steady_clock::now();
        auto elapsed = [&] {
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        };
        int progress_sec = 0;

        for (int slot = 0; ; slot = (slot + 1) % kInFlight) {
            [cb[slot] waitUntilCompleted];

            uint64_t salt;
            HexDigest hex;
            if (read_hit(res[slot], salt, hex)) {
                for (int i = 0; i < kInFlight; ++i) {
                    if (i == slot) continue;
                    [cb[i] waitUntilCompleted];
                    uint64_t os; HexDigest oh;
                    if (read_hit(res[i], os, oh) && os < salt) { salt = os; hex = oh; }
                }
                auto stamped = tpl; stamped.set_salt(salt);
                double secs = elapsed();
                if (progress_sec) fputc('\n', stderr);
                fprintf(stderr, "Found          %.40s\nTime           %.2fs\n"
                                "Throughput     %.2f GH/s\nHashes Tried   %.2fG\n",
                        hex.data(), secs, double(next_salt)/secs/1e9, double(next_salt)/1e9);
                return SolveResult{stamped.payload(), hex};
            }

            if (next_salt > kMaxSalt - kThreads) {
                for (int i = 0; i < kInFlight; ++i) [cb[i] waitUntilCompleted];
                return std::nullopt;
            }
            fire(slot, next_salt);
            next_salt += kThreads;

            int secs = int(elapsed());
            if (secs > progress_sec) {
                progress_sec = secs;
                fprintf(stderr, "%.2fG hashes | %.2f GH/s | %ds elapsed\n",
                        double(next_salt)/1e9, double(next_salt)/secs/1e9, secs);
            }
        }
    }
}
