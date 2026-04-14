// cpu_solver_shani.cpp - SHA-NI backend: 4-way interleaved SHA1

#include "cpu_solver_common.h"
#include <immintrin.h>
#include <atomic>

#define R0_3(P)   P##E=_mm_add_epi32(P##E,P##M0);P##E1=P##A;P##A=_mm_sha1rnds4_epu32(P##A,P##E,0);
#define R4_7(P)   P##E1=_mm_sha1nexte_epu32(P##E1,P##M1);P##E=P##A;P##A=_mm_sha1rnds4_epu32(P##A,P##E1,0);P##M0=_mm_sha1msg1_epu32(P##M0,P##M1);
#define R8_11(P)  P##E=_mm_sha1nexte_epu32(P##E,P##M2);P##E1=P##A;P##A=_mm_sha1rnds4_epu32(P##A,P##E,0);P##M1=_mm_sha1msg1_epu32(P##M1,P##M2);P##M0=_mm_xor_si128(P##M0,P##M2);
#define R12_15(P) P##E1=_mm_sha1nexte_epu32(P##E1,P##M3);P##E=P##A;P##M0=_mm_sha1msg2_epu32(P##M0,P##M3);P##A=_mm_sha1rnds4_epu32(P##A,P##E1,0);P##M2=_mm_sha1msg1_epu32(P##M2,P##M3);P##M1=_mm_xor_si128(P##M1,P##M3);
#define RE(P,fn) P##E=_mm_sha1nexte_epu32(P##E,P##M0);P##E1=P##A;P##M1=_mm_sha1msg2_epu32(P##M1,P##M0);P##A=_mm_sha1rnds4_epu32(P##A,P##E,fn);P##M3=_mm_sha1msg1_epu32(P##M3,P##M0);P##M2=_mm_xor_si128(P##M2,P##M0);
#define RO(P,fn) P##E1=_mm_sha1nexte_epu32(P##E1,P##M1);P##E=P##A;P##M2=_mm_sha1msg2_epu32(P##M2,P##M1);P##A=_mm_sha1rnds4_epu32(P##A,P##E1,fn);P##M0=_mm_sha1msg1_epu32(P##M0,P##M1);P##M3=_mm_xor_si128(P##M3,P##M1);
#define RE2(P,fn) P##E=_mm_sha1nexte_epu32(P##E,P##M2);P##E1=P##A;P##M3=_mm_sha1msg2_epu32(P##M3,P##M2);P##A=_mm_sha1rnds4_epu32(P##A,P##E,fn);P##M1=_mm_sha1msg1_epu32(P##M1,P##M2);P##M0=_mm_xor_si128(P##M0,P##M2);
#define RO2(P,fn) P##E1=_mm_sha1nexte_epu32(P##E1,P##M3);P##E=P##A;P##M0=_mm_sha1msg2_epu32(P##M0,P##M3);P##A=_mm_sha1rnds4_epu32(P##A,P##E1,fn);P##M2=_mm_sha1msg1_epu32(P##M2,P##M3);P##M1=_mm_xor_si128(P##M1,P##M3);
#define R64(P) P##E=_mm_sha1nexte_epu32(P##E,P##M0);P##E1=P##A;P##M1=_mm_sha1msg2_epu32(P##M1,P##M0);P##A=_mm_sha1rnds4_epu32(P##A,P##E,3);P##M3=_mm_sha1msg1_epu32(P##M3,P##M0);P##M2=_mm_xor_si128(P##M2,P##M0);
#define R68(P) P##E1=_mm_sha1nexte_epu32(P##E1,P##M1);P##E=P##A;P##M2=_mm_sha1msg2_epu32(P##M2,P##M1);P##A=_mm_sha1rnds4_epu32(P##A,P##E1,3);P##M3=_mm_xor_si128(P##M3,P##M1);
#define R72(P) P##E=_mm_sha1nexte_epu32(P##E,P##M2);P##E1=P##A;P##M3=_mm_sha1msg2_epu32(P##M3,P##M2);P##A=_mm_sha1rnds4_epu32(P##A,P##E,3);
#define R76(P) P##E1=_mm_sha1nexte_epu32(P##E1,P##M3);P##E=P##A;P##A=_mm_sha1rnds4_epu32(P##A,P##E1,3);
#define RFIN_ABCD(P) P##A=_mm_add_epi32(P##A,ABCD_INIT);

#define SHA1_80(M0,M1,M2,M3,AI,EI,AS,ES,ABCD,E0,E1) \
    ABCD=AI;E0=EI;AS=ABCD;ES=E0; \
    E0=_mm_add_epi32(E0,M0);E1=ABCD;ABCD=_mm_sha1rnds4_epu32(ABCD,E0,0); \
    E1=_mm_sha1nexte_epu32(E1,M1);E0=ABCD;ABCD=_mm_sha1rnds4_epu32(ABCD,E1,0);M0=_mm_sha1msg1_epu32(M0,M1); \
    E0=_mm_sha1nexte_epu32(E0,M2);E1=ABCD;ABCD=_mm_sha1rnds4_epu32(ABCD,E0,0);M1=_mm_sha1msg1_epu32(M1,M2);M0=_mm_xor_si128(M0,M2); \
    E1=_mm_sha1nexte_epu32(E1,M3);E0=ABCD;M0=_mm_sha1msg2_epu32(M0,M3);ABCD=_mm_sha1rnds4_epu32(ABCD,E1,0);M2=_mm_sha1msg1_epu32(M2,M3);M1=_mm_xor_si128(M1,M3); \
    E0=_mm_sha1nexte_epu32(E0,M0);E1=ABCD;M1=_mm_sha1msg2_epu32(M1,M0);ABCD=_mm_sha1rnds4_epu32(ABCD,E0,0);M3=_mm_sha1msg1_epu32(M3,M0);M2=_mm_xor_si128(M2,M0); \
    E1=_mm_sha1nexte_epu32(E1,M1);E0=ABCD;M2=_mm_sha1msg2_epu32(M2,M1);ABCD=_mm_sha1rnds4_epu32(ABCD,E1,1);M0=_mm_sha1msg1_epu32(M0,M1);M3=_mm_xor_si128(M3,M1); \
    E0=_mm_sha1nexte_epu32(E0,M2);E1=ABCD;M3=_mm_sha1msg2_epu32(M3,M2);ABCD=_mm_sha1rnds4_epu32(ABCD,E0,1);M1=_mm_sha1msg1_epu32(M1,M2);M0=_mm_xor_si128(M0,M2); \
    E1=_mm_sha1nexte_epu32(E1,M3);E0=ABCD;M0=_mm_sha1msg2_epu32(M0,M3);ABCD=_mm_sha1rnds4_epu32(ABCD,E1,1);M2=_mm_sha1msg1_epu32(M2,M3);M1=_mm_xor_si128(M1,M3); \
    E0=_mm_sha1nexte_epu32(E0,M0);E1=ABCD;M1=_mm_sha1msg2_epu32(M1,M0);ABCD=_mm_sha1rnds4_epu32(ABCD,E0,1);M3=_mm_sha1msg1_epu32(M3,M0);M2=_mm_xor_si128(M2,M0); \
    E1=_mm_sha1nexte_epu32(E1,M1);E0=ABCD;M2=_mm_sha1msg2_epu32(M2,M1);ABCD=_mm_sha1rnds4_epu32(ABCD,E1,1);M0=_mm_sha1msg1_epu32(M0,M1);M3=_mm_xor_si128(M3,M1); \
    E0=_mm_sha1nexte_epu32(E0,M2);E1=ABCD;M3=_mm_sha1msg2_epu32(M3,M2);ABCD=_mm_sha1rnds4_epu32(ABCD,E0,2);M1=_mm_sha1msg1_epu32(M1,M2);M0=_mm_xor_si128(M0,M2); \
    E1=_mm_sha1nexte_epu32(E1,M3);E0=ABCD;M0=_mm_sha1msg2_epu32(M0,M3);ABCD=_mm_sha1rnds4_epu32(ABCD,E1,2);M2=_mm_sha1msg1_epu32(M2,M3);M1=_mm_xor_si128(M1,M3); \
    E0=_mm_sha1nexte_epu32(E0,M0);E1=ABCD;M1=_mm_sha1msg2_epu32(M1,M0);ABCD=_mm_sha1rnds4_epu32(ABCD,E0,2);M3=_mm_sha1msg1_epu32(M3,M0);M2=_mm_xor_si128(M2,M0); \
    E1=_mm_sha1nexte_epu32(E1,M1);E0=ABCD;M2=_mm_sha1msg2_epu32(M2,M1);ABCD=_mm_sha1rnds4_epu32(ABCD,E1,2);M0=_mm_sha1msg1_epu32(M0,M1);M3=_mm_xor_si128(M3,M1); \
    E0=_mm_sha1nexte_epu32(E0,M2);E1=ABCD;M3=_mm_sha1msg2_epu32(M3,M2);ABCD=_mm_sha1rnds4_epu32(ABCD,E0,2);M1=_mm_sha1msg1_epu32(M1,M2);M0=_mm_xor_si128(M0,M2); \
    E1=_mm_sha1nexte_epu32(E1,M3);E0=ABCD;M0=_mm_sha1msg2_epu32(M0,M3);ABCD=_mm_sha1rnds4_epu32(ABCD,E1,3);M2=_mm_sha1msg1_epu32(M2,M3);M1=_mm_xor_si128(M1,M3); \
    E0=_mm_sha1nexte_epu32(E0,M0);E1=ABCD;M1=_mm_sha1msg2_epu32(M1,M0);ABCD=_mm_sha1rnds4_epu32(ABCD,E0,3);M3=_mm_sha1msg1_epu32(M3,M0);M2=_mm_xor_si128(M2,M0); \
    E1=_mm_sha1nexte_epu32(E1,M1);E0=ABCD;M2=_mm_sha1msg2_epu32(M2,M1);ABCD=_mm_sha1rnds4_epu32(ABCD,E1,3);M3=_mm_xor_si128(M3,M1); \
    E0=_mm_sha1nexte_epu32(E0,M2);E1=ABCD;M3=_mm_sha1msg2_epu32(M3,M2);ABCD=_mm_sha1rnds4_epu32(ABCD,E0,3); \
    E1=_mm_sha1nexte_epu32(E1,M3);E0=ABCD;ABCD=_mm_sha1rnds4_epu32(ABCD,E1,3); \
    E0=_mm_sha1nexte_epu32(E0,ES);ABCD=_mm_add_epi32(ABCD,AS);

__attribute__((target("sha,sse4.1,ssse3")))
static inline bool check_and_store(__m128i A, __m128i E_before_fin, __m128i E0_INIT,
                                   uint32_t mask0, uint32_t target0,
                                   uint32_t mask1, uint32_t target1,
                                   uint32_t pn, uint64_t salt,
                                   uint64_t salt_start,
                                   std::atomic<bool>& global_found,
                                   WorkerResult& result) {
    __m128i ABCD_out = _mm_shuffle_epi32(A, 0x1B);
    uint32_t h0 = (uint32_t)_mm_extract_epi32(ABCD_out, 0);
    if (__builtin_expect((h0 & mask0) == target0, 0)) {
        if (pn <= 8 || ((uint32_t)_mm_extract_epi32(ABCD_out, 1) & mask1) == target1) {
            __m128i E_final = _mm_sha1nexte_epu32(E_before_fin, E0_INIT);
            result.salt = salt;
            _mm_storeu_si128((__m128i*)result.hash, ABCD_out);
            result.hash[4] = (uint32_t)_mm_extract_epi32(E_final, 3);
            result.found = true;
            result.hashes = salt - salt_start + 1;
            global_found.store(true, std::memory_order_relaxed);
            return true;
        }
    }
    return false;
}

__attribute__((target("sha,sse4.1,ssse3")))
void sha_ni_worker(const CPUParams& p, uint64_t salt_start, uint64_t salt_end,
                          std::atomic<bool>& global_found, WorkerResult& result) {
    result.found = false;
    result.hashes = 0;

    const __m128i ABCD_INIT = _mm_shuffle_epi32(
            _mm_setr_epi32((int)p.pre_state[0], (int)p.pre_state[1],
                           (int)p.pre_state[2], (int)p.pre_state[3]), 0x1B);
        const __m128i E0_INIT = _mm_set_epi32((int)p.pre_state[4], 0, 0, 0);
        const __m128i MSG3_C = p.msg3_const;
        const uint32_t mask0 = p.mask0, target0 = p.target0;
        const uint32_t mask1 = p.mask1, target1 = p.target1;
        const uint32_t pn = p.prefix_len;

        uint64_t salt = salt_start;
        while (salt < salt_end) {
            __m128i M0_base = _mm_set_epi32(
                (int)salt_lut[(salt >> 44) & 0xF], (int)salt_lut[(salt >> 40) & 0xF],
                (int)salt_lut[(salt >> 36) & 0xF], (int)salt_lut[(salt >> 32) & 0xF]);
            __m128i M1_base = _mm_set_epi32(
                (int)salt_lut[(salt >> 28) & 0xF], (int)salt_lut[(salt >> 24) & 0xF],
                (int)salt_lut[(salt >> 20) & 0xF], (int)salt_lut[(salt >> 16) & 0xF]);

            uint64_t inner_end = (salt | 0xFFFF) + 1;
            if (inner_end > salt_end) inner_end = salt_end;

            if (global_found.load(std::memory_order_relaxed)) return;

            uint64_t s = salt;
            uint64_t quad_end = s + ((inner_end - s) & ~3ULL);

            for (; s < quad_end; s += 4) {
                __m128i aM0=M0_base,aM1=M1_base,aM3=MSG3_C;
                __m128i bM0=M0_base,bM1=M1_base,bM3=MSG3_C;
                __m128i cM0=M0_base,cM1=M1_base,cM3=MSG3_C;
                __m128i dM0=M0_base,dM1=M1_base,dM3=MSG3_C;

                __m128i aM2=_mm_set_epi32((int)salt_lut[(s>>12)&0xF],(int)salt_lut[(s>>8)&0xF],(int)salt_lut[(s>>4)&0xF],(int)salt_lut[s&0xF]);
                uint64_t s1=s+1;
                __m128i bM2=_mm_set_epi32((int)salt_lut[(s1>>12)&0xF],(int)salt_lut[(s1>>8)&0xF],(int)salt_lut[(s1>>4)&0xF],(int)salt_lut[s1&0xF]);
                uint64_t s2=s+2;
                __m128i cM2=_mm_set_epi32((int)salt_lut[(s2>>12)&0xF],(int)salt_lut[(s2>>8)&0xF],(int)salt_lut[(s2>>4)&0xF],(int)salt_lut[s2&0xF]);
                uint64_t s3=s+3;
                __m128i dM2=_mm_set_epi32((int)salt_lut[(s3>>12)&0xF],(int)salt_lut[(s3>>8)&0xF],(int)salt_lut[(s3>>4)&0xF],(int)salt_lut[s3&0xF]);

                __m128i aA=ABCD_INIT,aE=E0_INIT,aE1;
                __m128i bA=ABCD_INIT,bE=E0_INIT,bE1;
                __m128i cA=ABCD_INIT,cE=E0_INIT,cE1;
                __m128i dA=ABCD_INIT,dE=E0_INIT,dE1;

                R0_3(a) R0_3(b) R0_3(c) R0_3(d)
                R4_7(a) R4_7(b) R4_7(c) R4_7(d)
                R8_11(a) R8_11(b) R8_11(c) R8_11(d)
                R12_15(a) R12_15(b) R12_15(c) R12_15(d)
                RE(a,0) RE(b,0) RE(c,0) RE(d,0)
                RO(a,1) RO(b,1) RO(c,1) RO(d,1)
                RE2(a,1) RE2(b,1) RE2(c,1) RE2(d,1)
                RO2(a,1) RO2(b,1) RO2(c,1) RO2(d,1)
                RE(a,1) RE(b,1) RE(c,1) RE(d,1)
                RO(a,1) RO(b,1) RO(c,1) RO(d,1)
                RE2(a,2) RE2(b,2) RE2(c,2) RE2(d,2)
                RO2(a,2) RO2(b,2) RO2(c,2) RO2(d,2)
                RE(a,2) RE(b,2) RE(c,2) RE(d,2)
                RO(a,2) RO(b,2) RO(c,2) RO(d,2)
                RE2(a,2) RE2(b,2) RE2(c,2) RE2(d,2)
                RO2(a,3) RO2(b,3) RO2(c,3) RO2(d,3)
                R64(a) R64(b) R64(c) R64(d)
                R68(a) R68(b) R68(c) R68(d)
                R72(a) R72(b) R72(c) R72(d)
                R76(a) R76(b) R76(c) R76(d)
                RFIN_ABCD(a) RFIN_ABCD(b) RFIN_ABCD(c) RFIN_ABCD(d)

                if (check_and_store(aA, aE, E0_INIT, mask0, target0, mask1, target1, pn, s,   salt_start, global_found, result)) return;
                if (check_and_store(bA, bE, E0_INIT, mask0, target0, mask1, target1, pn, s+1, salt_start, global_found, result)) return;
                if (check_and_store(cA, cE, E0_INIT, mask0, target0, mask1, target1, pn, s+2, salt_start, global_found, result)) return;
                if (check_and_store(dA, dE, E0_INIT, mask0, target0, mask1, target1, pn, s+3, salt_start, global_found, result)) return;
            }

            for (; s < inner_end; ++s) {
                __m128i M0=M0_base,M1=M1_base,M3=MSG3_C;
                __m128i M2=_mm_set_epi32((int)salt_lut[(s>>12)&0xF],(int)salt_lut[(s>>8)&0xF],(int)salt_lut[(s>>4)&0xF],(int)salt_lut[s&0xF]);
                __m128i ABCD,E0,E1,AS,ES;
                SHA1_80(M0,M1,M2,M3,ABCD_INIT,E0_INIT,AS,ES,ABCD,E0,E1);
                ABCD=_mm_shuffle_epi32(ABCD,0x1B);
                uint32_t h0=(uint32_t)_mm_extract_epi32(ABCD,0);
                if (__builtin_expect((h0&mask0)==target0,0)) {
                    if (pn<=8||((uint32_t)_mm_extract_epi32(ABCD,1)&mask1)==target1) {
                        result.salt=s;_mm_storeu_si128((__m128i*)result.hash,ABCD);
                        result.hash[4]=(uint32_t)_mm_extract_epi32(E0,3);
                        result.found=true;result.hashes=s-salt_start+1;
                        global_found.store(true,std::memory_order_relaxed);return;
                    }
                }
            }
            salt = s;
            result.hashes = salt - salt_start;
        }
}
