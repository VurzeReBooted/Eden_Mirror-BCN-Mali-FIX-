// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Small, allocation free ASTC LDR 4x4 encoder used to re-encode BCn textures for GPUs that can
// sample ASTC but not BCn. It only emits a restricted subset of the format (one partition, 4x4
// weight grid, direct endpoint modes 0/4/8/12, void-extent blocks); the layout of this subset was
// verified against Eden's ASTC decoder (astc.cpp).

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>

#include "video_core/textures/astc_encode.h"

namespace Tegra::Texture::ASTC {
namespace {

constexpr u32 BlockTexels = 16;
// Sum of squared errors (16 texels, all channels) below which further configs are not tried.
constexpr u32 GoodEnoughError = 24;

// ------------------------------------------------------------------------------------------
// Static tables
// ------------------------------------------------------------------------------------------

enum QKind : u8 { Bits8 = 0, Trit6 = 1, Trit4 = 2 };

/// Nearest-value lookup for one endpoint range. code = (trit digit << 6) | low bits, or the
/// plain 8 bit value for Bits8.
struct QTab {
    std::array<u8, 256> val{};
    std::array<u8, 256> code{};
};

struct LevelSet {
    u32 levels = 0;
    u32 nbits = 0;
    std::array<u8, 16> uw{};      // unquantized weights 0..64
    std::array<u8, 65> nearest{}; // weight index whose unquantized weight is closest to 0..64
};

constexpr u32 ReplicateBits(u32 v, u32 nb, u32 to) {
    u32 r = 0;
    int sh = static_cast<int>(to) - static_cast<int>(nb);
    v &= (1U << nb) - 1;
    while (sh > -static_cast<int>(nb)) {
        r |= sh >= 0 ? (v << sh) : (v >> -sh);
        sh -= static_cast<int>(nb);
    }
    return r;
}

// Trit endpoint unquantization (mirrors the decoder).
u32 UnquantTrit(u32 nb, u32 m, u32 d) {
    const u32 a = (m & 1) ? 0x1FF : 0;
    u32 b = 0;
    u32 c = 0;
    switch (nb) {
    case 4: {
        c = 22;
        const u32 dcb = (m >> 1) & 7;
        b = (dcb << 6) | dcb;
        break;
    }
    case 6: {
        c = 5;
        const u32 f = (m >> 1) & 0x1F;
        b = (f << 4) | (f >> 4);
        break;
    }
    default:
        break;
    }
    u32 t = d * c + b;
    t ^= a;
    return (a & 0x80) | (t >> 2);
}

// Decodes a packed 8 bit trit block T into 5 trits (mirrors the decoder).
void TritsFromT(u32 T, u32 t[5]) {
    auto bit = [](u32 v, u32 i) { return (v >> i) & 1U; };
    auto rng = [](u32 v, u32 s, u32 e) { return (v >> s) & ((1U << (e - s + 1)) - 1U); };
    u32 C;
    if (rng(T, 2, 4) == 7) {
        C = (rng(T, 5, 7) << 2) | rng(T, 0, 1);
        t[4] = t[3] = 2;
    } else {
        C = rng(T, 0, 4);
        if (rng(T, 5, 6) == 3) {
            t[4] = 2;
            t[3] = bit(T, 7);
        } else {
            t[4] = bit(T, 7);
            t[3] = rng(T, 5, 6);
        }
    }
    if (rng(C, 0, 1) == 3) {
        t[2] = 2;
        t[1] = bit(C, 4);
        t[0] = (bit(C, 3) << 1) | (bit(C, 2) & ~bit(C, 3) & 1U);
    } else if (rng(C, 2, 3) == 3) {
        t[2] = 2;
        t[1] = 2;
        t[0] = rng(C, 0, 1);
    } else {
        t[2] = bit(C, 4);
        t[1] = rng(C, 2, 3);
        t[0] = (bit(C, 1) << 1) | (bit(C, 0) & ~bit(C, 1) & 1U);
    }
}

struct Tables {
    std::array<QTab, 3> q;
    std::array<LevelSet, 17> ls; // indexed by level count (2, 4, 8, 16)
    std::array<u8, 243> trit_inv{};

    Tables() {
        for (u32 v = 0; v < 256; ++v) {
            q[Bits8].val[v] = static_cast<u8>(v);
            q[Bits8].code[v] = static_cast<u8>(v);
        }
        BuildTrit(Trit6, 6);
        BuildTrit(Trit4, 4);

        for (u32 levels : {2U, 4U, 8U, 16U}) {
            LevelSet& s = ls[levels];
            s.levels = levels;
            s.nbits = levels == 2 ? 1 : levels == 4 ? 2 : levels == 8 ? 3 : 4;
            for (u32 i = 0; i < levels; ++i) {
                const u32 r = ReplicateBits(i, s.nbits, 6);
                s.uw[i] = static_cast<u8>(r > 32 ? r + 1 : r);
            }
            for (u32 pos = 0; pos <= 64; ++pos) {
                u32 best = 0;
                for (u32 i = 1; i < levels; ++i) {
                    if (std::abs(static_cast<int>(s.uw[i]) - static_cast<int>(pos)) <
                        std::abs(static_cast<int>(s.uw[best]) - static_cast<int>(pos))) {
                        best = i;
                    }
                }
                s.nearest[pos] = static_cast<u8>(best);
            }
        }
        // Smallest T for every trit tuple guarantees the bits dropped from a partial group are 0.
        for (u32 T = 256; T-- > 0;) {
            u32 t[5];
            TritsFromT(T, t);
            trit_inv[t[0] + 3 * t[1] + 9 * t[2] + 27 * t[3] + 81 * t[4]] = static_cast<u8>(T);
        }
    }

    void BuildTrit(QKind kind, u32 nb) {
        std::array<u8, 192> values{};
        std::array<u8, 192> codes{};
        u32 count = 0;
        for (u32 d = 0; d < 3; ++d) {
            for (u32 m = 0; m < (1U << nb); ++m) {
                values[count] = static_cast<u8>(UnquantTrit(nb, m, d));
                codes[count] = static_cast<u8>((d << 6) | m);
                ++count;
            }
        }
        for (u32 v = 0; v < 256; ++v) {
            u32 best = 0;
            int best_d = 1 << 20;
            for (u32 i = 0; i < count; ++i) {
                const int dd = std::abs(static_cast<int>(values[i]) - static_cast<int>(v));
                if (dd < best_d) {
                    best_d = dd;
                    best = i;
                }
            }
            q[kind].val[v] = values[best];
            q[kind].code[v] = codes[best];
        }
    }
};

const Tables& GetTables() {
    static const Tables tables;
    return tables;
}

// ------------------------------------------------------------------------------------------
// Block configurations
// ------------------------------------------------------------------------------------------

struct Cfg {
    bool dual;
    u8 levels;
    u8 cem;
    QKind qk;
};

constexpr Cfg kGray16{false, 16, 0, Bits8};
constexpr Cfg kGrayA16{false, 16, 4, Bits8};
constexpr Cfg kGrayADual4{true, 4, 4, Bits8};
constexpr Cfg kRgb8{false, 8, 8, Bits8};
constexpr Cfg kRgb16{false, 16, 8, Trit6};
constexpr Cfg kRgb4{false, 4, 8, Bits8};
constexpr Cfg kRgba8{false, 8, 12, Trit6};
constexpr Cfg kRgba4{false, 4, 12, Bits8};
constexpr Cfg kRgbaDual4{true, 4, 12, Trit4};
constexpr Cfg kRgbaDual2{true, 2, 12, Bits8};

constexpr u32 NumValues(u32 cem) {
    return ((cem >> 2) + 1) * 2;
}

constexpr u32 WeightBits(const Cfg& c) {
    const u32 nb = c.levels == 2 ? 1 : c.levels == 4 ? 2 : c.levels == 8 ? 3 : 4;
    return nb * BlockTexels * (c.dual ? 2 : 1);
}

constexpr u32 BlockMode(const Cfg& c) {
    switch (c.levels) {
    case 2:
        return 0x441; // dual plane only (single plane 1 bit weights are illegal for 4x4)
    case 4:
        return c.dual ? 0x442 : 0x042;
    case 8:
        return c.dual ? 0x453 : 0x053;
    default:
        return 0x242; // single plane only (dual plane 16 levels is illegal)
    }
}

// ------------------------------------------------------------------------------------------
// Bit writer
// ------------------------------------------------------------------------------------------

struct Bits128 {
    u64 lo = 0;
    u64 hi = 0;

    // n <= 32, pos + n <= 128
    void Put(u32 pos, u64 val, u32 n) {
        val &= (u64{1} << n) - 1;
        if (pos < 64) {
            lo |= val << pos;
            if (pos + n > 64) {
                hi |= val >> (64 - pos);
            }
        } else {
            hi |= val << (pos - 64);
        }
    }

    void Store(u8* out) const {
        for (u32 i = 0; i < 8; ++i) {
            out[i] = static_cast<u8>(lo >> (8 * i));
            out[8 + i] = static_cast<u8>(hi >> (8 * i));
        }
    }
};

constexpr u64 Reverse64(u64 v) {
    v = ((v >> 1) & 0x5555555555555555ULL) | ((v & 0x5555555555555555ULL) << 1);
    v = ((v >> 2) & 0x3333333333333333ULL) | ((v & 0x3333333333333333ULL) << 2);
    v = ((v >> 4) & 0x0F0F0F0F0F0F0F0FULL) | ((v & 0x0F0F0F0F0F0F0F0FULL) << 4);
    v = ((v >> 8) & 0x00FF00FF00FF00FFULL) | ((v & 0x00FF00FF00FF00FFULL) << 8);
    v = ((v >> 16) & 0x0000FFFF0000FFFFULL) | ((v & 0x0000FFFF0000FFFFULL) << 16);
    return (v >> 32) | (v << 32);
}

void WriteVoidExtent(u8* out, u32 r, u32 g, u32 b, u32 a) {
    Bits128 bits;
    bits.Put(0, 0x1FC, 9);
    bits.Put(10, 3, 2);
    for (u32 i = 0; i < 4; ++i) {
        bits.Put(12 + 13 * i, 0x1FFF, 13);
    }
    bits.Put(64, r * 257, 16);
    bits.Put(80, g * 257, 16);
    bits.Put(96, b * 257, 16);
    bits.Put(112, a * 257, 16);
    bits.Store(out);
}

/// One chosen encoding of a block. Endpoint codes are in wire order: for every channel of the
/// colour endpoint mode, (endpoint 0, endpoint 1). Decoded endpoint values are kept alongside.
struct Cand {
    Cfg cfg{};
    u32 err = ~0U;
    u8 code[8]{};
    int e0[4]{};
    int e1[4]{};
    u8 w[2][BlockTexels]{};
};

void WriteBlock(u8* out, const Cand& c) {
    const Tables& t = GetTables();
    const Cfg& cfg = c.cfg;
    const LevelSet& ls = t.ls[cfg.levels];
    const u32 wbits = WeightBits(cfg);
    const u32 avail = 128 - wbits - 17 - (cfg.dual ? 2 : 0);
    const u32 nvals = NumValues(cfg.cem);

    Bits128 bits;
    bits.Put(0, BlockMode(cfg), 11);
    bits.Put(13, cfg.cem, 4);

    // Endpoint values
    u32 cursor = 17;
    const u32 limit = 17 + avail;
    auto append = [&](u32 val, u32 n) {
        if (cursor >= limit) {
            return;
        }
        n = std::min(n, limit - cursor);
        bits.Put(cursor, val, n);
        cursor += n;
    };
    if (cfg.qk == Bits8) {
        for (u32 i = 0; i < nvals; ++i) {
            append(c.code[i], 8);
        }
    } else {
        const u32 nb = cfg.qk == Trit6 ? 6 : 4;
        const u32 mask = (1U << nb) - 1;
        // `limit` truncates the last group to the decoder's bit count; the dropped bits are 0
        // because the smallest T is used for every trit tuple.
        for (u32 g = 0; g < nvals; g += 5) {
            u32 m[5]{};
            u32 d[5]{};
            for (u32 i = 0; i < 5 && g + i < nvals; ++i) {
                m[i] = c.code[g + i] & mask;
                d[i] = c.code[g + i] >> 6;
            }
            const u32 T = t.trit_inv[d[0] + 3 * d[1] + 9 * d[2] + 27 * d[3] + 81 * d[4]];
            append(m[0], nb);
            append(T & 3, 2);
            append(m[1], nb);
            append((T >> 2) & 3, 2);
            append(m[2], nb);
            append((T >> 4) & 1, 1);
            append(m[3], nb);
            append((T >> 5) & 3, 2);
            append(m[4], nb);
            append((T >> 7) & 1, 1);
        }
    }
    if (cfg.dual) {
        bits.Put(17 + avail, 3, 2); // CCS = 3: alpha has its own weights
    }

    // Weights: stream bit s lives at block bit 127 - s.
    u64 stream = 0;
    u32 pos = 0;
    for (u32 i = 0; i < BlockTexels; ++i) {
        for (u32 p = 0; p < (cfg.dual ? 2U : 1U); ++p) {
            stream |= static_cast<u64>(c.w[p][i]) << pos;
            pos += ls.nbits;
        }
    }
    bits.hi |= Reverse64(stream);
    bits.Store(out);
}

// ------------------------------------------------------------------------------------------
// Line fitting
// ------------------------------------------------------------------------------------------

inline int Lerp(int a, int b, int w) {
    const int c = (257 * (a * (64 - w) + b * w) + 32) >> 6;
    return (255 * c + 32768) >> 16;
}

struct LineFit {
    int e0[4]{};
    int e1[4]{};
    u8 c0[4]{};
    u8 c1[4]{};
    u8 w[BlockTexels]{};
    u32 err = ~0U;
};

/// Picks the weight closest to the projection of every texel on the (decoded) endpoint segment
/// and returns the squared error of the resulting decoded texels.
u32 Assign(const int (*pts)[4], int nc, const int* e0, const int* e1, const LevelSet& ls, u8* w) {
    int d[4];
    int den = 0;
    for (int c = 0; c < nc; ++c) {
        d[c] = e1[c] - e0[c];
        den += d[c] * d[c];
    }
    const float scale = den > 0 ? 64.0f / static_cast<float>(den) : 0.0f;
    u32 total = 0;
    for (u32 i = 0; i < BlockTexels; ++i) {
        int dot = 0;
        for (int c = 0; c < nc; ++c) {
            dot += (pts[i][c] - e0[c]) * d[c];
        }
        const int pos = std::clamp(static_cast<int>(static_cast<float>(dot) * scale + 0.5f), 0, 64);
        const u32 k = ls.nearest[pos];
        w[i] = static_cast<u8>(k);
        for (int c = 0; c < nc; ++c) {
            const int df = pts[i][c] - Lerp(e0[c], e1[c], ls.uw[k]);
            total += static_cast<u32>(df * df);
        }
    }
    return total;
}

void FitLine(const int (*pts)[4], int nc, const LevelSet& ls, const QTab& qt, LineFit& out) {
    float mean[4]{};
    int mn[4];
    int mx[4];
    for (int c = 0; c < nc; ++c) {
        mn[c] = 255;
        mx[c] = 0;
    }
    for (u32 i = 0; i < BlockTexels; ++i) {
        for (int c = 0; c < nc; ++c) {
            mean[c] += static_cast<float>(pts[i][c]);
            mn[c] = std::min(mn[c], pts[i][c]);
            mx[c] = std::max(mx[c], pts[i][c]);
        }
    }
    for (int c = 0; c < nc; ++c) {
        mean[c] *= 1.0f / 16.0f;
    }

    float f0[4];
    float f1[4];
    if (nc == 1) {
        f0[0] = static_cast<float>(mn[0]);
        f1[0] = static_cast<float>(mx[0]);
    } else {
        float cov[4][4]{};
        for (u32 i = 0; i < BlockTexels; ++i) {
            float dv[4];
            for (int c = 0; c < nc; ++c) {
                dv[c] = static_cast<float>(pts[i][c]) - mean[c];
            }
            for (int a = 0; a < nc; ++a) {
                for (int b = a; b < nc; ++b) {
                    cov[a][b] += dv[a] * dv[b];
                }
            }
        }
        for (int a = 0; a < nc; ++a) {
            for (int b = 0; b < a; ++b) {
                cov[a][b] = cov[b][a];
            }
        }
        float axis[4];
        for (int c = 0; c < nc; ++c) {
            axis[c] = static_cast<float>(mx[c] - mn[c]);
        }
        for (int it = 0; it < 4; ++it) {
            float nv[4]{};
            float big = 0.0f;
            for (int a = 0; a < nc; ++a) {
                for (int b = 0; b < nc; ++b) {
                    nv[a] += cov[a][b] * axis[b];
                }
                big = std::max(big, std::abs(nv[a]));
            }
            if (big <= 1e-6f) {
                break;
            }
            const float inv = 1.0f / big;
            for (int c = 0; c < nc; ++c) {
                axis[c] = nv[c] * inv;
            }
        }
        float tmin = 1e30f;
        float tmax = -1e30f;
        for (u32 i = 0; i < BlockTexels; ++i) {
            float t = 0.0f;
            for (int c = 0; c < nc; ++c) {
                t += (static_cast<float>(pts[i][c]) - mean[c]) * axis[c];
            }
            tmin = std::min(tmin, t);
            tmax = std::max(tmax, t);
        }
        float norm = 0.0f;
        for (int c = 0; c < nc; ++c) {
            norm += axis[c] * axis[c];
        }
        if (norm > 1e-12f) {
            const float inv = 1.0f / norm; // t above was a scaled projection
            tmin *= inv;
            tmax *= inv;
        }
        for (int c = 0; c < nc; ++c) {
            f0[c] = mean[c] + tmin * axis[c];
            f1[c] = mean[c] + tmax * axis[c];
        }
    }

    auto quantize = [&](float f) {
        return std::clamp(static_cast<int>(f + 0.5f), 0, 255);
    };

    LineFit best;
    for (int iter = 0; iter < 4; ++iter) {
        LineFit cur;
        for (int c = 0; c < nc; ++c) {
            const int q0 = quantize(f0[c]);
            const int q1 = quantize(f1[c]);
            cur.e0[c] = qt.val[q0];
            cur.e1[c] = qt.val[q1];
            cur.c0[c] = qt.code[q0];
            cur.c1[c] = qt.code[q1];
        }
        cur.err = Assign(pts, nc, cur.e0, cur.e1, ls, cur.w);
        if (cur.err >= best.err) {
            break;
        }
        best = cur;
        if (best.err == 0) {
            break;
        }
        // Least squares refit of the endpoints for the current weights.
        float saa = 0.0f;
        float sab = 0.0f;
        float sbb = 0.0f;
        for (u32 i = 0; i < BlockTexels; ++i) {
            const float s = static_cast<float>(ls.uw[best.w[i]]) * (1.0f / 64.0f);
            saa += (1.0f - s) * (1.0f - s);
            sab += (1.0f - s) * s;
            sbb += s * s;
        }
        const float det = saa * sbb - sab * sab;
        if (det < 1e-4f) {
            break;
        }
        const float idet = 1.0f / det;
        for (int c = 0; c < nc; ++c) {
            float sap = 0.0f;
            float sbp = 0.0f;
            for (u32 i = 0; i < BlockTexels; ++i) {
                const float s = static_cast<float>(ls.uw[best.w[i]]) * (1.0f / 64.0f);
                const float p = static_cast<float>(pts[i][c]);
                sap += (1.0f - s) * p;
                sbp += s * p;
            }
            f0[c] = (sbb * sap - sab * sbp) * idet;
            f1[c] = (saa * sbp - sab * sap) * idet;
        }
    }
    out = best;
}

// ------------------------------------------------------------------------------------------
// Generic block encoder
// ------------------------------------------------------------------------------------------

void ApplyOrdering(Cand& c) {
    if (c.cfg.cem < 8) {
        return;
    }
    // Blue contraction is used by the decoder iff sum(E1.rgb) < sum(E0.rgb): avoid it by swapping.
    const int s0 = c.e0[0] + c.e0[1] + c.e0[2];
    const int s1 = c.e1[0] + c.e1[1] + c.e1[2];
    if (s1 >= s0) {
        return;
    }
    const int nch = static_cast<int>(NumValues(c.cfg.cem) / 2);
    for (int i = 0; i < nch; ++i) {
        std::swap(c.e0[i], c.e1[i]);
        std::swap(c.code[2 * i], c.code[2 * i + 1]);
    }
    const u8 top = static_cast<u8>(c.cfg.levels - 1);
    for (u32 p = 0; p < 2; ++p) {
        for (u32 i = 0; i < BlockTexels; ++i) {
            c.w[p][i] = static_cast<u8>(top - c.w[p][i]);
        }
    }
}

/// Fits one configuration to the block, result is stored in `out` (ordering already applied).
void EvalConfig(const int (*px)[4], const Cfg& cfg, Cand& out) {
    const Tables& t = GetTables();
    const LevelSet& ls = t.ls[cfg.levels];
    const QTab& qt = t.q[cfg.qk];
    out = Cand{};
    out.cfg = cfg;

    int pts[BlockTexels][4];
    LineFit fit;
    auto store = [&](const LineFit& f, int nc, int first) {
        for (int c = 0; c < nc; ++c) {
            out.e0[first + c] = f.e0[c];
            out.e1[first + c] = f.e1[c];
            out.code[2 * (first + c)] = f.c0[c];
            out.code[2 * (first + c) + 1] = f.c1[c];
        }
    };

    if (!cfg.dual) {
        int nc;
        switch (cfg.cem) {
        case 0:
            nc = 1;
            for (u32 i = 0; i < BlockTexels; ++i) {
                pts[i][0] = px[i][1];
            }
            break;
        case 4:
            nc = 2;
            for (u32 i = 0; i < BlockTexels; ++i) {
                pts[i][0] = px[i][1];
                pts[i][1] = px[i][3];
            }
            break;
        case 8:
            nc = 3;
            for (u32 i = 0; i < BlockTexels; ++i) {
                std::memcpy(pts[i], px[i], 3 * sizeof(int));
            }
            break;
        default:
            nc = 4;
            std::memcpy(pts, px, sizeof(pts));
            break;
        }
        FitLine(pts, nc, ls, qt, fit);
        store(fit, nc, 0);
        std::memcpy(out.w[0], fit.w, BlockTexels);
        out.err = fit.err;
    } else {
        const int nc = cfg.cem == 4 ? 1 : 3;
        for (u32 i = 0; i < BlockTexels; ++i) {
            if (nc == 1) {
                pts[i][0] = px[i][1];
            } else {
                std::memcpy(pts[i], px[i], 3 * sizeof(int));
            }
        }
        FitLine(pts, nc, ls, qt, fit);
        store(fit, nc, 0);
        std::memcpy(out.w[0], fit.w, BlockTexels);
        out.err = fit.err;
        for (u32 i = 0; i < BlockTexels; ++i) {
            pts[i][0] = px[i][3];
        }
        LineFit afit;
        FitLine(pts, 1, ls, qt, afit);
        store(afit, 1, nc);
        std::memcpy(out.w[1], afit.w, BlockTexels);
        out.err += afit.err;
    }
    ApplyOrdering(out);
}

void EncodeBlock(const int (*px)[4], u8* out) {
    // Constant / grey / opaque classification
    bool constant = true;
    bool grey = true;
    bool opaque = true;
    for (u32 i = 0; i < BlockTexels; ++i) {
        constant &= px[i][0] == px[0][0] && px[i][1] == px[0][1] && px[i][2] == px[0][2] &&
                    px[i][3] == px[0][3];
        grey &= px[i][0] == px[i][1] && px[i][1] == px[i][2];
        opaque &= px[i][3] == 255;
    }
    if (constant) {
        WriteVoidExtent(out, static_cast<u32>(px[0][0]), static_cast<u32>(px[0][1]),
                        static_cast<u32>(px[0][2]), static_cast<u32>(px[0][3]));
        return;
    }

    std::array<Cfg, 4> list{};
    u32 count = 0;
    if (grey) {
        if (opaque) {
            list[count++] = kGray16;
        } else {
            list[count++] = kGrayA16;
            list[count++] = kGrayADual4;
        }
    } else if (opaque) {
        list[count++] = kRgb4;
        list[count++] = kRgb8;
        list[count++] = kRgb16;
    } else {
        list[count++] = kRgba4;
        list[count++] = kRgba8;
        list[count++] = kRgbaDual4;
        list[count++] = kRgbaDual2;
    }

    Cand best;
    Cand cur;
    for (u32 i = 0; i < count; ++i) {
        EvalConfig(px, list[i], cur);
        if (cur.err < best.err) {
            best = cur;
        }
        if (best.err <= GoodEnoughError) {
            break;
        }
    }
    WriteBlock(out, best);
}

bool CheckSizes(u64 in_bytes, u64 have_in, u64 out_bytes, u64 have_out, u32 w, u32 h, u32 s) {
    return w != 0 && h != 0 && s != 0 && have_in >= in_bytes && have_out >= out_bytes;
}

// ------------------------------------------------------------------------------------------
// BC1 helpers
// ------------------------------------------------------------------------------------------

void Expand565(u32 c, int rgb[3]) {
    rgb[0] = static_cast<int>(((c & 0xF800) >> 8) | ((c & 0xE000) >> 13));
    rgb[1] = static_cast<int>(((c & 0x07E0) >> 3) | ((c & 0x0600) >> 9));
    rgb[2] = static_cast<int>(((c & 0x001F) << 3) | ((c & 0x001C) >> 2));
}

/// Same arithmetic as the host BCn decoder (bc_decoder) for BC1 with punch-through alpha.
void DecodeBC1Block(const u8* blk, int px[BlockTexels][4]) {
    const u32 c0 = blk[0] | (blk[1] << 8);
    const u32 c1 = blk[2] | (blk[3] << 8);
    const u32 idx = blk[4] | (blk[5] << 8) | (blk[6] << 16) | (static_cast<u32>(blk[7]) << 24);
    int pal[4][4];
    Expand565(c0, pal[0]);
    Expand565(c1, pal[1]);
    pal[0][3] = pal[1][3] = pal[2][3] = pal[3][3] = 255;
    for (int k = 0; k < 3; ++k) {
        if (c0 > c1) {
            pal[2][k] = (pal[0][k] * 2 + pal[1][k]) / 3;
            pal[3][k] = (pal[1][k] * 2 + pal[0][k]) / 3;
        } else {
            pal[2][k] = (pal[0][k] + pal[1][k]) >> 1;
            pal[3][k] = 0;
        }
    }
    if (c0 <= c1) {
        pal[3][3] = 0;
    }
    for (u32 i = 0; i < BlockTexels; ++i) {
        std::memcpy(px[i], pal[(idx >> (2 * i)) & 3], 4 * sizeof(int));
    }
}

constexpr std::array<u8, 4> kBcToAstc{0, 3, 1, 2};

/// Direct 4-colour BC1 -> ASTC single plane, 4 level, CEM 8 block.
void TranscodeBC1Direct(const u8* blk, u8* out) {
    const Tables& t = GetTables();
    const LevelSet& ls = t.ls[4];
    const u32 c0 = blk[0] | (blk[1] << 8);
    const u32 c1 = blk[2] | (blk[3] << 8);
    const u32 idx = blk[4] | (blk[5] << 8) | (blk[6] << 16) | (static_cast<u32>(blk[7]) << 24);

    int ex0[3];
    int ex1[3];
    Expand565(c0, ex0);
    Expand565(c1, ex1);

    Cand cand;
    cand.cfg = Cfg{false, 4, 8, Bits8};
    u32 cnt[4]{};
    for (u32 i = 0; i < BlockTexels; ++i) {
        const u32 j = kBcToAstc[(idx >> (2 * i)) & 3];
        cand.w[0][i] = static_cast<u8>(j);
        ++cnt[j];
    }
    for (u32 j = 0; j < 4; ++j) {
        if (cnt[j] == BlockTexels) { // a single palette entry is used: exact constant block
            int p[3];
            for (int k = 0; k < 3; ++k) {
                const int pal[4] = {ex0[k], (2 * ex0[k] + ex1[k]) / 3, (ex0[k] + 2 * ex1[k]) / 3,
                                    ex1[k]};
                p[k] = pal[j];
            }
            WriteVoidExtent(out, static_cast<u32>(p[0]), static_cast<u32>(p[1]),
                            static_cast<u32>(p[2]), 255);
            return;
        }
    }

    // Per channel: search +-2 around the expanded endpoints for the least squared error against
    // the true BC1 palette, weighted by how often each palette entry is used.
    for (int k = 0; k < 3; ++k) {
        const int target[4] = {ex0[k], (2 * ex0[k] + ex1[k]) / 3, (ex0[k] + 2 * ex1[k]) / 3,
                               ex1[k]};
        int best = 1 << 30;
        int b0 = ex0[k];
        int b1 = ex1[k];
        for (int d0 = -2; d0 <= 2; ++d0) {
            const int a = std::clamp(ex0[k] + d0, 0, 255);
            for (int d1 = -2; d1 <= 2; ++d1) {
                const int b = std::clamp(ex1[k] + d1, 0, 255);
                int sse = 0;
                for (u32 j = 0; j < 4; ++j) {
                    const int df = Lerp(a, b, ls.uw[j]) - target[j];
                    sse += static_cast<int>(cnt[j]) * df * df;
                }
                if (sse < best) {
                    best = sse;
                    b0 = a;
                    b1 = b;
                }
            }
        }
        cand.e0[k] = b0;
        cand.e1[k] = b1;
        cand.code[2 * k] = static_cast<u8>(b0);
        cand.code[2 * k + 1] = static_cast<u8>(b1);
    }
    ApplyOrdering(cand);
    WriteBlock(out, cand);
}

} // namespace

void EncodeRGBA8ToAstc4x4(std::span<const u8> rgba8, u32 width, u32 height, u32 slices,
                          std::span<u8> output, bool srgb) {
    // Both profiles encode the stored values as-is and measure the error in the encoded space.
    static_cast<void>(srgb);
    const u32 bx = (width + 3) / 4;
    const u32 by = (height + 3) / 4;
    const u64 in_bytes = u64{width} * height * 4 * slices;
    const u64 out_bytes = u64{bx} * by * slices * AstcBlockBytes;
    if (!CheckSizes(in_bytes, rgba8.size(), out_bytes, output.size(), width, height, slices)) {
        return;
    }
    static_cast<void>(GetTables());

    u8* dst = output.data();
    for (u32 z = 0; z < slices; ++z) {
        const u8* slice = rgba8.data() + u64{z} * width * height * 4;
        for (u32 y = 0; y < by; ++y) {
            for (u32 x = 0; x < bx; ++x) {
                int px[BlockTexels][4];
                for (u32 j = 0; j < 4; ++j) {
                    const u32 sy = std::min(y * 4 + j, height - 1);
                    for (u32 i = 0; i < 4; ++i) {
                        const u32 sx = std::min(x * 4 + i, width - 1);
                        const u8* p = slice + (u64{sy} * width + sx) * 4;
                        int* q = px[j * 4 + i];
                        q[0] = p[0];
                        q[1] = p[1];
                        q[2] = p[2];
                        q[3] = p[3];
                    }
                }
                EncodeBlock(px, dst);
                dst += AstcBlockBytes;
            }
        }
    }
}

void TranscodeBC1ToAstc4x4(std::span<const u8> bc1_blocks, u32 width, u32 height, u32 slices,
                           std::span<u8> output, bool srgb) {
    static_cast<void>(srgb);
    const u32 bx = (width + 3) / 4;
    const u32 by = (height + 3) / 4;
    const u64 blocks = u64{bx} * by * slices;
    if (!CheckSizes(blocks * 8, bc1_blocks.size(), blocks * AstcBlockBytes, output.size(), width,
                    height, slices)) {
        return;
    }
    static_cast<void>(GetTables());

    const u8* src = bc1_blocks.data();
    u8* dst = output.data();
    for (u64 b = 0; b < blocks; ++b, src += 8, dst += AstcBlockBytes) {
        const u32 c0 = src[0] | (src[1] << 8);
        const u32 c1 = src[2] | (src[3] << 8);
        if (c0 > c1) {
            TranscodeBC1Direct(src, dst);
        } else {
            int px[BlockTexels][4];
            DecodeBC1Block(src, px);
            EncodeBlock(px, dst);
        }
    }
}

} // namespace Tegra::Texture::ASTC
