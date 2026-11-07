// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <span>

#include "common/common_types.h"

namespace Tegra::Texture::ASTC {

/// Size in bytes of one ASTC block (any footprint).
constexpr u32 AstcBlockBytes = 16;

/// Encodes tightly packed RGBA8 texels (width * height * 4 bytes per slice, slices back to back)
/// into ASTC LDR 4x4 blocks (ceil(w/4) * ceil(h/4) blocks per slice, row-major, slices back to back).
/// Texels outside the image (edge blocks) replicate the border texels.
/// `srgb` states that RGB are sRGB-encoded values; they are encoded as-is (error is measured in
/// the encoded space for both profiles, alpha is always linear), so the output bits are identical.
/// Only a restricted, decoder-verified subset of the ASTC block modes is emitted (single partition,
/// 4x4 weight grid, direct luminance/RGB/RGBA endpoints, or void-extent blocks).
/// `output` must hold at least slices * ceil(w/4) * ceil(h/4) * AstcBlockBytes bytes and `rgba8`
/// at least slices * w * h * 4 bytes, otherwise nothing is written. Pure function, thread-safe.
void EncodeRGBA8ToAstc4x4(std::span<const u8> rgba8, u32 width, u32 height, u32 slices,
                          std::span<u8> output, bool srgb);

/// Transcodes BC1 blocks (8 bytes each, ceil(w/4) * ceil(h/4) per slice, no row padding) to ASTC
/// LDR 4x4 blocks with the same layout as EncodeRGBA8ToAstc4x4. Blocks in 4-colour mode
/// (c0 > c1 as raw u16) are converted directly (near lossless: max error of a few LSB), blocks in
/// 3-colour/punch-through mode (c0 <= c1) are decoded and run through the generic encoder.
/// `output` must hold at least slices * blocks_x * blocks_y * AstcBlockBytes bytes.
void TranscodeBC1ToAstc4x4(std::span<const u8> bc1_blocks, u32 width, u32 height, u32 slices,
                           std::span<u8> output, bool srgb);

} // namespace Tegra::Texture::ASTC
