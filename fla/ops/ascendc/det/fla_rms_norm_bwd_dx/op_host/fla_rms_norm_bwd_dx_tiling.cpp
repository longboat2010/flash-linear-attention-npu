/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file fla_rms_norm_bwd_dx_tiling.cpp
 * \brief Tiling of the deterministic RMSNorm backward input-gradient operator.
 *
 * The reduction length is padded to a power of two (dFold) so the kernel can fold
 * the row sum in a fixed order; the number of cores and the rows per core are
 * computed identically here and in fla_rms_norm_bwd_dg, which consumes the partial
 * sums written by this operator.
 */

#include <algorithm>

#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"
#include "../op_kernel/fla_rms_norm_bwd_dx_struct.h"

namespace optiling {
namespace {
constexpr int64_t MIN_FOLD = 64;
constexpr int64_t FOLD_ALIGN = 8;    // dFold granularity of the large-d path (32B of fp32)
constexpr int64_t MAX_D = 65536;     // same envelope as the vendor RMSNorm backward
constexpr int64_t PART_ROWS = 64;    // rows of dgammaPart, must match the host-side allocation
// Tiled kernel path: one contiguous DMA per tile of rows. Must match the constants in
// op_kernel/fla_rms_norm_bwd_dx.cpp.
constexpr int64_t TILED_MAX_D = 512;
constexpr int64_t TILED_RED_MASK = 64;      // WholeReduceSum repeat length (fp32 elements)
constexpr int64_t TILED_MAX_ELEMS = 4096;   // elements of one tile buffer
constexpr int64_t TILED_MAX_ROWS = 64;
// Largest d handled by the row-at-a-time path; beyond that the kernel walks the row in
// column chunks so that a row never has to fit the vector buffers. Must match kRowMaxD
// in op_kernel/fla_rms_norm_bwd_dx.cpp.
constexpr int64_t ROW_MAX_D = 4096;
constexpr int64_t CHUNK_ELEMS = 2048;       // fp32 elements per column chunk of the large-d path

constexpr size_t INPUT_X = 0;
constexpr size_t INPUT_RSTD = 1;
constexpr size_t INPUT_GAMMA = 2;
constexpr size_t INPUT_DY = 3;

int64_t NextPow2AtLeast(int64_t value)
{
    int64_t result = MIN_FOLD;
    while (result < value) {
        result <<= 1;
    }
    return result;
}

// Width of one dgammaPart row. The row path needs a power of two so that it can fold
// the row length down by halving; the chunked path only needs the partial row to stay
// 32B aligned. fla_rms_norm_bwd_dg derives the same value from the dgammaPart shape.
int64_t FoldLen(int64_t d)
{
    if (d <= ROW_MAX_D) {
        return NextPow2AtLeast(d);
    }
    return (d + FOLD_ALIGN - 1) / FOLD_ALIGN * FOLD_ALIGN;
}
} // namespace

struct FlaRmsNormBwdDxCompileInfo {};

static ge::graphStatus Tiling4FlaRmsNormBwdDx(gert::TilingContext *context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetRawTilingData());
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int64_t vectorCoreNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
    OP_CHECK_IF(vectorCoreNum <= 0, OP_LOGE(context, "invalid AIV core num: %ld", vectorCoreNum),
                return ge::GRAPH_FAILED);

    auto xShapePtr = context->GetInputShape(INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    const gert::Shape &xShape = xShapePtr->GetStorageShape();
    int64_t ndim = static_cast<int64_t>(xShape.GetDimNum());
    OP_CHECK_IF(ndim < 1, OP_LOGE(context, "x must have at least one dimension"), return ge::GRAPH_FAILED);
    int64_t d = xShape.GetDim(ndim - 1);
    int64_t xSize = xShape.GetShapeSize();
    OP_CHECK_IF(d <= 0 || xSize <= 0 || xSize % d != 0, OP_LOGE(context, "invalid x shape, size=%ld d=%ld", xSize, d),
                return ge::GRAPH_FAILED);
    int64_t m = xSize / d;
    OP_CHECK_IF(d > MAX_D, OP_LOGE(context, "unsupported hidden size d=%ld (max %ld)", d, MAX_D),
                return ge::GRAPH_FAILED);

    auto rstdShapePtr = context->GetInputShape(INPUT_RSTD);
    OP_CHECK_NULL_WITH_CONTEXT(context, rstdShapePtr);
    int64_t rstdSize = rstdShapePtr->GetStorageShape().GetShapeSize();
    OP_CHECK_IF(rstdSize != m, OP_LOGE(context, "rstd numel %ld != rows %ld", rstdSize, m),
                return ge::GRAPH_FAILED);

    auto gammaShapePtr = context->GetInputShape(INPUT_GAMMA);
    OP_CHECK_NULL_WITH_CONTEXT(context, gammaShapePtr);
    int64_t gammaSize = gammaShapePtr->GetStorageShape().GetShapeSize();
    OP_CHECK_IF(gammaSize != d, OP_LOGE(context, "gamma numel %ld != hidden size %ld", gammaSize, d),
                return ge::GRAPH_FAILED);

    auto dyShapePtr = context->GetInputShape(INPUT_DY);
    OP_CHECK_NULL_WITH_CONTEXT(context, dyShapePtr);
    int64_t dySize = dyShapePtr->GetStorageShape().GetShapeSize();
    OP_CHECK_IF(dySize != xSize, OP_LOGE(context, "dy numel %ld != x numel %ld", dySize, xSize),
                return ge::GRAPH_FAILED);

    // dgammaPart is allocated by the caller with PART_ROWS rows; the output shape is not
    // visible to tiling in the aclnn path, so the core count is clamped by the contract.
    int64_t usedCoreNum = std::min(std::min(vectorCoreNum, m), PART_ROWS);
    if (usedCoreNum < 1) {
        usedCoreNum = 1;
    }
    int64_t rowsPerCore = (m + usedCoreNum - 1) / usedCoreNum;
    // Row blocks are contiguous and rowsPerCore long, so ceil(m / rowsPerCore) cores are
    // enough. Dropping the extra cores is required for correctness, not just for
    // tidiness: a launched core whose rowStart >= m writes no dgammaPart row, and
    // fla_rms_norm_bwd_dg adds every row below usedCoreNum.
    usedCoreNum = (m + rowsPerCore - 1) / rowsPerCore;
    int64_t dFold = FoldLen(d);
    int64_t chunkElems = (d > ROW_MAX_D) ? CHUNK_ELEMS : 0;

    // The tiled path needs no fold padding, a whole number of reduce repeats per row and
    // a tile that fits the kernel buffers. 0 tells the kernel to use the row path.
    int64_t rowsPerTile = 0;
    if (dFold == d && d <= TILED_MAX_D && d % TILED_RED_MASK == 0) {
        rowsPerTile = TILED_MAX_ELEMS / d;
        rowsPerTile = std::min(rowsPerTile, TILED_MAX_ROWS);
        if (rowsPerTile * d > TILED_MAX_ELEMS) {
            rowsPerTile = 0;
        }
    }

    auto tiling = context->GetTilingData<FlaRmsNormBwdDxTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->m = m;
    tiling->d = d;
    tiling->dFold = dFold;
    tiling->rowsPerCore = rowsPerCore;
    tiling->usedCoreNum = usedCoreNum;
    tiling->rowsPerTile = rowsPerTile;
    tiling->chunkElems = chunkElems;

    OP_LOGD(context,
            "FlaRmsNormBwdDx tiling: m=%ld d=%ld dFold=%ld rowsPerCore=%ld usedCoreNum=%ld rowsPerTile=%ld "
            "chunkElems=%ld",
            m, d, dFold, rowsPerCore, usedCoreNum, rowsPerTile, chunkElems);

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));
    size_t *workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParse4FlaRmsNormBwdDx(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FlaRmsNormBwdDx)
    .Tiling(Tiling4FlaRmsNormBwdDx)
    .TilingParse<FlaRmsNormBwdDxCompileInfo>(TilingParse4FlaRmsNormBwdDx);

} // namespace optiling
