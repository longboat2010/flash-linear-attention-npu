/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file fla_rms_norm_bwd_dg_tiling.cpp
 * \brief Tiling of the deterministic RMSNorm weight-gradient reduce operator.
 *
 * The number of valid partial-sum rows is recovered exactly the same way as in
 * fla_rms_norm_bwd_dx (min(AIV core num, rows)) so the two operators agree on how
 * many rows of dgammaPart were written.
 */

#include <algorithm>

#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"
#include "../op_kernel/fla_rms_norm_bwd_dg_struct.h"

namespace optiling {
namespace {
constexpr int64_t MIN_FOLD = 64;
constexpr int64_t FOLD_ALIGN = 8;   // dFold granularity of the large-d path (32B of fp32)
constexpr int64_t MAX_D = 65536;    // same envelope as the vendor RMSNorm backward
constexpr int64_t ROW_MAX_D = 4096; // d above this uses the chunked path in fla_rms_norm_bwd_dx
constexpr int64_t PART_ROWS = 64;   // rows of dgammaPart, must match the host-side allocation

constexpr size_t INPUT_X = 0;
constexpr size_t INPUT_GAMMA = 1;
constexpr size_t INPUT_DGAMMA_PART = 2;

int64_t NextPow2AtLeast(int64_t value)
{
    int64_t result = MIN_FOLD;
    while (result < value) {
        result <<= 1;
    }
    return result;
}

// Must match FoldLen() in fla_rms_norm_bwd_dx_tiling.cpp: the width of one dgammaPart row.
int64_t FoldLen(int64_t d)
{
    if (d <= ROW_MAX_D) {
        return NextPow2AtLeast(d);
    }
    return (d + FOLD_ALIGN - 1) / FOLD_ALIGN * FOLD_ALIGN;
}
} // namespace

struct FlaRmsNormBwdDgCompileInfo {};

static ge::graphStatus Tiling4FlaRmsNormBwdDg(gert::TilingContext *context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetRawTilingData());
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int64_t vectorCoreNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
    OP_CHECK_IF(vectorCoreNum <= 0, OP_LOGE(context, "invalid AIV core num: %ld", vectorCoreNum),
                return ge::GRAPH_FAILED);

    auto gammaShapePtr = context->GetInputShape(INPUT_GAMMA);
    OP_CHECK_NULL_WITH_CONTEXT(context, gammaShapePtr);
    int64_t d = gammaShapePtr->GetStorageShape().GetShapeSize();
    OP_CHECK_IF(d <= 0 || d > MAX_D, OP_LOGE(context, "unsupported hidden size d=%ld (max %ld)", d, MAX_D),
                return ge::GRAPH_FAILED);

    auto xShapePtr = context->GetInputShape(INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    int64_t xSize = xShapePtr->GetStorageShape().GetShapeSize();
    OP_CHECK_IF(xSize <= 0 || xSize % d != 0, OP_LOGE(context, "invalid x numel %ld for d=%ld", xSize, d),
                return ge::GRAPH_FAILED);
    int64_t m = xSize / d;

    auto partShapePtr = context->GetInputShape(INPUT_DGAMMA_PART);
    OP_CHECK_NULL_WITH_CONTEXT(context, partShapePtr);
    const gert::Shape &partShape = partShapePtr->GetStorageShape();
    OP_CHECK_IF(partShape.GetDimNum() != 2, OP_LOGE(context, "dgammaPart must be 2-D"), return ge::GRAPH_FAILED);
    int64_t dFold = partShape.GetDim(1);
    OP_CHECK_IF(dFold != FoldLen(d), OP_LOGE(context, "dgammaPart second dim %ld != dFold %ld", dFold, FoldLen(d)),
                return ge::GRAPH_FAILED);

    // Must equal the number of dgammaPart rows written by FlaRmsNormBwdDx, which
    // clamps to PART_ROWS as well: the two operators share one fixed contract.
    // The second step mirrors the dx tiling: only ceil(m / rowsPerCore) cores actually
    // receive rows, and only those rows are written.
    int64_t usedCoreNum = std::min(std::min(vectorCoreNum, m), PART_ROWS);
    if (usedCoreNum < 1) {
        usedCoreNum = 1;
    }
    const int64_t rowsPerCore = (m + usedCoreNum - 1) / usedCoreNum;
    usedCoreNum = (m + rowsPerCore - 1) / rowsPerCore;
    OP_CHECK_IF(partShape.GetDim(0) < usedCoreNum,
                OP_LOGE(context, "dgammaPart first dim %ld < usedCoreNum %ld", partShape.GetDim(0), usedCoreNum),
                return ge::GRAPH_FAILED);

    auto tiling = context->GetTilingData<FlaRmsNormBwdDgTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->d = d;
    tiling->dFold = dFold;
    tiling->usedCoreNum = usedCoreNum;

    OP_LOGD(context, "FlaRmsNormBwdDg tiling: d=%ld dFold=%ld usedCoreNum=%ld", d, dFold, usedCoreNum);

    context->SetBlockDim(1);
    size_t *workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParse4FlaRmsNormBwdDg(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FlaRmsNormBwdDg)
    .Tiling(Tiling4FlaRmsNormBwdDg)
    .TilingParse<FlaRmsNormBwdDgCompileInfo>(TilingParse4FlaRmsNormBwdDg);

} // namespace optiling
