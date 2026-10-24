/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "aclnn_fla_rms_norm_bwd_dx.h"
#include "fla_rms_norm_bwd_dx.h"
#include <dlfcn.h>
#include <new>

#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/contiguous.h"
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

struct FlaRmsNormBwdDxParams {
    const aclTensor *x = nullptr;
    const aclTensor *rstd = nullptr;
    const aclTensor *gamma = nullptr;
    const aclTensor *dy = nullptr;
    const aclTensor *dx = nullptr;
    const aclTensor *dgammaPart = nullptr;
};

static aclnnStatus CheckNotNull(FlaRmsNormBwdDxParams params)
{
    CHECK_COND(params.x != nullptr, ACLNN_ERR_PARAM_NULLPTR, "x must not be nullptr.");
    CHECK_COND(params.rstd != nullptr, ACLNN_ERR_PARAM_NULLPTR, "rstd must not be nullptr.");
    CHECK_COND(params.gamma != nullptr, ACLNN_ERR_PARAM_NULLPTR, "gamma must not be nullptr.");
    CHECK_COND(params.dy != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dy must not be nullptr.");
    CHECK_COND(params.dx != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dx must not be nullptr.");
    CHECK_COND(params.dgammaPart != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dgammaPart must not be nullptr.");
    return ACLNN_SUCCESS;
}

// Format / shape / dtype are fixed by the op definition and by the caller
// (see FLANpuOpApi.cpp), which allocates ``dx`` and ``dgammaPart`` itself.
static aclnnStatus CheckParams(FlaRmsNormBwdDxParams params)
{
    CHECK_RET(CheckNotNull(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus DataContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus ParamsDataContiguous(FlaRmsNormBwdDxParams &params, aclOpExecutor *executorPtr)
{
    CHECK_COND(DataContiguous(params.x, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous x failed.");
    CHECK_COND(DataContiguous(params.rstd, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous rstd failed.");
    CHECK_COND(DataContiguous(params.gamma, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous gamma failed.");
    CHECK_COND(DataContiguous(params.dy, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous dy failed.");
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFlaRmsNormBwdDxGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *rstd,
    const aclTensor *gamma,
    const aclTensor *dy,
    const aclTensor *dx,
    const aclTensor *dgammaPart,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    FlaRmsNormBwdDxParams params{x, rstd, gamma, dy, dx, dgammaPart};
    L2_DFX_PHASE_1(aclnnFlaRmsNormBwdDx, DFX_IN(x, rstd, gamma, dy), DFX_OUT(dx, dgammaPart));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();

    auto ret = CheckParams(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_COND(ParamsDataContiguous(params, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "ParamsDataContiguous failed.");

    auto result = l0op::FlaRmsNormBwdDx(params.x, params.rstd, params.gamma, params.dy, params.dx,
                                        params.dgammaPart, executorPtr);
    CHECK_RET(result != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(result, params.dx, executorPtr);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFlaRmsNormBwdDx(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                 aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFlaRmsNormBwdDx);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "This is an error in FlaRmsNormBwdDx launch aicore.");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
