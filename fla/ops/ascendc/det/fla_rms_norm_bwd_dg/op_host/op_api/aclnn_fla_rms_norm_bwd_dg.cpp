/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "aclnn_fla_rms_norm_bwd_dg.h"
#include "fla_rms_norm_bwd_dg.h"
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

struct FlaRmsNormBwdDgParams {
    const aclTensor *x = nullptr;
    const aclTensor *gamma = nullptr;
    const aclTensor *dgammaPart = nullptr;
    const aclTensor *dgamma = nullptr;
};

static aclnnStatus CheckNotNull(FlaRmsNormBwdDgParams params)
{
    CHECK_COND(params.x != nullptr, ACLNN_ERR_PARAM_NULLPTR, "x must not be nullptr.");
    CHECK_COND(params.gamma != nullptr, ACLNN_ERR_PARAM_NULLPTR, "gamma must not be nullptr.");
    CHECK_COND(params.dgammaPart != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dgammaPart must not be nullptr.");
    CHECK_COND(params.dgamma != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dgamma must not be nullptr.");
    return ACLNN_SUCCESS;
}

// Format / shape / dtype are fixed by the op definition and by the caller
// (see FLANpuOpApi.cpp), which allocates ``dgamma`` itself.
static aclnnStatus CheckParams(FlaRmsNormBwdDgParams params)
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

static aclnnStatus ParamsDataContiguous(FlaRmsNormBwdDgParams &params, aclOpExecutor *executorPtr)
{
    CHECK_COND(DataContiguous(params.x, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous x failed.");
    CHECK_COND(DataContiguous(params.gamma, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous gamma failed.");
    CHECK_COND(DataContiguous(params.dgammaPart, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous dgammaPart failed.");
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFlaRmsNormBwdDgGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *gamma,
    const aclTensor *dgammaPart,
    const aclTensor *dgamma,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    FlaRmsNormBwdDgParams params{x, gamma, dgammaPart, dgamma};
    L2_DFX_PHASE_1(aclnnFlaRmsNormBwdDg, DFX_IN(x, gamma, dgammaPart), DFX_OUT(dgamma));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();

    auto ret = CheckParams(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_COND(ParamsDataContiguous(params, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "ParamsDataContiguous failed.");

    auto result = l0op::FlaRmsNormBwdDg(params.x, params.gamma, params.dgammaPart, params.dgamma, executorPtr);
    CHECK_RET(result != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(result, params.dgamma, executorPtr);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFlaRmsNormBwdDg(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                 aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFlaRmsNormBwdDg);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "This is an error in FlaRmsNormBwdDg launch aicore.");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
