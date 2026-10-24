/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include "fla_rms_norm_bwd_dx.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(FlaRmsNormBwdDx);

const aclTensor *FlaRmsNormBwdDx(
    const aclTensor *x,
    const aclTensor *rstd,
    const aclTensor *gamma,
    const aclTensor *dy,
    const aclTensor *dx,
    const aclTensor *dgammaPart,
    aclOpExecutor *executor)
{
    L0_DFX(FlaRmsNormBwdDx, x, rstd, gamma, dy, dx, dgammaPart);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(FlaRmsNormBwdDx,
                                           OP_INPUT(x, rstd, gamma, dy),
                                           OP_OUTPUT(dx, dgammaPart));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return nullptr;
    }
    return dx;
}

} // namespace l0op
