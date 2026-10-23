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
#include "fla_rms_norm_bwd_dg.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(FlaRmsNormBwdDg);

const aclTensor *FlaRmsNormBwdDg(
    const aclTensor *x,
    const aclTensor *gamma,
    const aclTensor *dgammaPart,
    const aclTensor *dgamma,
    aclOpExecutor *executor)
{
    L0_DFX(FlaRmsNormBwdDg, x, gamma, dgammaPart, dgamma);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(FlaRmsNormBwdDg,
                                           OP_INPUT(x, gamma, dgammaPart),
                                           OP_OUTPUT(dgamma));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return nullptr;
    }
    return dgamma;
}

} // namespace l0op
