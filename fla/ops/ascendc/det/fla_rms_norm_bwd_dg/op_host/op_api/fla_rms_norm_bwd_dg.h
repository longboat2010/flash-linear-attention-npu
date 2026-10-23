/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef OP_API_INC_LEVEL0_FLA_RMS_NORM_BWD_DG_H
#define OP_API_INC_LEVEL0_FLA_RMS_NORM_BWD_DG_H

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor *FlaRmsNormBwdDg(
    const aclTensor *x,
    const aclTensor *gamma,
    const aclTensor *dgammaPart,
    const aclTensor *dgamma,
    aclOpExecutor *executor);
}

#endif
