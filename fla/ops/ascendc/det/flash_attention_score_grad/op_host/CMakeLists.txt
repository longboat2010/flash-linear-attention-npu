# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Ported from ops-transformer 9.0.0 (attention/flash_attention_score_grad). Only the arch32
# implementation (Ascend 910B / 910_93) is shipped here; the ascend950/regbase sources and the
# per-op tests are not part of this tree.
add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnInner PRIVATE
        flash_attention_score_grad_def.cpp
    )
endif()

add_ops_compile_options(
        OP_NAME FlashAttentionScoreGrad
        OPTIONS --cce-auto-sync=off
                --cce-long-call=true
                -Wno-deprecated-declarations
                -Werror
                # Kernel helper headers this op pulls in from the upstream tree
                # (util.h / pse.h / dropmask.h) live in ../common, next to this op.
                -I${CMAKE_CURRENT_SOURCE_DIR}/../common
)
add_modules_sources_with_soc(
    # 该变量用于判断op_api目录是否从op_host移出
    OP_API_INDEPENDENT ON
    # op_api新的地址
    OP_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../op_api
    OPTYPE flash_attention_score_grad
    ACLNNTYPE aclnn_inner
)
