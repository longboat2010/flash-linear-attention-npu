/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file fla_rms_norm_bwd_dg_struct.h
 * \brief Tiling data of the deterministic RMSNorm weight-gradient reduce operator.
 */

#ifndef FLA_RMS_NORM_BWD_DG_STRUCT_H
#define FLA_RMS_NORM_BWD_DG_STRUCT_H

struct FlaRmsNormBwdDgTilingData {
    int64_t d;           // hidden size
    int64_t dFold;       // padded length of one partial-sum row
    int64_t usedCoreNum; // number of valid partial rows written by the dx operator
};

#endif // FLA_RMS_NORM_BWD_DG_STRUCT_H
