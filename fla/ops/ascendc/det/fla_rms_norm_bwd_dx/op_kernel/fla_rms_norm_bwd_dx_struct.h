/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file fla_rms_norm_bwd_dx_struct.h
 * \brief Tiling data of the deterministic RMSNorm backward input-gradient operator.
 */

#ifndef FLA_RMS_NORM_BWD_DX_STRUCT_H
#define FLA_RMS_NORM_BWD_DX_STRUCT_H

struct FlaRmsNormBwdDxTilingData {
    int64_t m;           // number of rows
    int64_t d;           // hidden size
    int64_t dFold;       // padded row length of one dgammaPart row (>= d, see the tiling)
    int64_t rowsPerCore; // contiguous rows handled by one core
    int64_t usedCoreNum; // number of launched vector cores
    int64_t rowsPerTile; // rows of one tiled-path tile, 0 disables the tiled path
    int64_t chunkElems;  // columns per chunk of the large-d path, 0 disables the chunked path
};

#endif // FLA_RMS_NORM_BWD_DX_STRUCT_H
