/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file fla_rms_norm_bwd_dg.cpp
 * \brief Deterministic RMSNorm backward weight-gradient reduce (single core).
 *
 * dgammaPart holds one fp32 partial-sum row per vector core, written by
 * fla_rms_norm_bwd_dx. This kernel adds the rows in ascending core order on a
 * single core, so the result does not depend on any cross-core scheduling.
 *
 * A partial row can be much wider than the vector buffers (dgammaPart rows are dFold
 * wide, and d may be up to 64K), so the reduction walks the columns in fixed
 * kChunkElems-wide chunks. Each column is still summed in ascending core order, which
 * makes the result identical to a single Add over the whole row and independent of the
 * chunk width.
 */

#include "kernel_operator.h"
#include "fla_rms_norm_bwd_dg_struct.h"

using namespace AscendC;

namespace {

constexpr uint32_t kChunkElems = 4096;   // fp32 elements per column chunk
constexpr uint32_t kAlignBytes = 32;

template <typename T>
__aicore__ inline void CastFromFloat(const LocalTensor<T> &dst, const LocalTensor<float> &src, uint32_t count)
{
    if constexpr (sizeof(T) == sizeof(float)) {
        Muls(dst, src, 1.0f, count);
    } else {
        Cast(dst, src, RoundMode::CAST_RINT, count);
    }
}

__aicore__ inline uint32_t AlignUp32(uint32_t bytes)
{
    return (bytes + kAlignBytes - 1) / kAlignBytes * kAlignBytes;
}

__aicore__ inline void CopyFloats(const LocalTensor<float> &dst, const GlobalTensor<float> &src,
                                  uint64_t elemOffset, uint32_t count)
{
    DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> pad{false, 0, 0, 0};
    DataCopyPad(dst, src[elemOffset], params, pad);
}

template <typename TG>
class FlaRmsNormBwdDgKernel {
public:
    __aicore__ inline FlaRmsNormBwdDgKernel() {}

    __aicore__ inline void Init(GM_ADDR dgammaPart, GM_ADDR dgamma, const FlaRmsNormBwdDgTilingData *tilingData)
    {
        d_ = tilingData->d;
        dFold_ = tilingData->dFold;
        usedCoreNum_ = tilingData->usedCoreNum;

        partGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dgammaPart));
        dgGm_.SetGlobalBuffer(reinterpret_cast<__gm__ TG *>(dgamma));

        chunkElems_ = (d_ < static_cast<int64_t>(kChunkElems)) ? static_cast<uint32_t>(d_) : kChunkElems;
        pipe_.InitBuffer(partInQue_, 2, AlignUp32(chunkElems_ * static_cast<uint32_t>(sizeof(float))));
        pipe_.InitBuffer(accBuf_, AlignUp32(chunkElems_ * static_cast<uint32_t>(sizeof(float))));
        pipe_.InitBuffer(dgOutQue_, 1, AlignUp32(chunkElems_ * static_cast<uint32_t>(sizeof(TG))));

        acc_ = accBuf_.Get<float>();
        evMte3Mte2_ = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
    }

    __aicore__ inline void Process()
    {
        for (int64_t col = 0; col < d_; col += static_cast<int64_t>(chunkElems_)) {
            // Number of columns of the output in this chunk, and the number of partial
            // columns to read: the last chunk also covers the zero padding of dFold.
            const uint32_t len = static_cast<uint32_t>(d_ - col < static_cast<int64_t>(chunkElems_)
                                                           ? d_ - col
                                                           : static_cast<int64_t>(chunkElems_));
            const int64_t readLeft = dFold_ - col;
            const uint32_t lenRead = static_cast<uint32_t>(readLeft < static_cast<int64_t>(chunkElems_)
                                                               ? readLeft
                                                               : static_cast<int64_t>(chunkElems_));

            Duplicate(acc_, 0.0f, lenRead);
            PipeBarrier<PIPE_V>();
            for (int64_t core = 0; core < usedCoreNum_; ++core) {
                LocalTensor<float> part = partInQue_.AllocTensor<float>();
                CopyFloats(part, partGm_, static_cast<uint64_t>(core) * static_cast<uint64_t>(dFold_) +
                                               static_cast<uint64_t>(col),
                           lenRead);
                partInQue_.EnQue(part);

                LocalTensor<float> partD = partInQue_.DeQue<float>();
                Add(acc_, acc_, partD, lenRead);
                // the next iteration's Add reads acc_ again; keep the barrier explicit
                PipeBarrier<PIPE_V>();
                partInQue_.FreeTensor(partD);
            }

            LocalTensor<TG> out = dgOutQue_.AllocTensor<TG>();
            CastFromFloat(out, acc_, len);
            PipeBarrier<PIPE_V>();
            dgOutQue_.EnQue(out);
            LocalTensor<TG> outD = dgOutQue_.DeQue<TG>();
            DataCopyExtParams params{1, static_cast<uint32_t>(len * sizeof(TG)), 0, 0, 0};
            DataCopyPad(dgGm_[static_cast<uint64_t>(col)], outD, params);
            dgOutQue_.FreeTensor(outD);
            // the next chunk reuses the output buffer: wait for the store to land
            SetFlag<HardEvent::MTE3_MTE2>(evMte3Mte2_);
            WaitFlag<HardEvent::MTE3_MTE2>(evMte3Mte2_);
        }
        PipeBarrier<PIPE_ALL>();
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, 2> partInQue_;
    TQue<QuePosition::VECOUT, 1> dgOutQue_;
    TBuf<TPosition::VECCALC> accBuf_;
    LocalTensor<float> acc_;
    GlobalTensor<float> partGm_;
    GlobalTensor<TG> dgGm_;

    int64_t d_ = 0;
    int64_t dFold_ = 0;
    int64_t usedCoreNum_ = 0;
    uint32_t chunkElems_ = 0;
    TEventID evMte3Mte2_ = 0;
};

} // namespace

extern "C" __global__ __aicore__ void fla_rms_norm_bwd_dg(GM_ADDR x, GM_ADDR gamma, GM_ADDR dgammaPart,
                                                          GM_ADDR dgamma, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)x;
    (void)gamma;
    (void)workspace;
    REGISTER_TILING_DEFAULT(FlaRmsNormBwdDgTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    FlaRmsNormBwdDgKernel<DTYPE_DGAMMA> kernel;
    kernel.Init(dgammaPart, dgamma, &tilingData);
    kernel.Process();
}
