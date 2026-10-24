/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file fla_rms_norm_bwd_dx.cpp
 * \brief Deterministic RMSNorm backward (vector only).
 *
 * Each core owns a contiguous block of rows and computes, per row:
 *     xhat = x * rstd
 *     dz   = dy * gamma
 *     dot  = mean(dz * xhat)                 (fixed order)
 *     dx   = rstd * (dz - xhat * dot)
 * and accumulates
 *     dgamma += dy * xhat                    (fp32, per core)
 *
 * The per-core dgamma partial sums are written to dgammaPart[coreIdx][:] and reduced
 * afterwards by fla_rms_norm_bwd_dg in ascending core order, which keeps the whole
 * weight gradient free of cross-core reductions and therefore bitwise reproducible.
 *
 * Three implementations live here:
 *
 *  - tiled (used when dFold == d, d <= kMaxTiledD and d is a multiple of 64, i.e. every
 *    power-of-two head size up to 512): one contiguous DMA per tile of rows for x, dy
 *    and dx, whole-tile widening, one WholeReduceSum for all row sums (64-element
 *    chunks, results read back in ascending chunk order) and a fixed binary row fold
 *    for dgamma. Only ~4 vector instructions per row are issued, instead of the ~20 of
 *    the row-at-a-time path; that path was 3-5x slower than the vendor operator for
 *    the small-d / many-row shapes of q35.
 *  - row at a time (d <= kRowMaxD, any alignment): every row is zero padded to dFold =
 *    NextPow2AtLeast(d) so that all vector instructions of one row run with a single
 *    element count, and the row sum is folded down pairwise in a fixed order.
 *  - chunked (d > kRowMaxD): the row no longer fits the vector buffers, so it is walked
 *    in kChunkElems-wide column chunks. The row sum is the ascending sum of the
 *    per-chunk folds, and the dgamma partial of the core is kept in dgammaPart and
 *    updated chunk by chunk (only this core touches that row). Both orders are fixed,
 *    so the result is still reproducible. This path costs a second pass over x and dy,
 *    which is why it is not used below kRowMaxD.
 *
 * All three are deterministic by construction: no cross-core reduction, no atomic, no
 * dependence on core scheduling, and every summation order is a pure function of the
 * inputs and the (deterministic) tiling.
 *
 * Rows are DMAed with DataCopyPad because the hidden size is not assumed to be a
 * multiple of the 32B block: only the element count of a row is known to be legal.
 *
 * NOTE: this kernel used to be built with --cce-auto-sync=off. That option made the
 * cores hang (vector core timeout) and must not be re-enabled; the explicit
 * PipeBarrier<PIPE_V>() calls below are kept as cheap extra protection for the
 * vector dependency chain, on top of the default compiler auto synchronization.
 */

#include "kernel_operator.h"
#include "fla_rms_norm_bwd_dx_struct.h"

using namespace AscendC;

namespace {

constexpr uint32_t FOLD_LEN = 8;       // fold down to one 32B block, then sum its lanes scalar
constexpr uint32_t kNumFloatBufs = 7;  // xF dyF gammaF xhat dz prod dsum
constexpr uint32_t kAlignBytes = 32;

constexpr uint32_t kMaxTiledD = 512;        // tiled path only for these head sizes
constexpr uint32_t kRedMask = 64;           // fp32 elements per WholeReduceSum repeat
constexpr uint32_t kMaxTileElems = 4096;    // elements of one tile buffer (rowsPerTile * d)
constexpr uint32_t kMaxRowsPerTile = 64;    // scalar bookkeeping arrays are this long
constexpr uint32_t kMaxRowsLoad = kMaxRowsPerTile + 8; // rstd window, 8 rows of alignment slack
constexpr uint32_t kChunkElems = 2048;      // fp32 elements per column chunk (large-d path)
constexpr uint32_t kRowMaxD = 4096;         // largest d handled by the row-at-a-time path

__aicore__ inline uint32_t AlignUp32(uint32_t bytes)
{
    return (bytes + kAlignBytes - 1) / kAlignBytes * kAlignBytes;
}

template <typename T>
__aicore__ inline void CastToFloat(const LocalTensor<float> &dst, const LocalTensor<T> &src, uint32_t count)
{
    if constexpr (sizeof(T) == sizeof(float)) {
        Muls(dst, src, 1.0f, count);
    } else {
        Cast(dst, src, RoundMode::CAST_NONE, count);
    }
}

template <typename T>
__aicore__ inline void CastFromFloat(const LocalTensor<T> &dst, const LocalTensor<float> &src, uint32_t count)
{
    if constexpr (sizeof(T) == sizeof(float)) {
        Muls(dst, src, 1.0f, count);
    } else {
        Cast(dst, src, RoundMode::CAST_RINT, count);
    }
}

template <typename T>
__aicore__ inline void CopyRowIn(const LocalTensor<T> &dst, const GlobalTensor<T> &src, uint64_t elemOffset,
                                 uint32_t count)
{
    DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> pad{false, 0, 0, 0};
    DataCopyPad(dst, src[elemOffset], params, pad);
}

template <typename T>
__aicore__ inline void CopyRowOut(const GlobalTensor<T> &dst, const LocalTensor<T> &src, uint64_t elemOffset,
                                  uint32_t count)
{
    DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    DataCopyPad(dst[elemOffset], src, params);
}

__aicore__ inline void CopyFloats(const LocalTensor<float> &dst, const GlobalTensor<float> &src,
                                  uint64_t elemOffset, uint32_t count)
{
    DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> pad{false, 0, 0, 0};
    DataCopyPad(dst, src[elemOffset], params, pad);
}

__aicore__ inline void CopyFloatsOut(const GlobalTensor<float> &dst, const LocalTensor<float> &src,
                                     uint64_t elemOffset, uint32_t count)
{
    DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
    DataCopyPad(dst[elemOffset], src, params);
}

template <typename TX, typename TG>
class FlaRmsNormBwdDxKernel {
public:
    __aicore__ inline FlaRmsNormBwdDxKernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR rstd, GM_ADDR gamma, GM_ADDR dy, GM_ADDR dx,
                                GM_ADDR dgammaPart, const FlaRmsNormBwdDxTilingData *tilingData)
    {
        m_ = tilingData->m;
        d_ = tilingData->d;
        dFold_ = tilingData->dFold;
        rowsPerCore_ = tilingData->rowsPerCore;
        chunkElems_ = static_cast<uint32_t>(tilingData->chunkElems);
        blockIdx_ = static_cast<uint32_t>(GetBlockIdx());

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ TX *>(x));
        rstdGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(rstd));
        gammaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ TG *>(gamma));
        dyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ TX *>(dy));
        dxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ TX *>(dx));
        dgPartGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dgammaPart));

        dElems_ = static_cast<uint32_t>(d_);
        dFoldElems_ = static_cast<uint32_t>(dFold_);
        padElems_ = dFoldElems_ - dElems_;

        rowsPerTile_ = static_cast<uint32_t>(tilingData->rowsPerTile);
        useTiled_ = rowsPerTile_ > 0 && dFoldElems_ == dElems_ && dElems_ <= kMaxTiledD &&
                    (dElems_ % kRedMask) == 0 && rowsPerTile_ <= kMaxRowsPerTile &&
                    rowsPerTile_ * dElems_ <= kMaxTileElems;
        if (!useTiled_) {
            rowsPerTile_ = 0;
        }
        // The row path folds a power-of-two row length, so it is only valid up to
        // kRowMaxD; anything wider must go through the chunked path even if the tiling
        // did not ask for it.
        // The chunked path stores the trailing dgammaPart padding as part of its last
        // chunk, which is only possible when that padding is shorter than one 32B block
        // (always true when dFold = AlignUp(d, 8), i.e. for d > kRowMaxD).
        useChunked_ = !useTiled_ && (chunkElems_ > 0 || dElems_ > kRowMaxD) && padElems_ < FOLD_LEN;
        if (useChunked_) {
            if (chunkElems_ == 0) {
                chunkElems_ = kChunkElems;
            }
            chunkElems_ = (chunkElems_ < dElems_) ? chunkElems_ : dElems_;
            // the per-chunk fold halves the chunk length, so it must be a power of two
            while ((chunkElems_ & (chunkElems_ - 1)) != 0) {
                chunkElems_ &= chunkElems_ - 1;
            }
            if (chunkElems_ < FOLD_LEN) {
                chunkElems_ = FOLD_LEN;
            }
            numChunks_ = (dElems_ + chunkElems_ - 1) / chunkElems_;
        }

        evVS_ = GetTPipePtr()->AllocEventID<HardEvent::V_S>();
        evSV_ = GetTPipePtr()->AllocEventID<HardEvent::S_V>();
        evVMte3_ = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
        evMte2S_ = GetTPipePtr()->AllocEventID<HardEvent::MTE2_S>();
        evMte2V_ = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
        evVMte2_ = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
        evMte3Mte2_ = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
        evMte3V_ = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
        evSMte2_ = GetTPipePtr()->AllocEventID<HardEvent::S_MTE2>();

        if (useTiled_) {
            InitTiled();
        } else if (useChunked_) {
            InitChunked();
        } else {
            InitRowAtATime();
        }
    }

    __aicore__ inline void Process()
    {
        const int64_t rowStart = static_cast<int64_t>(blockIdx_) * rowsPerCore_;
        int64_t rowEnd = rowStart + rowsPerCore_;
        if (rowEnd > m_) {
            rowEnd = m_;
        }
        if (useTiled_) {
            for (int64_t row = rowStart; row < rowEnd;) {
                int64_t rows = rowEnd - row;
                if (rows > static_cast<int64_t>(rowsPerTile_)) {
                    rows = static_cast<int64_t>(rowsPerTile_);
                }
                ProcessTile(row, rows);
                row += rows;
            }
        } else if (useChunked_) {
            for (int64_t row = rowStart; row < rowEnd; ++row) {
                ProcessRowChunked(row, row == rowStart);
            }
        } else {
            for (int64_t row = rowStart; row < rowEnd; ++row) {
                ProcessOneRow(row);
            }
        }
        // the chunked path updates the partial in dgammaPart chunk by chunk and has no
        // separate accumulator to flush
        if (!useChunked_) {
            WritePartial();
        }
    }

    __aicore__ inline void Release()
    {
        GetTPipePtr()->ReleaseEventID<HardEvent::V_S>(evVS_);
        GetTPipePtr()->ReleaseEventID<HardEvent::S_V>(evSV_);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(evVMte3_);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_S>(evMte2S_);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(evMte2V_);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(evVMte2_);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(evMte3Mte2_);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(evMte3V_);
        GetTPipePtr()->ReleaseEventID<HardEvent::S_MTE2>(evSMte2_);
    }

private:
    // =====================================================================================
    // tiled path
    // =====================================================================================

    __aicore__ inline void InitTiled()
    {
        const uint32_t tileElems = rowsPerTile_ * dElems_;
        const uint32_t stageBytes = AlignUp32(tileElems * static_cast<uint32_t>(sizeof(TX)));
        const uint32_t gBytes = AlignUp32(dElems_ * static_cast<uint32_t>(sizeof(TG)));
        pipe_.InitBuffer(stageBuf_, stageBytes * 2);
        pipe_.InitBuffer(gammaStageBuf_, gBytes);
        pipe_.InitBuffer(tileBuf_, AlignUp32((tileElems * 5 + dElems_) * sizeof(float)) +
                                     AlignUp32((tileElems / kRedMask) * sizeof(float)) +
                                     AlignUp32(kMaxRowsLoad * sizeof(float)));

        LocalTensor<TX> stage = stageBuf_.Get<TX>();
        xStage_ = stage[0];
        dyStage_ = stage[tileElems];

        LocalTensor<float> tile = tileBuf_.Get<float>();
        uint32_t off = 0;
        xF_ = tile[off];        off += tileElems;
        dyF_ = tile[off];       off += tileElems;
        xhat_ = tile[off];      off += tileElems;
        dz_ = tile[off];        off += tileElems;
        gammaB_ = tile[off];    off += tileElems;   // gamma replicated once per tile row
        dsum_ = tile[off];      off += dElems_;
        // WholeReduceSum writes one value per repeat, contiguously (dstRepeatStride = 1)
        dotRed_ = tile[off];    off += tileElems / kRedMask;
        rstdS_ = tile[off];

        Duplicate(dsum_, 0.0f, dElems_);
        PipeBarrier<PIPE_V>();

        // gamma: loaded once, then replicated so that dz = dy * gamma is one whole-tile op
        LocalTensor<TG> gammaIn = gammaStageBuf_.Get<TG>();
        DataCopy(gammaIn, gammaGm_[0], dElems_);
        SetFlag<HardEvent::MTE2_V>(evMte2V_);
        WaitFlag<HardEvent::MTE2_V>(evMte2V_);
        CastToFloat(gammaB_, gammaIn, dElems_);
        PipeBarrier<PIPE_V>();
        for (uint32_t row = 1; row < rowsPerTile_; ++row) {
            Muls(gammaB_[row * dElems_], gammaB_, 1.0f, dElems_);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProcessTile(int64_t rowStart, int64_t nRows)
    {
        const uint32_t elems = static_cast<uint32_t>(nRows) * dElems_;
        const uint64_t gmOff = static_cast<uint64_t>(rowStart) * static_cast<uint64_t>(d_);

        // ---- WAR: the previous tile's readers of the staging buffers must be done
        //      before this tile's DMAs overwrite them (no-ops for the first tile).
        SetFlag<HardEvent::V_MTE2>(evVMte2_);
        WaitFlag<HardEvent::V_MTE2>(evVMte2_);
        SetFlag<HardEvent::MTE3_MTE2>(evMte3Mte2_);
        WaitFlag<HardEvent::MTE3_MTE2>(evMte3Mte2_);
        SetFlag<HardEvent::S_MTE2>(evSMte2_);
        WaitFlag<HardEvent::S_MTE2>(evSMte2_);

        // ---- one contiguous DMA per tensor for the whole tile
        DataCopy(xStage_, xGm_[gmOff], elems);
        DataCopy(dyStage_, dyGm_[gmOff], elems);
        // rstd: a 32B-aligned window that covers rows [rowStart, rowStart + nRows)
        const int64_t rstdStart = rowStart & ~static_cast<int64_t>(7);
        const uint32_t rstdOff = static_cast<uint32_t>(rowStart - rstdStart);
        const uint32_t rstdLen = static_cast<uint32_t>(nRows) + rstdOff;
        DataCopyExtParams rstdParams{1, static_cast<uint32_t>(rstdLen * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> rstdPad{false, 0, 0, 0.0f};
        DataCopyPad(rstdS_, rstdGm_[rstdStart], rstdParams, rstdPad);

        SetFlag<HardEvent::MTE2_S>(evMte2S_);
        WaitFlag<HardEvent::MTE2_S>(evMte2S_);
        float rs[kMaxRowsPerTile];
        for (uint32_t row = 0; row < static_cast<uint32_t>(nRows); ++row) {
            rs[row] = rstdS_.GetValue(rstdOff + row);
        }
        SetFlag<HardEvent::S_V>(evSV_);
        WaitFlag<HardEvent::S_V>(evSV_);
        SetFlag<HardEvent::MTE2_V>(evMte2V_);
        WaitFlag<HardEvent::MTE2_V>(evMte2V_);

        // ---- widen x / dy
        CastToFloat(xF_, xStage_, elems);
        CastToFloat(dyF_, dyStage_, elems);
        PipeBarrier<PIPE_V>();

        // ---- xhat = x * rstd (one instruction per row, the broadcast is a scalar)
        for (uint32_t row = 0; row < static_cast<uint32_t>(nRows); ++row) {
            Muls(xhat_[row * dElems_], xF_[row * dElems_], rs[row], dElems_);
        }
        PipeBarrier<PIPE_V>();

        // ---- dz = dy * gamma
        Mul(dz_, dyF_, gammaB_, elems);
        PipeBarrier<PIPE_V>();

        // ---- dgamma: row product, fixed binary row fold, accumulate into the core partial
        Mul(xF_, dyF_, xhat_, elems);   // xF_ is free from here on and is reused as scratch
        PipeBarrier<PIPE_V>();
        FoldRows(xF_, static_cast<uint32_t>(nRows));
        Add(dsum_, dsum_, xF_, dElems_);
        PipeBarrier<PIPE_V>();

        // ---- row sums of dz * xhat, 64-element chunks in ascending order
        Mul(dyF_, dz_, xhat_, elems);   // dyF_ is free from here on and is reused as scratch
        PipeBarrier<PIPE_V>();
        WholeReduceSum(dotRed_, dyF_, kRedMask, static_cast<uint8_t>(elems / kRedMask), 1, 1,
                       kRedMask / FOLD_LEN);
        SetFlag<HardEvent::V_S>(evVS_);
        WaitFlag<HardEvent::V_S>(evVS_);
        const uint32_t chunks = dElems_ / kRedMask;
        float dots[kMaxRowsPerTile];
        for (uint32_t row = 0; row < static_cast<uint32_t>(nRows); ++row) {
            float sum = 0.0f;
            for (uint32_t c = 0; c < chunks; ++c) {
                sum += dotRed_.GetValue(row * chunks + c);
            }
            dots[row] = sum / static_cast<float>(d_);
        }
        SetFlag<HardEvent::S_V>(evSV_);
        WaitFlag<HardEvent::S_V>(evSV_);

        // ---- dx = rstd * (dz - xhat * dot)
        for (uint32_t row = 0; row < static_cast<uint32_t>(nRows); ++row) {
            Muls(xF_[row * dElems_], xhat_[row * dElems_], dots[row], dElems_);
            Sub(dz_[row * dElems_], dz_[row * dElems_], xF_[row * dElems_], dElems_);
            Muls(dz_[row * dElems_], dz_[row * dElems_], rs[row], dElems_);
        }
        PipeBarrier<PIPE_V>();

        // ---- narrow and store (xStage_ is free again, so dx is written into it)
        CastFromFloat(xStage_, dz_, elems);
        SetFlag<HardEvent::V_MTE3>(evVMte3_);
        WaitFlag<HardEvent::V_MTE3>(evVMte3_);
        DataCopy(dxGm_[gmOff], xStage_, elems);
    }

    // Sum rows [0, rows) of a tile buffer into row 0, in a fixed binary fold order.
    // Every Add works on two disjoint row ranges: folding the top half onto the bottom
    // half, with an odd leftover row merged into row 0 first. An in-place Add whose
    // source partially overlaps the destination is not allowed.

    __aicore__ inline void FoldRows(const LocalTensor<float> &buf, uint32_t rows)
    {
        while (rows > 1) {
            if ((rows & 1U) != 0) {
                Add(buf, buf, buf[(rows - 1) * dElems_], dElems_);   // row 0 += last row
                PipeBarrier<PIPE_V>();
                --rows;
            }
            const uint32_t half = rows >> 1;
            Add(buf, buf, buf[half * dElems_], half * dElems_);      // rows [0,half) += rows [half,rows)
            PipeBarrier<PIPE_V>();
            rows = half;
        }
    }

    // =====================================================================================
    // row-at-a-time path (d <= kRowMaxD, any alignment)
    // =====================================================================================

    __aicore__ inline void InitRowAtATime()
    {
        const uint32_t xBytes = AlignUp32(dElems_ * static_cast<uint32_t>(sizeof(TX)));
        const uint32_t gBytes = AlignUp32(dElems_ * static_cast<uint32_t>(sizeof(TG)));
        const uint32_t fBytes = AlignUp32(dFoldElems_ * static_cast<uint32_t>(sizeof(float)));

        pipe_.InitBuffer(xInQue_, 1, xBytes);
        pipe_.InitBuffer(dyInQue_, 1, xBytes);
        pipe_.InitBuffer(dxOutQue_, 1, xBytes);
        pipe_.InitBuffer(gammaInQue_, 1, gBytes);
        pipe_.InitBuffer(workBuf_, fBytes * kNumFloatBufs);

        LocalTensor<float> work = workBuf_.Get<float>();
        uint32_t off = 0;
        xF_ = work[off];      off += dFoldElems_;
        dyF_ = work[off];     off += dFoldElems_;
        gammaF_ = work[off];  off += dFoldElems_;
        xhat_ = work[off];    off += dFoldElems_;
        dz_ = work[off];      off += dFoldElems_;
        prod_ = work[off];    off += dFoldElems_;
        dsum_ = work[off];

        // Zero the whole working set once. Every per-row write below only touches lanes
        // [0, dElems_), so the padding of [dElems_, dFoldElems_) stays zero and the fixed
        // fold treats the row as zero padded. Doing it once here also avoids a
        // Duplicate() on an element-aligned (not 32B aligned) offset, which the vector
        // unit does not allow when d is not a multiple of 8.
        for (uint32_t i = 0; i < kNumFloatBufs; ++i) {
            Duplicate(work[i * dFoldElems_], 0.0f, dFoldElems_);
        }
        PipeBarrier<PIPE_V>();

        // gamma is loaded once and reused for every row
        LocalTensor<TG> gammaIn = gammaInQue_.AllocTensor<TG>();
        CopyRowIn(gammaIn, gammaGm_, 0, dElems_);
        gammaInQue_.EnQue(gammaIn);
        LocalTensor<TG> gammaInD = gammaInQue_.DeQue<TG>();
        CastToFloat(gammaF_, gammaInD, dElems_);
        PipeBarrier<PIPE_V>();
        gammaInQue_.FreeTensor(gammaInD);
    }

    __aicore__ inline void ProcessOneRow(int64_t row)
    {
        const uint64_t gmOffset = static_cast<uint64_t>(row) * static_cast<uint64_t>(d_);

        // ---- x / dy load
        LocalTensor<TX> xIn = xInQue_.AllocTensor<TX>();
        CopyRowIn(xIn, xGm_, gmOffset, dElems_);
        xInQue_.EnQue(xIn);
        LocalTensor<TX> dyIn = dyInQue_.AllocTensor<TX>();
        CopyRowIn(dyIn, dyGm_, gmOffset, dElems_);
        dyInQue_.EnQue(dyIn);

        // ---- widen to fp32 and zero pad to dFold
        // the padding lanes [dElems_, dFoldElems_) were zeroed once in InitRowAtATime
        LocalTensor<TX> xInD = xInQue_.DeQue<TX>();
        CastToFloat(xF_, xInD, dElems_);
        PipeBarrier<PIPE_V>();
        xInQue_.FreeTensor(xInD);
        LocalTensor<TX> dyInD = dyInQue_.DeQue<TX>();
        CastToFloat(dyF_, dyInD, dElems_);
        PipeBarrier<PIPE_V>();
        dyInQue_.FreeTensor(dyInD);

        // ---- scalar broadcast of the row reciprocal std
        const float rstd = rstdGm_.GetValue(static_cast<uint64_t>(row));

        // ---- xhat = x * rstd ; dgamma += dy * xhat
        Muls(xhat_, xF_, rstd, dFoldElems_);
        PipeBarrier<PIPE_V>();
        Mul(prod_, dyF_, xhat_, dFoldElems_);
        PipeBarrier<PIPE_V>();
        Add(dsum_, dsum_, prod_, dFoldElems_);
        PipeBarrier<PIPE_V>();

        // ---- dz = dy * gamma ; row product
        Mul(dz_, dyF_, gammaF_, dFoldElems_);
        PipeBarrier<PIPE_V>();
        Mul(prod_, dz_, xhat_, dFoldElems_);
        PipeBarrier<PIPE_V>();

        // ---- deterministic row sum: pairwise fold down to FOLD_LEN lanes
        uint32_t activeLen = dFoldElems_;
        while (activeLen > FOLD_LEN) {
            const uint32_t half = activeLen >> 1;
            Add(prod_, prod_, prod_[half], half);
            PipeBarrier<PIPE_V>();
            activeLen = half;
        }

        SetFlag<HardEvent::V_S>(evVS_);
        WaitFlag<HardEvent::V_S>(evVS_);
        float rowSum = 0.0f;
        for (uint32_t i = 0; i < FOLD_LEN; ++i) {
            rowSum += prod_.GetValue(i);
        }
        const float dot = rowSum / static_cast<float>(d_);
        SetFlag<HardEvent::S_V>(evSV_);
        WaitFlag<HardEvent::S_V>(evSV_);

        // ---- dx = rstd * (dz - xhat * dot)
        Muls(xF_, xhat_, dot, dFoldElems_);
        PipeBarrier<PIPE_V>();
        Sub(dz_, dz_, xF_, dFoldElems_);
        PipeBarrier<PIPE_V>();
        Muls(dz_, dz_, rstd, dFoldElems_);
        PipeBarrier<PIPE_V>();

        // ---- narrow and store
        LocalTensor<TX> dxOut = dxOutQue_.AllocTensor<TX>();
        CastFromFloat(dxOut, dz_, dElems_);
        PipeBarrier<PIPE_V>();
        dxOutQue_.EnQue(dxOut);
        LocalTensor<TX> dxOutD = dxOutQue_.DeQue<TX>();
        CopyRowOut(dxGm_, dxOutD, gmOffset, dElems_);
        dxOutQue_.FreeTensor(dxOutD);
    }

    // =====================================================================================
    // chunked path (d > kRowMaxD)
    // =====================================================================================

    __aicore__ inline void InitChunked()
    {
        const uint32_t xBytes = AlignUp32(chunkElems_ * static_cast<uint32_t>(sizeof(TX)));
        const uint32_t gBytes = AlignUp32(chunkElems_ * static_cast<uint32_t>(sizeof(TG)));
        const uint32_t fBytes = AlignUp32(chunkElems_ * static_cast<uint32_t>(sizeof(float)));

        pipe_.InitBuffer(xInQue_, 1, xBytes);
        pipe_.InitBuffer(dyInQue_, 1, xBytes);
        pipe_.InitBuffer(dxOutQue_, 1, xBytes);
        pipe_.InitBuffer(gammaInQue_, 1, gBytes);
        pipe_.InitBuffer(workBuf_, fBytes * 5);   // xF dyF gF xhat prod
        pipe_.InitBuffer(partBuf_, fBytes * 2);   // partF_ (loaded partial), contrib_

        LocalTensor<float> work = workBuf_.Get<float>();
        xF_ = work[0];
        dyF_ = work[chunkElems_];
        gammaF_ = work[chunkElems_ * 2];
        xhat_ = work[chunkElems_ * 3];
        dz_ = work[chunkElems_ * 4];
        partF_ = partBuf_.Get<float>();
        contrib_ = partBuf_.Get<float>()[chunkElems_];

        // The first row of the core overwrites its whole partial row chunk by chunk, so
        // dgammaPart needs no initialisation here (and none from the caller).
    }

    __aicore__ inline uint32_t ChunkLen(uint32_t c) const
    {
        const uint64_t col = static_cast<uint64_t>(c) * chunkElems_;
        const uint32_t left = dElems_ - static_cast<uint32_t>(col);
        return (left < chunkElems_) ? left : chunkElems_;
    }

    // Load one column chunk of x, dy and gamma, widen to fp32 and zero the tail of the
    // chunk so that the fixed fold below can run over the whole chunk length.
    __aicore__ inline void LoadChunkWide(uint64_t rowOffset, uint32_t c, uint32_t len)
    {
        const uint64_t col = static_cast<uint64_t>(c) * chunkElems_;
        LoadChunk(xInQue_, xGm_, rowOffset + col, len);
        LoadChunk(dyInQue_, dyGm_, static_cast<uint64_t>(rowOffset) + col, len);
        LoadChunk(gammaInQue_, gammaGm_, col, len);

        LocalTensor<TX> xIn = xInQue_.DeQue<TX>();
        LocalTensor<TX> dyIn = dyInQue_.DeQue<TX>();
        LocalTensor<TG> gIn = gammaInQue_.DeQue<TG>();
        Duplicate(xF_, 0.0f, chunkElems_);
        Duplicate(dyF_, 0.0f, chunkElems_);
        Duplicate(gammaF_, 0.0f, chunkElems_);
        PipeBarrier<PIPE_V>();
        CastToFloat(xF_, xIn, len);
        CastToFloat(dyF_, dyIn, len);
        CastToFloat(gammaF_, gIn, len);
        PipeBarrier<PIPE_V>();
        xInQue_.FreeTensor(xIn);
        dyInQue_.FreeTensor(dyIn);
        gammaInQue_.FreeTensor(gIn);
    }

    __aicore__ inline void ProcessRowChunked(int64_t row, bool firstRow)
    {
        const uint64_t rowOffset = static_cast<uint64_t>(row) * static_cast<uint64_t>(d_);
        const uint64_t partOffset = static_cast<uint64_t>(blockIdx_) * static_cast<uint64_t>(dFold_);
        const float rstd = rstdGm_.GetValue(static_cast<uint64_t>(row));
        const uint32_t tail = dFoldElems_ - dElems_;   // < 8, keeps the partial row 32B padded

        // ---- pass 1: row sum of dz * xhat, and the dgamma partial of this core.
        //      The partial lives in dgammaPart[row of this core] and is updated chunk by
        //      chunk; only this core ever touches that row, and the chunks are visited in
        //      ascending order, so the accumulation order is fixed.
        float rowSum = 0.0f;
        for (uint32_t c = 0; c < numChunks_; ++c) {
            const uint64_t col = static_cast<uint64_t>(c) * chunkElems_;
            const uint32_t len = ChunkLen(c);
            const uint32_t lenPad = (c + 1 == numChunks_) ? (len + tail) : len;
            LoadChunkWide(rowOffset, c, len);

            Muls(xhat_, xF_, rstd, chunkElems_);
            PipeBarrier<PIPE_V>();
            Mul(contrib_, dyF_, xhat_, chunkElems_);
            PipeBarrier<PIPE_V>();

            if (firstRow) {
                // The first row of this core initialises its partial, so every chunk of it
                // is written straight out. This must cover *all* chunks, not just chunk 0:
                // the following rows read dgammaPart back, and anything the first row left
                // untouched would be uninitialised memory that the reduction then adds in.
                // The last chunk writes len + tail, so the trailing padding of the partial
                // row is initialised here as well (contrib_'s tail is zero because xhat_ and
                // dyF_ are zero padded to chunkElems_).
                SetFlag<HardEvent::V_MTE3>(evVMte3_);
                WaitFlag<HardEvent::V_MTE3>(evVMte3_);
                CopyFloatsOut(dgPartGm_, contrib_, partOffset + col, lenPad);
            } else {
                SetFlag<HardEvent::MTE3_MTE2>(evMte3Mte2_);
                WaitFlag<HardEvent::MTE3_MTE2>(evMte3Mte2_);
                CopyFloats(partF_, dgPartGm_, partOffset + col, lenPad);
                SetFlag<HardEvent::MTE2_V>(evMte2V_);
                WaitFlag<HardEvent::MTE2_V>(evMte2V_);
                Add(partF_, partF_, contrib_, lenPad);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(evVMte3_);
                WaitFlag<HardEvent::V_MTE3>(evVMte3_);
                CopyFloatsOut(dgPartGm_, partF_, partOffset + col, lenPad);
            }
            // The next iteration recomputes contrib_ (and partF_ is reloaded), so the
            // store issued above must have consumed them first. MTE3 -> V: without this
            // wait the vector unit overwrites contrib_ while the store is still reading
            // it, which makes dgamma depend on DMA timing.
            SetFlag<HardEvent::MTE3_V>(evMte3V_);
            WaitFlag<HardEvent::MTE3_V>(evMte3V_);

            // dz = dy * gamma ; row product ; fixed fold, ascending chunk order
            Mul(dz_, dyF_, gammaF_, chunkElems_);
            PipeBarrier<PIPE_V>();
            Mul(dz_, dz_, xhat_, chunkElems_);
            PipeBarrier<PIPE_V>();
            uint32_t activeLen = chunkElems_;
            while (activeLen > FOLD_LEN) {
                const uint32_t half = activeLen >> 1;
                Add(dz_, dz_, dz_[half], half);
                PipeBarrier<PIPE_V>();
                activeLen = half;
            }
            SetFlag<HardEvent::V_S>(evVS_);
            WaitFlag<HardEvent::V_S>(evVS_);
            float chunkSum = 0.0f;
            for (uint32_t i = 0; i < FOLD_LEN; ++i) {
                chunkSum += dz_.GetValue(i);
            }
            rowSum += chunkSum;
            SetFlag<HardEvent::S_V>(evSV_);
            WaitFlag<HardEvent::S_V>(evSV_);
        }
        const float dot = rowSum / static_cast<float>(d_);

        // ---- pass 2: dx = rstd * (dz - xhat * dot), chunk by chunk
        for (uint32_t c = 0; c < numChunks_; ++c) {
            const uint64_t col = static_cast<uint64_t>(c) * chunkElems_;
            const uint32_t len = ChunkLen(c);
            LoadChunkWide(rowOffset, c, len);

            Muls(xhat_, xF_, rstd, len);
            PipeBarrier<PIPE_V>();
            Mul(dz_, dyF_, gammaF_, len);
            PipeBarrier<PIPE_V>();
            // gamma is no longer needed in this pass, so its buffer is the temporary
            Muls(gammaF_, xhat_, dot, len);
            PipeBarrier<PIPE_V>();
            Sub(dz_, dz_, gammaF_, len);
            PipeBarrier<PIPE_V>();
            Muls(dz_, dz_, rstd, len);
            PipeBarrier<PIPE_V>();

            LocalTensor<TX> dxOut = dxOutQue_.AllocTensor<TX>();
            CastFromFloat(dxOut, dz_, len);
            PipeBarrier<PIPE_V>();
            dxOutQue_.EnQue(dxOut);
            LocalTensor<TX> dxOutD = dxOutQue_.DeQue<TX>();
            CopyRowOut(dxGm_, dxOutD, rowOffset + col, len);
            dxOutQue_.FreeTensor(dxOutD);
            SetFlag<HardEvent::MTE3_MTE2>(evMte3Mte2_);
            WaitFlag<HardEvent::MTE3_MTE2>(evMte3Mte2_);
        }
    }

    template <typename T>
    __aicore__ inline void LoadChunk(TQue<QuePosition::VECIN, 1> &que, const GlobalTensor<T> &gm,
                                     uint64_t offset, uint32_t len)
    {
        LocalTensor<T> buf = que.AllocTensor<T>();
        CopyRowIn(buf, gm, offset, len);
        que.EnQue(buf);
    }

    __aicore__ inline void WritePartial()
    {
        SetFlag<HardEvent::V_MTE3>(evVMte3_);
        WaitFlag<HardEvent::V_MTE3>(evVMte3_);
        DataCopy(dgPartGm_[static_cast<uint64_t>(blockIdx_) * static_cast<uint64_t>(dFold_)], dsum_, dFoldElems_);
        PipeBarrier<PIPE_ALL>();
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> xInQue_;
    TQue<QuePosition::VECIN, 1> dyInQue_;
    TQue<QuePosition::VECIN, 1> gammaInQue_;
    TQue<QuePosition::VECOUT, 1> dxOutQue_;
    TBuf<TPosition::VECCALC> workBuf_;
    TBuf<TPosition::VECCALC> stageBuf_;
    TBuf<TPosition::VECCALC> gammaStageBuf_;
    TBuf<TPosition::VECCALC> tileBuf_;
    TBuf<TPosition::VECCALC> partBuf_;

    LocalTensor<TX> xStage_;
    LocalTensor<TX> dyStage_;

    LocalTensor<float> xF_;
    LocalTensor<float> dyF_;
    LocalTensor<float> gammaF_;
    LocalTensor<float> gammaB_;
    LocalTensor<float> xhat_;
    LocalTensor<float> dz_;
    LocalTensor<float> prod_;
    LocalTensor<float> dsum_;
    LocalTensor<float> dotRed_;
    LocalTensor<float> rstdS_;
    LocalTensor<float> partF_;
    LocalTensor<float> contrib_;

    GlobalTensor<TX> xGm_;
    GlobalTensor<float> rstdGm_;
    GlobalTensor<TG> gammaGm_;
    GlobalTensor<TX> dyGm_;
    GlobalTensor<TX> dxGm_;
    GlobalTensor<float> dgPartGm_;

    int64_t m_ = 0;
    int64_t d_ = 0;
    int64_t dFold_ = 0;
    int64_t rowsPerCore_ = 0;
    uint32_t blockIdx_ = 0;
    uint32_t dElems_ = 0;
    uint32_t dFoldElems_ = 0;
    uint32_t padElems_ = 0;
    uint32_t rowsPerTile_ = 0;
    uint32_t chunkElems_ = 0;
    uint32_t numChunks_ = 0;
    bool useTiled_ = false;
    bool useChunked_ = false;

    TEventID evVS_ = 0;
    TEventID evSV_ = 0;
    TEventID evVMte3_ = 0;
    TEventID evMte2S_ = 0;
    TEventID evMte2V_ = 0;
    TEventID evVMte2_ = 0;
    TEventID evMte3Mte2_ = 0;
    TEventID evMte3V_ = 0;
    TEventID evSMte2_ = 0;
};

} // namespace

extern "C" __global__ __aicore__ void fla_rms_norm_bwd_dx(GM_ADDR x, GM_ADDR rstd, GM_ADDR gamma, GM_ADDR dy,
                                                          GM_ADDR dx, GM_ADDR dgammaPart, GM_ADDR workspace,
                                                          GM_ADDR tiling)
{
    (void)workspace;
    REGISTER_TILING_DEFAULT(FlaRmsNormBwdDxTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    FlaRmsNormBwdDxKernel<DTYPE_X, DTYPE_GAMMA> kernel;
    kernel.Init(x, rstd, gamma, dy, dx, dgammaPart, &tilingData);
    kernel.Process();
    kernel.Release();
}
