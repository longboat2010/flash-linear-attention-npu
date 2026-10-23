#include "chunk_kda_bwd.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "acl/acl_base_rt.h"
#include <string>

// Single source of truth for the split-reduce slot layout.  The op_kernel
// header only depends on <cstdint>, so the opapi translation unit includes it
// by relative path; do not re-mirror the stride here.  A stale mirror would
// make the slot plane the wrong size and the fold would read past its rows.
#include "../../op_kernel/kda_gate_bwd_post_common.h"

using KDA::KDA_GATE_POST_SLOT_FLOATS;

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ChunkKdaBwd);
OP_TYPE_REGISTER(KdaGateBwdPost);
OP_TYPE_REGISTER(KdaGateBwdPostVarlen);

ChunkKdaBwdOutputs ChunkKdaBwd(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *beta, const aclTensor *gk,
    const aclTensor *aqk, const aclTensor *akk,
    const aclTensor *wOptional, const aclTensor *qgOptional,
    const aclTensor *kgOptional, const aclTensor *vNewOptional,
    const aclTensor *hOptional, const aclTensor *dO,
    const aclTensor *rawGOptional, const aclTensor *aLogOptional,
    const aclTensor *dtBiasOptional, const aclTensor *cuSeqlensOptional,
    const aclTensor *chunkIndicesOptional,
    float scale, int64_t chunkSize, bool safeGate,
    bool useGateInKernel, float lowerBound,
    bool disableRecompute, bool useExp2, bool stateVFirst,
    const aclTensor *dq,
    const aclTensor *dk, const aclTensor *dv,
    const aclTensor *db, const aclTensor *dg,
    const aclTensor *dAOptional,
    const aclTensor *dBiasOptional, aclOpExecutor *executor)
{
    L0_DFX(ChunkKdaBwd, q, k, v, beta, gk, aqk, akk, wOptional,
           qgOptional, kgOptional, vNewOptional, hOptional, dO,
           rawGOptional, aLogOptional, dtBiasOptional,
           cuSeqlensOptional, chunkIndicesOptional, scale, chunkSize,
           safeGate, useGateInKernel, lowerBound, disableRecompute, useExp2,
           stateVFirst, dq, dk, dv, db, dg,
           dAOptional, dBiasOptional);

    // Kernel C writes the chunk-local, unscanned gate gradient into the
    // final dg storage.  The following AIV gate kernel consumes and updates
    // that storage in place, avoiding a full-size cross-kernel workspace.
    const aclTensor *dgBase = dg;

    const char *socName = aclrtGetSocName();
    const auto qShape = q->GetViewShape();
    // Keep the high-performance fused gate phase for the dense, aligned A5
    // safe-gate path.  Other layouts and SoCs continue to use the standalone
    // post kernel.
    const bool fuseGateInC = useGateInKernel && safeGate &&
        socName != nullptr &&
        std::string(socName).find("950") != std::string::npos &&
        cuSeqlensOptional == nullptr && qShape.GetDimNum() == 4 &&
        qShape.GetDim(1) > 0 && qShape.GetDim(2) % chunkSize == 0;
    const bool deferGatePost = useGateInKernel && !fuseGateInC;

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ChunkKdaBwd,
        OP_INPUT(q, k, v, beta, gk, aqk, akk, wOptional, qgOptional,
                 kgOptional, vNewOptional, hOptional, dO,
                 rawGOptional, aLogOptional, dtBiasOptional,
                 cuSeqlensOptional, chunkIndicesOptional),
        OP_OUTPUT(dq, dk, dv, db, dgBase, dAOptional, dBiasOptional),
        OP_ATTR(scale, chunkSize, safeGate, fuseGateInC, lowerBound,
                disableRecompute, useExp2, stateVFirst,
                deferGatePost));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "ADD_TO_LAUNCHER_LIST_AICORE ChunkKdaBwd failed.");
        return {};
    }
    if (deferGatePost) {
        if (cuSeqlensOptional != nullptr) {
            // Split-reduce: fan the per-chunk reduction over (head, part)
            // tasks into a slot plane in one launch, then fold the slots in a
            // second launch of this same op, ordered by the launch boundary.
            // The split is skipped for a non-rank-3 raw gate, on A5 (DAV_3510)
            // and when the slot allocation fails; each of those runs the
            // legacy single launch instead.
            const bool splitArchOk =
                op::GetCurrentPlatformInfo().GetCurNpuArch() !=
                NpuArch::DAV_3510;
            bool splitReduce =
                splitArchOk && safeGate &&
                rawGOptional != nullptr &&
                rawGOptional->GetViewShape().GetDimNum() == 3;
            // Always hand the kernel a real slot plane: this launcher takes no
            // null OPTIONAL output, and aliasing dg would let any future slot
            // write corrupt dg.  The legacy arm gets a one-element placeholder
            // that the kernel never touches.
            op::Shape slotShape;
            if (splitReduce) {
                slotShape.AppendDim(
                    rawGOptional->GetViewShape().GetDim(0));
                slotShape.AppendDim(
                    chunkIndicesOptional->GetViewShape().GetShapeSize() / 2);
                slotShape.AppendDim(KDA_GATE_POST_SLOT_FLOATS);
            } else {
                slotShape.AppendDim(1);
            }
            const aclTensor *partials = executor->AllocTensor(
                slotShape, DataType::DT_FLOAT, Format::FORMAT_ND);
            if (partials == nullptr && splitReduce) {
                // The legacy launch needs no slots, so a failed slot
                // allocation costs the split only, not the result.
                OP_LOGW("Gate post split slot allocation failed; using the "
                        "single legacy launch.");
                splitReduce = false;
                op::Shape placeholder;
                placeholder.AppendDim(1);
                partials = executor->AllocTensor(
                    placeholder, DataType::DT_FLOAT, Format::FORMAT_ND);
            }
            if (partials == nullptr) {
                OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                        "AllocTensor for gate post slots failed.");
                return {};
            }
            // Phase 1 writes the plane and phase 2 reads it; both launch with
            // the same tensor, which is what carries the partials across.
            const int64_t splitMode = splitReduce ? 1 : 0;
            ret = ADD_TO_LAUNCHER_LIST_AICORE(
                KdaGateBwdPostVarlen,
                OP_INPUT(dgBase, rawGOptional, aLogOptional, dtBiasOptional,
                         cuSeqlensOptional, chunkIndicesOptional),
                OP_OUTPUT(dg, dAOptional, dBiasOptional, partials),
                OP_ATTR(chunkSize, safeGate, lowerBound, false, splitMode));
            if (ret != ACLNN_SUCCESS) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                        "ADD_TO_LAUNCHER_LIST_AICORE GatePost produce failed.");
                return {};
            }
            if (splitReduce) {
                ret = ADD_TO_LAUNCHER_LIST_AICORE(
                    KdaGateBwdPostVarlen,
                    OP_INPUT(dgBase, rawGOptional, aLogOptional,
                             dtBiasOptional, cuSeqlensOptional,
                             chunkIndicesOptional),
                    OP_OUTPUT(dg, dAOptional, dBiasOptional, partials),
                    OP_ATTR(chunkSize, safeGate, lowerBound, false,
                            static_cast<int64_t>(2)));
                // A fold failure is reported by the shared check below.
            }
        } else {
            ret = ADD_TO_LAUNCHER_LIST_AICORE(
                KdaGateBwdPost,
                OP_INPUT(dgBase, rawGOptional, aLogOptional, dtBiasOptional,
                         cuSeqlensOptional, chunkIndicesOptional),
                OP_OUTPUT(dg, dAOptional, dBiasOptional),
                OP_ATTR(chunkSize, safeGate, lowerBound, false));
        }
        if (ret != ACLNN_SUCCESS) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "ADD_TO_LAUNCHER_LIST_AICORE GatePost failed.");
            return {};
        }
    }
    return {dq, dk, dv, db, dg, dAOptional, dBiasOptional};
}

} // namespace l0op
