#ifndef KDA_GATE_BWD_POST_COMMON_H
#define KDA_GATE_BWD_POST_COMMON_H

#include <cstdint>

namespace KDA {

// Per-(head, chunk) reduction slot of the split-reduce path: 128 dbias + 8 dA
// floats, of which dA[0] is live.  The produce launch writes the slots and a
// second launch of the same op folds them; the launch boundary on one stream
// orders the two phases, so no per-call token is needed.
constexpr int32_t KDA_GATE_POST_SLOT_FLOATS = 136;
constexpr int32_t KDA_GATE_POST_PAYLOAD_FLOATS = 136;
// The fold reads a batch of consecutive slots as one contiguous run, which is
// only correct while a slot holds nothing but its payload.
static_assert(KDA_GATE_POST_SLOT_FLOATS == KDA_GATE_POST_PAYLOAD_FLOATS,
              "split-reduce fold batches assume slot == payload width");

// Private tiling contract for the delivery fallback that applies the raw-gate
// chain rule after the fused ABC kernel has produced chunk-local accumulated
// dg.  Shared by the op_host tiling/op_api and the op_kernel that consumes it;
// not part of the public aclnn ABI.
#define KDA_GATE_POST_SCALAR_FIELDS \
    int32_t batch; \
    int32_t seqlen; \
    int32_t headNum; \
    int32_t keyDim; \
    int32_t chunkSize; \
    int32_t chunkNum; \
    int32_t chunkNumPerBatch; \
    int32_t usedCoreNum; \
    int8_t isVarLen; \
    int8_t hasDtBias; \
    int8_t inputScanned; \
    int8_t splitMode; \
    int32_t splitParts; \
    int32_t chunksPerPart; \
    float lowerBound;

struct KdaGateBwdPostScalarTilingData {
    KDA_GATE_POST_SCALAR_FIELDS
};

struct KdaGateBwdPostTilingData {
    KDA_GATE_POST_SCALAR_FIELDS
};

#undef KDA_GATE_POST_SCALAR_FIELDS

} // namespace KDA

#endif // KDA_GATE_BWD_POST_COMMON_H
