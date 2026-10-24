
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_FLA_RMS_NORM_BWD_DG_H_
#define ACLNN_FLA_RMS_NORM_BWD_DG_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnFlaRmsNormBwdDgGetWorkspaceSize
 * parameters :
 * x : required
 * gamma : required
 * dgammaPart : required
 * dgamma : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnFlaRmsNormBwdDgGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *gamma,
    const aclTensor *dgammaPart,
    const aclTensor *dgamma,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnFlaRmsNormBwdDg
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnFlaRmsNormBwdDg(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
