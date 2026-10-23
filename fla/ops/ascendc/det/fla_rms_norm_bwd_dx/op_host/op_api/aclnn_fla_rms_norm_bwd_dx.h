
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_FLA_RMS_NORM_BWD_DX_H_
#define ACLNN_FLA_RMS_NORM_BWD_DX_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnFlaRmsNormBwdDxGetWorkspaceSize
 * parameters :
 * x : required
 * rstd : required
 * gamma : required
 * dy : required
 * dx : required
 * dgammaPart : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnFlaRmsNormBwdDxGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *rstd,
    const aclTensor *gamma,
    const aclTensor *dy,
    const aclTensor *dx,
    const aclTensor *dgammaPart,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnFlaRmsNormBwdDx
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnFlaRmsNormBwdDx(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
