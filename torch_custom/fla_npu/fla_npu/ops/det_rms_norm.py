"""Bitwise-reproducible RMSNorm backward backed by our own Ascend C operators.

The vendor ``torch_npu.npu_rms_norm`` forward is kept unchanged; only the
backward is redirected to two Ascend C operators that avoid every cross-core
reduction and therefore do not depend on core scheduling:

``npu_fla_rms_norm_bwd_dx``
    one core per contiguous block of rows computes ``dx`` and accumulates a
    fp32 partial ``dgamma`` in a fixed fold order;
``npu_fla_rms_norm_bwd_dg``
    a single core adds the partial rows in ascending core order.

Both are implemented in ``fla/ops/ascendc/det/``.  No torch native math is
involved in the backward, and no deterministic-level switch is needed.

``import fla_npu`` installs this wrapper by default; ``install()`` is public for
callers that import fla_npu lazily, and ``uninstall()`` restores the vendor
operator.  Setting ``FLA_NPU_DISABLE_DET_BACKWARD=1`` before importing fla_npu
skips the install altogether.

The supported envelope matches the vendor operator: any hidden size up to
65536 (no alignment requirement), any rank >= 1, and fp16 / bf16 / fp32
activations with a gamma of the same dtype or fp32.  Hidden sizes above 4096
are walked in column chunks, which needs a second pass over the row.
Anything outside that falls back to the vendor backward, so a run never
breaks; every fallback is counted and reported at exit (see ``stats()`` /
``reasons()``).
"""

from __future__ import annotations

import warnings

import torch

_ORIG = None
_STATS = {"patched": 0, "fallback": 0}
_REASONS = {}
_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_MAX_D = 65536       # same envelope as the vendor operator
_ROW_MAX_D = 4096    # d above this runs one extra pass over the row (see the kernels)


def _supported(x, gamma):
    if not isinstance(x, torch.Tensor) or not isinstance(gamma, torch.Tensor):
        return "not a tensor"
    if x.dim() < 1 or gamma.dim() != 1:
        return f"rank x={x.dim()} gamma={gamma.dim()}"
    if x.shape[-1] != gamma.shape[0]:
        return f"d mismatch {x.shape[-1]} vs {gamma.shape[0]}"
    if x.dtype not in _DTYPES or gamma.dtype not in _DTYPES:
        return f"dtype {x.dtype}/{gamma.dtype}"
    if gamma.dtype not in (x.dtype, torch.float32):
        # gamma must match x or be fp32; the operator has no other combination
        return f"dtype {x.dtype}/{gamma.dtype}"
    d = int(x.shape[-1])
    if d <= 0 or d > _MAX_D:
        return f"d={d} unsupported"
    if x.numel() == 0:
        return "empty"
    if not x.is_npu:
        return "not on npu"
    return None


def rms_norm_bwd(x, rstd, gamma, dy):
    """dx and dgamma of a zero-centred RMSNorm, deterministic by construction."""
    d = int(dy.shape[-1])
    x2 = x.reshape(-1, d)
    if not x2.is_contiguous():
        _STATS["x_copy"] = _STATS.get("x_copy", 0) + 1
        x2 = x2.contiguous()
    dy2 = dy.reshape(-1, d)
    if not dy2.is_contiguous():
        _STATS["dy_copy"] = _STATS.get("dy_copy", 0) + 1
        dy2 = dy2.contiguous()
    r = rstd.reshape(-1).float()
    dx2, part = torch.ops.npu.npu_fla_rms_norm_bwd_dx(x2, r, gamma, dy2)
    dgamma = torch.ops.npu.npu_fla_rms_norm_bwd_dg(x2, gamma, part)
    return dx2.reshape(dy.shape), dgamma


class _DetRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, eps):
        y, rstd = _ORIG(x, gamma, epsilon=eps)
        ctx.save_for_backward(x.reshape(-1, x.shape[-1]), gamma, rstd.reshape(-1))
        ctx.mark_non_differentiable(rstd)
        return y, rstd

    @staticmethod
    def backward(ctx, dy, _drstd):
        x, gamma, rstd = ctx.saved_tensors
        dx, dgamma = rms_norm_bwd(x, rstd, gamma, dy)
        return dx, dgamma, None


def npu_rms_norm(x, gamma, epsilon=1e-6):
    reason = _supported(x, gamma)
    if reason is not None:
        if reason not in _REASONS:
            warnings.warn(
                f"[det_rms_norm] shape not supported by the deterministic Ascend C backward "
                f"({reason}); falling back to the vendor kernel, whose results are not "
                f"bitwise reproducible.",
                RuntimeWarning,
                stacklevel=2,
            )
        _STATS["fallback"] += 1
        _REASONS[reason] = _REASONS.get(reason, 0) + 1
        return _ORIG(x, gamma, epsilon=epsilon)
    _STATS["patched"] += 1
    return _DetRMSNorm.apply(x, gamma, float(epsilon))


def install():
    """Route ``torch_npu.npu_rms_norm`` through the Ascend C backward."""
    import torch_npu

    global _ORIG
    if _ORIG is not None:
        return
    _ORIG = torch_npu.npu_rms_norm
    torch_npu.npu_rms_norm = npu_rms_norm


def uninstall():
    import torch_npu

    global _ORIG
    if _ORIG is None:
        return
    torch_npu.npu_rms_norm = _ORIG
    _ORIG = None


def stats():
    return dict(_STATS)


def reasons():
    return dict(_REASONS)


def _report():
    if _ORIG is None:
        return
    import sys
    holders = []
    for name, mod in list(sys.modules.items()):
        try:
            if getattr(mod, "npu_rms_norm", None) is _ORIG:
                holders.append(name)
        except Exception:
            pass
    print(f"[det_rms_norm] final stats {_STATS} reasons {_REASONS} "
          f"stale_alias_holders={holders}", flush=True)


import atexit as _atexit

_atexit.register(_report)
