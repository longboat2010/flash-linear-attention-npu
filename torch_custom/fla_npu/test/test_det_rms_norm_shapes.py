"""Acceptance sweep for the deterministic Ascend C RMSNorm backward.

Checks, for every shape in the envelope the vendor operator accepts, that
``fla_npu``'s patched ``torch_npu.npu_rms_norm``

  1. handles the backward natively (no fallback, see ``det_rms_norm.stats()``),
  2. matches a float64 reference of the same formula (dx and dgamma),
  3. is bitwise reproducible: two identical calls give ``torch.equal`` results.

Run after the wheel is installed:

    python test/test_det_rms_norm_shapes.py

Exits non-zero on the first failing group; prints one line per case.
"""

import itertools
import sys

import torch
import torch_npu

import fla_npu  # noqa: F401  (installs the deterministic backward by default)
from fla_npu.ops import det_rms_norm

EPS = 1e-6
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
# relative tolerance of a single fused backward against the float64 reference
TOL = {"fp16": 3e-3, "bf16": 3e-2, "fp32": 3e-5}

D_LIST = [1, 2, 3, 7, 8, 15, 16, 17, 24, 31, 63, 64, 100, 127, 128, 255, 256,
          1000, 2048, 4096, 4097, 5120, 6144, 7168, 8192, 12288, 16384]
M_LIST = [1, 2, 23, 24, 25, 47, 48, 49, 63, 64, 65, 257]


def ref(x, rstd, gamma, dy):
    """float64 reference, using the rstd the vendor forward produced."""
    d = x.shape[-1]
    xd = x.double().reshape(-1, d)
    dyd = dy.double().reshape(-1, d)
    gd = gamma.double()
    r = rstd.double().reshape(-1, 1)
    xhat = xd * r
    dz = dyd * gd
    dot = (dz * xhat).mean(-1, keepdim=True)
    return (r * (dz - xhat * dot)).reshape(x.shape), (dyd * xhat).sum(0)


def run(x, gamma, dy):
    xr = x.detach().clone().requires_grad_(True)
    gr = gamma.detach().clone().requires_grad_(True)
    y, rstd = torch_npu.npu_rms_norm(xr, gr, epsilon=EPS)
    y.backward(dy)
    torch.npu.synchronize()
    return xr.grad, gr.grad, rstd


def rel(a, b):
    a = a.float()
    b = b.float()
    den = b.abs().max().clamp_min(1e-30)
    return ((a - b).abs().max() / den).item()


def one(d, m, dt_name, gdt_name, rank):
    dt = DTYPES[dt_name]
    gdt = DTYPES[gdt_name]
    if rank == 1:
        shape = (d,)
        m = 1
    elif rank == 2:
        shape = (m, d)
    elif rank == 3:
        shape = (2, max(1, m // 2), d)
    else:
        shape = (2, 2, max(1, m // 4), d)
    tag = f"d={d:6d} m={m:4d} rank={rank} {dt_name}/{gdt_name}"

    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dt, device="npu")
    gamma = torch.rand(d, dtype=gdt, device="npu") + 0.5
    dy = torch.randn(shape, dtype=dt, device="npu")

    before = det_rms_norm.stats()
    dx1, dg1, rstd = run(x, gamma, dy)
    after = det_rms_norm.stats()
    if after["fallback"] != before["fallback"]:
        return f"FALLBACK {tag} {det_rms_norm.reasons()}"
    if after["patched"] == before["patched"]:
        return f"NOT_PATCHED {tag}"

    ref_dx, ref_dg = ref(x, rstd, gamma, dy)
    tol = TOL[dt_name]
    e_dx = rel(dx1, ref_dx)
    e_dg = rel(dg1, ref_dg)
    if e_dx > tol or e_dg > tol:
        return f"ACCURACY {tag} dx={e_dx:.3e} dg={e_dg:.3e} tol={tol:.1e}"
    if not torch.isfinite(dx1.float()).all() or not torch.isfinite(dg1.float()).all():
        return f"NONFINITE {tag}"

    dx2, dg2, _ = run(x, gamma, dy)
    if not (torch.equal(dx1, dx2) and torch.equal(dg1, dg2)):
        return f"NONDET {tag}"

    print(f"ok  {tag}  dx={e_dx:.2e} dg={e_dg:.2e}", flush=True)
    return None


def main():
    failures = []
    cases = []
    # full d / dtype sweep at rank 2
    for dt_name in DTYPES:
        for d in D_LIST:
            cases.append((d, 64, dt_name, dt_name, 2))
    # gamma in fp32 (except when x is already fp32)
    for dt_name in ("fp16", "bf16"):
        for d in D_LIST:
            cases.append((d, 64, dt_name, "fp32", 2))
    # row counts around the core-partition boundaries and every rank
    for m in M_LIST:
        for rank in (1, 2, 3, 4):
            cases.append((128, m, "bf16", "bf16", rank))
    for d in (1, 100, 4097):
        for rank in (1, 3, 4):
            cases.append((d, 25, "bf16", "fp32", rank))

    for case in cases:
        err = one(*case)
        if err is not None:
            print("FAIL", err, flush=True)
            failures.append(err)

    print(f"\n{len(cases) - len(failures)}/{len(cases)} cases passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
