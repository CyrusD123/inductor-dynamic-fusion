"""Repro 3: dims that are equal at runtime but are different symbols at compile time.
(BladeDISC, Sec. 2.2 / 4.2: fusion needs *dim equality*, not dim values.)

    y * rsqrt(mean(x**2, dim=-1))     # x: (M, D1), y: (M, D2), D1 == D2 at runtime

With static shapes D1 = D2 = 512, so the reduction over x and the pointwise
over y share one iteration space and fuse into 1 kernel.

With dynamic shapes Dynamo gives D1 and D2 separate symbols. Broadcasting
only unifies the M dims, because the reduction result has shape (M, 1) and
never meets D1 against D2. The scheduler sees reduction numel M*D1 vs.
pointwise numel M*D2, logs "nodes numel incompatibility", and emits 2
kernels.

Adding torch._check(D1 == D2) gives Dynamo the missing equality fact. It
substitutes one symbol for the other and fusion comes back. A compiler pass
that recovered these equalities automatically (or emitted a guarded fused
version, as SoD2 does with multi-versioning) would fix this without user
annotations.

A realistic version: normalize one tensor with statistics from another of
the same width, e.g. scaling an update by the RMS of the parameter it
updates, or adaptive-norm variants where stats come from one stream and
are applied to another.
"""

import torch

from harness import Case, run_cases

M, D = 4096, 512


def stats_from_other(x, y):
    return y * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


def stats_from_other_checked(x, y):
    torch._check(x.shape[1] == y.shape[1])
    return y * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


def stats_from_self(x, y):  # control: reduction and consumer share x's symbols
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) + y


R = torch.randn
xy = lambda d: (R(M, D, device=d), R(M, D, device=d))
DYN = {0: [0, 1], 1: [0, 1]}

cases = [
    Case("y * rsqrt(mean(x^2))", stats_from_other, xy, DYN),
    Case("[fix] + torch._check(D1 == D2)", stats_from_other_checked, xy, DYN),
    Case("[ctrl] x * rsqrt(mean(x^2)) + y", stats_from_self, xy, DYN),
]

if __name__ == "__main__":
    run_cases(cases)
