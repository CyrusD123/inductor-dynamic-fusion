"""Repro 2 (smallest example): reduce over C, then split C into heads.

    y = softmax(x, dim=-1)            # x: (B, T, C), reduction over C
    y.view(B, T, H, C // H) * 2.0     # pointwise consumer on the split view

Static shapes: 1 kernel (reduction + consumer fused).
Dynamic C:     2 kernels, and the softmax output is written to memory and
               read back.

Why: Inductor fuses a reduction with a later pointwise only if the
pointwise iteration space has the same total size as the reduction's
(numel == xnumel * rnumel). Here that is

    pointwise: B*T*H*(C//H)      vs.     reduction: (B*T) * C

H*(C//H) == C holds for every concrete C the view accepts, but the
scheduler compares symbolic expressions. FloorDiv(C, H)*H does not simplify
to C, even with torch._check(C % H == 0). The scheduler logs
"cannot fuse ...: nodes numel incompatibility".

The same pattern shows up in real models, e.g. OLMo-2-style QK-norm
(RMSNorm over the full hidden dim, then reshape to heads for RoPE).

Controls in this file:
  * no split                       -> fuses (dynamic == static)
  * split written as -1 / unflatten -> still breaks (not a spelling issue)
  * torch._check(C % H == 0)       -> still breaks (divisibility unused)
  * "merge direction": pass (B,T,H,Dh) with Dh dynamic, flatten to C=H*Dh,
    then split back -> fuses, because (H*Dh)//H simplifies to Dh.
    This hints at a fix: rewrite a symbol known to be divisible by H as H*s'.
"""

import torch

from harness import Case, run_cases

B, T, H, C = 2, 512, 8, 512


# ---- the pattern (softmax and RMSNorm flavors) -------------------------------

def softmax_split(x):
    B, T, C = x.shape
    return torch.softmax(x, -1).view(B, T, H, C // H) * 2.0


def rmsnorm_split(x, w_head):
    """OLMo-2-style QK-norm: RMSNorm over all of C, then split into heads and
    apply a per-head-dim scale (a stand-in for RoPE / any pointwise)."""
    B, T, C = x.shape
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    return y.view(B, T, H, C // H) * w_head


def layernorm_split(x):
    B, T, C = x.shape
    y = torch.nn.functional.layer_norm(x, (C,))
    return y.view(B, T, H, C // H) * 2.0


# ---- controls -----------------------------------------------------------------

def no_split(x):
    return torch.softmax(x, -1) * 2.0


def softmax_split_minus1(x):
    B, T, C = x.shape
    return torch.softmax(x, -1).view(B, T, H, -1) * 2.0


def softmax_split_unflatten(x):
    return torch.softmax(x, -1).unflatten(-1, (H, -1)) * 2.0


def softmax_split_checked(x):
    B, T, C = x.shape
    torch._check(C % H == 0)
    return torch.softmax(x, -1).view(B, T, H, C // H) * 2.0


def merge_direction(x4):  # x4: (B, T, H, Dh) with Dh dynamic
    B, T, Hh, Dh = x4.shape
    y = torch.softmax(x4.reshape(B, T, Hh * Dh), -1)
    return y.view(B, T, Hh, Dh) * 2.0


R = torch.randn
x_in = lambda d: (R(B, T, C, device=d),)
DYN_C = {0: [1, 2]}  # T and C dynamic

cases = [
    Case("softmax -> view(B,T,H,C//H)", softmax_split, x_in, DYN_C),
    Case("rmsnorm -> view(B,T,H,C//H)", rmsnorm_split,
         lambda d: (R(B, T, C, device=d), R(C // H, device=d)), DYN_C),
    Case("layernorm -> view(B,T,H,C//H)", layernorm_split, x_in, DYN_C),
    Case("[ctrl] softmax, no split", no_split, x_in, DYN_C),
    Case("[ctrl] split as view(...,-1)", softmax_split_minus1, x_in, DYN_C),
    Case("[ctrl] split as unflatten", softmax_split_unflatten, x_in, DYN_C),
    Case("[ctrl] + torch._check(C%H==0)", softmax_split_checked, x_in, DYN_C),
    Case("[ctrl] merge direction (Dh dyn)", merge_direction,
         lambda d: (R(B, T, H, C // H, device=d),), {0: [1, 3]}),
]

if __name__ == "__main__":
    run_cases(cases)
