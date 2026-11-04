"""Repro 1: pytorch/pytorch#181108, XSA projection (inference / forward only).

Pattern: L2-normalize a (B,T,Hkv,D) tensor, broadcast it against a
(B,T,Hkv,rep,D) *view* of a (B,T,Hq,D) tensor, take a dot-product reduction
over D, and subtract the projection. Also emit two per-row side outputs.

Why it breaks with dynamic shapes: `output.view(B, T, Hkv, Hq // Hkv, D)`
splits the symbolic dim Hq into (Hkv, Hq//Hkv). With static shapes Inductor
sees concrete sizes and can tell that every buffer iterates the same
(B*T*Hq) x D space. With symbolic Hq/D it has to reason about
`Hkv * (s0 // Hkv) == s0` and about the reduction size D being the same
symbol in every node. When that reasoning fails, nodes end up in different
kernels.
"""

import torch
import torch.nn.functional as F

from harness import Case, run_cases

B, T, HKV, N_REP, D = 1, 2048, 4, 8, 64
HQ = HKV * N_REP


def xsa_unsqueeze(output, xv):
    head_dim = xv.shape[-1]
    n_kv = xv.shape[2]
    n_heads = output.shape[2]
    torch._check(output.shape[3] == head_dim)
    torch._check(n_heads % n_kv == 0)
    n_rep = n_heads // n_kv

    vn = F.normalize(xv, dim=-1, eps=1e-6).unsqueeze(3)
    y = output.view(*output.shape[:2], n_kv, n_rep, head_dim)
    ps = (y * vn).sum(dim=-1, keepdim=True)
    cos_sim = ps.squeeze(-1).abs()
    proj_frac = cos_sim / y.norm(dim=-1).clamp(min=1e-6)
    result = (y - ps * vn).view_as(output)
    return result, cos_sim, proj_frac


def xsa_expand(output, xv):
    Bb, Tt, Hq, Dd = output.shape
    Hkv = xv.shape[2]
    torch._check(xv.shape[3] == Dd)
    torch._check(Hq % Hkv == 0)
    gs = Hq // Hkv

    v_exp = xv[:, :, :, None, :].expand(Bb, Tt, Hkv, gs, Dd).reshape(Bb, Tt, Hq, Dd)
    vn = F.normalize(v_exp, p=2, dim=-1, eps=1e-6)
    coeff = (output * vn).sum(dim=-1, keepdim=True)
    projected = output - coeff * vn
    cg = coeff.squeeze(-1).view(Bb, Tt, Hkv, gs)
    cos_sim = cg.abs()
    yg = output.view(Bb, Tt, Hkv, gs, Dd)
    proj_frac = cos_sim / yg.norm(dim=-1).clamp(min=1e-6)
    return projected, cos_sim, proj_frac


def inputs(device):
    return (
        torch.randn(B, T, HQ, D, device=device),
        torch.randn(B, T, HKV, D, device=device),
    )


# Same dims the issue marks dynamic: Hq and D on `output`, D on `xv`.
DYN = {0: [2, 3], 1: [3]}

cases = [
    Case("xsa_unsqueeze (issue #181108)", xsa_unsqueeze, inputs, DYN),
    Case("xsa_expand    (issue #181108)", xsa_expand, inputs, DYN),
]

if __name__ == "__main__":
    run_cases(cases)
