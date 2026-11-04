"""Repro 4: the motivating examples from the BladeDISC and SoD2 papers, as PyTorch.

Both papers target compilers (XLA/MLIR-based, DNNFusion) that have no symbolic
shapes. Inductor does have them, and these examples are checks on that:
current Inductor fuses them with dynamic shapes, which narrows the project to
the cases where its symbolic reasoning is too weak (see repro_split_heads.py,
repro_dim_equality.py, repro_xsa.py).

BladeDISC Fig. 5: softmax (reduce-max, reduce-sum, broadcast, div) on a
  tensor<?x?> whose rows are a reshape of tensor<?x?x?>, i.e. S0 = S2*S3.
  Stitching requires proving the reduce's "group dim" equals S2*S3.
  Inductor: merged dims become a product s0*s1, so this fuses.

SoD2 Fig. 4: C = sigmoid(A) + B with A: [I',J',K'] and B: [I,J,K] of unknown
  shapes. Without knowing which dims are 1, the compiler must either not fuse
  or generate 2^3 = 8 versions. Inductor: Dynamo specializes size-0/1 dims,
  so broadcast dims are static 1 and the rest get guarded equal, giving
  one fused kernel.
"""

import torch

from harness import Case, run_cases

R = torch.randn


def bladedisc_softmax_reshape(x):  # x: (S2, S3, S1)
    S2, S3, S1 = x.shape
    y = torch.softmax(x.reshape(S2 * S3, S1), dim=-1)  # (S0, S1), S0 = S2*S3
    return y.reshape(S2, S3, S1)


def sod2_sigmoid_broadcast_add(a, b):  # a: (I,1,1), b: (I,J,K)
    return torch.sigmoid(a) + b


cases = [
    Case("BladeDISC Fig5 softmax+reshape", bladedisc_softmax_reshape,
         lambda d: (R(8, 256, 512, device=d),), {0: [0, 1, 2]}),
    Case("SoD2 Fig4 sigmoid(A)+B bcast", sod2_sigmoid_broadcast_add,
         lambda d: (R(64, 1, 1, device=d), R(64, 32, 128, device=d)), {0: [0], 1: [0, 1, 2]}),
]

if __name__ == "__main__":
    run_cases(cases)
