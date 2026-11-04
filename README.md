# Missed Inductor fusion under dynamic shapes: minimal repros

Each script compiles a small function twice, once with static shapes and once with
`torch._dynamo.mark_dynamic` on chosen dims. For each run it counts the generated
kernels and the materialized buffers, and checks the result against eager.

```bash
python -m venv .venv && .venv/bin/pip install torch numpy
PY=.venv/bin/python ./run_all.sh --device cuda      # or mps / cpu
.venv/bin/python repro_split_heads.py --why         # print the scheduler's "cannot fuse" reasons
.venv/bin/python repro_split_heads.py --show-code   # dump the generated code
```

## Results (torch 2.15.0, Apple M2 Pro, `--device mps`)

The Metal backend uses the same `SIMDScheduling` fusion logic as Triton, so it is a
local stand-in for it. **The CUDA/Triton results have not been verified yet. Run
these on a GPU first.**

| script | case | static kernels | dynamic kernels | why (from `--why`) |
|---|---|---|---|---|
| repro_split_heads | softmax → `view(B,T,H,C//H)` | 1 | **2** | `nodes numel incompatibility`: `B*T*H*(C//H)` vs `B*T*C` |
| | rmsnorm → split heads (OLMo-2 QK-norm) | 1 | **2** | same |
| | layernorm → split heads | 1 | **2** | same |
| | ctrl: no split | 1 | 1 | |
| | ctrl: `view(...,-1)`, `unflatten` | 1 | **2** | not a spelling issue |
| | ctrl: `+ torch._check(C % H == 0)` | 1 | **2** | divisibility fact isn't used |
| | ctrl: merge direction (Dh dynamic, C = H·Dh) | 1 | 1 | `(H*Dh)//H` simplifies |
| repro_dim_equality | `y * rsqrt(mean(x²))`, x,y same width | 1 | **2** | `nodes numel incompatibility`: `M*D1` vs `M*D2` |
| | fix: `+ torch._check(D1 == D2)` | 1 | 1 | symbols unified |
| repro_xsa (#181108) | `xsa_expand` | 1 | **2** | matches the issue's fwd numbers |
| | `xsa_unsqueeze` | 2 | 2 | the issue reports 2 → 4 on CUDA; probably Triton-specific |
| repro_paper_patterns | BladeDISC Fig. 5 softmax+reshape | 1 | 1 | Inductor already handles it |
| | SoD2 Fig. 4 sigmoid(A)+B broadcast | 1 | 1 | 0/1 specialization handles it |

On `--device cpu` the C++ backend always emits a single kernel, because it fuses
at the outer loop. The miss still shows there as extra materialized buffers:
softmax-split goes from 3 to 4, XSA from 4 to 6.

## Two root causes

1. **Floor-division from splitting a symbolic dim.** `C` → `(H, C//H)`. The
   scheduler requires `numel == xnumel*rnumel` to fuse a reduction with a
   pointwise consumer, and `H*(C//H)` is not provably equal to `C`. Possible fix:
   when a symbol `s` is known to be divisible by `H`, rewrite it as `H*s'`
   (the "merge direction" control shows that this form fuses). Another option is
   to make the numel check use divisibility facts.
2. **Equal-but-unproven dims.** These are different symbols with no op tying them
   together. Possible fix: infer the equality (BladeDISC-style symbolic dim
   propagation), or emit a guarded fused kernel with a runtime equality check
   (SoD2-style multi-versioning).
