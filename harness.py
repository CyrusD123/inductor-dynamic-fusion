"""Shared harness: compile a function with static vs dynamic shapes and count
the kernels TorchInductor generates.

Usage from a repro script:

    from harness import Case, run_cases
    run_cases([Case(name, fn, make_inputs, dynamic_dims), ...])

`dynamic_dims` maps input index -> list of dims to `mark_dynamic`.

Device is chosen with --device {cuda,cpu,mps} (default: cuda if available,
else cpu). Kernel counts are backend-specific: the Triton (cuda) backend is
the one the project is about; cpu/mps are useful for local iteration but
their fusion heuristics differ in places.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch._inductor.utils import run_and_get_code

# Always recompile from scratch so the scheduler (and its fusion logs) actually run.
torch._inductor.config.force_disable_caches = True

# One regex per backend for "a kernel definition in the generated wrapper".
KERNEL_RE = {
    "cuda": re.compile(r"^(triton_\w+) = async_compile\.triton\(", re.M),
    "cpu": re.compile(r"^(cpp_\w+) = async_compile\.cpp_pybinding\(", re.M),
    "mps": re.compile(r"^(\w+) = async_compile\.metal\(", re.M),
}
# Buffers allocated in the wrapper's `call()` = materialized tensors
# (intermediates + outputs). Fewer intermediates = more fusion.
ALLOC_RE = re.compile(r"^\s*(buf\d+) = empty_strided(?:_\w+)?\(", re.M)


class _FusionReasons(logging.Handler):
    """Collects the scheduler's "cannot fuse X with Y: <reason>" log lines."""

    def __init__(self):
        super().__init__(logging.DEBUG)
        self.lines: list[str] = []

    def emit(self, record):
        msg = record.getMessage()
        if "cannot fuse" in msg:
            self.lines.append(msg)


_FUSION_LOGGER = logging.getLogger("torch._inductor.scheduler.__fusion")


@dataclass
class Case:
    name: str
    fn: Callable
    make_inputs: Callable[[str], tuple]  # device -> tuple of tensors
    dynamic_dims: dict[int, list[int]] = field(default_factory=dict)
    note: str = ""


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def compile_and_count(case: Case, device: str, dynamic: bool, show_code: bool = False,
                      why: bool = False):
    torch._dynamo.reset()
    handler = _FusionReasons()
    if why:
        torch._logging.set_logs(fusion=True)
        _FUSION_LOGGER.addHandler(handler)
    inputs = case.make_inputs(device)
    if dynamic:
        for idx, dims in case.dynamic_dims.items():
            for d in dims:
                torch._dynamo.mark_dynamic(inputs[idx], d)
    else:
        # Make sure static really means static (no automatic_dynamic).
        pass

    compiled = torch.compile(case.fn, dynamic=None if dynamic else False, fullgraph=True)
    try:
        out, codes = run_and_get_code(compiled, *inputs)
    finally:
        if why:
            _FUSION_LOGGER.removeHandler(handler)
            torch._logging.set_logs()

    # Correctness check against eager.
    ref = case.fn(*inputs)
    outs = out if isinstance(out, (tuple, list)) else (out,)
    refs = ref if isinstance(ref, (tuple, list)) else (ref,)
    ok = all(torch.allclose(o, r, atol=1e-4, rtol=1e-4) for o, r in zip(outs, refs))

    kre = KERNEL_RE[device]
    kernels = [k for code in codes for k in kre.findall(code)]
    bufs = [b for code in codes for b in ALLOC_RE.findall(code)]
    if show_code:
        for code in codes:
            print(code)
    return kernels, len(bufs), ok, sorted(set(handler.lines))


def run_cases(cases: list[Case], argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--device", default=default_device(), choices=["cuda", "cpu", "mps"])
    p.add_argument("--show-code", action="store_true", help="dump generated wrapper+kernels")
    p.add_argument("--why", action="store_true", help="print the scheduler's 'cannot fuse' reasons")
    p.add_argument("--only", default=None, help="substring filter on case name")
    args = p.parse_args(argv)

    print(f"torch {torch.__version__}  device={args.device}")
    print(f"{'case':38s} {'shapes':8s} {'#kern':>5s} {'#bufs':>5s}  ok   kernels")
    print("-" * 100)
    any_regression = False
    for case in cases:
        if args.only and args.only not in case.name:
            continue
        results = {}
        for dynamic in (False, True):
            k, nb, ok, reasons = compile_and_count(case, args.device, dynamic, args.show_code, args.why)
            results[dynamic] = len(k)
            tag = "dynamic" if dynamic else "static"
            print(f"{case.name:38s} {tag:8s} {len(k):5d} {nb:5d}  {'Y' if ok else 'N'}    {', '.join(k)}")
            for r in reasons:
                print(f"{'':47s}why: {r}")
        if results[True] > results[False]:
            any_regression = True
            print(f"{'':38s} ^^^ dynamic shapes produced {results[True] - results[False]} extra kernel(s)")
        if case.note:
            print(f"{'':38s} note: {case.note}")
        print()
    return any_regression


if __name__ == "__main__":
    sys.exit("import this module from a repro script")
