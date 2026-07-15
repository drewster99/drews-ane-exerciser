"""Benchmark the exerciser model across Core ML compute-unit options and compare.

Compute units are chosen at load time (there is no per-engine-only mode from
Python). Core ML treats the selection as "eligible," not "guaranteed," and falls
back to another engine for any op the chosen one can't run:
    CPU_ONLY     - everything on CPU
    CPU_AND_GPU  - GPU eligible
    CPU_AND_NE   - Neural Engine eligible
    ALL          - GPU and Neural Engine both eligible

Each mode runs for a fixed wall-clock budget (default 15s) and reports throughput.

Author: Andrew Benson
"""

import sys
import time
import numpy as np
import coremltools as ct

MODEL_PATH = "exerciser.mlpackage"
INPUT_NAME = "image"
SHAPE = (1, 3, 1024, 1024)

seconds_per_mode = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0

# One fixed input reused everywhere keeps the comparison about compute units.
input_data = np.random.rand(*SHAPE).astype(np.float32)
feed = {INPUT_NAME: input_data}

MODES = [
    ("CPU only", ct.ComputeUnit.CPU_ONLY),
    ("CPU + GPU", ct.ComputeUnit.CPU_AND_GPU),
    ("CPU + ANE", ct.ComputeUnit.CPU_AND_NE),
    ("CPU+GPU+ANE", ct.ComputeUnit.ALL),
]


def benchmark(label, compute_units):
    print(f"Running {label} for {seconds_per_mode:.0f}s ...", flush=True)
    model = ct.models.MLModel(MODEL_PATH, compute_units=compute_units)
    model.predict(feed)  # warm up: first run pays compilation/load cost

    iterations = 0
    start = time.perf_counter()
    while time.perf_counter() - start < seconds_per_mode:
        model.predict(feed)
        iterations += 1
    elapsed = time.perf_counter() - start

    per_iter_ms = elapsed / iterations * 1000
    iters_per_s = iterations / elapsed
    print(f"  done: {iterations} iters, {per_iter_ms:.1f} ms/iter, {iters_per_s:.1f} iters/s\n", flush=True)
    return iterations, per_iter_ms, iters_per_s


print(f"Model: {MODEL_PATH}   input {SHAPE}   {seconds_per_mode:.0f}s per mode\n")

results = {label: benchmark(label, cu) for label, cu in MODES}

baseline_iters_per_s = results["CPU only"][2]
print("Summary")
print(f"  {'mode':12s} {'ms/iter':>9s} {'iters/s':>9s} {'speedup':>9s}")
for label, (iterations, per_iter_ms, iters_per_s) in results.items():
    speedup = iters_per_s / baseline_iters_per_s
    print(f"  {label:12s} {per_iter_ms:9.1f} {iters_per_s:9.1f} {speedup:8.2f}x")
