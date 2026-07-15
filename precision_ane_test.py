"""Build the exerciser at each Core ML compute precision and see which is ANE-eligible.

For each precision (FLOAT32, FLOAT16) we:
  1. Build an .mlpackage with that compute_precision.
  2. Ask Core ML's compute plan which device each op is placed on when the
     Neural Engine is made eligible -- this is the authoritative answer to
     "is it ANE-eligible," not an inference from throughput.
  3. Benchmark CPU only and CPU + ANE a couple of times each.

The Apple Neural Engine runs FLOAT16 only, so a FLOAT32 model is expected to
fall back to CPU/GPU even when CPU_AND_NE is requested.

Author: Andrew Benson
"""

import time
import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.models.compute_plan import MLComputePlan
from coremltools.models.compute_device import (
    MLCPUComputeDevice,
    MLGPUComputeDevice,
    MLNeuralEngineComputeDevice,
)

INPUT_NAME = "image"
BATCH, CHANNELS, HEIGHT, WIDTH = 1, 3, 1024, 1024
SHAPE = (BATCH, CHANNELS, HEIGHT, WIDTH)

NUM_LAYERS = 16
FILTERS = 64
KERNEL = 3

SECONDS_PER_MODE = 8.0
REPEATS = 2

PRECISIONS = [
    ("FLOAT32", ct.precision.FLOAT32),
    ("FLOAT16", ct.precision.FLOAT16),
]


def _conv_weight(out_channels, in_channels, kernel=KERNEL):
    return (np.random.rand(out_channels, in_channels, kernel, kernel).astype(np.float32) - 0.5) * 0.1


def build_model(precision, path):
    @mb.program(input_specs=[mb.TensorSpec(shape=SHAPE)])
    def prog(image):
        x = image
        in_channels = CHANNELS
        for _ in range(NUM_LAYERS):
            weight = _conv_weight(FILTERS, in_channels)
            x = mb.conv(x=x, weight=weight, pad_type="same")
            x = mb.relu(x=x)
            in_channels = FILTERS
        return x

    mlmodel = ct.convert(
        prog,
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=precision,
        minimum_deployment_target=ct.target.macOS13,
        inputs=[ct.TensorType(name=INPUT_NAME, shape=SHAPE)],
    )
    mlmodel.save(path)


def _device_kind(device):
    if isinstance(device, MLNeuralEngineComputeDevice):
        return "ANE"
    if isinstance(device, MLGPUComputeDevice):
        return "GPU"
    if isinstance(device, MLCPUComputeDevice):
        return "CPU"
    return type(device).__name__


def analyze_placement(path):
    """Return per-op preferred-device counts and ANE-supported count with NE eligible."""
    # The compute plan API needs a compiled .mlmodelc, which loading produces.
    model = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    plan = MLComputePlan.load_from_path(
        model.get_compiled_model_path(), compute_units=ct.ComputeUnit.CPU_AND_NE
    )
    program = plan.model_structure.program
    main = program.functions["main"]

    preferred_counts = {}
    ane_supported = 0
    total = 0
    for operation in main.block.operations:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
        if usage is None:
            continue
        total += 1
        preferred = _device_kind(usage.preferred_compute_device)
        preferred_counts[preferred] = preferred_counts.get(preferred, 0) + 1
        if any(_device_kind(d) == "ANE" for d in usage.supported_compute_devices):
            ane_supported += 1
    return total, preferred_counts, ane_supported


def benchmark(path, compute_units, seconds):
    model = ct.models.MLModel(path, compute_units=compute_units)
    feed = {INPUT_NAME: np.random.rand(*SHAPE).astype(np.float32)}
    model.predict(feed)  # warm up

    iterations = 0
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        model.predict(feed)
        iterations += 1
    elapsed = time.perf_counter() - start
    return iterations / elapsed, elapsed / iterations * 1000


def main():
    print(f"Model: {NUM_LAYERS} conv+relu layers, {FILTERS} filters, input {SHAPE}")
    print(f"Bench: {SECONDS_PER_MODE:.0f}s x {REPEATS} repeats per mode\n")

    for name, precision in PRECISIONS:
        path = f"exerciser_{name.lower()}.mlpackage"
        print(f"=== compute_precision = {name} ===")
        build_model(precision, path)

        total, preferred, ane_supported = analyze_placement(path)
        pref_str = ", ".join(f"{k}:{v}" for k, v in sorted(preferred.items()))
        print(f"  compute plan (NE eligible): {total} ops  preferred[{pref_str}]")
        print(f"  ops with ANE among supported devices: {ane_supported}/{total}")
        verdict = "ANE-ELIGIBLE" if preferred.get("ANE", 0) > 0 else "NOT on ANE (falls back)"
        print(f"  verdict: {verdict}")

        for mode_name, cu in [
            ("CPU only", ct.ComputeUnit.CPU_ONLY),
            ("CPU + ANE", ct.ComputeUnit.CPU_AND_NE),
        ]:
            runs = []
            for _ in range(REPEATS):
                iters_per_s, ms = benchmark(path, cu, SECONDS_PER_MODE)
                runs.append((iters_per_s, ms))
            avg_ips = sum(r[0] for r in runs) / len(runs)
            detail = "  ".join(f"{r[0]:.1f} it/s" for r in runs)
            print(f"    {mode_name:10s} runs: {detail}   avg {avg_ips:.1f} it/s")
        print()


if __name__ == "__main__":
    main()
