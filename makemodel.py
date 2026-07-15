"""Build a compute-heavy Core ML model for exercising the Apple Neural Engine.

The model is a deep stack of 3x3 convolutions over a largeish image. It computes
nothing meaningful; the point is to keep the ANE saturated so try_ane.py has
something heavy to measure. Turn up NUM_LAYERS / FILTERS to make it heavier.

Author: Andrew Benson
"""

import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb

INPUT_NAME = "image"
BATCH, CHANNELS, HEIGHT, WIDTH = 1, 3, 1024, 1024

# Number of conv layers and their width. Turn these up to make it heavier.
NUM_LAYERS = 16
FILTERS = 64
KERNEL = 3


def _conv_weight(out_channels, in_channels, kernel=KERNEL):
    # Small values keep activations from blowing up across many layers.
    return (np.random.rand(out_channels, in_channels, kernel, kernel).astype(np.float32) - 0.5) * 0.1


@mb.program(input_specs=[mb.TensorSpec(shape=(BATCH, CHANNELS, HEIGHT, WIDTH))])
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
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.macOS13,
    inputs=[ct.TensorType(name=INPUT_NAME, shape=(BATCH, CHANNELS, HEIGHT, WIDTH))],
)
mlmodel.save("exerciser.mlpackage")
print("saved exerciser.mlpackage")
