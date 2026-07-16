"""Hammer the Neural Engine continuously with a live sparkline of throughput.

Loads exerciser.mlpackage with CPU + ANE eligible and runs prediction in a tight
loop forever. Four times a second it overwrites a single status line with the
current throughput -- an exponential moving average (EMA) with a ~3s time
constant -- followed by a sparkline of that smoothed value over time. Ctrl-C to
stop.

The sparkline is linear, scaled from the lowest smoothed value ever seen (minus
10% headroom) up to the highest ever seen. The newest bar enters on the left and
older bars scroll right. Each bar is frozen the moment it is drawn, so the chart
never re-scales its whole width out from under you -- only the newest bar
reflects the latest all-time range.

Author: Andrew Benson
"""

import math
import signal
import sys
import time
from collections import deque

import numpy as np
import coremltools as ct

MODEL_PATH = "exerciser.mlpackage"
INPUT_NAME = "image"
SHAPE = (1, 3, 256, 256)

UPDATE_INTERVAL_SECONDS = 0.25
EMA_TIME_CONSTANT_SECONDS = 3.0
HISTORY_LENGTH = 30
SPARK_TICKS = "▁▂▃▄▅▆▇█"
FLOOR_HEADROOM_FRACTION = 0.10  # chart bottom sits 10% below the lowest value seen


def bar_for(value, low, high):
    """Pick the block character for ``value`` on a linear scale from ``low`` to ``high``."""
    span = high - low
    # A flat range maps to the top tick rather than dividing by zero.
    fraction = 1.0 if span <= 0 else min(1.0, max(0.0, (value - low) / span))
    return SPARK_TICKS[round(fraction * (len(SPARK_TICKS) - 1))]


model = ct.models.MLModel(MODEL_PATH, compute_units=ct.ComputeUnit.CPU_AND_NE)
feed = {INPUT_NAME: np.random.rand(*SHAPE).astype(np.float32)}
model.predict(feed)  # warm up: first run pays compilation/load cost

# Make Ctrl-C reliably kill the process: unblock SIGINT in case something masked
# it, then restore the OS default action so delivery terminates us immediately --
# even while the thread is inside a native predict() call. Done after warm-up so a
# disposition change inside Core ML's first predict can't undo it.
signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT})
signal.signal(signal.SIGINT, signal.SIG_DFL)

print("Spamming the ANE (CPU + ANE). Ctrl-C to stop.\n", flush=True)

history = deque(maxlen=HISTORY_LENGTH)
ema_iterations_per_second = None
lowest_seen = None
highest_seen = None
window_iterations = 0
window_start = time.perf_counter()
while True:
    model.predict(feed)
    window_iterations += 1

    elapsed = time.perf_counter() - window_start
    if elapsed >= UPDATE_INTERVAL_SECONDS:
        sample = window_iterations / elapsed
        # EMA weight derived from actual elapsed time keeps the ~3s time
        # constant honest even when a window runs long or short.
        alpha = 1.0 - math.exp(-elapsed / EMA_TIME_CONSTANT_SECONDS)
        if ema_iterations_per_second is None:
            ema_iterations_per_second = sample
        else:
            ema_iterations_per_second += alpha * (sample - ema_iterations_per_second)

        lowest_seen = ema_iterations_per_second if lowest_seen is None else min(lowest_seen, ema_iterations_per_second)
        highest_seen = ema_iterations_per_second if highest_seen is None else max(highest_seen, ema_iterations_per_second)

        # Freeze each bar as it is drawn (newest on the left) so old bars never
        # re-scale when a new all-time extreme shifts the range.
        bar = bar_for(ema_iterations_per_second, lowest_seen * (1.0 - FLOOR_HEADROOM_FRACTION), highest_seen)
        if history:
            history.appendleft(bar)
        else:
            # Fill the whole width from the first value so the chart starts full.
            history.extend([bar] * HISTORY_LENGTH)
        chart = "".join(history)
        sys.stdout.write(f"\r{ema_iterations_per_second:5.0f} it/s  {chart}")
        sys.stdout.flush()
        window_iterations = 0
        window_start = time.perf_counter()
