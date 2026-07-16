# ANE Exerciser

Builds a compute-heavy Core ML model and benchmarks it across Core ML compute
units to see how the Apple Neural Engine (ANE) compares to CPU and GPU.

## Quick start

```sh
./doit.sh
```

`doit.sh` does the whole thing end to end: it wipes any existing `.venv` and
`exerciser.mlpackage`, runs the setup script, builds a fresh model, and prints
the compute-unit comparison. Run it and read the summary table at the bottom.

The three steps it chains together are also usable on their own, described
below.

## The pieces

### 1. `setup.sh` — environment

```sh
./setup.sh
source .venv/bin/activate
```

Creates the `.venv` and installs dependencies. Safe to re-run. Needs Python 3.11
and, ideally, [`uv`](https://docs.astral.sh/uv/) (the script falls back to
`python3 -m venv` + `pip` if `uv` isn't installed).

### 2. `makemodel.py` — the model maker

```sh
python makemodel.py   # builds exerciser.mlpackage
```

Builds the compute-heavy Core ML model — a stack of 3x3 convolutions over a
large image, built directly with coremltools' MIL builder (no TensorFlow/Keras).
It does nothing meaningful; it just keeps the ANE busy. Turn up the layer count
and filter width near the top of the file to make it heavier.

### 3. `try_ane.py` — the benchmark

```sh
python try_ane.py     # benchmarks exerciser.mlpackage across CPU / GPU / Neural Engine
```

By default, `try_ane.py` runs a **60-second test** — each of the four compute
modes (CPU only, CPU + GPU, CPU + ANE, CPU + GPU + ANE) runs for a 15-second
wall-clock budget. Pass a per-mode duration in seconds to make it shorter or
longer:

```sh
python try_ane.py 5    # 5s per mode (20s total)
```

## Example output

```
Model: exerciser.mlpackage   input (1, 3, 1024, 1024)   15s per mode

Running CPU only for 15s ...
  done: 34 iters, 447.1 ms/iter, 2.2 iters/s

Running CPU + GPU for 15s ...
  done: 128 iters, 117.6 ms/iter, 8.5 iters/s

Running CPU + ANE for 15s ...
  done: 271 iters, 55.4 ms/iter, 18.1 iters/s

Running CPU+GPU+ANE for 15s ...
  done: 268 iters, 56.0 ms/iter, 17.9 iters/s

Summary
  mode           ms/iter   iters/s   speedup
  CPU only         447.1       2.2     1.00x
  CPU + GPU        117.6       8.5     3.80x
  CPU + ANE         55.4      18.1     8.07x
  CPU+GPU+ANE       56.0      17.9     7.99x
```

Numbers vary by hardware, thermal state, and macOS version; the example above is
illustrative. Core ML treats the selected compute units as *eligible*, not
*guaranteed*, and falls back to another engine for any op the chosen one can't
run — so the "CPU + ANE" and "CPU + GPU + ANE" rows generally reflect the ANE
doing most of the work.

## Live ANE monitor

```sh
python spam_the_ane.py
```

Runs prediction continuously with CPU + ANE eligible and, four times a second,
overwrites a single line with the current throughput (a ~3-second exponential
moving average, in iterations/sec) followed by a sparkline of that value over
time, scaled from just below the lowest rate seen up to the highest. Handy for
watching the ANE under sustained load — e.g. alongside
`powermetrics` or `asitop`. Ctrl-C to stop. Needs `exerciser.mlpackage` (run
`makemodel.py` first).

## Cleanup

```sh
./cleanup.sh
```

Removes everything the setup and build steps create — the `.venv` and any
`*.mlpackage`. Safe to re-run.

## Author

**Andrew Benson**

- GitHub: [@drewster99](https://github.com/drewster99)
- Bluesky: [@thedrewbenson.bsky.social](https://bsky.app/profile/thedrewbenson.bsky.social)
- X: [@TheDrewBenson](https://x.com/TheDrewBenson)

## License

Copyright 2026 Andrew Benson

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the
full text.
