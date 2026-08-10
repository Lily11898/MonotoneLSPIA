"""Repeatable benchmark of the public fitting interface."""

import statistics
import time

import numpy as np

from monotone_lspia import fit

rng = np.random.default_rng(20260730)
print("samples,control_points,median_seconds,sd_seconds,iterations,rmse")
for sample_size, control_points in zip((250, 1000, 5000, 10000), (12, 16, 24, 32), strict=True):
    x = np.linspace(10, 110, sample_size)
    y = np.log(x) + 0.03 * rng.standard_normal(sample_size)
    timings = []
    for _ in range(5):
        start = time.perf_counter()
        model = fit(x, y, num_control_points=control_points)
        timings.append(time.perf_counter() - start)
    print(
        f"{sample_size},{control_points},{statistics.median(timings):.8f},"
        f"{statistics.stdev(timings):.8f},{model.diagnostics.iterations},{model.rmse:.8g}"
    )
