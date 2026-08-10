# Fair timing benchmark interpretation

This timing-only benchmark uses identical public workflow boundaries. Every
timed call starts with in-memory `x` and `y` arrays and ends after model
construction, fitting, prediction on the same grid, and a common monotonicity
check. The total workflow includes roughness-parameter selection for the
penalized spline and pyGAM. The fixed workflow assumes that the selected
roughness parameter is already known.

Each condition used two untimed warm-ups followed by 20 timed repetitions.
The six method/workflow tasks were randomly reordered within every repetition.
All numerical libraries were restricted to one thread. All 1,560 timed calls
converged.

Across the nine target/noise timing conditions, the median total-workflow
speedup was 2.54x relative to the cross-validated penalized monotone B-spline
and 5.03x relative to pyGAM grid search. The condition ranges were 1.70--4.13x
and 3.66--11.79x, respectively.

The fixed-parameter result has the opposite interpretation for the penalized
spline. Its median competitor/MonotoneLSPIA time ratio was 0.200 across the
nine conditions, so its one-fit workflow was about five times faster when the
penalty was already known. The corresponding pyGAM ratio was 1.04, with a
range crossing one; neither method was uniformly faster in that comparison.

Scaling medians, including prediction at the same number of points as
observations, were:

| Observations | Workflow | MonotoneLSPIA | Penalized spline | pyGAM |
|---:|:---|---:|---:|---:|
| 120 | Total | 0.0185 s | 0.0467 s | 0.0928 s |
| 1,000 | Total | 0.0201 s | 0.0642 s | 0.1118 s |
| 10,000 | Total | 0.0526 s | 0.1365 s | 0.2370 s |
| 100,000 | Total | 0.4366 s | 1.3733 s | 2.0933 s |
| 120 | Fixed | 0.0185 s | 0.0033 s | 0.0232 s |
| 1,000 | Fixed | 0.0200 s | 0.0046 s | 0.0252 s |
| 10,000 | Fixed | 0.0531 s | 0.0131 s | 0.0476 s |
| 100,000 | Fixed | 0.4479 s | 0.0581 s | 0.4330 s |

The trial-level CSV and summary CSV report the first quartile, median, and
third quartile for every condition. Maximum sampled monotonicity violations
were at numerical precision: `3.33e-13` for MonotoneLSPIA, `4.44e-13` for the
penalized spline, and `3.46e-09` for pyGAM.

The supported computational claim is therefore specific: MonotoneLSPIA
reduces parameter-selection-inclusive workflow time over the tested range. It
is not a uniformly faster individual solver. The earlier 3.22x and 8.48x
values used unequal timing boundaries and are superseded by this benchmark.
