# Publication validation

The final SoftwareX accuracy evidence uses five methods:

1. unconstrained LSPIA;
2. scikit-learn `IsotonicRegression`;
3. a cross-validated penalized monotone B-spline implemented with SciPy/SLSQP;
4. pyGAM with its monotonic-increasing constraint and documented grid search;
5. MonotoneLSPIA.

All spline methods use 12 cubic basis functions. The experiment uses 120
observations, three target functions, noise standard deviations 0.02, 0.05,
and 0.10, and 100 paired deterministic Monte Carlo trials per condition.

Install the validation dependencies and run a two-trial smoke check first:

```bash
python -m pip install -e '.[benchmark]'
python validation/run_representative_fit.py
python validation/run_final_accuracy_comparison.py \
  --num-trials 2 --output results/final_accuracy_smoke
```

Run the final 100-trial experiment with:

```bash
python validation/run_final_accuracy_comparison.py --num-trials 100
```

If the completed focused three-method trial file is already present, the
same final figure can be produced without recomputing those expensive fits:

```bash
python validation/run_final_accuracy_comparison.py --num-trials 100 \
  --reuse-focused-results \
  results/value_proposition_optimized_default/accuracy_trials.csv
```

This compatibility path recomputes LSPIA and scikit-learn isotonic regression
with the same seeds and reuses only the matching MonotoneLSPIA, penalized
B-spline, and pyGAM accuracy rows. It does not reuse timing measurements.

Or reproduce both publication accuracy figures with one command:

```bash
python validation/run_comparative_validation.py --num-trials 100
```

To reproduce the complete publication workflow (representative fit, final
accuracy experiment, fair timing benchmark, Puromycin validation, and NASA
B0005 interpolation study), run:

```bash
python validation/run_publication_workflow.py --download-nasa
```

`--download-nasa` explicitly obtains `B0005.mat` from the archive linked by
the NASA PCoE repository and verifies its recorded SHA-256 digest. The source
file is stored locally but is ignored by Git and is not redistributed with
this package. Users who downloaded the archive manually can instead run
`python validation/fetch_nasa_b0005.py --archive PATH` or pass an existing
file with `--nasa-data PATH`.

The command writes `results/publication_workflow_manifest.json` with the exact
subcommands, environment, timestamps, and completion status. Before the full
run, verify all code paths quickly in a separate output directory:

```bash
python validation/run_publication_workflow.py --smoke --download-nasa \
  --output-root results/publication_smoke
```

Smoke results are workflow checks and must not replace the archived full
publication results. Use `--skip-nasa` only when the NASA source file is not
available; such a run is incomplete by definition.

The outputs are:

- `results/figure2_representative_fit.png` and `.pdf`;
- `results/final_accuracy/final_accuracy_trials.csv`;
- `results/final_accuracy/final_accuracy_summary.csv`;
- `results/final_accuracy/configuration.json`;
- `results/final_accuracy/figure3_recovery_rmse.png` and `.pdf`.

The older `run_noise_sweep.py` and `plot_noise_sweep.py` scripts retain the
superseded Denoise+LSPIA experiment for provenance only. They must not be used
for the final paper.

## Focused value-proposition benchmark

`run_value_proposition_benchmark.py` provides the existing focused comparison
with the penalized monotone B-spline and pyGAM, including the projection audit.
Its accuracy and convergence outputs remain valid. Timing values stored inside
that accuracy experiment are superseded by the fair timing benchmark below.

```bash
python validation/run_value_proposition_benchmark.py \
  --num-trials 100 --scale-sizes 120 1000 10000 100000 \
  --output results/value_proposition_optimized_default
```

## Fair timing benchmark

Publication timing claims must use the dedicated fair-workflow benchmark.
Every timed call starts with in-memory `x,y` arrays and ends after model
construction, fitting, prediction on a common grid, and a common monotonicity
check. The total workflow includes parameter selection; the fixed workflow
assumes that the selected parameter is already known.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python validation/run_fair_timing_benchmark.py \
  --warmups 2 --repeats 20 --scale-sizes 120 1000 10000 100000 \
  --output results/fair_timing_final
```

## Real-data examples

```bash
python examples/puromycin_example.py
python validation/run_puromycin_case_study.py
python validation/fetch_nasa_b0005.py
python validation/run_nasa_b0005_case_study.py
python validation/run_ccpp_case_study.py
```

The Puromycin validation leaves out one complete interior concentration
level at a time, keeping replicate predictor values together. It writes the
publication figure, group summaries, and auditable fold-level results to
`results/puromycin/`.

The CCPP script performs the deterministic 80/20 held-out comparison of
LSPIA, scikit-learn isotonic regression, the controlled penalized monotone
B-spline, pyGAM, and MonotoneLSPIA. It writes the publication figure, table
data, and configuration metadata to `results/ccpp/`.
