# Puromycin example data

`Puromycin.csv` contains the 23 observations from the `Puromycin` data frame
distributed with the R `datasets` package. The variables are substrate
concentration (`conc`), reaction rate (`rate`), and treatment state (`state`).

Source references:

- Bates, D. M., and Watts, D. G. (1988). *Nonlinear Regression Analysis and
  Its Applications*. Wiley, Appendix A1.3.
- Treloar, M. A. (1974). *Effects of Puromycin on Galactosyltransferase in
  Golgi Membranes*. Master's thesis, University of Toronto.
- R documentation:
  https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/Puromycin.html

The R `datasets` package is part of R, which is distributed under
`GPL-2.0-only OR GPL-3.0-only`. The CSV is kept separate from the BSD-licensed
MonotoneLSPIA source code and is included only as attributed example data.

## Combined Cycle Power Plant case-study data

`CCPP/Folds5x2_pp.xlsx` is the original UCI Combined Cycle Power Plant
workbook. It contains 9568 observations and is used for the large
engineering-domain validation in `validation/run_ccpp_case_study.py`. See
`CCPP/README.md` for variables, checksums, citation, scope, and license.

The dataset is licensed under CC BY 4.0 and is kept separate from the
BSD-licensed software source.

## NASA B0005 battery data

The NASA B0005 raw MAT file is not redistributed with this software. See
`NASA_B0005/README.md` for the official source, citation, verified checksum,
and the download helper used by the publication workflow.
