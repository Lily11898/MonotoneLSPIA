# UCI Combined Cycle Power Plant data

This directory contains the original `Folds5x2_pp.xlsx` workbook and
`Readme.txt` distributed by the UCI Machine Learning Repository.

MonotoneLSPIA uses the first worksheet, which contains 9568 observations and
the variables:

- `AT`: ambient temperature in degrees Celsius;
- `V`: exhaust vacuum in cm Hg;
- `AP`: ambient pressure in millibar;
- `RH`: relative humidity in percent;
- `PE`: net hourly electrical energy output in MW.

The SoftwareX case study fits `PE` as a decreasing univariate function of
`AT`. The other measured variables are deliberately omitted, so the example
is an operational-trend and software-scalability demonstration rather than a
causal or complete multivariate power model.

Source:

> Tüfekci, P., and Kaya, H. (2014). Combined Cycle Power Plant [Dataset].
> UCI Machine Learning Repository. https://doi.org/10.24432/C5002N

Dataset page:
https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant

License: Creative Commons Attribution 4.0 International (CC BY 4.0).

Integrity information:

- Official downloaded ZIP SHA-256:
  `cc7b2a4977c0a44e8221c91d9a7e5746b3c68186cff7e5c61c70af6432b98c7a`
- `Folds5x2_pp.xlsx` SHA-256:
  `ccd490981db2a2f079963b3d9f0aea30d9d338900a0285428dfc6385396f4651`
- `Readme.txt` SHA-256:
  `2d79c1fb5a91fa1fcbeb514546d9d3fc354be8e4bf8f4df316445e1e7089c137`

The data are kept separate from the BSD-licensed source code and retain
their CC BY 4.0 attribution requirements.
