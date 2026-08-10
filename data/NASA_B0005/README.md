# NASA B0005 lithium-ion battery aging data

- Official repository: <https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/>
- Official archive: <https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip>
- Dataset citation: B. Saha and K. Goebel (2007), *Battery Data Set*, NASA
  Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.
- Battery used: B0005
- Expected local filename: `B0005.mat`
- Expected SHA-256: `0eae4585baf3f200c09fe24c5ab884f1889679fc75206ca1aa19da704104f0b0`

The original MAT file is not redistributed with MonotoneLSPIA. Obtain and
verify it directly from the archive linked by the NASA repository:

```bash
python validation/fetch_nasa_b0005.py
```

Alternatively, download the official archive manually and run:

```bash
python validation/fetch_nasa_b0005.py \
  --archive /path/to/5.Battery-Data-Set.zip
```

The helper searches the outer and nested ZIP files, extracts only
`B0005.mat`, verifies the SHA-256 digest above, and writes the file locally.
The case-study script extracts the capacity value from each of the 168
discharge cycles. The cycle index is the predictor and discharge capacity in
ampere-hours is the response.
