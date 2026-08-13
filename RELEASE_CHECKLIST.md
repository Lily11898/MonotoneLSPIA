# Release checklist

- [x] Add permanent repository and documentation URLs to `pyproject.toml`.
- [x] Run `python -m pytest` on all supported Python versions.
- [x] Run the short and complete validation pipelines.
- [x] Build with `python -m build` and install the wheel in a clean environment.
- [x] Confirm that `data/NASA_B0005/B0005.mat` is absent from every release
  archive; only the source URL, citation, checksum, and download helper ship.
- [x] Confirm version agreement in `pyproject.toml`, `__init__.py`, `api.py`,
  `CITATION.cff`, `codemeta.json`, and `CHANGELOG.md`.
- [x] Add the release date to `CITATION.cff`, `codemeta.json`, and
  `CHANGELOG.md` immediately before creating the release.
- [ ] Create the GitHub v1.0.1 release and archive it with Zenodo. Replace the
  concept DOI (`10.5281/zenodo.21900832`) with the new version DOI wherever a
  version-specific archive is required.
- [x] Insert the permanent repository URL and DOI in the manuscript metadata.
