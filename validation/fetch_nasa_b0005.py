"""Download B0005 from the archive linked by NASA and verify its checksum.

The source data are intentionally not redistributed with MonotoneLSPIA.  This
helper performs an explicit user-requested download from the archive linked on
the NASA PCoE repository page, searches nested ZIP files for ``B0005.mat``, and
writes the file only after its SHA-256 digest has been verified.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tempfile
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).parents[1]
OFFICIAL_ARCHIVE = (
    "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"
)
DEFAULT_OUTPUT = ROOT / "data" / "NASA_B0005" / "B0005.mat"
EXPECTED_SHA256 = "0eae4585baf3f200c09fe24c5ab884f1889679fc75206ca1aa19da704104f0b0"
CHUNK_SIZE = 1024 * 1024
MAX_NESTING_DEPTH = 4


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_b0005(archive: zipfile.ZipFile, depth: int = 0) -> bytes:
    """Return B0005.mat bytes from an outer or nested ZIP archive."""
    if depth > MAX_NESTING_DEPTH:
        raise FileNotFoundError("B0005.mat was not found within the nesting limit")

    for member in archive.infolist():
        if not member.is_dir() and Path(member.filename).name.casefold() == "b0005.mat":
            return archive.read(member)

    for member in archive.infolist():
        if member.is_dir() or Path(member.filename).suffix.casefold() != ".zip":
            continue
        try:
            with zipfile.ZipFile(BytesIO(archive.read(member))) as nested:
                return find_b0005(nested, depth + 1)
        except (FileNotFoundError, zipfile.BadZipFile):
            continue

    raise FileNotFoundError("B0005.mat was not found in the downloaded archive")


def download_archive(destination: Path) -> None:
    """Stream the NASA-linked archive to a temporary local file."""
    request = urllib.request.Request(
        OFFICIAL_ARCHIVE,
        headers={"User-Agent": "MonotoneLSPIA/1.0 reproducibility helper"},
    )
    with (
        urllib.request.urlopen(request, timeout=120) as response,
        destination.open("wb") as handle,
    ):
        shutil.copyfileobj(response, handle, length=CHUNK_SIZE)


def install_b0005(archive_path: Path, output: Path, *, force: bool) -> None:
    """Extract, verify, and atomically install B0005.mat."""
    if output.exists():
        current_digest = sha256(output)
        if current_digest == EXPECTED_SHA256:
            print(f"B0005.mat is already present and verified: {output}")
            return
        if not force:
            raise FileExistsError(
                f"Refusing to replace checksum-mismatched file: {output}; use --force"
            )

    with zipfile.ZipFile(archive_path) as archive:
        payload = find_b0005(archive)
    payload_digest = hashlib.sha256(payload).hexdigest()
    if payload_digest != EXPECTED_SHA256:
        raise ValueError(
            "Downloaded B0005.mat failed SHA-256 verification: "
            f"expected {EXPECTED_SHA256}, received {payload_digest}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix="B0005.",
        suffix=".tmp",
        dir=output.parent,
        delete=False,
    ) as handle:
        temporary_output = Path(handle.name)
        handle.write(payload)
    try:
        os.replace(temporary_output, output)
    finally:
        temporary_output.unlink(missing_ok=True)
    print(f"Saved verified B0005.mat: {output}")
    print(f"SHA-256: {payload_digest}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Use a manually downloaded official archive instead of network access",
    )
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()

    if arguments.archive is not None:
        install_b0005(arguments.archive, arguments.output, force=arguments.force)
        return

    with tempfile.TemporaryDirectory(prefix="monotone_lspia_nasa_") as directory:
        archive_path = Path(directory) / "Battery_Data_Set.zip"
        print(f"Downloading the NASA-linked archive: {OFFICIAL_ARCHIVE}")
        download_archive(archive_path)
        install_b0005(archive_path, arguments.output, force=arguments.force)


if __name__ == "__main__":
    main()
