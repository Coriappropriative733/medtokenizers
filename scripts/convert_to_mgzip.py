#!/usr/bin/env python3
"""Convert .nii.gz files to mgzip format (multi-member gzip) for parallel decompression.

Mgzip files are backwards-compatible with standard gzip but can be decompressed
in parallel, providing 3-5x faster loading on multi-core systems.

Usage:
    python scripts/convert_to_mgzip.py /path/to/data --workers 8
"""

import argparse
import gzip
import os
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import mgzip


def is_mgzip(filepath: str) -> bool:
    """Check if a gzip file has multiple members (i.e. is already mgzip).

    Standard gzip writes a single member; mgzip writes several concatenated
    members. ``zlib.decompressobj`` (unlike ``gzip.GzipFile``) stops at the end
    of the first member, so we decompress that member and check whether a second
    gzip header (magic ``1f 8b``) immediately follows.
    """
    try:
        with open(filepath, "rb") as f:
            if f.read(2) != b"\x1f\x8b":
                return False
            f.seek(0)
            decompressor = zlib.decompressobj(zlib.MAX_WBITS | 16)
            while not decompressor.eof:
                chunk = f.read(1 << 20)
                if not chunk:
                    return False  # truncated or single member
                decompressor.decompress(chunk)
            # First member finished; check for a following gzip header.
            trailing = decompressor.unused_data or f.read(2)
            return trailing[:2] == b"\x1f\x8b"
    except Exception:
        return False


def convert_file(
    gz_path: Path, blocksize: int = 4 * 1024 * 1024
) -> tuple[Path, bool, str]:
    """Convert a single .nii.gz file to mgzip format in-place."""
    if is_mgzip(str(gz_path)):
        return gz_path, True, "already mgzip"

    tmp_path = gz_path.with_suffix(".tmp.gz")
    try:
        with gzip.open(gz_path, "rb") as f_in:
            data = f_in.read()

        with mgzip.open(str(tmp_path), "wb", thread=0, blocksize=blocksize) as f_out:
            f_out.write(data)

        os.replace(tmp_path, gz_path)
        return gz_path, True, "converted"
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        return gz_path, False, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Convert NIfTI dataset to mgzip format"
    )
    parser.add_argument("data_dir", type=Path, help="Dataset directory")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers")
    parser.add_argument(
        "--blocksize", type=int, default=4, help="Block size in MB (default: 4)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Just count files")
    args = parser.parse_args()

    gz_files = list(args.data_dir.rglob("*.nii.gz"))
    print(f"Found {len(gz_files)} .nii.gz files")

    if args.dry_run:
        return

    blocksize = args.blocksize * 1024 * 1024
    success = 0
    skipped = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(convert_file, f, blocksize): f for f in gz_files}

        for i, future in enumerate(as_completed(futures)):
            path, ok, msg = future.result()
            if ok:
                if msg == "already mgzip":
                    skipped += 1
                else:
                    success += 1
            else:
                failed += 1
                print(f"FAILED: {path}: {msg}")

            if (i + 1) % 100 == 0:
                print(
                    f"Progress: {i + 1}/{len(gz_files)} (converted={success}, skipped={skipped}, failed={failed})"
                )

    print(f"\nDone: {success} converted, {skipped} already mgzip, {failed} failed")


if __name__ == "__main__":
    main()
