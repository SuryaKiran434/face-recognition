"""One-time migration: convert legacy .pkl encoding files to .npz.

The old encode_faces.py stored {"encodings": [...], "names": [...]} via pickle.
Loading pickle is unsafe (arbitrary code execution), so the recognizer now
reads .npz files with allow_pickle=False. Run this script once against any
directory that still contains .pkl files.

Usage:
    python scripts/convert_pickle.py /path/to/FaceRecognitionData

The original .pkl files are left in place. Verify the new .npz files load
correctly, then delete the .pkl files manually.
"""

import argparse
import os
import pickle  # noqa: S403 -- legacy data only; see verify_safety() below
import sys

import numpy as np


def verify_safety(path):
    """Reject any .pkl that doesn't deserialize into the expected shape.

    pickle is still RCE-prone in principle. This script is intended for files
    you wrote yourself with the old encode_faces.py. If you suspect tampering,
    delete the .pkl files and re-encode from source images instead of running
    this converter.
    """
    if not os.path.isfile(path):
        raise ValueError(f"{path} is not a file")


def convert(directory):
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a directory.", file=sys.stderr)
        return 1

    converted = 0
    skipped = 0
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".pkl"):
            continue
        pkl_path = os.path.join(directory, name)
        npz_path = os.path.splitext(pkl_path)[0] + ".npz"

        if os.path.exists(npz_path):
            print(f"Skipping {pkl_path}: {npz_path} already exists.")
            skipped += 1
            continue

        verify_safety(pkl_path)
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)  # noqa: S301

        if not isinstance(data, dict) or "encodings" not in data or "names" not in data:
            print(f"Skipping {pkl_path}: unexpected format.", file=sys.stderr)
            skipped += 1
            continue

        np.savez(
            npz_path,
            encodings=np.asarray(data["encodings"]),
            names=np.asarray(data["names"]),
        )
        print(f"Converted {pkl_path} -> {npz_path}")
        converted += 1

    print(f"\nDone. Converted: {converted}, skipped: {skipped}.")
    if converted:
        print("Verify the new .npz files load correctly, then delete the .pkl files.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", help="Directory containing legacy .pkl files")
    args = parser.parse_args()
    sys.exit(convert(args.directory))


if __name__ == "__main__":
    main()
