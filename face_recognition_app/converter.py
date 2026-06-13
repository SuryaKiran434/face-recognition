"""One-time migration helpers: convert legacy .pkl encodings to .npz."""

from __future__ import annotations

import os
import pickle  # noqa: S403 -- legacy data migration only

import numpy as np


def convert_pickle_directory(directory):
    """Convert .pkl files in directory to sibling .npz files.

    Returns (converted_count, skipped_count). Skips files when a .npz
    sibling already exists or when the pickle's shape is unexpected.
    Raises FileNotFoundError when directory is missing.

    Note: pickle deserialization is RCE-prone. This function is intended
    for files you wrote yourself with the legacy encode_faces.py. If a
    .pkl might be tampered with, delete it and re-encode from source
    images instead of running this migration.
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(directory)

    converted = 0
    skipped = 0
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".pkl"):
            continue
        pkl_path = os.path.join(directory, name)
        npz_path = os.path.splitext(pkl_path)[0] + ".npz"

        if os.path.exists(npz_path):
            skipped += 1
            continue

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)  # noqa: S301

        if not isinstance(data, dict) or "encodings" not in data or "names" not in data:
            skipped += 1
            continue

        np.savez(
            npz_path,
            encodings=np.asarray(data["encodings"]),
            names=np.asarray(data["names"]),
        )
        converted += 1

    return converted, skipped
