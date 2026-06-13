"""Pure numpy helpers for face matching. No heavy dependencies."""

from __future__ import annotations

import numpy as np


def largest_face(face_locations):
    """Return the (top, right, bottom, left) tuple with the largest bbox area."""
    return max(face_locations, key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]))


def match_faces(known_encodings, known_names, face_encodings, threshold):
    """Match each face encoding against the known set.

    Args:
        known_encodings: (N, D) array of stored face encodings.
        known_names: length-N sequence of names parallel to known_encodings.
        face_encodings: (M, D) array (or list) of encodings to match.
        threshold: max euclidean distance to count as a match.

    Returns a list of length M: the matched name when min distance < threshold,
    else "Unknown".
    """
    enc_array = np.asarray(face_encodings)
    if enc_array.size == 0:
        return []

    diffs = known_encodings[np.newaxis, :, :] - enc_array[:, np.newaxis, :]
    dists = np.linalg.norm(diffs, axis=2)
    best_idx = np.argmin(dists, axis=1)
    best_dist = dists[np.arange(len(enc_array)), best_idx]

    return [
        known_names[int(i)] if d < threshold else "Unknown"
        for i, d in zip(best_idx, best_dist)
    ]
