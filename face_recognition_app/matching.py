"""Pure numpy helpers for face matching. No heavy dependencies."""

from __future__ import annotations

import numpy as np


def largest_face(face_locations):
    """Return the (top, right, bottom, left) tuple with the largest bbox area."""
    return max(face_locations, key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]))


def squared_norms(encodings):
    """Row-wise ||v||^2 for an (N, D) encoding array.

    Hoisted out of match_faces so the known set's norms are computed once when
    the encodings are loaded rather than on every processed frame.
    """
    arr = np.asarray(encodings)
    if arr.size == 0:
        return np.zeros(len(arr), dtype=float)
    return np.einsum("ij,ij->i", arr, arr)


def match_faces(known_encodings, known_names, face_encodings, threshold,
                known_sq=None):
    """Match each face encoding against the known set.

    Args:
        known_encodings: (N, D) array of stored face encodings.
        known_names: length-N sequence of names parallel to known_encodings.
        face_encodings: (M, D) array (or list) of encodings to match.
        threshold: max euclidean distance to count as a match.
        known_sq: optional precomputed squared_norms(known_encodings). Pass it
            to skip recomputing the known set's norms on every frame.

    Returns a list of length M: the matched name when min distance < threshold,
    else "Unknown".
    """
    enc_array = np.asarray(face_encodings)
    if enc_array.size == 0:
        return []

    known = np.asarray(known_encodings)
    if known.size == 0:
        # Nothing to match against: everyone is unknown.
        return ["Unknown"] * len(enc_array)

    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b, so the (M, N) distance matrix
    # falls out of one BLAS matmul with no (M, N, D) intermediate.
    if known_sq is None:
        known_sq = squared_norms(known)
    else:
        known_sq = np.asarray(known_sq)
    enc_sq = np.einsum("ij,ij->i", enc_array, enc_array)

    d2 = enc_sq[:, None] + known_sq[None, :] - 2.0 * (enc_array @ known.T)
    # Cancellation can push near-identical vectors a hair below zero; clamping
    # also keeps ties between duplicate encodings exact, so argmin picks the
    # first one just as the old elementwise implementation did.
    np.maximum(d2, 0.0, out=d2)

    best_idx = np.argmin(d2, axis=1)
    best_d2 = d2[np.arange(len(enc_array)), best_idx]

    # Compare squared distances to skip M*N sqrt calls. threshold <= 0 can
    # never match, and squaring would wrongly flip a negative threshold.
    thresh_sq = threshold * threshold if threshold > 0 else -1.0

    return [
        known_names[int(i)] if d < thresh_sq else "Unknown"
        for i, d in zip(best_idx, best_d2)
    ]
