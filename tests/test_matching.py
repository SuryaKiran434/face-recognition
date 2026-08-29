import tracemalloc

import numpy as np
import pytest

from face_recognition_app.matching import largest_face, match_faces, squared_norms


def test_largest_face_picks_biggest_bbox():
    faces = [
        (0, 10, 10, 0),    # 10x10 = 100
        (0, 50, 50, 0),    # 50x50 = 2500
        (0, 20, 20, 0),    # 20x20 = 400
    ]
    assert largest_face(faces) == (0, 50, 50, 0)


def test_match_returns_closest_name_below_threshold():
    known = np.array([[0.0, 0.0], [1.0, 1.0]])
    names = np.array(["alice", "bob"])
    face_encs = np.array([[0.05, 0.05]])
    assert match_faces(known, names, face_encs, threshold=0.5) == ["alice"]


def test_match_returns_unknown_when_distance_exceeds_threshold():
    known = np.array([[0.0, 0.0]])
    names = np.array(["alice"])
    face_encs = np.array([[10.0, 10.0]])
    assert match_faces(known, names, face_encs, threshold=0.5) == ["Unknown"]


def test_match_handles_multiple_faces_in_frame():
    known = np.array([[0.0, 0.0], [1.0, 1.0]])
    names = np.array(["alice", "bob"])
    face_encs = np.array([[0.0, 0.0], [1.0, 1.0]])
    assert match_faces(known, names, face_encs, threshold=0.5) == ["alice", "bob"]


def test_match_empty_input_returns_empty_list():
    known = np.array([[0.0, 0.0]])
    names = np.array(["alice"])
    assert match_faces(known, names, [], threshold=0.5) == []


def test_match_threshold_boundary_is_strict_less_than():
    # distance == threshold should NOT match (must be strictly less)
    known = np.array([[0.0, 0.0]])
    names = np.array(["alice"])
    face_encs = np.array([[0.5, 0.0]])  # distance = 0.5
    assert match_faces(known, names, face_encs, threshold=0.5) == ["Unknown"]


# --- Equivalence with the original broadcast implementation -----------------
#
# match_faces used to materialise an (M, N, D) difference array just to reduce
# it to (M, N).  The replacement uses the ||a-b||^2 = ||a||^2 + ||b||^2 - 2a.b
# identity via a single matmul.  The reference below is that original code,
# kept verbatim so we can assert the fast path returns IDENTICAL results.


def _legacy_match_faces(known_encodings, known_names, face_encodings, threshold):
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


def _assert_same(known, names, encs, threshold):
    expected = _legacy_match_faces(known, names, encs, threshold)
    assert match_faces(known, names, encs, threshold) == expected
    # The precomputed-norms path must agree too.
    assert match_faces(known, names, encs, threshold,
                       known_sq=squared_norms(known)) == expected
    return expected


def test_equivalence_randomised_inputs():
    rng = np.random.default_rng(20240613)
    for n_known in (1, 2, 7, 64):
        for n_faces in (1, 3, 9):
            known = rng.normal(size=(n_known, 128))
            names = np.array([f"person{i}" for i in range(n_known)])
            encs = rng.normal(size=(n_faces, 128))
            for threshold in (0.0, 0.6, 5.0, 20.0, 1e6):
                _assert_same(known, names, encs, threshold)


def test_equivalence_near_identical_encodings():
    """Faces that are (almost) exactly a known encoding: this is where the
    a.b identity suffers cancellation and can go slightly negative."""
    rng = np.random.default_rng(99)
    known = rng.normal(size=(20, 128))
    names = np.array([f"person{i}" for i in range(20)])
    encs = np.vstack([
        known[3],                                    # exact copy
        known[11] + 1e-12,                           # a hair off
        known[0] * (1.0 + 1e-15),                    # relative perturbation
    ])
    assert _assert_same(known, names, encs, 0.6) == ["person3", "person11", "person0"]


def test_equivalence_no_faces():
    rng = np.random.default_rng(1)
    known = rng.normal(size=(5, 128))
    names = np.array([f"person{i}" for i in range(5)])
    assert _assert_same(known, names, np.empty((0, 128)), 0.6) == []
    assert _assert_same(known, names, [], 0.6) == []


def test_no_known_encodings_returns_unknown():
    """load_encodings returns np.empty((0, 128)) when the directory is empty.
    The old implementation raised on that (argmin over a zero-length axis);
    the new one degrades to 'Unknown' instead."""
    known = np.empty((0, 128))
    names = np.array([], dtype=str)
    encs = np.zeros((3, 128))

    with pytest.raises(ValueError):
        _legacy_match_faces(known, names, encs, 0.6)

    assert match_faces(known, names, encs, 0.6) == ["Unknown"] * 3
    assert match_faces(known, names, [], 0.6) == []


def test_equivalence_exact_threshold_boundary():
    """Distances chosen to land exactly on the threshold in binary floating
    point, so 'strictly less than' is tested with no rounding slop."""
    known = np.array([[0.0, 0.0], [0.0, 0.0]])
    names = np.array(["alice", "bob"])
    for threshold in (0.5, 0.25, 1.0, 2.0):
        encs = np.array([
            [threshold, 0.0],          # distance == threshold -> Unknown
            [threshold / 2, 0.0],      # inside -> alice
            [threshold * 2, 0.0],      # outside -> Unknown
        ])
        assert _assert_same(known, names, encs, threshold) == [
            "Unknown", "alice", "Unknown",
        ]

    # 3-4-5 triangle: distance is exactly 5.0 across both dimensions.
    known = np.array([[0.0, 0.0]])
    names = np.array(["alice"])
    encs = np.array([[3.0, 4.0]])
    assert _assert_same(known, names, encs, 5.0) == ["Unknown"]
    assert _assert_same(known, names, encs, 5.0000001) == ["alice"]


def test_equivalence_ties_pick_first_match():
    """Duplicate known encodings tie exactly; argmin must keep taking the
    lowest index, as the old elementwise implementation did."""
    known = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    names = np.array(["first", "second", "third"])
    encs = np.array([[1.0, 2.0], [1.5, 2.5]])
    assert _assert_same(known, names, encs, 0.9) == ["first", "first"]

    # Ties at equal distance either side of a known encoding.
    known = np.array([[-1.0, 0.0], [1.0, 0.0]])
    names = np.array(["left", "right"])
    encs = np.array([[0.0, 0.0]])
    assert _assert_same(known, names, encs, 2.0) == ["left"]

    # Tie between an exact hit and a duplicate, on random data.
    rng = np.random.default_rng(7)
    base = rng.normal(size=(4, 128))
    known = np.vstack([base, base])           # every row duplicated
    names = np.array([f"p{i}" for i in range(len(known))])
    _assert_same(known, names, base, 0.6)


def test_equivalence_single_known_encoding():
    rng = np.random.default_rng(31337)
    known = rng.normal(size=(1, 128))
    names = np.array(["solo"])
    encs = rng.normal(size=(4, 128))
    _assert_same(known, names, encs, 0.6)
    _assert_same(known, names, encs, 1e6)


def test_negative_threshold_never_matches():
    """threshold is squared internally; a negative one must not flip positive."""
    known = np.array([[0.0, 0.0]])
    names = np.array(["alice"])
    encs = np.array([[0.0, 0.0]])
    assert _assert_same(known, names, encs, -1.0) == ["Unknown"]


def test_no_large_intermediate_allocated():
    """The whole point: matching must not allocate an (M, N, D) array."""
    rng = np.random.default_rng(5)
    known = rng.normal(size=(1000, 128))
    names = np.array([f"person{i}" for i in range(1000)])
    encs = rng.normal(size=(5, 128))

    tracemalloc.start()
    try:
        match_faces(known, names, encs, 0.6, known_sq=squared_norms(known))
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    # (5, 1000, 128) float64 would be ~5 MB; the (5, 1000) result is ~40 KB.
    assert peak < 1_000_000, f"peak allocation {peak} bytes looks like an (M, N, D) buffer"
