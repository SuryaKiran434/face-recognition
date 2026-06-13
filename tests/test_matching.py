import numpy as np

from face_recognition_app.matching import largest_face, match_faces


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
