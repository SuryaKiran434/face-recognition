import pickle

import numpy as np
import pytest

from face_recognition_app.converter import convert_pickle_directory


def test_converts_valid_pkl_to_npz(tmp_path):
    pkl = tmp_path / "encodings.pkl"
    payload = {
        "encodings": [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])],
        "names": ["alice", "bob"],
    }
    pkl.write_bytes(pickle.dumps(payload))

    converted, skipped = convert_pickle_directory(str(tmp_path))
    assert converted == 1
    assert skipped == 0

    npz_path = tmp_path / "encodings.npz"
    assert npz_path.exists()
    with np.load(npz_path, allow_pickle=False) as data:
        assert data["encodings"].shape == (2, 3)
        assert list(data["names"]) == ["alice", "bob"]


def test_skips_when_npz_already_exists(tmp_path):
    pkl = tmp_path / "encodings.pkl"
    pkl.write_bytes(pickle.dumps({"encodings": [np.array([0.0])], "names": ["x"]}))
    (tmp_path / "encodings.npz").write_bytes(b"placeholder")

    converted, skipped = convert_pickle_directory(str(tmp_path))
    assert converted == 0
    assert skipped == 1


def test_skips_unexpected_format(tmp_path):
    pkl = tmp_path / "weird.pkl"
    pkl.write_bytes(pickle.dumps(["not", "a", "dict"]))

    converted, skipped = convert_pickle_directory(str(tmp_path))
    assert converted == 0
    assert skipped == 1


def test_raises_on_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        convert_pickle_directory(str(tmp_path / "nope"))


def test_ignores_non_pkl_files(tmp_path):
    (tmp_path / "readme.txt").write_text("hi")
    (tmp_path / "data.npz").write_bytes(b"placeholder")

    converted, skipped = convert_pickle_directory(str(tmp_path))
    assert converted == 0
    assert skipped == 0
