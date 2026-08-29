"""Tests for the image preprocessing path -- the repo's only Pillow consumer.

These cover the Pillow/pillow-heif API surface `preprocessor` actually calls
(`register_heif_opener`, `Image.open`, `thumbnail`, `convert("RGB")`,
`save(..., "JPEG", quality=90)`), so a Pillow upgrade that breaks image
handling fails CI instead of passing unnoticed.

Fixtures are synthesised with numpy into `tmp_path`: no camera, no network,
no committed binaries. Nothing here needs dlib or ultralytics.
"""

import os
import random
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

# `preprocessor` imports pillow_heif at module scope and registers the HEIF
# opener on import, so it cannot be imported at all without pillow-heif.
# Skip cleanly rather than erroring the whole file if the wheel is missing.
pytest.importorskip("pillow_heif", reason="pillow-heif required to import preprocessor")

from face_recognition_app.preprocessor import (  # noqa: E402
    preprocess_datasets,
    resize_image,
    sample_and_process,
)


# Encoding HEIF segfaults the interpreter if OpenCV has already been imported
# into the same process -- a native-library conflict between opencv-python and
# pillow-heif that predates (and is unrelated to) any Pillow version here.
# Other modules in this suite import cv2, so HEIC fixtures are encoded in a
# clean subprocess. Decoding HEIF in-process is unaffected, so the code path
# under test still runs in the pytest process where its assertions live.
_HEIC_WRITER = """
import sys
import numpy as np
import pillow_heif
from PIL import Image

pillow_heif.register_heif_opener()
path, width, height = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
arr = np.random.default_rng(0).integers(0, 256, (height, width, 3), dtype=np.uint8)
Image.fromarray(arr, "RGB").save(path, format="HEIF")
"""


def make_heic(path, width, height):
    """Write a real .heic via a subprocess; skip the test if unsupported.

    Skips rather than fails when this pillow-heif build has no HEIF encoder
    (some platform wheels ship without one), so the job stays green.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _HEIC_WRITER, str(path), str(width), str(height)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not os.path.exists(path):
        pytest.skip(f"pillow-heif cannot encode HEIF here: {proc.stderr.strip()[-200:]}")
    return str(path)


def make_image(path, width, height, mode="RGB", fmt=None, seed=0):
    """Write a deterministic synthetic image and return its path as str."""
    rng = np.random.default_rng(seed)
    channels = {"RGB": 3, "RGBA": 4, "L": 1}[mode]
    shape = (height, width) if channels == 1 else (height, width, channels)
    arr = rng.integers(0, 256, shape, dtype=np.uint8)
    Image.fromarray(arr, mode).save(str(path), fmt)
    return str(path)


# --- resize_image -----------------------------------------------------------


def test_downscales_oversized_image_within_bounds(tmp_path):
    src = make_image(tmp_path / "big.png", 1600, 1200)

    out = resize_image(src, max_width=800, max_height=800)

    with Image.open(out) as im:
        assert im.width <= 800 and im.height <= 800


def test_preserves_aspect_ratio_when_downscaling(tmp_path):
    src = make_image(tmp_path / "wide.png", 1600, 1200)

    with Image.open(resize_image(src, max_width=800, max_height=800)) as im:
        assert im.size == (800, 600)


def test_normalizes_to_rgb_jpeg(tmp_path):
    src = make_image(tmp_path / "shot.png", 400, 300)

    with Image.open(resize_image(src)) as im:
        assert im.format == "JPEG"
        assert im.mode == "RGB"


def test_replaces_non_jpeg_original(tmp_path):
    src = make_image(tmp_path / "shot.png", 400, 300)

    out = resize_image(src)

    assert out == str(tmp_path / "shot.jpg")
    assert os.path.exists(out)
    assert not os.path.exists(src), "original .png should be removed"


def test_keeps_jpeg_in_place_without_deleting_it(tmp_path):
    """The `new_path != image_path` guard: a .jpg input must survive."""
    src = make_image(tmp_path / "already.jpg", 320, 240, fmt="JPEG")

    out = resize_image(src)

    assert out == src
    assert os.path.exists(out), "in-place .jpg must not be deleted"


def test_leaves_small_image_undersized(tmp_path):
    src = make_image(tmp_path / "small.png", 120, 90)

    with Image.open(resize_image(src, max_width=800, max_height=800)) as im:
        assert im.size == (120, 90), "thumbnail must not upscale"


def test_strips_alpha_channel_for_jpeg_save(tmp_path):
    """JPEG cannot store alpha; convert("RGB") must run before save."""
    src = make_image(tmp_path / "alpha.png", 200, 150, mode="RGBA")

    with Image.open(resize_image(src)) as im:
        assert im.mode == "RGB"


def test_converts_grayscale_to_rgb(tmp_path):
    src = make_image(tmp_path / "gray.png", 200, 150, mode="L")

    with Image.open(resize_image(src)) as im:
        assert im.mode == "RGB"


def test_honours_custom_bounds(tmp_path):
    src = make_image(tmp_path / "big.png", 1000, 1000)

    with Image.open(resize_image(src, max_width=100, max_height=100)) as im:
        assert im.size == (100, 100)


# --- pillow-heif integration ------------------------------------------------


def test_heif_opener_is_registered():
    """Importing preprocessor must register HEIF with Pillow."""
    assert Image.registered_extensions().get(".heic") == "HEIF"
    assert "HEIF" in Image.OPEN


def test_reads_heic_through_registered_opener(tmp_path):
    src = make_heic(tmp_path / "photo.heic", 400, 360)

    with Image.open(src) as im:
        assert im.format == "HEIF"
        assert im.size == (400, 360)


def test_converts_heic_to_resized_jpeg(tmp_path):
    """End-to-end HEIC path: the compiled pillow-heif extension must still
    interoperate with whatever Pillow version is pinned."""
    src = make_heic(tmp_path / "photo.heic", 1000, 900)

    out = resize_image(src, max_width=800, max_height=800)

    assert out == str(tmp_path / "photo.jpg")
    assert not os.path.exists(src)
    with Image.open(out) as im:
        assert im.format == "JPEG"
        assert im.mode == "RGB"
        assert max(im.size) <= 800


# --- sample_and_process -----------------------------------------------------


def _seeded_source(tmp_path, count, ext="png", fmt=None):
    src = tmp_path / "src"
    src.mkdir()
    for i in range(count):
        make_image(src / f"p{i}.{ext}", 900, 850, fmt=fmt, seed=i)
    return src


def test_sample_and_process_samples_and_converts(tmp_path):
    random.seed(1234)
    src = _seeded_source(tmp_path, 5)
    dst = tmp_path / "dst"
    dst.mkdir()

    sample_and_process(str(src), str(dst), max_samples=3)

    produced = sorted(os.listdir(dst))
    assert len(produced) == 3
    assert all(name.endswith(".jpg") for name in produced)
    for name in produced:
        with Image.open(dst / name) as im:
            assert im.format == "JPEG"
            assert max(im.size) <= 800


def test_sample_and_process_leaves_sources_untouched(tmp_path):
    random.seed(7)
    src = _seeded_source(tmp_path, 4)
    dst = tmp_path / "dst"
    dst.mkdir()

    sample_and_process(str(src), str(dst), max_samples=2)

    assert sorted(os.listdir(src)) == ["p0.png", "p1.png", "p2.png", "p3.png"]


def test_sample_and_process_caps_at_available_files(tmp_path):
    random.seed(7)
    src = _seeded_source(tmp_path, 2)
    dst = tmp_path / "dst"
    dst.mkdir()

    sample_and_process(str(src), str(dst), max_samples=10)

    assert len(os.listdir(dst)) == 2


def test_sample_and_process_ignores_non_images(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "notes.txt").write_text("not an image")
    dst = tmp_path / "dst"
    dst.mkdir()

    sample_and_process(str(src), str(dst), max_samples=5)

    assert os.listdir(dst) == []


def test_preprocess_datasets_walks_person_folders(tmp_path):
    random.seed(99)
    root = tmp_path / "raw"
    for person in ("alice", "bob"):
        person_dir = root / person
        person_dir.mkdir(parents=True)
        make_image(person_dir / "a.png", 900, 900)
    dst = tmp_path / "out"

    preprocess_datasets([str(root)], str(dst), max_samples_per_folder=5)

    assert sorted(os.listdir(dst)) == ["alice", "bob"]
    assert os.listdir(dst / "alice") == ["a.jpg"]
