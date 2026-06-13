import json

import pytest

from face_recognition_app.config import Config, coerce_camera_source


def test_load_full_config(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "encodings_dir": "/tmp/enc",
        "face_recognition_threshold": 0.5,
        "resize_factor": 0.25,
        "process_frame_interval": 3,
        "face_detection_model": "cnn",
    }))
    cfg = Config.load(p)
    assert cfg.encodings_dir == "/tmp/enc"
    assert cfg.face_recognition_threshold == 0.5
    assert cfg.resize_factor == 0.25
    assert cfg.process_frame_interval == 3
    assert cfg.face_detection_model == "cnn"


def test_face_detection_model_defaults_to_hog(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "encodings_dir": "/tmp/enc",
        "face_recognition_threshold": 0.6,
        "resize_factor": 0.5,
        "process_frame_interval": 2,
    }))
    cfg = Config.load(p)
    assert cfg.face_detection_model == "hog"


def test_numeric_fields_coerced_from_strings(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "encodings_dir": "/tmp/enc",
        "face_recognition_threshold": "0.6",
        "resize_factor": "0.5",
        "process_frame_interval": "2",
    }))
    cfg = Config.load(p)
    assert cfg.face_recognition_threshold == 0.6
    assert cfg.resize_factor == 0.5
    assert cfg.process_frame_interval == 2


def test_camera_source_defaults_to_zero(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "encodings_dir": "/tmp/enc",
        "face_recognition_threshold": 0.6,
        "resize_factor": 0.5,
        "process_frame_interval": 2,
    }))
    cfg = Config.load(p)
    assert cfg.camera_source == 0


def test_camera_source_accepts_integer(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "encodings_dir": "/tmp/enc",
        "face_recognition_threshold": 0.6,
        "resize_factor": 0.5,
        "process_frame_interval": 2,
        "camera_source": 1,
    }))
    cfg = Config.load(p)
    assert cfg.camera_source == 1


def test_camera_source_accepts_numeric_string(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "encodings_dir": "/tmp/enc",
        "face_recognition_threshold": 0.6,
        "resize_factor": 0.5,
        "process_frame_interval": 2,
        "camera_source": "2",
    }))
    cfg = Config.load(p)
    assert cfg.camera_source == 2


def test_camera_source_preserves_rtsp_url(tmp_path):
    url = "rtsp://user:pass@192.168.1.50:554/stream1"
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "encodings_dir": "/tmp/enc",
        "face_recognition_threshold": 0.6,
        "resize_factor": 0.5,
        "process_frame_interval": 2,
        "camera_source": url,
    }))
    cfg = Config.load(p)
    assert cfg.camera_source == url


def test_coerce_camera_source_passes_int_through():
    assert coerce_camera_source(0) == 0
    assert coerce_camera_source(7) == 7


def test_coerce_camera_source_parses_numeric_strings():
    assert coerce_camera_source("0") == 0
    assert coerce_camera_source("12") == 12


def test_coerce_camera_source_leaves_urls_untouched():
    assert coerce_camera_source("http://192.168.1.10:4747/video") == \
        "http://192.168.1.10:4747/video"


def test_coerce_camera_source_rejects_bool():
    with pytest.raises(TypeError):
        coerce_camera_source(True)


def test_coerce_camera_source_rejects_float():
    with pytest.raises(TypeError):
        coerce_camera_source(3.14)
