import json

from face_recognition_app.config import Config


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
