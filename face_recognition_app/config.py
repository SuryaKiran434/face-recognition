"""Shared configuration loader."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"


def coerce_camera_source(value):
    """Normalize a camera source value.

    Accepts an int (device index) or a string. Strings that parse as an
    integer become device indices; anything else (RTSP/HTTP URL, GStreamer
    pipeline) is returned unchanged for cv2.VideoCapture to handle.
    """
    if isinstance(value, bool):
        # bool is an int subclass — reject explicitly to catch JSON true/false typos.
        raise TypeError(f"camera_source must be int or str, got bool")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return value
    raise TypeError(f"camera_source must be int or str, got {type(value).__name__}")


@dataclass(frozen=True)
class Config:
    encodings_dir: str
    face_recognition_threshold: float
    resize_factor: float
    process_frame_interval: int
    face_detection_model: str = "hog"
    camera_source: Union[int, str] = 0

    @classmethod
    def load(cls, path=None):
        path = Path(path) if path else DEFAULT_CONFIG_PATH
        with open(path, "r") as f:
            data = json.load(f)
        return cls(
            encodings_dir=str(data["encodings_dir"]),
            face_recognition_threshold=float(data["face_recognition_threshold"]),
            resize_factor=float(data["resize_factor"]),
            process_frame_interval=int(data["process_frame_interval"]),
            face_detection_model=str(data.get("face_detection_model", "hog")),
            camera_source=coerce_camera_source(data.get("camera_source", 0)),
        )
