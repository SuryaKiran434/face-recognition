"""Shared configuration loader."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"


@dataclass(frozen=True)
class Config:
    encodings_dir: str
    face_recognition_threshold: float
    resize_factor: float
    process_frame_interval: int
    face_detection_model: str = "hog"

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
        )
