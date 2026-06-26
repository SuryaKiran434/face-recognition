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
class DetectionConfig:
    """Object-detection settings (the YOLO person/package stage)."""

    enabled: bool = True
    weights: str = "yolov8n.pt"
    confidence: float = 0.4
    # Proximity gate: a person only counts as "near" (and thus triggers an
    # event) when their bounding box occupies at least this fraction of the
    # frame area. 0 disables the gate (any detected person triggers).
    near_min_area_ratio: float = 0.0
    # Temporal smoothing for the recognised name: number of recent frames the
    # majority vote runs over. 1 disables smoothing (raw per-frame result).
    label_smoothing_window: int = 5

    @classmethod
    def from_dict(cls, data):
        data = data or {}
        return cls(
            enabled=bool(data.get("enabled", True)),
            weights=str(data.get("weights", "yolov8n.pt")),
            confidence=float(data.get("confidence", 0.4)),
            near_min_area_ratio=float(data.get("near_min_area_ratio", 0.0)),
            label_smoothing_window=int(data.get("label_smoothing_window", 5)),
        )


@dataclass(frozen=True)
class EventsConfig:
    """Detection-event behaviour (debounce, cooldown, snapshots, retention)."""

    debounce_frames: int = 3
    cooldown_seconds: float = 120.0
    retention_days: int = 14
    snapshots_dir: str = "snapshots"
    log_path: str = "events.jsonl"

    @classmethod
    def from_dict(cls, data):
        data = data or {}
        return cls(
            debounce_frames=int(data.get("debounce_frames", 3)),
            cooldown_seconds=float(data.get("cooldown_seconds", 120.0)),
            retention_days=int(data.get("retention_days", 14)),
            snapshots_dir=str(data.get("snapshots_dir", "snapshots")),
            log_path=str(data.get("log_path", "events.jsonl")),
        )


@dataclass(frozen=True)
class NotifyConfig:
    """Email-notification settings (credentials live in .env, never here)."""

    enabled: bool = True
    email_on: tuple = ("known", "unknown", "likely_delivery")

    @classmethod
    def from_dict(cls, data):
        data = data or {}
        return cls(
            enabled=bool(data.get("enabled", True)),
            email_on=tuple(data.get("email_on", ["known", "unknown", "likely_delivery"])),
        )


@dataclass(frozen=True)
class Config:
    encodings_dir: str
    face_recognition_threshold: float
    resize_factor: float
    process_frame_interval: int
    face_detection_model: str = "hog"
    camera_source: Union[int, str] = 0
    detection: DetectionConfig = DetectionConfig()
    events: EventsConfig = EventsConfig()
    notify: NotifyConfig = NotifyConfig()

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
            detection=DetectionConfig.from_dict(data.get("detection")),
            events=EventsConfig.from_dict(data.get("events")),
            notify=NotifyConfig.from_dict(data.get("notify")),
        )
