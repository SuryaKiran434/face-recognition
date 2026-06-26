"""Real-time face recognition from webcam."""

from __future__ import annotations

import glob
import logging
import os
import time

import cv2
import face_recognition
import numpy as np

from face_recognition_app import notify
from face_recognition_app.config import Config
from face_recognition_app.decision import DoorStatus, classify, classify_people
from face_recognition_app.detector import (
    PERSON_LABEL,
    build_detector,
    carried_objects,
    count_people,
)
from face_recognition_app.events import EventGate, purge_old, save_event
from face_recognition_app.matching import match_faces


logger = logging.getLogger(__name__)

# Banner colours per door status (BGR).
_STATUS_COLORS = {
    "known": (0, 180, 0),
    "unknown": (0, 165, 255),
    "likely_delivery": (255, 80, 0),
}

_STATUS_TEXT = {
    "known": lambda s: f"KNOWN: {s.name}",
    "unknown": lambda s: "UNKNOWN",
    "likely_delivery": lambda s: "LIKELY DELIVERY",
}


def draw_detections(frame, detections):
    """Outline carried objects (bags/suitcases). Person boxes are skipped —
    faces are already boxed and full-body boxes just add clutter."""
    for det in detections:
        if det.label == "person":
            continue
        x1, y1, x2, y2 = det.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
        cv2.putText(frame, det.label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)


def draw_status_banner(frame, status):
    """Draw a top banner summarising the door status (nothing when 'none')."""
    if status.label == "none":
        return
    color = _STATUS_COLORS.get(status.label, (128, 128, 128))
    text = _STATUS_TEXT.get(status.label, lambda s: s.label.upper())(status)
    width = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (width, 40), color, -1)
    cv2.putText(frame, text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    if status.reasons:
        cv2.putText(frame, "; ".join(status.reasons), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def load_encodings(encodings_dir):
    encodings_list = []
    names_list = []
    for file_path in sorted(glob.glob(os.path.join(encodings_dir, "*.npz"))):
        logger.info("Loading encodings from %s", file_path)
        # allow_pickle=False prevents arbitrary code execution from a
        # malicious .npz dropped into the encodings directory.
        with np.load(file_path, allow_pickle=False) as data:
            encodings_list.append(data["encodings"])
            names_list.append(data["names"])
    if not encodings_list:
        return np.empty((0, 128)), np.array([], dtype=str)
    return (np.concatenate(encodings_list, axis=0),
            np.concatenate(names_list, axis=0))


def process_frame(frame, known_encodings, known_names, cfg):
    small_bgr = cv2.resize(frame, (0, 0), fx=cfg.resize_factor, fy=cfg.resize_factor)
    small_frame = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
    scale_factor = frame.shape[1] / small_frame.shape[1]

    face_locations = face_recognition.face_locations(
        small_frame, model=cfg.face_detection_model
    )
    if not face_locations:
        return [], []

    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    face_locations = [
        (int(top * scale_factor), int(right * scale_factor),
         int(bottom * scale_factor), int(left * scale_factor))
        for top, right, bottom, left in face_locations
    ]

    names = match_faces(
        known_encodings, known_names, face_encodings, cfg.face_recognition_threshold
    )
    return face_locations, names


def _person_boxes(detections):
    return [d.box for d in detections if d.label == PERSON_LABEL]


def _carried_boxes(detections):
    return [d.box for d in detections if d.label != PERSON_LABEL]


def emit_event(frame, people, cfg, send_email=True):
    """Persist a detection event (frame + per-person crops + log) and, when
    enabled, email it. Never raises — failures are logged so the capture loop
    keeps running."""
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    try:
        event = save_event(
            frame, people, cfg.events.snapshots_dir, cfg.events.log_path, timestamp
        )
    except Exception:
        logger.exception("Failed to save detection event")
        return None

    should_email = (
        send_email
        and cfg.notify.enabled
        and any(p.label in cfg.notify.email_on for p in people)
    )
    if should_email:
        try:
            code = notify.send_event_email(event)
            if code != 0:
                logger.warning("Email not sent (code %d)", code)
        except Exception:
            logger.exception("Failed to send event email")
    return event


def run_recognizer(cfg: Config, send_email=True, headless=False):
    known_encodings, known_names = load_encodings(cfg.encodings_dir)
    logger.info("Loaded %d encodings", len(known_encodings))
    if len(known_encodings) == 0:
        raise ValueError(f"no .npz encoding files found in {cfg.encodings_dir}")

    detector = build_detector(cfg)
    purge_old(cfg.events.snapshots_dir, cfg.events.log_path, cfg.events.retention_days)
    gate = EventGate(
        debounce_frames=cfg.events.debounce_frames,
        cooldown_seconds=cfg.events.cooldown_seconds,
    )

    logger.info("Opening video source: %r", cfg.camera_source)
    video_capture = cv2.VideoCapture(cfg.camera_source)
    if not video_capture.isOpened():
        raise RuntimeError(f"could not open video source: {cfg.camera_source!r}")

    last_face_locations = []
    last_face_names = []
    last_detections = []
    last_people = []
    last_status = DoorStatus("none", None, 0.0, ())

    try:
        frame_count = 0
        while True:
            ret, frame = video_capture.read()
            if not ret:
                logger.warning("Failed to capture frame; exiting loop")
                break

            frame_count += 1
            if frame_count % cfg.process_frame_interval == 0:
                start_time = time.time()
                last_face_locations, last_face_names = process_frame(
                    frame, known_encodings, known_names, cfg
                )
                last_detections = detector.detect(frame)

                person_count = count_people(last_detections)
                present = person_count > 0 or len(last_face_names) > 0
                dwell = gate.dwell_seconds()

                faces = list(zip(last_face_locations, last_face_names))
                last_people = classify_people(
                    _person_boxes(last_detections),
                    faces,
                    _carried_boxes(last_detections),
                    dwell_seconds=dwell,
                    dwell_threshold=cfg.detection.person_dwell_seconds,
                )
                last_status = classify(
                    last_face_names, person_count,
                    carried_objects(last_detections),
                    dwell_seconds=dwell,
                    dwell_threshold=cfg.detection.person_dwell_seconds,
                )
                logger.debug("Frame processed in %.2fs -> %s",
                             time.time() - start_time, last_status.label)

                # One event per visit: fire after debounce, then cooldown.
                if gate.observe(present) and last_people:
                    event = emit_event(frame, last_people, cfg, send_email=send_email)
                    if event:
                        logger.info("Detection event: %s", event.summary)

            if headless:
                continue

            for (top, right, bottom, left), name in zip(last_face_locations, last_face_names):
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, name, (left, bottom + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            draw_detections(frame, last_detections)
            draw_status_banner(frame, last_status)

            cv2.imshow("Face Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        video_capture.release()
        cv2.destroyAllWindows()


def run_once(cfg: Config, image_path, send_email=True):
    """Process a single still image end-to-end (no camera): detect, classify
    everyone, save the event, and optionally email it. Useful for testing the
    full pipeline and email wiring. Returns the DoorEvent (or None)."""
    known_encodings, known_names = load_encodings(cfg.encodings_dir)
    if len(known_encodings) == 0:
        raise ValueError(f"no .npz encoding files found in {cfg.encodings_dir}")

    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"could not read image: {image_path!r}")

    detector = build_detector(cfg)
    face_locations, face_names = process_frame(frame, known_encodings, known_names, cfg)
    detections = detector.detect(frame)
    people = classify_people(
        _person_boxes(detections),
        list(zip(face_locations, face_names)),
        _carried_boxes(detections),
    )
    if not people:
        logger.info("No person found in %s; nothing to do", image_path)
        return None
    return emit_event(frame, people, cfg, send_email=send_email)
