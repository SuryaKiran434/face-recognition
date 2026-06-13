"""Real-time face recognition from webcam."""

from __future__ import annotations

import glob
import logging
import os
import time

import cv2
import face_recognition
import numpy as np

from face_recognition_app.config import Config
from face_recognition_app.matching import match_faces


logger = logging.getLogger(__name__)


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


def run_recognizer(cfg: Config):
    known_encodings, known_names = load_encodings(cfg.encodings_dir)
    logger.info("Loaded %d encodings", len(known_encodings))
    if len(known_encodings) == 0:
        raise ValueError(f"no .npz encoding files found in {cfg.encodings_dir}")

    logger.info("Opening video source: %r", cfg.camera_source)
    video_capture = cv2.VideoCapture(cfg.camera_source)
    if not video_capture.isOpened():
        raise RuntimeError(f"could not open video source: {cfg.camera_source!r}")

    last_face_locations = []
    last_face_names = []

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
                logger.debug("Frame processed in %.2fs", time.time() - start_time)

            for (top, right, bottom, left), name in zip(last_face_locations, last_face_names):
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, name, (left, bottom + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow("Face Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        video_capture.release()
        cv2.destroyAllWindows()
