import glob
import os
import sys
import time
from pathlib import Path

import click
import cv2
import face_recognition
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from face_recognition_app.config import Config


def load_encodings(encodings_dir):
    encodings_list = []
    names_list = []
    for file_path in sorted(glob.glob(os.path.join(encodings_dir, "*.npz"))):
        click.echo(f"Loading encodings from {file_path}...")
        # allow_pickle=False prevents arbitrary code execution from a
        # malicious .npz dropped into the encodings directory.
        with np.load(file_path, allow_pickle=False) as data:
            encodings_list.append(data["encodings"])
            names_list.append(data["names"])
    if not encodings_list:
        return np.empty((0, 128)), np.array([], dtype=str)
    return np.concatenate(encodings_list, axis=0), np.concatenate(names_list, axis=0)


def _process_frame(frame, known_encodings, known_names, cfg):
    # Resize first (smaller buffer), then color-convert.
    small_bgr = cv2.resize(frame, (0, 0), fx=cfg.resize_factor, fy=cfg.resize_factor)
    small_frame = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
    scale_factor = frame.shape[1] / small_frame.shape[1]

    face_locations = face_recognition.face_locations(small_frame, model=cfg.face_detection_model)
    if not face_locations:
        return [], []

    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    face_locations = [
        (int(top * scale_factor), int(right * scale_factor),
         int(bottom * scale_factor), int(left * scale_factor))
        for top, right, bottom, left in face_locations
    ]

    # Broadcast subtraction across all faces in the frame at once.
    enc_array = np.asarray(face_encodings)
    diffs = known_encodings[np.newaxis, :, :] - enc_array[:, np.newaxis, :]
    dists = np.linalg.norm(diffs, axis=2)
    best_idx = np.argmin(dists, axis=1)
    best_dist = dists[np.arange(len(enc_array)), best_idx]

    names = [
        known_names[int(i)] if d < cfg.face_recognition_threshold else "Unknown"
        for i, d in zip(best_idx, best_dist)
    ]
    return face_locations, names


def run_recognizer(cfg):
    known_encodings, known_names = load_encodings(cfg.encodings_dir)
    click.echo(f"Loaded {len(known_encodings)} encodings.")
    if len(known_encodings) == 0:
        click.echo(f"Error: no .npz encoding files found in {cfg.encodings_dir}.", err=True)
        sys.exit(1)

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        click.echo("Error: Could not open video stream.", err=True)
        sys.exit(1)

    last_face_locations = []
    last_face_names = []

    try:
        frame_count = 0
        while True:
            ret, frame = video_capture.read()
            if not ret:
                click.echo("Error: Failed to capture frame.", err=True)
                break

            frame_count += 1
            if frame_count % cfg.process_frame_interval == 0:
                start_time = time.time()
                last_face_locations, last_face_names = _process_frame(
                    frame, known_encodings, known_names, cfg
                )
                end_time = time.time()
                click.echo(f"Frame processed in {end_time - start_time:.2f} seconds")

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


@click.command()
@click.option("--config", "config_path", default=None,
              type=click.Path(dir_okay=False),
              help="Path to config.json. Defaults to repo-root config.json.")
def main(config_path):
    cfg = Config.load(config_path)
    run_recognizer(cfg)


if __name__ == "__main__":
    main()
