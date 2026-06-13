import glob
import json
import os
import sys
import time
from pathlib import Path

import cv2
import face_recognition
import numpy as np


CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"

with open(CONFIG_PATH, "r") as config_file:
    config = json.load(config_file)

encodings_dir = config["encodings_dir"]
threshold = float(config["face_recognition_threshold"])
resize_factor = float(config["resize_factor"])
process_frame_interval = int(config["process_frame_interval"])
detection_model = config.get("face_detection_model", "hog")


def load_encodings(encodings_dir):
    encodings_list = []
    names_list = []
    for file_path in sorted(glob.glob(os.path.join(encodings_dir, "*.npz"))):
        print(f"Loading encodings from {file_path}...")
        # allow_pickle=False prevents arbitrary code execution from a
        # malicious .npz dropped into the encodings directory.
        with np.load(file_path, allow_pickle=False) as data:
            encodings_list.append(data["encodings"])
            names_list.append(data["names"])
    if not encodings_list:
        return np.empty((0, 128)), np.array([], dtype=str)
    return np.concatenate(encodings_list, axis=0), np.concatenate(names_list, axis=0)


known_encodings, known_names = load_encodings(encodings_dir)
print(f"Loaded {len(known_encodings)} encodings.")

if len(known_encodings) == 0:
    print(f"Error: no .npz encoding files found in {encodings_dir}.", file=sys.stderr)
    sys.exit(1)


def process_frame(frame):
    # Resize first (smaller buffer), then color-convert — ~4x less work
    # at resize_factor=0.5.
    small_bgr = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)
    small_frame = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
    scale_factor = frame.shape[1] / small_frame.shape[1]

    face_locations = face_recognition.face_locations(small_frame, model=detection_model)
    if not face_locations:
        return [], []

    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    face_locations = [
        (int(top * scale_factor), int(right * scale_factor),
         int(bottom * scale_factor), int(left * scale_factor))
        for top, right, bottom, left in face_locations
    ]

    # Single broadcast subtraction across all faces in the frame, rather
    # than calling face_distance() per face. Shape: (n_faces, n_known).
    enc_array = np.asarray(face_encodings)
    diffs = known_encodings[np.newaxis, :, :] - enc_array[:, np.newaxis, :]
    dists = np.linalg.norm(diffs, axis=2)
    best_idx = np.argmin(dists, axis=1)
    best_dist = dists[np.arange(len(enc_array)), best_idx]

    names = [
        known_names[int(i)] if d < threshold else "Unknown"
        for i, d in zip(best_idx, best_dist)
    ]
    return face_locations, names


def main():
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Could not open video stream.", file=sys.stderr)
        sys.exit(1)

    # Cache last detections so non-detection frames still draw rectangles.
    last_face_locations = []
    last_face_names = []

    try:
        frame_count = 0
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Error: Failed to capture frame.", file=sys.stderr)
                break

            frame_count += 1
            if frame_count % process_frame_interval == 0:
                start_time = time.time()
                last_face_locations, last_face_names = process_frame(frame)
                end_time = time.time()
                print(f"Frame processed in {end_time - start_time:.2f} seconds")

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


if __name__ == "__main__":
    main()
