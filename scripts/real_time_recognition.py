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


def load_encodings(encodings_dir):
    encodings_list = []
    names_list = []
    for file_path in sorted(glob.glob(os.path.join(encodings_dir, "*.npz"))):
        print(f"Loading encodings from {file_path}...")
        # allow_pickle=False is critical: it prevents arbitrary code execution
        # from a malicious .npz dropped into the encodings directory.
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
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    small_frame = cv2.resize(rgb_frame, (0, 0), fx=resize_factor, fy=resize_factor)
    scale_factor = frame.shape[1] / small_frame.shape[1]

    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    face_locations = [
        (int(top * scale_factor), int(right * scale_factor),
         int(bottom * scale_factor), int(left * scale_factor))
        for top, right, bottom, left in face_locations
    ]

    names = []
    for face_encoding in face_encodings:
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        min_distance = np.min(distances)
        if min_distance < threshold:
            names.append(known_names[int(np.argmin(distances))])
        else:
            names.append("Unknown")

    return face_locations, names


def main():
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Could not open video stream.", file=sys.stderr)
        sys.exit(1)

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
                face_locations, face_names = process_frame(frame)
                end_time = time.time()

                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.putText(frame, name, (left, bottom + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                print(f"Frame processed in {end_time - start_time:.2f} seconds")

            cv2.imshow("Face Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
