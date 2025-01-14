import os
import json
import pickle
import face_recognition
import cv2
import numpy as np
import time

# Load configuration
config_path = "config.json"
with open(config_path, "r") as config_file:
    config = json.load(config_file)

encodings_dir = os.getenv("ENCODINGS_DIR", config["encodings_dir"])
threshold = float(os.getenv("FACE_RECOGNITION_THRESHOLD", config["face_recognition_threshold"]))
resize_factor = float(os.getenv("RESIZE_FACTOR", config["resize_factor"]))
process_frame_interval = int(os.getenv("PROCESS_FRAME_INTERVAL", config["process_frame_interval"]))

# Initialize known encodings and names
known_encodings = []
known_names = []

# Load all .pkl files in the directory
def load_encodings(encodings_dir):
    global known_encodings, known_names
    for file_name in os.listdir(encodings_dir):
        if file_name.endswith(".pkl"):
            file_path = os.path.join(encodings_dir, file_name)
            print(f"Loading encodings from {file_path}...")
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                known_encodings.extend(data["encodings"])
                known_names.extend(data["names"])
    print(f"Loaded {len(known_encodings)} encodings.")

load_encodings(encodings_dir)

# Convert known encodings to a NumPy array for better performance
known_encodings = np.array(known_encodings)

# Initialize video capture
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Function to process a single frame
def process_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize frame for faster processing
    small_frame = cv2.resize(rgb_frame, (0, 0), fx=resize_factor, fy=resize_factor)
    scale_factor = frame.shape[1] / small_frame.shape[1]

    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    # Scale face locations back to the original frame size
    face_locations = [
        (int(top * scale_factor), int(right * scale_factor), int(bottom * scale_factor), int(left * scale_factor))
        for top, right, bottom, left in face_locations
    ]

    names = []
    for face_encoding in face_encodings:
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        if len(distances) > 0:
            min_distance = np.min(distances)
            if min_distance < threshold:  # Configurable threshold for face recognition
                best_match_index = np.argmin(distances)
                names.append(known_names[best_match_index])
            else:
                names.append("Unknown")
        else:
            names.append("Unknown")

    return face_locations, names

# Main loop
frame_count = 0
while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    frame_count += 1
    if frame_count % process_frame_interval == 0:
        start_time = time.time()
        face_locations, face_names = process_frame(frame)
        end_time = time.time()

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            # Draw the name below the face
            cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        print(f"Frame processed in {end_time - start_time:.2f} seconds")

    # Display the resulting image
    cv2.imshow("Face Recognition", frame)

    # Break loop on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
