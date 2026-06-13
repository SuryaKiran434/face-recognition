import os
import sys

import face_recognition
import numpy as np


def _largest_face(face_locations):
    return max(face_locations, key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]))


def encode_faces(dataset_path, output_file, model="cnn"):
    """
    Encodes all faces in the dataset directory and saves them to a .npz file.

    Args:
        dataset_path (str): Path to the dataset.
        output_file (str): Path to save the encodings (.npz).
        model (str): "hog" (fast, CPU) or "cnn" (accurate, GPU recommended).
    """
    if not os.path.isdir(dataset_path):
        print(f"Error: dataset path {dataset_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    known_encodings = []
    known_names = []

    person_folders = sorted(
        f for f in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, f))
    )

    for person_name in person_folders:
        person_folder = os.path.join(dataset_path, person_name)

        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)

            if not os.path.isfile(image_path) or not image_name.lower().endswith(
                (".jpg", ".jpeg", ".png", ".heic")
            ):
                continue

            print(f"Processing {image_path}...")

            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image, model=model)
            if not face_locations:
                continue

            if len(face_locations) > 1:
                print(f"  Warning: {len(face_locations)} faces found, using largest.")
                face_locations = [_largest_face(face_locations)]

            face_encodings = face_recognition.face_encodings(image, face_locations)
            if face_encodings:
                known_encodings.append(face_encodings[0])
                known_names.append(person_name)

    if not known_encodings:
        print("Error: no faces encoded.", file=sys.stderr)
        sys.exit(1)

    np.savez(
        output_file,
        encodings=np.asarray(known_encodings),
        names=np.asarray(known_names),
    )
    print(f"Encodings saved to {output_file} ({len(known_encodings)} faces)")


if __name__ == "__main__":
    dataset_path = "/Users/suryakiran/Preprocessed_Faces"
    output_file = "/Users/suryakiran/FaceRecognitionData/face_encodings.npz"
    encode_faces(dataset_path, output_file)
