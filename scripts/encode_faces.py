import os
import sys
from concurrent.futures import ProcessPoolExecutor

import face_recognition
import numpy as np


_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".heic")


def _largest_face(face_locations):
    return max(face_locations, key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]))


def _encode_one(task):
    """Worker: load one image and return (person_name, encoding) or None."""
    person_name, image_path, model = task
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image, model=model)
    if not face_locations:
        return None
    if len(face_locations) > 1:
        face_locations = [_largest_face(face_locations)]
    encs = face_recognition.face_encodings(image, face_locations)
    if not encs:
        return None
    return person_name, encs[0]


def _collect_tasks(dataset_path, model):
    person_folders = sorted(
        f for f in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, f))
    )
    for person_name in person_folders:
        person_folder = os.path.join(dataset_path, person_name)
        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            if not os.path.isfile(image_path):
                continue
            if not image_name.lower().endswith(_IMAGE_EXTS):
                continue
            yield person_name, image_path, model


def encode_faces(dataset_path, output_file, model="cnn", workers=None):
    """
    Encode all faces in the dataset directory and save them as .npz.

    Args:
        dataset_path: Root directory containing one subfolder per person.
        output_file: Path to write the .npz encodings file.
        model: "hog" (fast, CPU) or "cnn" (accurate, needs CUDA dlib).
        workers: Parallel worker count. Defaults to 1 for "cnn" (to avoid
                 GPU/RAM contention) and cpu_count-1 for "hog".
    """
    if not os.path.isdir(dataset_path):
        print(f"Error: dataset path {dataset_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if workers is None:
        workers = 1 if model == "cnn" else max(1, (os.cpu_count() or 2) - 1)

    tasks = list(_collect_tasks(dataset_path, model))
    print(f"Encoding {len(tasks)} image(s) with {workers} worker(s), model={model}...")

    known_encodings = []
    known_names = []

    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, result in enumerate(ex.map(_encode_one, tasks, chunksize=4), 1):
            if result is not None:
                person_name, encoding = result
                known_encodings.append(encoding)
                known_names.append(person_name)
            if i % 50 == 0:
                print(f"  {i}/{len(tasks)} processed ({len(known_encodings)} encoded)")

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
