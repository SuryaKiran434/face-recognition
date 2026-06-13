import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import click
import face_recognition
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from face_recognition_app.config import Config


_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".heic")


def _largest_face(face_locations):
    return max(face_locations, key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]))


def _encode_one(task):
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
    if not os.path.isdir(dataset_path):
        click.echo(f"Error: dataset path {dataset_path} does not exist.", err=True)
        sys.exit(1)

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if workers is None:
        workers = 1 if model == "cnn" else max(1, (os.cpu_count() or 2) - 1)

    tasks = list(_collect_tasks(dataset_path, model))
    click.echo(f"Encoding {len(tasks)} image(s) with {workers} worker(s), model={model}...")

    known_encodings = []
    known_names = []

    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, result in enumerate(ex.map(_encode_one, tasks, chunksize=4), 1):
            if result is not None:
                person_name, encoding = result
                known_encodings.append(encoding)
                known_names.append(person_name)
            if i % 50 == 0:
                click.echo(f"  {i}/{len(tasks)} processed ({len(known_encodings)} encoded)")

    if not known_encodings:
        click.echo("Error: no faces encoded.", err=True)
        sys.exit(1)

    np.savez(
        output_file,
        encodings=np.asarray(known_encodings),
        names=np.asarray(known_names),
    )
    click.echo(f"Encodings saved to {output_file} ({len(known_encodings)} faces)")


@click.command()
@click.option("--dataset", required=True,
              type=click.Path(exists=True, file_okay=False, dir_okay=True),
              help="Root directory with one subfolder per person.")
@click.option("--output", "output_file", type=click.Path(dir_okay=False),
              help="Output .npz path. Defaults to <encodings_dir>/face_encodings.npz.")
@click.option("--model", type=click.Choice(["hog", "cnn"]), default="cnn",
              show_default=True,
              help='Detection model. "hog" is fast on CPU; "cnn" needs GPU.')
@click.option("--workers", type=int, default=None,
              help="Parallel worker count. Default: cpu-1 (hog) or 1 (cnn).")
@click.option("--config", "config_path", default=None,
              type=click.Path(dir_okay=False),
              help="Path to config.json (used only to default --output).")
def main(dataset, output_file, model, workers, config_path):
    if output_file is None:
        cfg = Config.load(config_path)
        output_file = os.path.join(cfg.encodings_dir, "face_encodings.npz")
    encode_faces(dataset, output_file, model=model, workers=workers)


if __name__ == "__main__":
    main()
