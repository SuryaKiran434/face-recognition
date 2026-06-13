"""CLI entrypoint: build face encodings from a per-person image dataset."""

import os
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from face_recognition_app.config import Config
from face_recognition_app.encoder import encode_faces


@click.command()
@click.option("--dataset", required=True,
              type=click.Path(exists=True, file_okay=False),
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

    def on_progress(i, total, encoded):
        if i % 50 == 0:
            click.echo(f"  {i}/{total} processed ({encoded} encoded)")

    try:
        n = encode_faces(dataset, output_file, model=model, workers=workers,
                         on_progress=on_progress)
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    click.echo(f"Encodings saved to {output_file} ({n} faces)")


if __name__ == "__main__":
    main()
