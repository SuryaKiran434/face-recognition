"""CLI entrypoint: preprocess raw face image datasets."""

import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from face_recognition_app.logging_setup import configure as configure_logging
from face_recognition_app.preprocessor import preprocess_datasets


@click.command()
@click.option("--src", "src_dirs", multiple=True, required=True,
              type=click.Path(exists=True, file_okay=False),
              help="Source dataset directory. Pass --src multiple times for multiple sources.")
@click.option("--dst", "dst_dir", required=True, type=click.Path(file_okay=False),
              help="Destination directory for preprocessed images.")
@click.option("--max-samples", default=50, show_default=True, type=int,
              help="Maximum number of images to sample per folder.")
@click.option("-v", "--verbose", is_flag=True, help="Enable DEBUG logging.")
def main(src_dirs, dst_dir, max_samples, verbose):
    configure_logging(verbose=verbose)
    preprocess_datasets(list(src_dirs), dst_dir, max_samples)


if __name__ == "__main__":
    main()
