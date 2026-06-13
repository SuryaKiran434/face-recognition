"""CLI entrypoint: convert legacy .pkl encoding files to .npz.

The old encode_faces.py stored {"encodings": [...], "names": [...]} via pickle.
Loading pickle is unsafe (arbitrary code execution), so the recognizer now
reads .npz files with allow_pickle=False. Run this script once against any
directory that still contains .pkl files.

Usage:
    python scripts/convert_pickle.py /path/to/FaceRecognitionData

The original .pkl files are left in place. Verify the new .npz files load
correctly, then delete the .pkl files manually.
"""

import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from face_recognition_app.converter import convert_pickle_directory


@click.command(help=__doc__)
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
def main(directory):
    try:
        converted, skipped = convert_pickle_directory(directory)
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    click.echo(f"Converted: {converted}, skipped: {skipped}.")
    if converted:
        click.echo("Verify the new .npz files load, then delete the .pkl files.")


if __name__ == "__main__":
    main()
