"""CLI entrypoint: real-time face recognition from webcam."""

import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from face_recognition_app.config import Config
from face_recognition_app.recognizer import run_recognizer


@click.command()
@click.option("--config", "config_path", default=None,
              type=click.Path(dir_okay=False),
              help="Path to config.json. Defaults to repo-root config.json.")
def main(config_path):
    cfg = Config.load(config_path)

    def on_load(n):
        click.echo(f"Loaded {n} encodings.")

    def on_frame(elapsed):
        click.echo(f"Frame processed in {elapsed:.2f} seconds")

    try:
        run_recognizer(cfg, on_load=on_load, on_frame=on_frame)
    except (ValueError, RuntimeError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
