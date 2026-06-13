"""CLI entrypoint: real-time face recognition from webcam."""

import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from face_recognition_app.config import Config
from face_recognition_app.logging_setup import configure as configure_logging
from face_recognition_app.recognizer import run_recognizer


@click.command()
@click.option("--config", "config_path", default=None,
              type=click.Path(dir_okay=False),
              help="Path to config.json. Defaults to repo-root config.json.")
@click.option("-v", "--verbose", is_flag=True, help="Enable DEBUG logging.")
def main(config_path, verbose):
    configure_logging(verbose=verbose)
    cfg = Config.load(config_path)
    try:
        run_recognizer(cfg)
    except (ValueError, RuntimeError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
