"""CLI entrypoint: real-time face recognition from webcam or network camera."""

import dataclasses
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from face_recognition_app.config import Config, coerce_camera_source
from face_recognition_app.logging_setup import configure as configure_logging
from face_recognition_app.recognizer import run_recognizer


@click.command()
@click.option("--config", "config_path", default=None,
              type=click.Path(dir_okay=False),
              help="Path to config.json. Defaults to repo-root config.json.")
@click.option("--camera", "camera_override", default=None,
              help=(
                  "Override camera_source from config. Accepts a device index "
                  '(0, 1, ...) or a URL (rtsp://..., http://...). '
                  'Examples: --camera 1 (second local camera); '
                  '--camera rtsp://user:pass@192.168.1.50:554/stream1 '
                  '(IP camera); --camera http://192.168.1.50:4747/video (DroidCam).'
              ))
@click.option("-v", "--verbose", is_flag=True, help="Enable DEBUG logging.")
def main(config_path, camera_override, verbose):
    configure_logging(verbose=verbose)
    cfg = Config.load(config_path)
    if camera_override is not None:
        cfg = dataclasses.replace(cfg, camera_source=coerce_camera_source(camera_override))
    try:
        run_recognizer(cfg)
    except (ValueError, RuntimeError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
