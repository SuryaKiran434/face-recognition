"""CLI entrypoint: real-time face recognition from webcam or network camera."""

import dataclasses
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from face_recognition_app.config import Config, coerce_camera_source
from face_recognition_app.logging_setup import configure as configure_logging
from face_recognition_app.recognizer import run_once, run_recognizer


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
@click.option("--no-email", "no_email", is_flag=True,
              help="Detect and save snapshots but do not send emails.")
@click.option("--headless", is_flag=True,
              help="Run without the preview window (e.g. for an always-on setup).")
@click.option("--once", "once_image", default=None,
              type=click.Path(dir_okay=False, exists=True),
              help="Process a single image end-to-end (no camera) and exit. "
                   "Useful for testing detection + email wiring.")
@click.option("-v", "--verbose", is_flag=True, help="Enable DEBUG logging.")
def main(config_path, camera_override, no_email, headless, once_image, verbose):
    configure_logging(verbose=verbose)
    cfg = Config.load(config_path)
    if camera_override is not None:
        cfg = dataclasses.replace(cfg, camera_source=coerce_camera_source(camera_override))
    try:
        if once_image:
            event = run_once(cfg, once_image, send_email=not no_email)
            click.echo(event.summary if event else "No person detected.")
        else:
            run_recognizer(cfg, send_email=not no_email, headless=headless)
    except (ValueError, RuntimeError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
