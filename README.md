# face-recognition

Real-time face recognition from a webcam or network camera, plus the offline
tooling to build the face-encoding database it matches against.

The app captures video, detects faces in each frame, computes 128-dimension
encodings, and labels each face with the closest known person (or `Unknown`)
using a Euclidean-distance threshold.

## Requirements

- Python 3.10+
- A webcam (or an RTSP/HTTP camera URL)
- Build tooling for [dlib](http://dlib.net/) — `dlib` compiles from source on
  first install, which can take several minutes. On macOS this needs the Xcode
  command line tools (`xcode-select --install`); on Debian/Ubuntu, `cmake` and
  `build-essential`.

Pinned dependencies live in [`requirements.txt`](requirements.txt).

## Quick start

```bash
./run.sh
```

`run.sh` creates a local virtualenv at `./venv`, installs the pinned
dependencies (only when `requirements.txt` changes), then launches real-time
recognition. A window titled **Face Recognition** opens showing the camera feed
with a box and name label on each detected face. Press **`q`** in that window to
quit.

Any extra arguments are forwarded to the recognizer:

```bash
./run.sh --camera 1       # use the second local camera
./run.sh --verbose        # DEBUG logging
```

To use a different interpreter for the venv, set `PYTHON`:

```bash
PYTHON=/usr/local/bin/python3.12 ./run.sh
```

## Configuration

Runtime settings are read from [`config.json`](config.json) at the repo root:

| Key | Meaning |
| --- | --- |
| `encodings_dir` | Directory scanned for `*.npz` encoding files at startup. |
| `face_recognition_threshold` | Max Euclidean distance to count as a match (lower = stricter). |
| `resize_factor` | Frame downscale factor before detection (lower = faster, less accurate). |
| `process_frame_interval` | Process every Nth frame; intermediate frames reuse the last result. |
| `face_detection_model` | `hog` (fast, CPU) or `cnn` (accurate, needs a GPU). |
| `camera_source` | Device index (`0`, `1`, …) or a URL (`rtsp://…`, `http://…`). |

`camera_source` accepts an integer, a numeric string, or a camera URL; the
`--camera` flag overrides it for a single run, e.g.
`--camera rtsp://user:pass@192.168.1.50:554/stream1`.

## Building the encoding database

The recognizer matches against `*.npz` files in `encodings_dir`. Each file holds
an `encodings` array and a parallel `names` array. Generate one from a dataset of
labelled images:

```text
dataset/
  alice/   img1.jpg img2.jpg ...
  bob/     img1.jpg img2.png ...
```

```bash
# (optional) sample, resize, and normalize raw images to JPEG first
venv/bin/python scripts/preprocessing.py --src raw_dataset --dst dataset --max-samples 50

# build encodings (defaults to <encodings_dir>/face_encodings.npz)
venv/bin/python scripts/encode_faces.py --dataset dataset --model hog
```

`encode_faces.py` encodes the largest face per image in parallel. Use
`--model cnn` for higher accuracy on a GPU, `--output` to choose the `.npz`
path, and `--workers` to set the parallel worker count.

### Migrating legacy `.pkl` encodings

Older versions stored encodings as pickle files. Loading pickle is unsafe
(arbitrary code execution), so the recognizer now reads `.npz` with
`allow_pickle=False`. Convert any legacy files you trust:

```bash
venv/bin/python scripts/convert_pickle.py /path/to/FaceRecognitionData
```

This writes sibling `.npz` files and leaves the originals in place; verify the
`.npz` files load, then delete the `.pkl` files manually.

## Project layout

```
face_recognition_app/      importable library package
  config.py                  config.json loader and camera-source parsing
  recognizer.py              webcam capture loop and per-frame recognition
  encoder.py                 build encodings from an image dataset
  preprocessor.py            sample/resize/normalize raw images
  matching.py                pure numpy distance matching (no heavy deps)
  converter.py               legacy .pkl -> .npz migration
  logging_setup.py           shared logging configuration
scripts/                   CLI entrypoints (thin wrappers over the package)
  real_time_recognition.py
  encode_faces.py
  preprocessing.py
  convert_pickle.py
tests/                     pytest unit tests for config, matching, converter
config.json                runtime configuration
requirements.txt           pinned dependencies
run.sh                     bootstrap deps and launch the app
```

## Running the tests

The unit tests cover the pure logic (config loading, distance matching, and the
pickle migration) and need no camera or model downloads:

```bash
venv/bin/python -m pip install pytest
venv/bin/python -m pytest
```
