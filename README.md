# face-recognition

A door monitor for macOS/Linux. It watches a webcam or network camera, works out
**who is at the door** — a known person, a stranger, or a likely delivery — and
emails you a snapshot when someone actually turns up.

Face recognition is the core signal, but it is not the whole app. Each processed
frame also runs a YOLOv8 object detector (people and carried bags), applies a
proximity gate, smooths the recognised name over recent frames, and passes the
result through a debounce/cooldown gate so you get **one email per visit, not one
per frame**.

Alongside the live app there is offline tooling to build the face-encoding
database it matches against.

---

## Architecture

The capture loop lives in `face_recognition_app/recognizer.py`. Every
`process_frame_interval`-th frame goes through the full pipeline; frames in
between simply redraw the last result, which is what keeps the preview smooth.

```
                       camera_source (device index | rtsp:// | http://)
                                    │
                                    ▼
                          ┌──────────────────┐
                          │  cv2 frame read  │
                          └────────┬─────────┘
                                   │  every Nth frame (process_frame_interval)
                                   ▼
       ┌─────────────────────────────────────────────────────────┐
       │ process_frame()                                          │
       │   downscale by resize_factor  ──►  BGR to RGB            │
       │   face_locations(model=hog|cnn)                          │
       │   face_encodings()  ──►  128-D vectors                   │
       │   match_faces(..., known_sq)  ──►  name | "Unknown"      │
       │   (locations rescaled back to full-frame pixels)         │
       └───────────────────────┬─────────────────────────────────┘
                               │  faces = [(box, name), ...]
                               ▼
       ┌─────────────────────────────────────────────────────────┐
       │ detector.detect(frame)   — YOLOv8n via ultralytics       │
       │   classes: person, backpack, handbag, suitcase           │
       │   (NullDetector when disabled or ultralytics is absent)  │
       └───────────────────────┬─────────────────────────────────┘
                               │  detections
                               ▼
       ┌─────────────────────────────────────────────────────────┐
       │ LabelSmoother — majority vote over the last N frames     │
       │   stabilises the largest face's name (known <-> Unknown) │
       └───────────────────────┬─────────────────────────────────┘
                               ▼
       ┌─────────────────────────────────────────────────────────┐
       │ proximity gate: near_person_boxes(near_min_area_ratio)   │
       │ classify_people(): attach each face to the person box    │
       │   whose area contains its centre, plus carried objects   │
       │     known face                    -> "known"             │
       │     unknown + carrying a bag      -> "likely_delivery"   │
       │     unknown                       -> "unknown"           │
       │ aggregate_status() -> the on-screen banner               │
       └───────────────────────┬─────────────────────────────────┘
                               ▼
       ┌─────────────────────────────────────────────────────────┐
       │ EventGate.observe(present)                               │
       │   fires only after `debounce_frames` consecutive frames  │
       │   with somebody there, once per presence, then waits     │
       │   `cooldown_seconds` before the next visit can fire      │
       └───────────────────────┬─────────────────────────────────┘
                               │ True
                               ▼
       ┌─────────────────────────────────────────────────────────┐
       │ save_event(): full frame JPEG + one padded crop per      │
       │   person into snapshots/, one JSON line into events.jsonl│
       │ notify.send_event_email(): HTML mail over Gmail SMTP_SSL │
       │   with each person inlined as cid: image + status badge  │
       │   (only when the person's label is in notify.email_on)   │
       └─────────────────────────────────────────────────────────┘
```

Failures inside the event stage are logged, never raised — a broken SMTP
connection must not take the camera loop down. `purge_old()` runs once at
startup and deletes snapshots and log lines older than `retention_days`.

### Why the matcher looks the way it does

`match_faces` computes the whole (M, N) distance matrix from one BLAS matmul
using the identity `||a - b||^2 = ||a||^2 + ||b||^2 - 2·a·b`, instead of
materialising an (M, N, 128) broadcast. Compared to the old elementwise
version that is roughly **13x faster with about 85x lower peak memory** on a
realistic encoding set.

Two consequences are worth knowing:

- It compares **squared** distances against `threshold²`, which avoids M·N
  `sqrt` calls. A non-positive threshold can never match (squaring would
  otherwise flip the sign of a negative threshold).
- The known set's row norms are hoisted out of the loop. `squared_norms()` is
  called once in `run_recognizer` when the encodings load, and the result is
  threaded through `process_frame(..., known_sq=)` into `match_faces` as an
  optional keyword argument. Omit it and it is recomputed per call, so the
  function still works standalone.

An empty known-encoding set is not an error: `match_faces` returns `"Unknown"`
for every input face rather than raising. (`run_recognizer` still refuses to
start with an empty database — that is a configuration mistake, not a runtime
state.)

---

## Requirements

- **Python 3.10+**
- A webcam, or an RTSP/HTTP camera URL
- Build tooling for [dlib](http://dlib.net/) — it **compiles from source** on
  first install and can take several minutes:
  - macOS: Xcode command line tools (`xcode-select --install`)
  - Debian/Ubuntu: `cmake` and `build-essential`
- Optional: a Gmail account with an App Password, if you want email alerts

Pinned runtime dependencies live in [`requirements.txt`](requirements.txt);
[`requirements-dev.txt`](requirements-dev.txt) adds `pytest` on top.

---

## Running locally

### 1. Virtualenv and dependencies

`run.sh` does this for you, but the manual equivalent is:

```bash
python3 -m venv venv
venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install -r requirements.txt   # dlib compiles here
```

### 2. Build the encoding database

The recognizer matches against `*.npz` files in `encodings_dir`. Each file holds
an `encodings` array and a parallel `names` array. Start from a dataset with one
folder per person:

```text
dataset/
  alice/   img1.jpg img2.jpg ...
  bob/     img1.jpg img2.png ...
```

```bash
# (optional) sample, resize to <=800px, and normalize to JPEG first.
# --src is repeatable; .heic input is supported via pillow-heif.
venv/bin/python scripts/preprocessing.py --src raw_dataset --dst dataset --max-samples 50

# build encodings -> <encodings_dir>/face_encodings.npz by default
venv/bin/python scripts/encode_faces.py --dataset dataset --model hog
```

`encode_faces.py` encodes the largest face per image, in parallel:

| Flag | Meaning |
| --- | --- |
| `--dataset` | Required. Root directory, one subfolder per person. |
| `--output` | Output `.npz` path. Defaults to `<encodings_dir>/face_encodings.npz`. |
| `--model` | `hog` (fast, CPU) or `cnn`. **Defaults to `cnn`** — pass `--model hog` if you have no GPU. |
| `--workers` | Parallel workers. Default: `cpu-1` for `hog`, `1` for `cnn`. |
| `--config` | Alternate `config.json`, used only to resolve the default `--output`. |

### 3. Configure `config.json`

Runtime settings are read from [`config.json`](config.json) at the repo root.
Top-level keys:

| Key | Default | Meaning |
| --- | --- | --- |
| `encodings_dir` | *required* | Directory scanned for `*.npz` encoding files at startup. |
| `face_recognition_threshold` | *required* | Max Euclidean distance to count as a match (lower = stricter). |
| `resize_factor` | *required* | Frame downscale factor before detection (lower = faster, less accurate). |
| `process_frame_interval` | *required* | Process every Nth frame; intermediate frames reuse the last result. |
| `face_detection_model` | `hog` | `hog` (fast, CPU) or `cnn` (accurate, needs a GPU). |
| `camera_source` | `0` | Device index (`0`, `1`, …) or a URL (`rtsp://…`, `http://…`). |

`"detection"` — the YOLO stage:

| Key | Default | Meaning |
| --- | --- | --- |
| `enabled` | `true` | `false` swaps in a `NullDetector` and runs face-only. |
| `weights` | `yolov8n.pt` | Weights file; ultralytics downloads and caches it on first use. |
| `confidence` | `0.4` | Minimum detection confidence. |
| `near_min_area_ratio` | `0.0` | Proximity gate: a person counts only when their box covers at least this fraction of the frame area. `0` disables it. The repo's `config.json` uses `0.45`. |
| `label_smoothing_window` | `5` | Frames in the majority vote for the recognised name. `1` disables smoothing. |

`"events"` — capture and retention:

| Key | Default | Meaning |
| --- | --- | --- |
| `debounce_frames` | `3` | Consecutive present-frames before an event fires. |
| `cooldown_seconds` | `120.0` | Minimum gap between events. |
| `retention_days` | `14` | Snapshots and log lines older than this are purged at startup. |
| `snapshots_dir` | `snapshots` | Where JPEGs are written. |
| `log_path` | `events.jsonl` | Append-only JSONL event log. |

`"notify"` — email:

| Key | Default | Meaning |
| --- | --- | --- |
| `enabled` | `true` | Master switch for email. |
| `email_on` | `["known", "unknown", "likely_delivery"]` | Only send when at least one person carries one of these labels. |

`camera_source` accepts an integer, a numeric string, or a camera URL. A JSON
`true`/`false` is rejected explicitly, since `bool` is an `int` subclass in
Python and would otherwise be read as device 0 or 1.

### 4. Email credentials (optional)

Copy [`.env.example`](.env.example) to `.env` (gitignored) and fill in
`SENDER_EMAIL`, `SENDER_APP_PASSWORD` (a Google **App password**, not your
account password), and `RECIPIENT_EMAIL`. If this repo has no `.env`,
`notify.py` falls back to `~/IdeaProjects/BrewAutomation/.env`, so an app
password already configured there is reused rather than copied around. A real
environment variable overrides either file.

Running with `--no-email` skips sending entirely and still writes snapshots.

### 5. Run it

```bash
./run.sh
```

`run.sh` creates `./venv` on first run, installs the pinned dependencies (only
when `requirements.txt` is newer than the install stamp), then launches the
recognizer. A window titled **Face Recognition** opens with a box and name on
each face, outlines around carried objects, and a colour-coded status banner.
Press **`q`** in that window to quit.

Any extra arguments are forwarded verbatim to
`scripts/real_time_recognition.py`:

| Flag | Meaning |
| --- | --- |
| `--config PATH` | Use a different `config.json`. |
| `--camera SRC` | Override `camera_source` for one run — `--camera 1`, `--camera rtsp://user:pass@192.168.1.50:554/stream1`, `--camera http://192.168.1.50:4747/video`. |
| `--no-email` | Detect and save snapshots, but send nothing. |
| `--headless` | No preview window — for an always-on setup. |
| `--once IMAGE` | Run the whole pipeline against a single still image and exit. Also logs the largest person's frame-area ratio, which is how you calibrate `near_min_area_ratio`. |
| `-v` / `--verbose` | DEBUG logging. |

```bash
./run.sh --camera 1
./run.sh --headless --no-email
./run.sh --once snapshots/20260626-193207_frame.jpg --no-email -v
```

To build the venv with a different interpreter, set `PYTHON`:

```bash
PYTHON=/usr/local/bin/python3.12 ./run.sh
```

`run.sh` also filters one specific piece of noise: some pyenv interpreters built
without blake2 support make `hashlib` print an "unsupported hash type" traceback
on every import. It is harmless, and the filter drops only that block.

---

## Running the tests

```bash
venv/bin/python -m pip install -r requirements-dev.txt
venv/bin/python -m pytest
```

**71 tests**, none of which need a camera, a model download, or a GPU.

More usefully: **the test suite requires neither dlib nor ultralytics.** No test
imports `recognizer` (the only dlib consumer), and the `ultralytics` import is
deferred into `YoloDetector.__init__`, so neither heavy dependency is reachable
from a test run. That is why CI installs only three packages instead of
`requirements.txt`:

```yaml
pip install numpy opencv-python-headless pytest
```

`opencv-python-headless` stands in for `opencv-python`: identical `cv2` API,
without the `libGL` system dependency the GitHub runner lacks. Installing the
full requirements file would compile dlib from source and pull in torch for no
benefit.

| Test file | Covers |
| --- | --- |
| `test_config.py` | `config.json` loading, defaults, camera-source coercion |
| `test_matching.py` | `squared_norms`, `match_faces`, empty inputs, threshold edges |
| `test_detector.py` | Area ratios, proximity filtering, detector fallback |
| `test_decision.py` | `classify` precedence and `aggregate_status` |
| `test_classify_people.py` | Face-to-person association and carried-object attribution |
| `test_smoothing.py` | `LabelSmoother` majority vote and reset |
| `test_events.py` | `EventGate` debounce/cooldown, `save_event`, `purge_old` |
| `test_converter.py` | Legacy `.pkl` -> `.npz` migration |

CI runs on every push to `main` and every pull request. The **`Tests (Python)`**
check is a required status check on `main`.

---

## Migrating legacy `.pkl` encodings

Older versions stored encodings as pickle files. Unpickling is unsafe
(arbitrary code execution), so the recognizer now loads `.npz` with
`allow_pickle=False`. Convert any legacy files you wrote yourself:

```bash
venv/bin/python scripts/convert_pickle.py /path/to/FaceRecognitionData
```

This writes sibling `.npz` files and leaves the originals in place; files that
already have a `.npz` sibling, or whose pickle isn't the expected
`{"encodings": ..., "names": ...}` dict, are skipped. Verify the `.npz` files
load, then delete the `.pkl` files manually. If a `.pkl` might have been
tampered with, delete it and re-encode from the source images instead.

---

## Project layout

```
face_recognition_app/      importable library package
  config.py                  config.json loader; camera-source parsing
  recognizer.py              capture loop, per-frame pipeline, drawing, run_once
  detector.py                YOLOv8n object detection + proximity helpers
  decision.py                pure door-status logic (known/unknown/delivery)
  smoothing.py               LabelSmoother majority vote over recent frames
  events.py                  EventGate debounce, snapshot capture, JSONL log, purge
  notify.py                  Gmail SMTP_SSL sender with inline snapshot images
  matching.py                matmul distance matching (numpy only, no heavy deps)
  encoder.py                 build encodings from an image dataset
  preprocessor.py            sample/resize/normalize raw images
  converter.py               legacy .pkl -> .npz migration
  logging_setup.py           shared logging configuration
scripts/                   CLI entrypoints (thin click wrappers over the package)
  real_time_recognition.py
  encode_faces.py
  preprocessing.py
  convert_pickle.py
tests/                     pytest unit tests (71, no dlib/ultralytics needed)
config.json                runtime configuration
.env.example               email-credential template
requirements.txt           pinned runtime dependencies
requirements-dev.txt       requirements.txt + pytest
run.sh                     bootstrap deps and launch the app
```

Generated at runtime and gitignored: `venv/`, `snapshots/`, `events.jsonl`,
`.env`, `*.npz`, `*.pt`.

---

## Known limitations

- `likely_delivery` is a **heuristic**, not a classifier. COCO has no
  "package"/"cardboard box" class and cannot read uniforms, so the app infers a
  delivery from an unknown person carrying a backpack, handbag, or suitcase.
  Replacing it with a trained classifier touches only `decision.py`.
- The proximity gate uses bounding-box **area** rather than height. A distant
  person is tall but thin, so area separates "at the door" from "on the street"
  far more reliably. Calibrate it with `--once` against a real doorway photo.
- Encodings are biometric data. `.gitignore` excludes `*.npz`, `*.pkl`, `*.npy`,
  `snapshots/`, and `events.jsonl` for that reason — keep it that way.
