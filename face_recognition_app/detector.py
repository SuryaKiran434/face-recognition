"""Object detection: find people and carried objects in a frame.

The capture loop talks to a small `Detector` interface so the backend is
swappable — YOLOv8n (ultralytics) on the Mac today, a lighter ONNX/Coral
backend on a Raspberry Pi later. The ultralytics/torch import is lazy so this
module (and the pure helpers below) can be imported without those heavy deps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

PERSON_LABEL = "person"

# COCO classes we care about: a person and a few "carryable" objects that hint
# at a delivery. Filtering at the model keeps inference output small.
CLASSES_OF_INTEREST = frozenset({"person", "backpack", "handbag", "suitcase"})


@dataclass(frozen=True)
class Detection:
    """A single detected object."""

    label: str
    confidence: float
    box: tuple[int, int, int, int]  # (x1, y1, x2, y2) in full-frame pixels


def count_people(detections):
    """Number of `person` detections."""
    return sum(1 for d in detections if d.label == PERSON_LABEL)


def carried_objects(detections):
    """Labels of detected non-person objects of interest (e.g. ["suitcase"])."""
    return [d.label for d in detections if d.label != PERSON_LABEL]


class NullDetector:
    """A detector that finds nothing — used when detection is disabled or the
    optional ultralytics dependency is unavailable."""

    def detect(self, frame_bgr):
        return []


class YoloDetector:
    """YOLOv8n object detector backed by ultralytics.

    The model weights (e.g. yolov8n.pt) are downloaded and cached by ultralytics
    on first use. Import is deferred to construction so importing this module
    never requires torch.
    """

    def __init__(self, weights="yolov8n.pt", confidence=0.4):
        from ultralytics import YOLO  # heavy (torch); imported lazily

        self._model = YOLO(weights)
        self._confidence = confidence
        self._names = self._model.names  # {class_id: name}
        # Restrict inference to the class ids we care about, when resolvable.
        self._classes = [
            cid for cid, name in self._names.items() if name in CLASSES_OF_INTEREST
        ] or None

    def detect(self, frame_bgr):
        results = self._model.predict(
            frame_bgr,
            conf=self._confidence,
            classes=self._classes,
            verbose=False,
        )
        detections = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = self._names.get(cls_id, str(cls_id))
                if label not in CLASSES_OF_INTEREST:
                    continue
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                detections.append(
                    Detection(label, float(box.conf[0]), (x1, y1, x2, y2))
                )
        return detections


def build_detector(cfg):
    """Build a detector from config, degrading gracefully.

    Returns a NullDetector when detection is disabled in config or when
    ultralytics/torch is not installed, so the app still runs (face-only).
    """
    if not cfg.detection.enabled:
        logger.info("Object detection disabled in config")
        return NullDetector()
    try:
        detector = YoloDetector(
            weights=cfg.detection.weights,
            confidence=cfg.detection.confidence,
        )
        logger.info("Loaded YOLO detector (%s)", cfg.detection.weights)
        return detector
    except ImportError:
        logger.warning(
            "ultralytics not installed; running without object detection. "
            "Install it (pip install -r requirements.txt) to enable "
            "person/package detection."
        )
        return NullDetector()
    except Exception:
        logger.exception("Failed to load YOLO detector; running face-only")
        return NullDetector()
