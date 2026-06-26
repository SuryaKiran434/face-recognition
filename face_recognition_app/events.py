"""Detection-event lifecycle: turn a per-frame person stream into discrete
events, capture snapshots (full frame + one crop per person), append a
structured log, and purge old data.

The firing logic (`EventGate`) is pure and clock-injected so it is unit
testable without a camera. The I/O helpers (`save_event`, `purge_old`) take
explicit paths and timestamps for the same reason.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass

import cv2

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DoorEvent:
    """One emitted detection event."""

    timestamp: str            # ISO-8601, local time
    summary: str              # e.g. "2 people: Surya (known), Unknown (likely delivery)"
    frame_path: str           # full annotated frame
    person_paths: tuple       # one crop per person, parallel to `people`
    people: tuple             # tuple of dicts: {label, name, reasons}


class EventGate:
    """Decides *when* a detection event fires.

    A person must be present for `debounce_frames` consecutive observations
    before an event fires — once per continuous presence. After the person
    leaves (an absence) and `cooldown_seconds` has elapsed, the next arrival can
    fire again. This yields "one email per visit, not per frame".
    """

    def __init__(self, debounce_frames=3, cooldown_seconds=120.0, clock=time.time):
        self._debounce = max(1, int(debounce_frames))
        self._cooldown = float(cooldown_seconds)
        self._clock = clock
        self._present_streak = 0
        self._fired_this_presence = False
        self._last_fire = None
        self._present_since = None

    def observe(self, present):
        """Feed one observation. Returns True exactly on the frame an event
        should fire."""
        now = self._clock()
        if not present:
            self._present_streak = 0
            self._fired_this_presence = False
            self._present_since = None
            return False

        self._present_streak += 1
        if self._present_since is None:
            self._present_since = now

        if self._fired_this_presence or self._present_streak < self._debounce:
            return False

        if self._last_fire is not None and (now - self._last_fire) < self._cooldown:
            # Still within cooldown from the previous event; don't re-fire, but
            # mark this presence as handled so we wait for a fresh arrival.
            self._fired_this_presence = True
            return False

        self._fired_this_presence = True
        self._last_fire = now
        return True

    def dwell_seconds(self):
        """How long the current presence has lasted (0 when nobody is here)."""
        if self._present_since is None:
            return 0.0
        return self._clock() - self._present_since


def _crop(frame, box, pad_ratio=0.15):
    """Crop `box` (x1,y1,x2,y2) from frame with padding, clamped to bounds."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    pw = int((x2 - x1) * pad_ratio)
    ph = int((y2 - y1) * pad_ratio)
    x1 = max(0, x1 - pw)
    y1 = max(0, y1 - ph)
    x2 = min(w, x2 + pw)
    y2 = min(h, y2 + ph)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def _label_text(person):
    if person.label == "known":
        return f"{person.name} (known)"
    if person.label == "likely_delivery":
        return "Unknown (likely delivery)"
    return "Unknown"


def save_event(frame, people, snapshots_dir, log_path, timestamp):
    """Write the full frame + one crop per person, append a JSONL log line, and
    return a DoorEvent. `timestamp` is an ISO string supplied by the caller."""
    os.makedirs(snapshots_dir, exist_ok=True)
    stamp = timestamp.replace(":", "").replace("-", "").replace("T", "-")[:15]

    frame_name = f"{stamp}_frame.jpg"
    frame_path = os.path.join(snapshots_dir, frame_name)
    cv2.imwrite(frame_path, frame)

    person_paths = []
    people_meta = []
    for idx, person in enumerate(people):
        crop = _crop(frame, person.box)
        if crop is None or crop.size == 0:
            path = ""
        else:
            path = os.path.join(snapshots_dir, f"{stamp}_person{idx + 1}_{person.label}.jpg")
            cv2.imwrite(path, crop)
        person_paths.append(path)
        people_meta.append({
            "label": person.label,
            "name": person.name,
            "reasons": list(person.reasons),
            "image": path,
        })

    summary = _summarize(people)
    record = {
        "ts": timestamp,
        "summary": summary,
        "frame": frame_path,
        "people": people_meta,
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")

    logger.info("Event saved: %s", summary)
    return DoorEvent(
        timestamp=timestamp,
        summary=summary,
        frame_path=frame_path,
        person_paths=tuple(p for p in person_paths if p),
        people=tuple(people_meta),
    )


def _summarize(people):
    if not people:
        return "person detected"
    n = len(people)
    head = f"{n} person" if n == 1 else f"{n} people"
    return head + ": " + ", ".join(_label_text(p) for p in people)


def purge_old(snapshots_dir, log_path, retention_days, now=None):
    """Delete snapshots and trim log lines older than retention_days."""
    now = now if now is not None else time.time()
    cutoff = now - retention_days * 86400

    removed = 0
    if os.path.isdir(snapshots_dir):
        for name in os.listdir(snapshots_dir):
            path = os.path.join(snapshots_dir, name)
            try:
                if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                    os.remove(path)
                    removed += 1
            except OSError:
                logger.debug("Could not stat/remove %s", path)

    if os.path.isfile(log_path):
        try:
            cutoff_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(cutoff))
            with open(log_path) as f:
                lines = f.readlines()
            kept = [ln for ln in lines if _line_ts(ln) >= cutoff_iso]
            if len(kept) != len(lines):
                with open(log_path, "w") as f:
                    f.writelines(kept)
        except OSError:
            logger.debug("Could not trim log %s", log_path)

    if removed:
        logger.info("Retention: removed %d old snapshot(s)", removed)
    return removed


def _line_ts(line):
    try:
        return json.loads(line).get("ts", "")
    except (ValueError, TypeError):
        return ""
