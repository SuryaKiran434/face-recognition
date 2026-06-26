import json
import os
import time

import numpy as np

from face_recognition_app.decision import PersonResult
from face_recognition_app.events import EventGate, purge_old, save_event


class FakeClock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


def test_gate_fires_once_after_debounce():
    clock = FakeClock()
    gate = EventGate(debounce_frames=3, cooldown_seconds=100, clock=clock)
    assert gate.observe(True) is False   # 1
    assert gate.observe(True) is False   # 2
    assert gate.observe(True) is True    # 3 -> fire
    assert gate.observe(True) is False   # already fired this presence
    assert gate.observe(True) is False


def test_gate_does_not_fire_without_presence():
    gate = EventGate(debounce_frames=2, clock=FakeClock())
    assert gate.observe(False) is False
    assert gate.observe(True) is False
    assert gate.observe(False) is False  # streak reset
    assert gate.observe(True) is False   # streak back to 1


def test_gate_refires_after_absence_and_cooldown():
    clock = FakeClock()
    gate = EventGate(debounce_frames=1, cooldown_seconds=60, clock=clock)
    assert gate.observe(True) is True    # first event at t=0
    gate.observe(False)                  # person leaves -> presence reset
    clock.t = 61                         # cooldown elapsed
    assert gate.observe(True) is True    # new visit fires


def test_gate_suppresses_within_cooldown():
    clock = FakeClock()
    gate = EventGate(debounce_frames=1, cooldown_seconds=60, clock=clock)
    assert gate.observe(True) is True    # fire at t=0
    gate.observe(False)
    clock.t = 30                         # still in cooldown
    assert gate.observe(True) is False


def test_gate_dwell_tracks_presence():
    clock = FakeClock()
    gate = EventGate(debounce_frames=5, clock=clock)
    gate.observe(True)
    clock.t = 7
    assert gate.dwell_seconds() == 7
    gate.observe(False)
    assert gate.dwell_seconds() == 0.0


def _blank(w=200, h=200):
    return np.full((h, w, 3), 255, dtype=np.uint8)


def test_save_event_writes_frame_crops_and_log(tmp_path):
    snaps = tmp_path / "snaps"
    log = tmp_path / "events.jsonl"
    people = [
        PersonResult("known", "Surya", (10, 10, 80, 120), ("recognised Surya",)),
        PersonResult("likely_delivery", None, (100, 20, 180, 150),
                     ("unknown face", "carrying an object")),
    ]
    event = save_event(_blank(), people, str(snaps), str(log), "2026-06-26T18:00:00")

    assert os.path.isfile(event.frame_path)
    assert len(event.person_paths) == 2
    for p in event.person_paths:
        assert os.path.isfile(p)
    assert "2 people" in event.summary

    record = json.loads(log.read_text().strip())
    assert record["summary"] == event.summary
    assert len(record["people"]) == 2
    assert record["people"][0]["name"] == "Surya"


def test_purge_old_removes_aged_snapshots_and_log_lines(tmp_path):
    snaps = tmp_path / "snaps"
    snaps.mkdir()
    old = snaps / "old.jpg"
    old.write_bytes(b"x")
    os.utime(old, (0, 0))  # epoch -> very old
    new = snaps / "new.jpg"
    new.write_bytes(b"y")

    log = tmp_path / "events.jsonl"
    log.write_text(
        json.dumps({"ts": "2000-01-01T00:00:00", "summary": "old"}) + "\n"
        + json.dumps({"ts": "2999-01-01T00:00:00", "summary": "new"}) + "\n"
    )

    now = time.time()  # cutoff = now - 14d sits between year 2000 and 2999
    removed = purge_old(str(snaps), str(log), retention_days=14, now=now)

    assert removed == 1
    assert not old.exists()
    assert new.exists()
    kept = [json.loads(ln) for ln in log.read_text().splitlines()]
    assert len(kept) == 1
    assert kept[0]["summary"] == "new"
