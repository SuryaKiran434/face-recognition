"""Pure decision logic: turn per-frame signals into a door status.

No heavy dependencies, no I/O — this is the unit-tested heart of the door
system. It combines face-recognition results (known/unknown) with object
detections (carried bags) and dwell time into a single labelled verdict.

Honest limitation: "likely_delivery" is a heuristic. COCO has no
"package/cardboard box" class and cannot read uniforms, so we infer delivery
from an unknown person who is either carrying a bag/suitcase or lingering at
the door. Replacing this with a trained classifier touches only this module.
"""

from __future__ import annotations

from dataclasses import dataclass

# COCO object classes we treat as "carrying something" — a weak delivery hint.
CARRIED_OBJECT_LABELS = frozenset({"backpack", "handbag", "suitcase"})

# The sentinel name produced by matching.match_faces for an unrecognised face.
UNKNOWN_NAME = "Unknown"


@dataclass(frozen=True)
class DoorStatus:
    """The verdict for a single frame (or a stabilised visit).

    label: one of "known", "unknown", "likely_delivery", "none".
    name: the recognised person's name when label == "known", else None.
    confidence: rough confidence of the driving signal (0..1, heuristic).
    reasons: human-readable explanations, useful for logs and email bodies.
    """

    label: str
    name: str | None
    confidence: float
    reasons: tuple[str, ...]


def classify(
    face_names,
    person_count,
    carried_objects,
    dwell_seconds=0.0,
    dwell_threshold=8.0,
):
    """Classify what is happening at the door this frame.

    Args:
        face_names: names from matching.match_faces (each a known name or
            "Unknown"); empty when no face was found.
        person_count: number of people from the object detector (0 when none /
            detection disabled).
        carried_objects: iterable of detected object labels (e.g. ["suitcase"]).
        dwell_seconds: how long a person has been continuously present.
        dwell_threshold: dwell (seconds) above which lingering counts as a
            delivery hint.

    Returns a DoorStatus. Precedence: a known face always wins (it's the
    reliable signal); otherwise an unknown person is escalated to
    "likely_delivery" when carrying something or lingering, else "unknown".
    """
    known_names = [n for n in face_names if n and n != UNKNOWN_NAME]
    carried = [obj for obj in carried_objects if obj in CARRIED_OBJECT_LABELS]
    person_present = person_count > 0 or len(face_names) > 0

    # 1. Nobody there.
    if not person_present:
        return DoorStatus("none", None, 0.0, ())

    # 2. A known face wins outright.
    if known_names:
        name = known_names[0]
        reasons = (f"recognised {name}",)
        if len(known_names) > 1:
            reasons += (f"{len(known_names)} known faces present",)
        return DoorStatus("known", name, 1.0, reasons)

    # 3. Unknown person — look for delivery hints.
    delivery_reasons = []
    if carried:
        delivery_reasons.append("carrying " + ", ".join(sorted(set(carried))))
    if dwell_seconds >= dwell_threshold:
        delivery_reasons.append(f"lingering {dwell_seconds:.0f}s at the door")

    if delivery_reasons:
        return DoorStatus(
            "likely_delivery",
            None,
            0.6,
            ("unknown face", *delivery_reasons),
        )

    # 4. Plain unknown person.
    return DoorStatus("unknown", None, 0.5, ("unknown person at the door",))


@dataclass(frozen=True)
class PersonResult:
    """One person detected in a frame, with their classification and the
    bounding box used to crop their image for the email."""

    label: str            # "known" | "unknown" | "likely_delivery"
    name: str | None
    box: tuple            # (x1, y1, x2, y2) in full-frame pixels
    reasons: tuple


def _face_box_to_xyxy(face_trbl):
    """Convert a face_recognition (top, right, bottom, left) box to (x1,y1,x2,y2)."""
    top, right, bottom, left = face_trbl
    return (left, top, right, bottom)


def _center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _contains(box, point):
    x1, y1, x2, y2 = box
    px, py = point
    return x1 <= px <= x2 and y1 <= py <= y2


def classify_people(
    person_boxes,
    faces,
    carried_boxes=(),
    dwell_seconds=0.0,
    dwell_threshold=8.0,
):
    """Classify every person in a frame for a multi-person email.

    Args:
        person_boxes: list of (x1,y1,x2,y2) person boxes from the detector.
        faces: list of ((top,right,bottom,left), name) from face recognition.
        carried_boxes: list of (x1,y1,x2,y2) boxes for carried objects.
        dwell_seconds / dwell_threshold: lingering hint for delivery.

    Returns a list of PersonResult — one per person, each with the box to crop.
    Faces are associated to the person box that contains their centre; a face
    matching no person box becomes its own entry (so a recognised face is never
    dropped). When no person boxes exist, each face is treated as a person.
    """
    faces_xyxy = [(_face_box_to_xyxy(box), name) for box, name in faces]
    results = []
    used_faces = set()

    for pbox in person_boxes:
        names_here = []
        for i, (fbox, name) in enumerate(faces_xyxy):
            if i in used_faces:
                continue
            if _contains(pbox, _center(fbox)):
                used_faces.add(i)
                names_here.append(name)
        carrying = any(_contains(pbox, _center(cb)) for cb in carried_boxes)
        results.append(_person_result(pbox, names_here, carrying,
                                       dwell_seconds, dwell_threshold))

    # Recognised/seen faces not inside any person box still get their own entry.
    for i, (fbox, name) in enumerate(faces_xyxy):
        if i not in used_faces:
            results.append(_person_result(fbox, [name], False,
                                           dwell_seconds, dwell_threshold))

    return results


def _person_result(box, names, carrying, dwell_seconds, dwell_threshold):
    box = tuple(int(v) for v in box)
    known = [n for n in names if n and n != UNKNOWN_NAME]
    if known:
        return PersonResult("known", known[0], box, (f"recognised {known[0]}",))

    reasons = []
    if carrying:
        reasons.append("carrying an object")
    if dwell_seconds >= dwell_threshold:
        reasons.append(f"lingering {dwell_seconds:.0f}s")
    if reasons:
        return PersonResult("likely_delivery", None, box, ("unknown face", *reasons))
    return PersonResult("unknown", None, box, ("unknown person",))
