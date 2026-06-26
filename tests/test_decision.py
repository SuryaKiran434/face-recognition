from face_recognition_app.decision import DoorStatus, classify


def test_no_person_no_faces_is_none():
    status = classify(face_names=[], person_count=0, carried_objects=[])
    assert status.label == "none"
    assert status.name is None


def test_known_face_wins():
    status = classify(face_names=["Surya"], person_count=1, carried_objects=[])
    assert status.label == "known"
    assert status.name == "Surya"
    assert status.confidence == 1.0


def test_known_face_wins_even_with_package_and_dwell():
    # A recognised person carrying a bag and lingering is still "known".
    status = classify(
        face_names=["Surya"],
        person_count=1,
        carried_objects=["suitcase"],
        dwell_seconds=30.0,
        dwell_threshold=8.0,
    )
    assert status.label == "known"
    assert status.name == "Surya"


def test_unknown_person_no_hints_is_unknown():
    status = classify(face_names=["Unknown"], person_count=1, carried_objects=[])
    assert status.label == "unknown"
    assert status.name is None


def test_person_detected_without_any_face_is_unknown():
    # YOLO sees a person but no face was resolved (turned away / too far).
    status = classify(face_names=[], person_count=1, carried_objects=[])
    assert status.label == "unknown"


def test_unknown_carrying_object_is_likely_delivery():
    status = classify(
        face_names=["Unknown"], person_count=1, carried_objects=["suitcase"]
    )
    assert status.label == "likely_delivery"
    assert any("carrying" in r for r in status.reasons)


def test_unknown_lingering_is_likely_delivery():
    status = classify(
        face_names=["Unknown"],
        person_count=1,
        carried_objects=[],
        dwell_seconds=10.0,
        dwell_threshold=8.0,
    )
    assert status.label == "likely_delivery"
    assert any("lingering" in r for r in status.reasons)


def test_dwell_below_threshold_stays_unknown():
    status = classify(
        face_names=["Unknown"],
        person_count=1,
        carried_objects=[],
        dwell_seconds=3.0,
        dwell_threshold=8.0,
    )
    assert status.label == "unknown"


def test_non_interest_objects_are_ignored_for_delivery():
    # A detected "dog" (not a carryable) must not trigger delivery.
    status = classify(
        face_names=["Unknown"], person_count=1, carried_objects=["dog"]
    )
    assert status.label == "unknown"


def test_multiple_known_faces_picks_first_and_notes_count():
    status = classify(
        face_names=["Surya", "Alice"], person_count=2, carried_objects=[]
    )
    assert status.label == "known"
    assert status.name == "Surya"
    assert any("2 known faces" in r for r in status.reasons)


def test_returns_frozen_dataclass():
    status = classify(face_names=[], person_count=0, carried_objects=[])
    assert isinstance(status, DoorStatus)
