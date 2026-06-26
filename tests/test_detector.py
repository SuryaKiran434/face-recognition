from face_recognition_app.detector import (
    Detection,
    NullDetector,
    carried_objects,
    count_people,
)


def _det(label):
    return Detection(label=label, confidence=0.9, box=(0, 0, 10, 10))


def test_count_people_counts_only_persons():
    dets = [_det("person"), _det("person"), _det("suitcase")]
    assert count_people(dets) == 2


def test_count_people_empty():
    assert count_people([]) == 0


def test_carried_objects_excludes_persons():
    dets = [_det("person"), _det("backpack"), _det("handbag")]
    assert carried_objects(dets) == ["backpack", "handbag"]


def test_null_detector_finds_nothing():
    assert NullDetector().detect(object()) == []
