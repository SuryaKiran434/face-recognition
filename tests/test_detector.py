from face_recognition_app.detector import (
    Detection,
    NullDetector,
    carried_objects,
    count_people,
    near_person_boxes,
    person_height_ratio,
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


def test_person_height_ratio():
    # box (x1,y1,x2,y2) height 240 in a 480-tall frame -> 0.5
    assert person_height_ratio((0, 120, 50, 360), 480) == 0.5
    assert person_height_ratio((0, 0, 10, 10), 0) == 0.0


def test_near_person_boxes_filters_by_height():
    frame_h = 480
    near = (0, 0, 50, 300)     # height 300 -> ratio 0.625
    far = (0, 0, 50, 100)      # height 100 -> ratio ~0.21
    result = near_person_boxes([near, far], frame_h, min_ratio=0.5)
    assert result == [near]


def test_near_person_boxes_disabled_returns_all():
    boxes = [(0, 0, 10, 50), (0, 0, 10, 100)]
    assert near_person_boxes(boxes, 480, min_ratio=0.0) == boxes
