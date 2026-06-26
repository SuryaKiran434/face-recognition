from face_recognition_app.detector import (
    Detection,
    NullDetector,
    carried_objects,
    count_people,
    near_person_boxes,
    person_area_ratio,
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


def test_person_area_ratio():
    # box 320x240 in a 640x480 frame -> (320*240)/(640*480) = 0.25
    assert person_area_ratio((0, 0, 320, 240), 640, 480) == 0.25
    assert person_area_ratio((0, 0, 10, 10), 0, 0) == 0.0


def test_near_person_boxes_filters_by_area():
    w, h = 640, 480
    near = (0, 0, 400, 360)    # area ratio = (400*360)/(640*480) ~ 0.47
    far = (0, 0, 80, 300)      # tall but thin -> (80*300)/307200 ~ 0.078
    result = near_person_boxes([near, far], w, h, min_ratio=0.3)
    assert result == [near]


def test_near_person_boxes_disabled_returns_all():
    boxes = [(0, 0, 10, 50), (0, 0, 10, 100)]
    assert near_person_boxes(boxes, 640, 480, min_ratio=0.0) == boxes
