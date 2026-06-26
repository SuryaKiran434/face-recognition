from face_recognition_app.decision import classify_people


# Faces are (top, right, bottom, left); person boxes are (x1, y1, x2, y2).

def test_no_people_no_faces_returns_empty():
    assert classify_people([], []) == []


def test_known_face_inside_person_box_is_known():
    person = (0, 0, 100, 200)          # x1,y1,x2,y2
    face = (10, 60, 50, 20)            # top,right,bottom,left -> centre (40,30)
    people = classify_people([person], [(face, "Surya")])
    assert len(people) == 1
    assert people[0].label == "known"
    assert people[0].name == "Surya"
    assert people[0].box == (0, 0, 100, 200)


def test_unknown_face_in_person_box_is_unknown():
    person = (0, 0, 100, 200)
    face = (10, 60, 50, 20)
    people = classify_people([person], [(face, "Unknown")])
    assert people[0].label == "unknown"


def test_unknown_person_carrying_object_is_likely_delivery():
    person = (0, 0, 100, 200)
    face = (10, 60, 50, 20)
    carried = (30, 100, 70, 180)      # centre (50,140) inside person box
    people = classify_people([person], [(face, "Unknown")], carried_boxes=[carried])
    assert people[0].label == "likely_delivery"


def test_multiple_people_each_get_their_own_result():
    p1 = (0, 0, 100, 200)
    p2 = (200, 0, 320, 200)
    f1 = (10, 60, 50, 20)              # centre (40,30) -> in p1
    f2 = (10, 300, 50, 240)           # centre (270,30) -> in p2
    people = classify_people([p1, p2], [(f1, "Surya"), (f2, "Unknown")])
    assert len(people) == 2
    labels = {p.label for p in people}
    assert labels == {"known", "unknown"}


def test_face_outside_any_person_box_becomes_its_own_entry():
    person = (0, 0, 50, 50)
    far_face = (300, 360, 340, 320)   # centre (340,320) -> outside person box
    people = classify_people([person], [(far_face, "Surya")])
    # one for the empty person box (unknown) + one for the orphan known face
    assert len(people) == 2
    assert any(p.label == "known" and p.name == "Surya" for p in people)


def test_no_person_boxes_falls_back_to_faces():
    face = (10, 60, 50, 20)
    people = classify_people([], [(face, "Surya")])
    assert len(people) == 1
    assert people[0].label == "known"
