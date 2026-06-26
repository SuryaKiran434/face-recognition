from face_recognition_app.smoothing import LabelSmoother


def test_holds_stable_through_single_misread():
    s = LabelSmoother(window=5)
    for _ in range(3):
        assert s.update("Surya") == "Surya"
    # one stray Unknown should NOT flip the stable label
    assert s.update("Unknown") == "Surya"
    assert s.update("Surya") == "Surya"


def test_switches_after_sustained_change():
    s = LabelSmoother(window=5)
    for _ in range(3):
        s.update("Surya")
    # three Unknowns in the window of five -> majority -> switch
    s.update("Unknown")
    s.update("Unknown")
    assert s.update("Unknown") == "Unknown"


def test_returns_current_value_before_majority_forms():
    s = LabelSmoother(window=5)
    # first ever value: no stable yet, so it returns the current best guess
    assert s.update("Surya") == "Surya"


def test_reset_clears_history():
    s = LabelSmoother(window=3)
    for _ in range(3):
        s.update("Surya")
    s.reset()
    assert s.update("Unknown") == "Unknown"


def test_window_of_one_disables_smoothing():
    s = LabelSmoother(window=1)
    assert s.update("Surya") == "Surya"
    assert s.update("Unknown") == "Unknown"
    assert s.update("Surya") == "Surya"
