"""Temporal smoothing for flickery per-frame labels.

Face recognition can flip between a known name and "Unknown" frame-to-frame
when the match distance hovers near the threshold. LabelSmoother holds a stable
label and only switches when a new value wins a majority of the recent window,
so brief misreads don't change the displayed/emailed identity.
"""

from __future__ import annotations

from collections import Counter, deque


class LabelSmoother:
    """Majority-vote smoother over a sliding window of recent values.

    The stable label changes only when some value reaches a majority of the
    window (`window // 2 + 1` occurrences). Until a majority forms, the current
    most-common value is returned so there is always something to show.
    """

    def __init__(self, window=5):
        self._window = max(1, int(window))
        self._min_count = self._window // 2 + 1
        self._history = deque(maxlen=self._window)
        self._stable = None

    def update(self, value):
        self._history.append(value)
        most_common, count = Counter(self._history).most_common(1)[0]
        if count >= self._min_count:
            self._stable = most_common
        return self._stable if self._stable is not None else most_common

    def reset(self):
        self._history.clear()
        self._stable = None
