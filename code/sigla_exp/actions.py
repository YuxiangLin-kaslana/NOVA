from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np


ACTION_NAMES = (
    "wait",
    "alarm",
    "suppress",
    "inspect",
    "request_evidence",
    "recalibrate",
    "escalate",
)
ACTION_TO_ID = {name: idx for idx, name in enumerate(ACTION_NAMES)}


@dataclass(frozen=True)
class Event:
    onset: int
    end: int


def find_events(labels: Sequence[int]) -> List[Event]:
    labels_arr = np.asarray(labels).astype(bool)
    events: List[Event] = []
    in_event = False
    start = 0
    for idx, value in enumerate(labels_arr):
        if value and not in_event:
            start = idx
            in_event = True
        elif not value and in_event:
            events.append(Event(onset=start, end=idx - 1))
            in_event = False
    if in_event:
        events.append(Event(onset=start, end=len(labels_arr) - 1))
    return events


def _next_event(end_idx: int, events: Iterable[Event]) -> Event | None:
    future_events = [event for event in events if event.onset >= end_idx]
    if not future_events:
        return None
    return min(future_events, key=lambda event: event.onset)


def weak_action_label(
    end_idx: int,
    window_label: int,
    events: Sequence[Event],
    l_min: int,
    l_max: int,
) -> int:
    """Build a simple precursor-aware behavior-cloning target.

    Stable region -> wait.
    Valid precursor window [onset - l_max, onset - l_min] -> alarm.
    Late pre-onset and post-onset windows -> escalate.
    """
    if window_label:
        return ACTION_TO_ID["escalate"]

    event = _next_event(end_idx, events)
    if event is None:
        return ACTION_TO_ID["wait"]

    lead_time = event.onset - end_idx
    if l_min <= lead_time <= l_max:
        return ACTION_TO_ID["alarm"]
    if 0 <= lead_time < l_min:
        return ACTION_TO_ID["request_evidence"]
    return ACTION_TO_ID["wait"]


def weak_action_labels(
    end_indices: np.ndarray,
    window_labels: np.ndarray,
    point_labels: np.ndarray,
    l_min: int,
    l_max: int,
) -> np.ndarray:
    events = find_events(point_labels)
    return np.asarray(
        [
            weak_action_label(int(end_idx), int(label), events, l_min, l_max)
            for end_idx, label in zip(end_indices, window_labels)
        ],
        dtype=np.int64,
    )


def event_regions(length: int, labels: Sequence[int], l_min: int, l_max: int) -> np.ndarray:
    """Return per-time-step region ids for analysis.

    0 stable, 1 valid precursor, 2 late warning, 3 post/on-event.
    """
    regions = np.zeros(length, dtype=np.int64)
    for event in find_events(labels):
        precursor_start = max(0, event.onset - l_max)
        precursor_end = max(0, event.onset - l_min)
        late_start = max(0, event.onset - l_min + 1)
        regions[precursor_start : precursor_end + 1] = 1
        regions[late_start : event.onset] = 2
        regions[event.onset : event.end + 1] = 3
    return regions

