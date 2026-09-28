from dataclasses import dataclass, field
from typing import Hashable


PERSON_CLASS = 0
VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck in COCO


@dataclass
class TrackState:
    previous_y: float
    last_seen_frame: int
    counted_directions: set[str] = field(default_factory=set)


class LineCounter:
    """Count tracked objects only when their center actually crosses a horizontal line."""

    def __init__(self, line_y: float, hysteresis: float = 4.0, max_idle_frames: int = 90):
        if line_y < 0:
            raise ValueError("line_y must be non-negative")
        self.line_y = float(line_y)
        self.hysteresis = max(0.0, float(hysteresis))
        self.max_idle_frames = max_idle_frames
        self.tracks: dict[Hashable, TrackState] = {}
        self.counts = {"person": 0, "vehicle": 0, "up": 0, "down": 0}

    def update(self, track_id: Hashable, class_id: int, center_y: float, frame_index: int) -> str | None:
        current_y = float(center_y)
        state = self.tracks.get(track_id)
        if state is None:
            self.tracks[track_id] = TrackState(current_y, frame_index)
            self._expire(frame_index)
            return None

        direction = None
        crossed_down = state.previous_y <= self.line_y - self.hysteresis and current_y >= self.line_y + self.hysteresis
        crossed_up = state.previous_y >= self.line_y + self.hysteresis and current_y <= self.line_y - self.hysteresis
        if crossed_down and "down" not in state.counted_directions:
            direction = "down"
        elif crossed_up and "up" not in state.counted_directions:
            direction = "up"

        if direction:
            state.counted_directions.add(direction)
            self.counts[direction] += 1
            if class_id == PERSON_CLASS:
                self.counts["person"] += 1
            elif class_id in VEHICLE_CLASSES:
                self.counts["vehicle"] += 1

        state.previous_y = current_y
        state.last_seen_frame = frame_index
        self._expire(frame_index)
        return direction

    def _expire(self, frame_index: int) -> None:
        expired = [
            track_id
            for track_id, state in self.tracks.items()
            if frame_index - state.last_seen_frame > self.max_idle_frames
        ]
        for track_id in expired:
            del self.tracks[track_id]
