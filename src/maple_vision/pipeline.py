from .contracts import Frame, FrameSource, Observation, Perception, Status


class NoOpPerception:
    def observe(self, frame: Frame) -> Observation:
        return Observation(frame.sequence, frame.timestamp_ms, Status.UNKNOWN)


def run_once(source: FrameSource, perception: Perception) -> Observation | None:
    frame = source.read()
    if frame is None:
        return None
    return perception.observe(frame)
