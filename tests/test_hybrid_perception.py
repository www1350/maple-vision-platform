from maple_vision.agent import HybridPerception
from maple_vision.contracts import Detection, Status
from maple_vision.scene import SceneRenderer
from maple_vision.simulation import Simulator


class Detector:
    def __init__(self, detections):
        self.detections = detections

    def detect(self, frame):
        return self.detections


def test_dynamic_objects_come_only_from_model_and_terrain_from_pixels():
    frame = SceneRenderer().render(Simulator().snapshot())
    model_player = Detection("player", (25, 146, 39, 168), 0.9)
    observation = HybridPerception(Detector((model_player,))).observe(frame)
    assert observation.status is Status.KNOWN
    assert [d for d in observation.detections if d.class_id == "player"] == [model_player]
    assert not [d for d in observation.detections if d.class_id == "monster"]
    assert any(d.class_id == "platform" for d in observation.detections)
    assert observation.source_id == "onnx-objects+fixture-terrain-hud-v1"


def test_missing_yolo_player_is_unknown_even_when_palette_can_see_it():
    frame = SceneRenderer().render(Simulator().snapshot())
    observation = HybridPerception(Detector(())).observe(frame)
    assert observation.status is Status.UNKNOWN
    assert observation.confidence == 0
