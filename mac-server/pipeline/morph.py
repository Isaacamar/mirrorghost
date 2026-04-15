import time
import random


class MorphState:
    """
    Tracks how far the face has converged toward the subject.
    weight=0.0: ControlNet has no influence — SD generates any random face.
    weight=1.0: ControlNet fully enforces the subject's face geometry.
    """

    def __init__(self):
        self.weight: float = 0.0
        self.target_weight: float = 0.0
        self.session_start: float = time.time()
        self.frame_count: int = 0

    def advance(self, increment: float = 0.018):
        """Nudge morph forward — called on timer or manual trigger."""
        self.target_weight = min(1.0, self.target_weight + increment)

    def get_weight(self) -> float:
        """Smoothly interpolate toward target. Tiny noise keeps it organic."""
        self.weight += (self.target_weight - self.weight) * 0.05
        noise = random.gauss(0, 0.003)
        return max(0.0, min(1.0, self.weight + noise))

    def reset(self):
        self.weight = 0.0
        self.target_weight = 0.0
        self.frame_count = 0
        self.session_start = time.time()
