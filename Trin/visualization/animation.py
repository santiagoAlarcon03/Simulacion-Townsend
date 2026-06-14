class AnimationController:
    def __init__(self) -> None:
        self.enabled = True
        self.speed = 1.0

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def set_speed(self, speed: float) -> None:
        self.speed = max(float(speed), 0.1)
