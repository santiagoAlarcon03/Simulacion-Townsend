class DataLogger:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.times = []
        self.counts = []
        self.currents = []

    def log(self, time_s: float, count: int, current_a: float) -> None:
        self.times.append(float(time_s))
        self.counts.append(int(count))
        self.currents.append(float(current_a))
