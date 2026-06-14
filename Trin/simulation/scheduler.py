class SimulationScheduler:
    def __init__(self, dt, callback) -> None:
        self.dt = dt
        self.callback = callback
        self.running = False

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False

    def tick(self) -> None:
        if self.running:
            self.callback(self.dt)
