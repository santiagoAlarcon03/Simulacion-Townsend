from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QHBoxLayout, QMainWindow, QVBoxLayout, QWidget

from app.controls_panel import ControlsPanel
from app.plot_panel import PlotPanel
from app.simulation_view import SimulationView
from config import SimulationConfig
from simulation.data_logger import DataLogger
from simulation.engine import SimulationEngine


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Townsend Discharge 3D")

        self.config = SimulationConfig()
        self.engine = SimulationEngine(self.config)
        self.logger = DataLogger()

        self.controls = ControlsPanel()
        self.view = SimulationView()
        self.plot_panel = PlotPanel()

        self._running = False

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(self.view)
        right_layout.addWidget(self.plot_panel)

        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.addWidget(self.controls)
        root_layout.addWidget(right_panel, stretch=1)
        self.setCentralWidget(root)

        self.controls.start_requested.connect(self.start_simulation)
        self.controls.pause_toggled.connect(self.set_paused)
        self.controls.reset_requested.connect(self.reset_simulation)

        self.timer = QTimer(self)
        self.timer.setInterval(self.config.frame_dt_ms)
        self.timer.timeout.connect(self.on_tick)
        self.timer.start()

        self.reset_simulation()
        self.view.set_domain(self.config.xy_extent, self.config.gap_distance)

    def start_simulation(self) -> None:
        stage = self.controls.selected_stage()
        count = self.controls.particle_count()
        self.engine.reset(stage, count)
        self.logger.reset()
        self._running = True
        self.controls.set_pause_state(False)

    def set_paused(self, paused: bool) -> None:
        self._running = not paused

    def reset_simulation(self) -> None:
        stage = self.controls.selected_stage()
        count = self.controls.particle_count()
        self.engine.reset(stage, count)
        self.logger.reset()
        self._running = False
        self.controls.set_pause_state(True)
        if self.engine.state is not None:
            self.view.update_particles(self.engine.state.positions)
            self.controls.set_stage_readback(self.engine.state.stage)
        self.plot_panel.update_plot(self.logger.times, self.logger.counts)

    def on_tick(self) -> None:
        if not self._running:
            return
        state, metrics = self.engine.step(self.config.dt)
        if state is None or metrics is None:
            return
        self.logger.log(state.time, metrics["count"], metrics["current"])
        self.view.update_particles(state.positions)
        self.controls.set_stage_readback(state.stage)
        self.plot_panel.update_plot(self.logger.times, self.logger.counts)
