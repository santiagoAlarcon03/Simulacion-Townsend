from PyQt6.QtWidgets import QVBoxLayout, QWidget

from visualization.renderer3d import Renderer3D


class SimulationView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.renderer = Renderer3D(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.renderer.widget)

    def update_particles(self, electron_positions, neutral_positions=None) -> None:
        self.renderer.update_particles(electron_positions, neutral_positions)

    def set_domain(self, xy_extent: float, gap_distance: float) -> None:
        self.renderer.set_domain(xy_extent, gap_distance)
