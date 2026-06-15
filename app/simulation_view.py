from PyQt6.QtWidgets import QVBoxLayout, QWidget
from visualization.renderer3d import Renderer3D

class SimulationView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.renderer = Renderer3D(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.renderer.widget)

    # 🎯 SOLUCIÓN VISUAL: Declaramos explícitamente todos los parámetros en su orden real
    def update_particles(self, electron_positions, neutral_positions=None, ion_positions=None, recombined_positions=None) -> None:
        """Envía las matrices de coordenadas directamente al renderizador 3D en orden estricto."""
        
        # Ahora el flujo de datos es directo, transparente y libre de errores de indexación
        self.renderer.update_particles(
            electron_positions, 
            neutral_positions, 
            ion_positions, 
            recombined_positions
        )

    def set_domain(self, xy_extent: float, gap_distance: float) -> None:
        self.renderer.set_domain(xy_extent, gap_distance)