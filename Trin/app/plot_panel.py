import numpy as np
from PyQt6.QtWidgets import QVBoxLayout, QWidget
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class PlotPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=(4, 3))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        
        # Configuramos el diseño estático una sola vez para ahorrar CPU
        self._setup_axes()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0) # Optimiza espacio en la UI
        layout.addWidget(self.canvas)

    def _setup_axes(self) -> None:
        """Configura el estilo estático del gráfico."""
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Particles")
        self.ax.grid(True, alpha=0.3)

    def update_plot(self, times, counts, ion_counts=None, *args) -> None:
        # 🛡️ VALIDACIÓN CRÍTICA: Si no hay datos, no hacemos nada para no romper Matplotlib
        if times is None or len(times) == 0 or len(counts) == 0:
            return

        # Convertimos a arrays de NumPy de forma segura para evitar desajustes de tamaño
        times = np.asarray(times)
        counts = np.asarray(counts)

        self.ax.clear()
        
        # 🔵 Gráfica de Electrones
        self.ax.plot(times, counts, color="tab:blue", alpha=0.6, label="Electrons")
        
        # 🟠 Suavizado de Electrones
        if len(counts) >= 5:
            window = 5
            kernel = np.ones(window) / window
            smooth = np.convolve(counts, kernel, mode="valid")
            smooth_times = times[window - 1 :]
            self.ax.plot(smooth_times, smooth, color="tab:orange", label="Electrons Avg")
            
        # 🔴 ¡RESTAURAMOS LOS IONES POSITIVOS AQUÍ!
        # De este modo la gráfica te mostrará en tiempo real cómo caen los iones al recombinarse
        if ion_counts is not None and len(ion_counts) == len(times):
            ion_counts = np.asarray(ion_counts)
            self.ax.plot(times, ion_counts, color="tab:red", alpha=0.7, linestyle="--", label="Positive Ions")

        # Reestablecemos las etiquetas tras el clear()
        self._setup_axes()
        self.ax.legend(loc="upper left")
        
        # Refresco asíncrono eficiente para evitar tirones en PyQt6
        self.canvas.draw_idle()