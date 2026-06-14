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
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Particles")
        self.ax.grid(True, alpha=0.3)

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

    def update_plot(self, times, counts, ion_counts=None, *args) -> None:
        self.ax.clear()
        
        # 🔵 Gráfica de Electrones (Original)
        self.ax.plot(times, counts, color="tab:blue", alpha=0.6, label="Electrons")
        
        # 🟠 Suavizado de Electrones (Original)
        if len(counts) >= 5:
            window = 5
            kernel = np.ones(window) / window
            smooth = np.convolve(counts, kernel, mode="valid")
            smooth_times = times[window - 1 :]
            self.ax.plot(smooth_times, smooth, color="tab:orange", label="Electrons Avg")
            
        # ❌ ELIMINAMOS LA LÍNEA DE IONES DE AQUÍ PARA QUE NO SE MUESTRE

        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Particles")
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc="upper left")
        self.canvas.draw_idle()