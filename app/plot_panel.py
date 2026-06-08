import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class PlotPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # =====================================================
        # CONTADORES
        # =====================================================

        self.electron_label = QLabel("⚡ Electrons: 0")
        self.ion_label = QLabel("🟠 Positive Ions: 0")

        self.electron_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.ion_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.electron_label.setStyleSheet("""
            QLabel {
                color: #FFD700;
                font-size: 15px;
                font-weight: bold;
                background-color: rgba(255,215,0,25);
                border-radius: 8px;
                padding: 6px;
            }
        """)

        self.ion_label.setStyleSheet("""
            QLabel {
                color: #FF6A00;
                font-size: 15px;
                font-weight: bold;
                background-color: rgba(255,106,0,25);
                border-radius: 8px;
                padding: 6px;
            }
        """)

        # =====================================================
        # FIGURA
        # =====================================================

        self.figure = Figure(
            figsize=(4, 3),
            tight_layout=True,
        )

        self.canvas = FigureCanvas(self.figure)

        self.ax = self.figure.add_subplot(111)

        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Particles")
        self.ax.grid(True, alpha=0.3)

        # =====================================================
        # LAYOUT DE CONTADORES
        # =====================================================

        counters_layout = QHBoxLayout()

        counters_layout.addWidget(
            self.electron_label,
            stretch=1,
        )

        counters_layout.addWidget(
            self.ion_label,
            stretch=1,
        )

        # =====================================================
        # LAYOUT PRINCIPAL
        # =====================================================

        layout = QVBoxLayout(self)

        layout.addLayout(counters_layout)

        layout.addWidget(
            self.canvas,
            stretch=1,
        )

    def update_plot(
        self,
        times,
        counts,
        ion_counts,
    ) -> None:

        # =====================================================
        # ACTUALIZAR CONTADORES
        # =====================================================

        electron_count = (
            counts[-1]
            if len(counts) > 0
            else 0
        )

        ion_count = (
            ion_counts[-1]
            if len(ion_counts) > 0
            else 0
        )

        self.electron_label.setText(
            f"⚡ Electrons: {electron_count:,}"
        )

        self.ion_label.setText(
            f"🟠 Positive Ions: {ion_count:,}"
        )

        # =====================================================
        # LIMPIAR GRÁFICA
        # =====================================================

        self.ax.clear()

        # =====================================================
        # ELECTRONES
        # =====================================================

        self.ax.plot(
            times,
            counts,
            color="tab:blue",
            linewidth=2,
            label="Electrons",
        )

        # =====================================================
        # IONES POSITIVOS
        # =====================================================

        self.ax.plot(
            times,
            ion_counts,
            color="tab:red",
            linewidth=2,
            label="Positive Ions",
        )

        # =====================================================
        # SUAVIZADO
        # =====================================================

        if len(counts) >= 5:

            window = 5

            kernel = np.ones(window) / window

            smooth = np.convolve(
                counts,
                kernel,
                mode="valid",
            )

            smooth_times = times[window - 1:]

            self.ax.plot(
                smooth_times,
                smooth,
                color="tab:orange",
                linewidth=2,
                linestyle="--",
                label="Electron Avg",
            )

        # =====================================================
        # ESTILO
        # =====================================================

        self.ax.set_title(
            "Townsend Discharge Evolution"
        )

        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Particles")

        self.ax.grid(
            True,
            alpha=0.3,
        )

        self.ax.legend(
            loc="upper left",
        )

        self.canvas.draw_idle()