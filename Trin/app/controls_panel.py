from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from simulation.state import STAGE_LABELS, Stage


class ControlsPanel(QWidget):
    start_requested = pyqtSignal()
    pause_toggled = pyqtSignal(bool)
    reset_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.stage_combo = QComboBox()
        for stage, label in STAGE_LABELS.items():
            self.stage_combo.addItem(label, stage)

        self.particle_spin = QSpinBox()
        self.particle_spin.setRange(1, 20000)
        self.particle_spin.setValue(200)

        self.start_button = QPushButton("Start")
        self.pause_button = QPushButton("Pause")
        self.pause_button.setCheckable(True)
        self.reset_button = QPushButton("Reset")

        self.stage_value = QLabel("-")

        form = QFormLayout()
        form.addRow("Start stage", self.stage_combo)
        form.addRow("Initial particles", self.particle_spin)
        form.addRow("Current stage", self.stage_value)

        buttons = QHBoxLayout()
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.pause_button)
        buttons.addWidget(self.reset_button)

        box = QGroupBox("Simulation Controls")
        box_layout = QVBoxLayout(box)
        box_layout.addLayout(form)
        box_layout.addLayout(buttons)

        layout = QVBoxLayout(self)
        layout.addWidget(box)
        layout.addStretch(1)

        self.start_button.clicked.connect(self.start_requested.emit)
        self.pause_button.toggled.connect(self._on_pause_toggled)
        self.reset_button.clicked.connect(self.reset_requested.emit)

    def _on_pause_toggled(self, checked: bool) -> None:
        self.pause_button.setText("Resume" if checked else "Pause")
        self.pause_toggled.emit(checked)

    def selected_stage(self) -> Stage:
        data = self.stage_combo.currentData(Qt.ItemDataRole.UserRole)
        return data if isinstance(data, Stage) else Stage.INITIAL_ELECTRONS

    def particle_count(self) -> int:
        return int(self.particle_spin.value())

    def set_pause_state(self, paused: bool) -> None:
        self.pause_button.blockSignals(True)
        self.pause_button.setChecked(paused)
        self.pause_button.setText("Resume" if paused else "Pause")
        self.pause_button.blockSignals(False)

    def set_stage_readback(self, stage: Stage) -> None:
        self.stage_value.setText(STAGE_LABELS.get(stage, "Unknown"))
