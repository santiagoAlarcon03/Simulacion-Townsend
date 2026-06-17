"""
Panel de controles para la simulación de Townsend.

Este módulo define la clase ControlsPanel, encargada de proporcionar
la interfaz gráfica para controlar la simulación:

- Seleccionar la etapa inicial.
- Definir el número inicial de partículas.
- Definir la cantidad de partículas neutras de fondo.
- Iniciar la simulación.
- Pausar/Reanudar la simulación.
- Reiniciar la simulación.
- Agregar electrones libres manualmente.
- Agregar partículas neutras manualmente.

La comunicación con el resto de la aplicación se realiza mediante señales
(PyQt Signals), desacoplando la interfaz gráfica de la lógica de simulación.
"""

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
    """
    Panel lateral de control de la simulación.

    Contiene todos los controles que permiten al usuario interactuar
    con la simulación de descarga de Townsend.

    Señales emitidas:
    -----------------
    start_requested:
        Se emite cuando se presiona el botón Start.

    pause_toggled(bool):
        Se emite cuando el usuario pausa o reanuda la simulación.

    reset_requested:
        Se emite cuando se solicita reiniciar la simulación.

    add_electron_requested:
        Se emite al agregar un electrón libre.

    add_neutral_requested:
        Se emite al agregar una partícula neutra.
    """

    # Señales utilizadas para comunicarse con el controlador principal.
    start_requested = pyqtSignal()
    pause_toggled = pyqtSignal(bool)
    reset_requested = pyqtSignal()
    add_electron_requested = pyqtSignal()
    add_neutral_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """
        Inicializa todos los componentes visuales del panel.

        Parámetros
        ----------
        parent : QWidget | None
            Widget padre opcional.
        """
        super().__init__(parent)

        # ------------------------------------------------------------
        # Selector de etapa inicial de la simulación
        # ------------------------------------------------------------
        self.stage_combo = QComboBox()

        # Se cargan todas las etapas disponibles definidas en STAGE_LABELS.
        # El texto visible es la descripción y el dato asociado es el enum Stage.
        for stage, label in STAGE_LABELS.items():
            self.stage_combo.addItem(label, stage)

        # ------------------------------------------------------------
        # Selector de cantidad inicial de partículas (Electrones Semilla)
        # ------------------------------------------------------------
        self.particle_spin = QSpinBox()

        # Permite entre 1 y 20.000 partículas.
        self.particle_spin.setRange(1, 20000)

        # Valor inicial por defecto.
        self.particle_spin.setValue(200)

        # ------------------------------------------------------------
        # ⚛️ NUEVO: Selector de cantidad de partículas neutras de fondo
        # ------------------------------------------------------------
        self.neutral_spin = QSpinBox()
        
        # Rango amplio para el fondo de gas neutro
        self.neutral_spin.setRange(0, 50000)
        
        # Valor inicial por defecto razonable
        self.neutral_spin.setValue(1000)

        # ------------------------------------------------------------
        # Botones de control
        # ------------------------------------------------------------

        # Iniciar simulación.
        self.start_button = QPushButton("Start")

        # Pausar/Reanudar simulación.
        self.pause_button = QPushButton("Pause")

        # Permite mantener estado presionado/no presionado.
        self.pause_button.setCheckable(True)

        # Reiniciar simulación.
        self.reset_button = QPushButton("Reset")

        # Agregar electrón libre manualmente.
        self.add_electron_button = QPushButton("Agregar electrón libre")

        # Agregar partícula neutra manualmente.
        self.add_neutral_button = QPushButton("Agregar partícula neutra")

        # ------------------------------------------------------------
        # Etiqueta informativa de la etapa actual
        # ------------------------------------------------------------
        self.stage_value = QLabel("-")

        # ------------------------------------------------------------
        # Formulario de parámetros
        # ------------------------------------------------------------
        form = QFormLayout()

        form.addRow("Start stage", self.stage_combo)
        form.addRow("Initial particles", self.particle_spin)
        form.addRow("Initial neutrals", self.neutral_spin) # ⚛️ Añadido al diseño del formulario
        form.addRow("Current stage", self.stage_value)

        # ------------------------------------------------------------
        # Distribución horizontal de botones
        # ------------------------------------------------------------
        buttons = QHBoxLayout()

        buttons.addWidget(self.start_button)
        buttons.addWidget(self.add_electron_button)
        buttons.addWidget(self.add_neutral_button)
        buttons.addWidget(self.pause_button)
        buttons.addWidget(self.reset_button)

        # ------------------------------------------------------------
        # Caja agrupadora principal
        # ------------------------------------------------------------
        box = QGroupBox("Simulation Controls")

        box_layout = QVBoxLayout(box)
        box_layout.addLayout(form)
        box_layout.addLayout(buttons)

        # ------------------------------------------------------------
        # Layout principal del panel
        # ------------------------------------------------------------
        layout = QVBoxLayout(self)

        layout.addWidget(box)

        # Espacio flexible para mantener el contenido arriba.
        layout.addStretch(1)

        # ------------------------------------------------------------
        # Conexión de señales de los botones
        # ------------------------------------------------------------

        # Al presionar Start se emite start_requested.
        self.start_button.clicked.connect(self.start_requested.emit)

        # Al agregar un electrón se emite add_electron_requested.
        self.add_electron_button.clicked.connect(
            self.add_electron_requested.emit
        )

        # Al agregar una partícula neutra se emite add_neutral_requested.
        self.add_neutral_button.clicked.connect(
            self.add_neutral_requested.emit
        )

        # Gestiona el comportamiento del botón Pause.
        self.pause_button.toggled.connect(self._on_pause_toggled)

        # Reinicio de la simulación.
        self.reset_button.clicked.connect(self.reset_requested.emit)

    def _on_pause_toggled(self, checked: bool) -> None:
        """Gestiona el cambio de estado del botón de pausa."""
        self.pause_button.setText("Resume" if checked else "Pause")
        self.pause_toggled.emit(checked)

    def selected_stage(self) -> Stage:
        """Obtiene la etapa seleccionada actualmente por el usuario."""
        data = self.stage_combo.currentData(Qt.ItemDataRole.UserRole)

        return data if isinstance(data, Stage) else Stage.INITIAL_ELECTRONS

    def particle_count(self) -> int:
        """Obtiene la cantidad de partículas (electrones semilla) configurada por el usuario."""
        return int(self.particle_spin.value())

    # ⚛️ NUEVO MÉTODO DE LECTURA DE NEUTRAS
    def neutral_particle_count(self) -> int:
        """
        Obtiene la cantidad de partículas neutras configurada por el usuario.

        Retorna
        -------
        int
            Número inicial de partículas neutras de fondo.
        """
        return int(self.neutral_spin.value())

    def set_pause_state(self, paused: bool) -> None:
        """Actualiza visualmente el estado del botón de pausa sin emitir señales."""
        self.pause_button.blockSignals(True)
        self.pause_button.setChecked(paused)
        self.pause_button.setText("Resume" if paused else "Pause")
        self.pause_button.blockSignals(False)

    def set_stage_readback(self, stage: Stage) -> None:
        """Actualiza la etiqueta que muestra la etapa actual de la simulación."""
        self.stage_value.setText(
            STAGE_LABELS.get(stage, "Unknown")
        )