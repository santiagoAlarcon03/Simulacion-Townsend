from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QHBoxLayout, QMainWindow, QVBoxLayout, QWidget

from app.controls_panel import ControlsPanel
from app.plot_panel import PlotPanel
from app.simulation_view import SimulationView
from config import SimulationConfig
from simulation.data_logger import DataLogger
from simulation.engine import SimulationEngine


class MainWindow(QMainWindow):
    """Ventana principal de la aplicación que coordina la interfaz gráfica y la física.
    
    Esta clase actúa como el Controlador (en una arquitectura MVC simplificada). 
    Se encarga de orquestar el flujo de datos entre el panel de controles de usuario 
    (`ControlsPanel`), la vista tridimensional de partículas (`SimulationView`), 
    el módulo de gráficas (`PlotPanel`), el motor de simulación (`SimulationEngine`) 
    y el registrador histórico (`DataLogger`).
    """

    def __init__(self) -> None:
        """Inicializa los componentes de la interfaz, configura layouts y conecta señales de Qt."""
        super().__init__()
        self.setWindowTitle("Townsend Discharge 3D")

        # --- INSTANCIACIÓN DE NÚCLEOS DE DATOS Y FÍSICA ---
        self.config = SimulationConfig()
        self.engine = SimulationEngine(self.config)
        self.logger = DataLogger()

        # --- INSTANCIACIÓN DE COMPONENTES DE INTERFAZ (UI) ---
        self.controls = ControlsPanel()
        self.view = SimulationView()
        self.plot_panel = PlotPanel()

        # Bandera de control para pausar/reanudar los cálculos físicos
        self._running = False

        # --- DISEÑO Y DISTRIBUCIÓN DE ESPACIOS (LAYOUTS) ---
        # Panel Derecho: Vista 3D arriba, Gráficas de métricas abajo
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(self.view)
        right_layout.addWidget(self.plot_panel)

        # Contenedor Raíz: Controles a la izquierda, Panel Derecho ocupando el resto
        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.addWidget(self.controls)
        root_layout.addWidget(right_panel, stretch=1)
        self.setCentralWidget(root)

        # --- INTERCONEXIÓN DE SEÑALES Y RANURAS (SIGNALS & SLOTS) ---
        self.controls.start_requested.connect(self.start_simulation)
        self.controls.pause_toggled.connect(self.set_paused)
        self.controls.reset_requested.connect(self.reset_simulation)
        self.controls.add_electron_requested.connect(self.add_electron)
        self.controls.add_neutral_requested.connect(self.add_neutral)

        # --- CONFIGURACIÓN DEL TEMPORIZADOR DEL GAME LOOP ---
        self.timer = QTimer(self)
        self.timer.setInterval(self.config.frame_dt_ms)
        self.timer.timeout.connect(self.on_tick)
        self.timer.start()

        # --- ESTADO INICIAL ---
        self.reset_simulation()
        self.view.set_domain(self.config.xy_extent, self.config.gap_distance)

    def start_simulation(self) -> None:
        """Inicia de forma activa la ejecución física de la simulación.
        
        Extrae los parámetros configurados por el usuario en el panel frontal,
        reinicia los buffers del motor y del logger, y activa el procesamiento dinámico.
        """
        stage = self.controls.selected_stage()
        count = self.controls.particle_count()
        self.engine.reset(stage, count)
        self.logger.reset()
        self._running = True
        self.controls.set_pause_state(False)

    def set_paused(self, paused: bool) -> None:
        """Alterna el estado de pausa del procesamiento físico sin borrar los datos actuales.

        Args:
            paused (bool): True si se desea pausar la física, False si se desea reanudar.
        """
        self._running = not paused

    def reset_simulation(self) -> None:
        """Detiene la simulación y restablece todas las variables a sus condiciones iniciales.
        
        Limpia los gráficos, lee los controles de configuración actuales e inyecta 
        las partículas semilla en el motor visualizaciones para preparar un nuevo ciclo.
        """
        stage = self.controls.selected_stage()
        count = self.controls.particle_count()
        self.engine.reset(stage, count)
        self.logger.reset()
        self._running = False
        self.controls.set_pause_state(True)
        
        # Sincroniza la UI inmediatamente con el estado vacío o inicializado
        if self.engine.state is not None:
            self.view.update_particles(
                self.engine.state.positions, 
                self.engine.state.neutral_positions, 
                self.engine.state.ion_positions
            )
            self.controls.set_stage_readback(self.engine.state.stage)
            
        self.plot_panel.update_plot(
            self.logger.times, 
            self.logger.counts,
            self.logger.ion_counts,
        )

    def on_tick(self) -> None:
        """Subrutina cíclica principal activada por el QTimer (Game Loop de la UI).
        
        Si el sistema no está pausado, solicita un paso de tiempo diferencial (dt) al 
        motor físico, extrae las nuevas métricas obtenidas, las almacena en el logger 
        e instruye a los paneles visuales (3D y 2D) redibujar el nuevo estado de las partículas.
        """
        if not self._running:
            return
            
        # Ejecuta cálculos físicos de un paso de tiempo
        state, metrics = self.engine.step(self.config.dt)
        if state is None or metrics is None:
            return
            
        # Almacena en el histórico (incluyendo conteo de iones)
        self.logger.log(
            state.time, 
            metrics["count"], 
            metrics["current"],
            metrics["ion_count"],
        )
        
        # Actualiza la posición de todas las especies de partículas en la escena 3D
        self.view.update_particles(state.positions, state.neutral_positions, state.ion_positions)
        # Actualiza el indicador de texto que muestra la etapa física deducida
        self.controls.set_stage_readback(state.stage)
        # Refresca las curvas del osciloscopio gráfico en tiempo real
        self.plot_panel.update_plot(self.logger.times, self.logger.counts, self.logger.ion_counts)

    def add_electron(self) -> None:
        """Inyecta manualmente un único electrón libre en el sistema en tiempo de ejecución.
        
        Fuerza una actualización visual inmediata en 3D y en los paneles de lectura 
        para que el usuario perciba instantáneamente la inserción, incluso en estado de pausa.
        """
        self.engine.add_electron()
        if self.engine.state is not None:
            self.view.update_particles(
                self.engine.state.positions, 
                self.engine.state.neutral_positions, 
                self.engine.state.ion_positions
            )
            self.controls.set_stage_readback(self.engine.state.stage)
            self.plot_panel.update_plot(
                self.logger.times, 
                self.logger.counts,
                self.logger.ion_counts,
            )

    def add_neutral(self) -> None:
        """Inyecta manualmente una partícula del gas neutro de fondo en el sistema.
        
        Fuerza una actualización visual de forma inmediata en las mallas correspondientes.
        """
        self.engine.add_neutral()
        if self.engine.state is not None:
            self.view.update_particles(
                self.engine.state.positions, 
                self.engine.state.neutral_positions,
                self.engine.state.ion_positions
            )
            self.controls.set_stage_readback(self.engine.state.stage)
            self.plot_panel.update_plot(
                self.logger.times, 
                self.logger.counts,
                self.logger.ion_counts,
            )