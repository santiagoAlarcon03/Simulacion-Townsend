import traceback
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QHBoxLayout, QMainWindow, QVBoxLayout, QWidget, QLabel, QFrame
from PyQt6.QtGui import QFont
import numpy as np

from app.controls_panel import ControlsPanel
from app.plot_panel import PlotPanel
from app.simulation_view import SimulationView
from config import SimulationConfig
from simulation.data_logger import DataLogger
from simulation.engine import SimulationEngine


class MainWindow(QMainWindow):
    """Ventana principal de la aplicación que coordina la interfaz gráfica y la física."""

    def __init__(self) -> None:
        """Inicializa los componentes de la interfaz, configura layouts y conecta señales de Qt."""
        super().__init__()
        self.setWindowTitle("Townsend Discharge 3D")
        self.resize(1100, 750)

        # --- INSTANCIACIÓN DE NÚCLEOS DE DATOS Y FÍSICA ---
        self.config = SimulationConfig()
        self.engine = SimulationEngine(self.config)
        self.logger = DataLogger()

        # --- INSTANCIACIÓN DE COMPONENTES DE INTERFAZ (UI) ---
        self.controls = ControlsPanel()
        self.view = SimulationView()
        self.plot_panel = PlotPanel()

        # --- CONTENEDOR ESTILIZADO PARA LOS CONTADORES VISUALES ---
        self.counter_frame = QFrame()
        self.counter_frame.setStyleSheet("""
            QFrame {
                background-color: #1E1E24;
                border: 1px solid #3A3A42;
                border-radius: 8px;
            }
            QLabel {
                border: none;
                background-color: transparent;
            }
        """)
        
        counter_layout = QVBoxLayout(self.counter_frame)
        counter_layout.setSpacing(12)
        counter_layout.setContentsMargins(15, 15, 15, 15)
        
        font_title = QFont("Segoe UI", 10, QFont.Weight.Bold)
        font_counters = QFont("Consolas", 11, QFont.Weight.Bold)
        
        lbl_panel_title = QLabel("MONITOREO DE ESPECIES")
        lbl_panel_title.setFont(font_title)
        lbl_panel_title.setStyleSheet("color: #8A8A93;")
        counter_layout.addWidget(lbl_panel_title)
        
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("background-color: #3A3A42; max-height: 1px;")
        counter_layout.addWidget(line)
        
        self.lbl_electrons = QLabel("⚡ Electrones:    0")
        self.lbl_electrons.setFont(font_counters)
        self.lbl_electrons.setStyleSheet("color: #FFD700;")
        
        self.lbl_ions = QLabel("➕ Iones (+):     0")
        self.lbl_ions.setFont(font_counters)
        self.lbl_ions.setStyleSheet("color: #FF4500;")
        
        self.lbl_neutrals = QLabel("⚛️ Neutras:       0")
        self.lbl_neutrals.setFont(font_counters)
        self.lbl_neutrals.setStyleSheet("color: #1E90FF;")

        self.lbl_recombined = QLabel("🔮 Desionizados:  0")
        self.lbl_recombined.setFont(font_counters)
        self.lbl_recombined.setStyleSheet("color: #BA55D3;")
        
        counter_layout.addWidget(self.lbl_electrons)
        counter_layout.addWidget(self.lbl_ions)
        counter_layout.addWidget(self.lbl_neutrals)
        counter_layout.addWidget(self.lbl_recombined)

        self._running = False

        # --- LAYOUTS ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.controls)
        left_layout.addWidget(self.counter_frame)
        left_layout.addStretch(1)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(self.view, stretch=2)
        right_layout.addWidget(self.plot_panel, stretch=1)

        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.addWidget(left_panel)
        root_layout.addWidget(right_panel, stretch=1)
        self.setCentralWidget(root)

        # --- SEÑALES ---
        self.controls.start_requested.connect(self.start_simulation)
        self.controls.pause_toggled.connect(self.set_paused)
        self.controls.reset_requested.connect(self.reset_simulation)
        self.controls.add_electron_requested.connect(self.add_electron)
        self.controls.add_neutral_requested.connect(self.add_neutral)

        # --- TEMPORIZADOR ---
        self.timer = QTimer(self)
        self.timer.setInterval(self.config.frame_dt_ms)
        self.timer.timeout.connect(self.on_tick)
        self.timer.start()

        self.reset_simulation()
        self.view.set_domain(self.config.xy_extent, self.config.gap_distance)

    def start_simulation(self) -> None:
        stage = self.controls.selected_stage()
        count = self.controls.particle_count()
        neutral_count = self.controls.neutral_particle_count()
        
        self.engine.reset(stage, count, neutral_count)
        self.logger.reset()
        self._running = True
        self.controls.set_pause_state(False)

    def set_paused(self, paused: bool) -> None:
        self._running = not paused

    def reset_simulation(self) -> None:
        stage = self.controls.selected_stage()
        count = self.controls.particle_count()
        neutral_count = self.controls.neutral_particle_count()
        
        self.engine.reset(stage, count, neutral_count)
        self.logger.reset()
        self._running = False
        self.controls.set_pause_state(True)
        
        if self.engine.state is not None:
            recombined_pos = getattr(self.engine.state, 'recombined_pos', None)
            self.view.update_particles(
                self.engine.state.positions,
                self.engine.state.neutral_positions,
                self.engine.state.ion_positions,
                recombined_pos
            )
            self.controls.set_stage_readback(self.engine.state.stage)
            
        self.plot_panel.update_plot(
            self.logger.times, 
            self.logger.counts,
            self.logger.ion_counts,
        )
        self.lbl_electrons.setText("⚡ Electrones:    0")
        self.lbl_ions.setText("➕ Iones (+):     0")
        self.lbl_neutrals.setText("⚛️ Neutras:       0")
        self.lbl_recombined.setText("🔮 Desionizados:  0")

    def on_tick(self) -> None:
        """Subrutina cíclica principal (Game Loop de la UI) con depuración inyectada."""
        if not self._running:
            return
            
        state, metrics = self.engine.step(self.config.dt)
        if state is None or metrics is None:
            return

        # ========================================================
        # 🔍 INTERCEPTOR DE DATOS DE SIMULACIÓN PARA CONSOLA
        # ========================================================
        try:
            e_size = len(state.positions) if state.positions is not None else 0
            n_size = len(state.neutral_positions) if state.neutral_positions is not None else 0
            i_size = len(state.ion_positions) if state.ion_positions is not None else 0
            
            recombined_pos = getattr(state, 'recombined_pos', None)
            r_size = len(recombined_pos) if recombined_pos is not None else 0
            
            print("\n--- 🛰️ [CONSOLA DIAGNÓSTICO WINDOW] ---")
            print(f"    ⚡ Electrones en memoria: {e_size}")
            print(f"    🔹 Neutras en memoria:    {n_size}")
            print(f"    🔴 IONES EN MEMORIA:      {i_size}  <--- ¡VERIFICA ESTO!")
            print(f"    🔮 Recombinados en frame: {r_size}")
            print("---------------------------------------\n")
        except Exception as e:
            print(f"⚠️ Error generando logs: {e}")
        # ========================================================
            
        self.logger.log(
            state.time, 
            metrics["count"], 
            metrics["current"],
            metrics.get("ion_count", 0),
        )
        
        # Sincronización tridimensional
        self.view.update_particles(
            state.positions, 
            state.neutral_positions, 
            state.ion_positions, 
            recombined_pos
        )
        
        self.controls.set_stage_readback(state.stage)
        self.plot_panel.update_plot(self.logger.times, self.logger.counts, self.logger.ion_counts)

        # Actualización de la tarjeta de la interfaz
        e_count = metrics.get("count", 0)
        i_count = metrics.get("ion_count", 0)
        n_count = len(state.neutral_positions) if state.neutral_positions is not None else 0
        r_count = getattr(state, 'total_recombinations', 0) if hasattr(state, 'total_recombinations') else metrics.get("collision_events", 0)
        
        self.lbl_electrons.setText(f"⚡ Electrones:    {e_count:<5}")
        self.lbl_ions.setText(f"➕ Iones (+):     {i_count:<5}")
        self.lbl_neutrals.setText(f"⚛️ Neutras:        {n_count:<5}")
        self.lbl_recombined.setText(f"🔮 Desionizados:  {r_count:<5}")

    def add_electron(self) -> None:
        self.engine.add_electron()
        if self.engine.state is not None:
            recombined_pos = getattr(self.engine.state, 'recombined_pos', None)
            self.view.update_particles(
                self.engine.state.positions, 
                self.engine.state.neutral_positions, 
                self.engine.state.ion_positions,
                recombined_pos
            )
            self.controls.set_stage_readback(self.engine.state.stage)
            self.plot_panel.update_plot(
                self.logger.times, 
                self.logger.counts,
                self.logger.ion_counts,
            )

    def add_neutral(self) -> None:
        self.engine.add_neutral()
        if self.engine.state is not None:
            recombined_pos = getattr(self.engine.state, 'recombined_pos', None)
            self.view.update_particles(
                self.engine.state.positions, 
                self.engine.state.neutral_positions,
                self.engine.state.ion_positions,
                recombined_pos
            )
            self.controls.set_stage_readback(self.engine.state.stage)
            self.plot_panel.update_plot(
                self.logger.times, 
                self.logger.counts,
                self.logger.ion_counts,
            )