import numpy as np
from PyQt6.QtWidgets import QLabel

# ==========================================
# CLASE: RENDERIZADOR 3D (PyVistaQt + PyQt6)
# ==========================================
class Renderer3D:
    """Gestiona la visualización tridimensional en tiempo real de la simulación.
    
    Esta clase actúa como un envoltorio (wrapper) alrededor de PyVistaQt para incrustar
    una ventana gráfica interactiva de 3D dentro de la interfaz de usuario PyQt6. Se encarga
    of mapear las coordenadas físicas de las partículas a mallas poligonales.
    """

    def __init__(self, parent=None) -> None:
        """Inicializa la ventana gráfica 3D y configura los actores iniciales."""
        self.available = False
        self.widget = QLabel("3D view requires PyVistaQt.")

        self._plotter = None

        self._electron_mesh = None
        self._electron_actor = None

        self._neutral_mesh = None
        self._neutral_actor = None

        self._ion_mesh = None
        self._ion_actor = None

        # --- NUEVO: Actor y malla para desionización/recombinación ---
        self._recombined_mesh = None
        self._recombined_actor = None

        self._domain_actor = None
        self._cathode_actor = None
        self._anode_actor = None

        try:
            import pyvista as pv
            from pyvistaqt import QtInteractor
        except Exception as exc:
            self.widget.setText(f"3D view unavailable: {exc}")
            return

        self.available = True
        self._pv = pv

        self._plotter = QtInteractor(parent)
        self.widget = self._plotter

        self._plotter.set_background("black")
        self._plotter.add_axes()

        self._first_render = True

        # =====================================================
        # MALLAS INDEPENDIENTES INICIALES
        # =====================================================
        self._electron_mesh = pv.PolyData(np.zeros((1, 3)))
        self._electron_actor = self._plotter.add_mesh(
            self._electron_mesh, render_points_as_spheres=True, point_size=12, color="yellow"
        )

        self._neutral_mesh = pv.PolyData(np.zeros((1, 3)))
        self._neutral_actor = self._plotter.add_mesh(
            self._neutral_mesh, render_points_as_spheres=True, point_size=7, color="dodgerblue", opacity=0.35
        )

        self._ion_mesh = pv.PolyData(np.zeros((1, 3)))
        self._ion_actor = self._plotter.add_mesh(
            self._ion_mesh, render_points_as_spheres=True, point_size=10, color="orangered", ambient=0.5, diffuse=0.8
        )

        # NUEVO: Registro inicial del actor morado
        self._recombined_mesh = pv.PolyData(np.zeros((1, 3)))
        self._recombined_actor = self._plotter.add_mesh(
            self._recombined_mesh, render_points_as_spheres=True, point_size=14, color="purple", ambient=0.6
        )

        self._plotter.view_isometric()
        self._set_default_domain()

    # 🎯 CORRECCIÓN TOTAL: Eliminamos *args y declaramos las 4 especies fijas
    def update_particles(self, electron_positions, neutral_positions=None, ion_positions=None, recombined_positions=None) -> None:
        if not self.available:
            return

        # 1. Electrones (Amarillo)
        if electron_positions is None or len(electron_positions) == 0:
            electron_points = np.empty((0, 3))
        else:
            electron_points = np.asarray(electron_positions, dtype=float)

        # 2. Neutras (Azul)
        if neutral_positions is None or len(neutral_positions) == 0:
            neutral_points = np.empty((0, 3))
        else:
            neutral_points = np.asarray(neutral_positions, dtype=float)

        # 3. Iones Positivos (Naranja/Rojo)
        if ion_positions is None or len(ion_positions) == 0:
            ion_points = np.empty((0, 3))
        else:
            # 💥 Al usar asarray directo evitamos problemas de referencias circulares in-place
            ion_points = np.asarray(ion_positions, dtype=float)

        # 4. Recombinadas (Morado)
        if recombined_positions is None or len(recombined_positions) == 0:
            recombined_points = np.empty((0, 3))
        else:
            recombined_points = np.asarray(recombined_positions, dtype=float)

        # Renderizar de forma segura las 4 especies usando el reemplazo dinámico
        self._replace_actor("electron", electron_points, "yellow", 12)
        self._replace_actor("neutral", neutral_points, "dodgerblue", 7)
        self._replace_actor("ion", ion_points, "orangered", 10)
        self._replace_actor("recombined", recombined_points, "purple", 14)

        if self._first_render:
            self._plotter.reset_camera()
            self._first_render = False

        self._plotter.render()

    def _replace_actor(self, kind: str, points: np.ndarray, color: str, size: int) -> None:
        """Remueve de forma eficiente el actor gráfico anterior y dibuja uno nuevo."""
        if not self.available:
            return

        # 🛡️ FILTRO DE SEGURIDAD: Si por error llega "ions" con 's', lo forzamos a "ion"
        if kind == "ions":
            kind = "ion"

        actor_attr = f"_{kind}_actor"
        mesh_attr = f"_{kind}_mesh"
        
        # Verificación preventiva: si el atributo no existe en el __init__, detenemos el flujo
        if not hasattr(self, actor_attr):
            print(f"⚠️ Error en Renderer3D: El atributo {actor_attr} no existe en la clase.")
            return

        actor = getattr(self, actor_attr)
        
        # 1. REMOCIÓN DEL ACTOR CADUCO: Limpiamos el búfer viejo de la escena de PyVista
        if actor is not None:
            try:
                self._plotter.remove_actor(actor)
            except Exception:
                pass # Evita caídas si el actor ya fue eliminado de forma externa
            
        # 2. ESCENARIO VACÍO: Si no hay partículas en este frame, limpiamos las referencias
        if points is None or points.size == 0 or len(points) == 0:
            setattr(self, mesh_attr, None)
            setattr(self, actor_attr, None)
            return
            
        # 3. RENDERIZADO REGENERATIVO: Construimos la PolyData con los nuevos datos reducidos
        mesh = self._pv.PolyData(points)
        
        # Inyectamos la geometría fresca en el motor de renderizado
        new_actor = self._plotter.add_mesh(
            mesh,
            render_points_as_spheres=True,
            point_size=size,
            color=color,
            opacity=0.35 if kind == "neutral" else 1.0,
            ambient=0.5 if kind == "ion" else 0.0,  # Mantenemos las propiedades de tus iones
            diffuse=0.8 if kind == "ion" else 1.0
        )
        
        # Guardamos de manera segura las nuevas referencias en los slots de la clase
        setattr(self, mesh_attr, mesh)
        setattr(self, actor_attr, new_actor)

    def set_domain(self, xy_extent: float, gap_distance: float) -> None:
        if not self.available:
            return
        self._add_domain_box(xy_extent, gap_distance)
        self._set_camera_for_domain(xy_extent, gap_distance)

    def _set_default_domain(self) -> None:
        self._add_domain_box(0.005, 0.01)
        self._set_camera_for_domain(0.005, 0.01)

    def _add_domain_box(self, xy_extent: float, gap_distance: float) -> None:
        if self._domain_actor is not None: self._plotter.remove_actor(self._domain_actor)
        if self._cathode_actor is not None: self._plotter.remove_actor(self._cathode_actor)
        if self._anode_actor is not None: self._plotter.remove_actor(self._anode_actor)
            
        bounds = (-xy_extent, xy_extent, -xy_extent, xy_extent, 0.0, gap_distance)
        box = self._pv.Box(bounds=bounds)
        self._domain_actor = self._plotter.add_mesh(box, style="wireframe", color="gray", opacity=0.35)

        cathode_plane = self._pv.Plane(center=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0), i_size=2.0*xy_extent, j_size=2.0*xy_extent)
        self._cathode_actor = self._plotter.add_mesh(cathode_plane, color="silver", opacity=0.4, show_edges=True)

        anode_plane = self._pv.Plane(center=(0.0, 0.0, gap_distance), direction=(0.0, 0.0, 1.0), i_size=2.0*xy_extent, j_size=2.0*xy_extent)
        self._anode_actor = self._plotter.add_mesh(anode_plane, color="goldenrod", opacity=0.4, show_edges=True)

    def _set_camera_for_domain(self, xy_extent: float, gap_distance: float) -> None:
        center = (0.0, 0.0, gap_distance * 0.5)
        distance = max(xy_extent, gap_distance) * 6.0
        self._plotter.camera_position = [(distance, distance, distance), center, (0.0, 0.0, 1.0)]
        self._plotter.reset_camera()

    def clear(self) -> None:
        if not self.available: return
        for attr in ["_electron_actor", "_neutral_actor", "_ion_actor", "_recombined_actor"]:
            actor = getattr(self, attr)
            if actor is not None: self._plotter.remove_actor(actor)
        self._electron_mesh = self._pv.PolyData(np.zeros((1, 3)))
        self._neutral_mesh = None
        self._ion_mesh = None
        self._recombined_mesh = None
        self._electron_actor = self._plotter.add_mesh(self._electron_mesh, render_points_as_spheres=True, point_size=12, color="yellow")
        self._neutral_actor = None
        self._ion_actor = None
        self._recombined_actor = None
        self._first_render = True
        self._plotter.render()


# ==========================================
# CLASE: MOTOR DE SIMULACIÓN DE CHOQUES
# ==========================================
class TownsendSimulation:
    """Motor de cálculo físico microscópico para descargas de tipo Townsend."""

    def __init__(self, xy_extent=0.005, gap_distance=0.01, num_neutrals=250):
        self.xy_extent = xy_extent
        self.gap_distance = gap_distance
        
        # ⚡ VALORES CALIBRADOS: Campo eléctrico aumentado para asegurar ionización
        self.E_field = np.array([0.0, 0.0, 1.2e6]) 
        self.dt = 4e-11  # Paso de tiempo optimizado para visualización
        
        self.e_charge = 1.602e-19                    
        self.e_mass = 9.109e-31                      
        self.ion_mass = 6.633e-26  # Masa de un ion de Argón
        
        # 🎯 RADIO DE COLISIÓN OPTIMIZADO: Facilita que los electrones toquen los átomos
        self.collision_radius = 0.0006                
        self.ionization_energy = 15.6 * self.e_charge 

        # 🌟 Parámetros de auto-sostenibilidad y fases
        self.gamma = 0.06  # 6% de probabilidad de emisión secundaria en el cátodo
        self.current_phase = "Descarga No Sostenida (Avalanchas aisladas)"

        # Distribución del gas neutro inicial
        self.neutral_pos = (np.random.rand(num_neutrals, 3) - 0.5) * 2 * xy_extent
        self.neutral_pos[:, 2] = np.random.rand(num_neutrals) * gap_distance 

        # Configuración de Electrones Semilla (Inyectamos 2 para garantizar flujo visual inicial)
        self.electron_pos = np.array([[0.0, 0.0, 0.0005], [0.0005, -0.0005, 0.0005]])
        self.electron_vel = np.array([[0.0, 0.0, 1e4], [0.0, 0.0, 1e4]])

        # Vectores dinámicos
        self.ion_pos = np.empty((0, 3))
        self.ion_vel = np.empty((0, 3))
        self.recombined_pos = np.empty((0, 3))

    def step(self):
        """Calcula el siguiente paso temporal integrando fuerzas con escalas visibles."""
        
        # --- A. CINEMÁTICA DE LOS IONES ACELERADA VISUALMENTE ---
        if len(self.ion_pos) > 0:
            # Multiplicamos la aceleración del ion por un factor artificial (ej. 500) 
            # para compensar su enorme masa (73,000x) y que puedas verlos bajar en tiempo real
            ion_acceleration = (self.e_charge * self.E_field) / self.ion_mass
            self.ion_vel -= ion_acceleration * self.dt * 500.0  # <- FACTOR VISUAL
            self.ion_pos += self.ion_vel * self.dt
            
            # Detectar impactos en el cátodo inferior (Z <= 0)
            iones_en_catodo = self.ion_pos[:, 2] <= 0
            num_impactos = np.sum(iones_en_catodo)
            
            if num_impactos > 0 and self.gamma > 0:
                electrones_secundarios = []
                vel_secundarias = []
                for _ in range(num_impactos):
                    if np.random.rand() < self.gamma:
                        electrones_secundarios.append([0.0, 0.0, 0.0002])
                        vel_secundarias.append([0.0, 0.0, 1e4])
                
                if len(electrones_secundarios) > 0:
                    if len(self.electron_pos) == 0:
                        self.electron_pos = np.array(electrones_secundarios)
                        self.electron_vel = np.array(vel_secundarias)
                    else:
                        self.electron_pos = np.vstack([self.electron_pos, electrones_secundarios])
                        self.electron_vel = np.vstack([self.electron_vel, vel_secundarias])

            # Absorción estricta de iones
            ion_fuera = (self.ion_pos[:, 2] <= 0) | (self.ion_pos[:, 2] > self.gap_distance)
            self.ion_pos = self.ion_pos[~ion_fuera]
            self.ion_vel = self.ion_vel[~ion_fuera]

        # Reseteo dinámico de desionizaciones por frame
        self.recombined_pos = np.empty((0, 3))

        nuevos_electrones_pos = []
        nuevos_electrones_vel = []
        nuevos_iones_pos = []
        electrones_a_eliminar = []

        # --- B. CINEMÁTICA Y COLISIONES DE ELECTRONES ---
        if len(self.electron_pos) > 0:
            # Filtro drástico inicial: Eliminar CUALQUIER electrón que ya esté en el techo o fuera de los límites
            # Esto evita que se queden atascados arriba como se ve en tu imagen
            en_techo = (self.electron_pos[:, 2] >= self.gap_distance * 0.98) | (self.electron_pos[:, 2] < 0)
            # También limpiamos los límites laterales del cubo (X e Y)
            fuera_lados = (np.abs(self.electron_pos[:, 0]) > self.xy_extent) | (np.abs(self.electron_pos[:, 1]) > self.xy_extent)
            
            eliminados_frontera = en_techo | fuera_lados
            self.electron_pos = self.electron_pos[~eliminados_frontera]
            self.electron_vel = self.electron_vel[~eliminados_frontera]

        # Si tras la limpieza quedan electrones, procesamos su movimiento y colisiones
        if len(self.electron_pos) > 0:
            acceleration = (self.e_charge * self.E_field) / self.e_mass
            self.electron_vel += acceleration * self.dt
            self.electron_pos += self.electron_vel * self.dt

            for i, e_pos in enumerate(self.electron_pos):
                if len(self.neutral_pos) == 0:
                    continue
                    
                distancias = np.linalg.norm(self.neutral_pos - e_pos, axis=1)
                choques = np.where(distancias < self.collision_radius)[0]

                if len(choques) > 0:
                    idx_impacto = choques[0]
                    
                    # 🎯 FORZADO DE DESIONIZACIÓN VISUAL: 
                    # Cada colisión efectiva tiene alta probabilidad de ionizar y pintar morado
                    if np.random.rand() < 0.50:
                        for _ in range(2):
                            v_random = np.random.normal(0, 1, 3)
                            v_random /= np.linalg.norm(v_random)
                            nuevos_electrones_pos.append(e_pos.copy())
                            nuevos_electrones_vel.append(v_random * 1e5)
                        
                        electrones_a_eliminar.append(i)

                        # Crear Ion positivo (Esfera roja)
                        nuevos_iones_pos.append(self.neutral_pos[idx_impacto].copy())
                        
                        # Registrar punto morado efímero de desionización
                        self.recombined_pos = np.vstack([self.recombined_pos, self.neutral_pos[idx_impacto]])
                    else:
                        # Rebote elástico ordinario
                        v_random = np.random.normal(0, 1, 3)
                        self.electron_vel[i] = (v_random / np.linalg.norm(v_random)) * np.linalg.norm(self.electron_vel[i])

            # Aplicar cambios en arreglos
            if len(electrones_a_eliminar) > 0:
                self.electron_pos = np.delete(self.electron_pos, electrones_a_eliminar, axis=0)
                self.electron_vel = np.delete(self.electron_vel, electrones_a_eliminar, axis=0)

            if len(nuevos_electrones_pos) > 0:
                if len(self.electron_pos) == 0:
                    self.electron_pos = np.array(nuevos_electrones_pos)
                    self.electron_vel = np.array(nuevos_electrones_vel)
                else:
                    self.electron_pos = np.vstack([self.electron_pos, nuevos_electrones_pos])
                    self.electron_vel = np.vstack([self.electron_vel, nuevos_electrones_vel])

            if len(nuevos_iones_pos) > 0:
                if len(self.ion_pos) == 0:
                    self.ion_pos = np.array(nuevos_iones_pos)
                    self.ion_vel = np.zeros((len(nuevos_iones_pos), 3))
                else:
                    self.ion_pos = np.vstack([self.ion_pos, nuevos_iones_pos])
                    self.ion_vel = np.vstack([self.ion_vel, np.zeros((len(nuevos_iones_pos), 3))])

        # --- C. CONTROL DE FASES PARA TU BANNER ---
        total_activos = len(self.electron_pos) + len(self.ion_pos)
        num_recombinados = len(self.recombined_pos)

        if total_activos == 0:
            self.current_phase = "Descarga Extinguida (Sin ionización suficiente)"
        elif num_recombinados > 0 and len(self.electron_pos) == 0:
            self.current_phase = "Fase Townsend: Desionización Activa (Recombinación)"
        elif len(self.electron_pos) == 0 and len(self.ion_pos) > 0:
            self.current_phase = "Fase Townsend: Deriva de Iones en Retorno"
        elif total_activos > 0 and total_activos < 50:
            self.current_phase = "Fase Townsend: Avalanchas Primarias Obscuras"
        else:
            self.current_phase = "⚡ DESCARGA AUTO-SOSTENIDA (Ruptura Townsend) ⚡"



# ==========================================
# CÓDIGO DE EJECUCIÓN (Integración Calibrada con PyQt6)
# ==========================================
if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
    from PyQt6.QtCore import QTimer
    from PyQt6.QtGui import QFont

    app = QApplication(sys.argv)
    window = QMainWindow()
    window.setWindowTitle("Simulación de Avalancha Townsend Completa 3D")
    window.resize(800, 600)

    # 🌟 Contenedor principal y layout vertical para Banner + 3D
    main_widget = QWidget()
    layout = QVBoxLayout(main_widget)
    layout.setContentsMargins(5, 5, 5, 5)

    # 🏷️ Banner visual de las fases con telemetría de partículas integrada
    lbl_fase = QLabel("Fase Townsend: Inicializando...")
    lbl_fase.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
    lbl_fase.setStyleSheet("""
        QLabel {
            background-color: #1a1a24;
            color: #00ffcc;
            border: 2px solid #333344;
            border-radius: 5px;
            padding: 8px;
            qproperty-alignment: 'AlignCenter';
        }
    """)
    layout.addWidget(lbl_fase)

    renderer = Renderer3D(window)
    sim = TownsendSimulation()
    
    # Añadir el widget del renderizador al layout inferior
    layout.addWidget(renderer.widget, stretch=1)
    window.setCentralWidget(main_widget)
    
    renderer.set_domain(sim.xy_extent, sim.gap_distance)

    # Game Loop de renderizado acoplado
    def timer_event():
        # 1. Avanzar la física del plasma
        sim.step()
        
        # Extracción segura de arreglos para evitar desfases de memoria con PyVista
        e_pos = sim.electron_pos if len(sim.electron_pos) > 0 else np.empty((0, 3))
        n_pos = sim.neutral_pos if len(sim.neutral_pos) > 0 else np.empty((0, 3))
        i_pos = sim.ion_pos if len(sim.ion_pos) > 0 else np.empty((0, 3))
        r_pos = sim.recombined_pos if len(sim.recombined_pos) > 0 else np.empty((0, 3))
        
        # 2. 🔄 Pasamos todas las especies activas al renderizador
        renderer.update_particles(e_pos, n_pos, i_pos, r_pos)
        
        # 3. 🔄 ACTUALIZACIÓN CON TELEMETRÍA: Monitoreamos los contadores en tiempo real
        lbl_fase.setText(f"{sim.current_phase}  |  [ e⁻: {len(e_pos)}  •  Iones+: {len(i_pos)} ]")
        
        # 4. 🔄 CONTROL ESTÍTICO ROBUSTO DEL BANNER
        if "AUTO-SOSTENIDA" in sim.current_phase:
            lbl_fase.setStyleSheet("""
                background-color: #2b0040; color: #ff00ff; border: 2px solid #ff00ff; 
                border-radius: 5px; padding: 8px; qproperty-alignment: 'AlignCenter';
            """)
        elif "Desionización" in sim.current_phase or "Deriva" in sim.current_phase:
            # Color azul índigo/eléctrico para la fase donde los electrones ya se fueron y quedan los iones
            lbl_fase.setStyleSheet("""
                background-color: #0f0f2d; color: #7f7fff; border: 2px solid #4f4fff; 
                border-radius: 5px; padding: 8px; qproperty-alignment: 'AlignCenter';
            """)
        elif "Extinguida" in sim.current_phase:
            lbl_fase.setStyleSheet("""
                background-color: #2b1111; color: #ff5555; border: 2px solid #ff5555; 
                border-radius: 5px; padding: 8px; qproperty-alignment: 'AlignCenter';
            """)
        else:
            # Estado base (Cian) para avalanchas oscuras activas
            lbl_fase.setStyleSheet("""
                background-color: #1a1a24; color: #00ffcc; border: 2px solid #333344; 
                border-radius: 5px; padding: 8px; qproperty-alignment: 'AlignCenter';
            """)

        # 🔒 REINYECCIÓN ANTICORRUPCIÓN:
        # Solo se permite un nuevo disparo semilla si el gas se desionizó por completo 
        # y no queda absolutamente ningún portador de carga libre flotando.
        if len(sim.electron_pos) == 0 and len(sim.ion_pos) == 0:
            sim.electron_pos = np.array([[0.0, 0.0, 0.0005]])
            sim.electron_vel = np.array([[0.0, 0.0, 1e4]])

    timer = QTimer()
    timer.timeout.connect(timer_event)
    timer.start(16) 

    window.show()
    sys.exit(app.exec())