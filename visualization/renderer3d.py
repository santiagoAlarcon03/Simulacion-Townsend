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

    # 🔄 MODIFICADO CON *args: Elástica y blindada contra errores posicionales
    def update_particles(self, electron_positions, neutral_positions=None, *args) -> None:
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

        # 3. Iones Positivos (Naranja/Rojo) -> Primer argumento extra en args
        ion_points = np.empty((0, 3))
        if len(args) > 0 and args[0] is not None:
            if len(args[0]) > 0:
                ion_points = np.asarray(args[0], dtype=float)

        # 4. Recombinadas (Morado) -> Segundo argumento extra en args
        recombined_points = np.empty((0, 3))
        if len(args) > 1 and args[1] is not None:
            if len(args[1]) > 0:
                recombined_points = np.asarray(args[1], dtype=float)

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
        actor_attr = f"_{kind}_actor"
        mesh_attr = f"_{kind}_mesh"
        actor = getattr(self, actor_attr)
        
        if actor is not None:
            self._plotter.remove_actor(actor)
            
        if points.size == 0:
            setattr(self, mesh_attr, None)
            setattr(self, actor_attr, None)
            return
            
        mesh = self._pv.PolyData(points)
        actor = self._plotter.add_mesh(
            mesh,
            render_points_as_spheres=True,
            point_size=size,
            color=color,
            opacity=0.35 if kind == "neutral" else 1.0
        )
        
        setattr(self, mesh_attr, mesh)
        setattr(self, actor_attr, actor)

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

    def __init__(self, xy_extent=0.005, gap_distance=0.01, num_neutrals=150):
        self.xy_extent = xy_extent
        self.gap_distance = gap_distance
        
        # Campo eléctrico uniforme orientado en el eje vertical Z (V/m)
        self.E_field = np.array([0.0, 0.0, 5e5]) 
        self.dt = 1e-11  
        
        self.e_charge = 1.602e-19                    
        self.e_mass = 9.109e-31                      
        self.ion_mass = 6.633e-26  # Masa de un ion de Argón (aprox 73000 veces más pesado)
        self.collision_radius = 0.0003                
        self.ionization_energy = 15.6 * self.e_charge 

        # Distribución del gas neutro inicial
        self.neutral_pos = (np.random.rand(num_neutrals, 3) - 0.5) * 2 * xy_extent
        self.neutral_pos[:, 2] = np.random.rand(num_neutrals) * gap_distance 

        # Configuración de Electrones
        self.electron_pos = np.array([[0.0, 0.0, 0.0005]])
        self.electron_vel = np.array([[0.0, 0.0, 1e4]])

        # ➕ NUEVO: Vectores dinámicos para almacenar Iones positivos creados
        self.ion_pos = np.empty((0, 3))
        self.ion_vel = np.empty((0, 3))

        # 🔮 NUEVO: Almacenamiento de puntos efímeros de desionización/recombinación
        self.recombined_pos = np.empty((0, 3))

    def step(self):
        """Calcula el siguiente paso temporal integrando fuerzas de electrones e iones."""
        
        # --- A. CINEMÁTICA DE LOS IONES POSITIVOS ---
        # Los iones se mueven en sentido OPUESTO al campo eléctrico (hacia el cátodo Z=0)
        if len(self.ion_pos) > 0:
            ion_acceleration = (self.e_charge * self.E_field) / self.ion_mass
            self.ion_vel -= ion_acceleration * self.dt  # Signo menos porque van al cátodo
            self.ion_pos += self.ion_vel * self.dt
            
            # Condición de frontera para Iones: Absorción en el Cátodo (Z <= 0)
            ion_fuera = (self.ion_pos[:, 2] <= 0) | (self.ion_pos[:, 2] > self.gap_distance)
            self.ion_pos = self.ion_pos[~ion_fuera]
            self.ion_vel = self.ion_vel[~ion_fuera]

        # Al iniciar cada ciclo limpiamos el "destello" morado anterior para que sea dinámico
        self.recombined_pos = np.empty((0, 3))

        if len(self.electron_pos) == 0:
            return

        # --- B. CINEMÁTICA DE LOS ELECTRONES ---
        acceleration = (self.e_charge * self.E_field) / self.e_mass
        self.electron_vel += acceleration * self.dt
        self.electron_pos += self.electron_vel * self.dt

        nuevos_electrones = []
        electrones_a_eliminar = []
        nuevos_iones = []

        # --- C. DETECCIÓN DE IMPACTOS DE MONTE CARLO ---
        for i, e_pos in enumerate(self.electron_pos):
            if len(self.neutral_pos) == 0:
                continue
                
            distancias = np.linalg.norm(self.neutral_pos - e_pos, axis=1)
            choques = np.where(distancias < self.collision_radius)[0]

            if len(choques) > 0:
                idx_impacto = choques[0] # Tomamos el primer átomo chocado
                v_mag = np.linalg.norm(self.electron_vel[i])
                kinetic_energy = 0.5 * self.e_mass * (v_mag**2)

                if kinetic_energy > self.ionization_energy:
                    # ➕ FÍSICA DE IONIZACIÓN DE TOWNSEND:
                    # 1. El electrón arranca otro electrón. Nacen dos electrones.
                    for _ in range(2):
                        v_random = np.random.normal(0, 1, 3)
                        v_random /= np.linalg.norm(v_random)
                        v_val = np.sqrt(2 * (kinetic_energy - self.ionization_energy) / (2 * self.e_mass))
                        nuevos_electrones.append(v_random * v_val)
                    
                    electrones_a_eliminar.append(i)

                    # 2. El átomo neutro impactado se convierte en un Ion Positivo en ese punto exacto
                    nuevos_iones.append(self.neutral_pos[idx_impacto].copy())
                    
                    # 3. 🔮 FÍSICA DE DESIONIZACIÓN/RECOMBINACIÓN:
                    # Registramos el evento visual morado justo en el punto de la colisión ionizante
                    self.recombined_pos = np.vstack([self.recombined_pos, self.neutral_pos[idx_impacto]])

                else:
                    # COLISIÓN ELÁSTICA: Dispersión angular simple
                    v_random = np.random.normal(0, 1, 3)
                    self.electron_vel[i] = (v_random / np.linalg.norm(v_random)) * v_mag

        # Actualizar arreglos de electrones
        if len(electrones_a_eliminar) > 0:
            self.electron_pos = np.delete(self.electron_pos, electrones_a_eliminar, axis=0)
            self.electron_vel = np.delete(self.electron_vel, electrones_a_eliminar, axis=0)

        if len(nuevos_electrones) > 0:
            puntos_impacto = np.repeat([e_pos], len(nuevos_electrones), axis=0)
            self.electron_pos = np.vstack([self.electron_pos, puntos_impacto])
            self.electron_vel = np.vstack([self.electron_vel, nuevos_electrones])

        # Agregar los nuevos iones a la física activa
        if len(nuevos_iones) > 0:
            self.ion_pos = np.vstack([self.ion_pos, nuevos_iones])
            # Los iones nacen prácticamente en reposo térmico (velocidad muy baja)
            self.ion_vel = np.vstack([self.ion_vel, np.zeros((len(nuevos_iones), 3))])

        # Absorción de electrones en ánodo/cátodo
        fuera_rango = (self.electron_pos[:, 2] > self.gap_distance) | (self.electron_pos[:, 2] < 0)
        self.electron_pos = self.electron_pos[~fuera_rango]
        self.electron_vel = self.electron_vel[~fuera_rango]


# ==========================================
# CÓDIGO DE EJECUCIÓN (Integración con PyQt6)
# ==========================================
if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication, QMainWindow
    from PyQt6.QtCore import QTimer

    app = QApplication(sys.argv)
    window = QMainWindow()
    window.setWindowTitle("Simulación de Avalancha Townsend Completa 3D")
    window.resize(800, 600)

    renderer = Renderer3D(window)
    sim = TownsendSimulation()
    
    window.setCentralWidget(renderer.widget)
    renderer.set_domain(sim.xy_extent, sim.gap_distance)

    # Game Loop de renderizado acoplado
    def timer_event():
        sim.step()
        
        # 🔄 Pasamos los electrones, las neutras, los iones y las partículas recombinadas en orden
        renderer.update_particles(
            sim.electron_pos, 
            sim.neutral_pos, 
            sim.ion_pos, 
            sim.recombined_pos
        )
        
        # Reinyección si la avalancha se extingue por completo
        if len(sim.electron_pos) == 0:
            sim.electron_pos = np.array([[0.0, 0.0, 0.0005]])
            sim.electron_vel = np.array([[0.0, 0.0, 1e4]])

    timer = QTimer()
    timer.timeout.connect(timer_event)
    timer.start(16) 

    window.show()
    sys.exit(app.exec())