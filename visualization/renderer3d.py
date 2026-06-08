import numpy as np
from PyQt6.QtWidgets import QLabel

# ==========================================
# TU CLASE ORIGINAL (Mantenida intacta)
# ==========================================
class Renderer3D:
    def __init__(self, parent=None) -> None:
        self.available = False
        self.widget = QLabel("3D view requires PyVistaQt.")
        self._plotter = None
        self._electron_mesh = None
        self._electron_actor = None
        self._neutral_mesh = None
        self._neutral_actor = None
        self._domain_actor = None

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
        self._electron_mesh = pv.PolyData(np.zeros((1, 3)))
        self._neutral_mesh = None
        self._electron_actor = self._plotter.add_mesh(
            self._electron_mesh,
            render_points_as_spheres=True,
            point_size=12,
            color="yellow",
        )
        self._neutral_actor = None
        self._plotter.view_isometric()

        self._set_default_domain()

    def update_particles(self, electron_positions, neutral_positions=None) -> None:
        if not self.available:
            return
        if electron_positions is None or len(electron_positions) == 0:
            electron_points = np.empty((0, 3))
        else:
            electron_points = np.asarray(electron_positions, dtype=float)

        if neutral_positions is None or len(neutral_positions) == 0:
            neutral_points = np.empty((0, 3))
        else:
            neutral_points = np.asarray(neutral_positions, dtype=float)

        self._replace_actor("electron", electron_points, "yellow")
        self._replace_actor("neutral", neutral_points, "dodgerblue")
        if self._first_render:
            self._plotter.reset_camera()
            self._first_render = False
        self._plotter.render()

    def _replace_actor(self, kind: str, points: np.ndarray, color: str) -> None:
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
            point_size=12,
            color=color,
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
        if self._domain_actor is not None:
            self._plotter.remove_actor(self._domain_actor)
        bounds = (
            -xy_extent,
            xy_extent,
            -xy_extent,
            xy_extent,
            0.0,
            gap_distance,
        )
        box = self._pv.Box(bounds=bounds)
        self._domain_actor = self._plotter.add_mesh(
            box,
            style="wireframe",
            color="gray",
            opacity=0.35,
            line_width=1.0,
        )

    def _set_camera_for_domain(self, xy_extent: float, gap_distance: float) -> None:
        center = (0.0, 0.0, gap_distance * 0.5)
        distance = max(xy_extent, gap_distance) * 6.0
        self._plotter.camera_position = [
            (distance, distance, distance),
            center,
            (0.0, 0.0, 1.0),
        ]
        self._plotter.reset_camera()

    def clear(self) -> None:
        if not self.available:
            return
        if self._electron_actor is not None:
            self._plotter.remove_actor(self._electron_actor)
        if self._neutral_actor is not None:
            self._plotter.remove_actor(self._neutral_actor)
        self._electron_mesh = self._pv.PolyData(np.zeros((1, 3)))
        self._neutral_mesh = None
        self._electron_actor = self._plotter.add_mesh(
            self._electron_mesh,
            render_points_as_spheres=True,
            point_size=12,
            color="yellow",
        )
        self._neutral_actor = None
        self._first_render = True
        self._plotter.render()


# ==========================================
# NUEVA CLASE: MOTOR DE SIMULACIÓN DE CHOQUES
# ==========================================
class TownsendSimulation:
    def __init__(self, xy_extent=0.005, gap_distance=0.01, num_neutrals=150):
        self.xy_extent = xy_extent
        self.gap_distance = gap_distance
        
        # Campo eléctrico en dirección Z (V/m) que acelera los electrones
        self.E_field = np.array([0.0, 0.0, 5e5]) 
        self.dt = 1e-11  # Paso de tiempo de la simulación
        
        # Constantes físicas simplificadas
        self.e_charge = 1.602e-19
        self.e_mass = 9.109e-31
        self.collision_radius = 0.0003  # Distancia umbral para detectar choque
        self.ionization_energy = 15.6 * self.e_charge  # Energía de ionización (ej. Argón ~15.6 eV)

        # Inicializar partículas de gas neutro fijas o lentas
        self.neutral_pos = (np.random.rand(num_neutrals, 3) - 0.5) * 2 * xy_extent
        self.neutral_pos[:, 2] = np.random.rand(num_neutrals) * gap_distance # Distribución en Z

        # Inicializar electrones libres (comienzan abajo en Z=0)
        self.electron_pos = np.array([[0.0, 0.0, 0.0005]])
        self.electron_vel = np.array([[0.0, 0.0, 1e4]]) # Velocidad inicial hacia arriba

    def step(self):
        """Calcula el siguiente paso físico de la simulación."""
        if len(self.electron_pos) == 0:
            return

        # 1. Aceleración por el campo eléctrico (F = q*E -> a = q*E/m)
        acceleration = (self.e_charge * self.E_field) / self.e_mass
        self.electron_vel += acceleration * self.dt
        self.electron_pos += self.electron_vel * self.dt

        nuevos_electrones = []
        electrones_a_eliminar = []

        # 2. Verificar colisiones individuales (Electrón <-> Neutro)
        for i, e_pos in enumerate(self.electron_pos):
            # Calcular distancia de este electrón a todos los átomos neutros
            distancias = np.linalg.norm(self.neutral_pos - e_pos, axis=1)
            choques = np.where(distancias < self.collision_radius)[0]

            if len(choques) > 0:
                # ¡Ocurrió un golpe! Calculamos la energía cinética actual del electrón
                v_mag = np.linalg.norm(self.electron_vel[i])
                kinetic_energy = 0.5 * self.e_mass * (v_mag**2)

                if kinetic_energy > self.ionization_energy:
                    # --- CASO A: DESCARGA DE TOWNSEND (IONIZACIÓN) ---
                    # El electrón golpea fuerte y arranca otro electrón. 
                    # Ambos salen despedidos con direcciones aleatorias y menos energía.
                    for _ in range(2):
                        v_random = np.random.normal(0, 1, 3)
                        v_random /= np.linalg.norm(v_random)
                        # Comparten la energía restante tras la ionización
                        v_val = np.sqrt(2 * (kinetic_energy - self.ionization_energy) / (2 * self.e_mass))
                        nuevos_electrones.append(v_random * v_val)
                    
                    electrones_a_eliminar.append(i)
                else:
                    # --- CASO B: COLISIÓN ELÁSTICA ---
                    # El electrón rebota como una canica sin suficiente energía para ionizar
                    v_random = np.random.normal(0, 1, 3)
                    self.electron_vel[i] = (v_random / np.linalg.norm(v_random)) * v_mag

        # Actualizar las listas de electrones (añadir los nuevos por avalancha y remover duplicados rotos)
        if len(electrones_a_eliminar) > 0:
            self.electron_pos = np.delete(self.electron_pos, electrones_a_eliminar, axis=0)
            self.electron_vel = np.delete(self.electron_vel, electrones_a_eliminar, axis=0)

        if len(nuevos_electrones) > 0:
            # Los nuevos electrones nacen en el último punto de impacto conocido
            puntos_impacto = np.repeat([e_pos], len(nuevos_electrones), axis=0)
            self.electron_pos = np.vstack([self.electron_pos, puntos_impacto])
            self.electron_vel = np.vstack([self.electron_vel, nuevos_electrones])

        # 3. Filtrar electrones que ya salieron de las placas (Z > gap_distance)
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
    window.setWindowTitle("Simulación de Avalancha Townsend 3D")
    window.resize(800, 600)

    # 1. Instanciar renderizador y simulación
    renderer = Renderer3D(window)
    sim = TownsendSimulation()
    
    window.setCentralWidget(renderer.widget)
    renderer.set_domain(sim.xy_extent, sim.gap_distance)

    # 2. Bucle de animación usando un QTimer de PyQt
    def timer_event():
        sim.step()
        # Inyectar las posiciones calculadas al renderizador
        renderer.update_particles(sim.electron_pos, sim.neutral_pos)
        
        # Si se quedan sin electrones, reiniciar con uno nuevo para mantener la simulación viva
        if len(sim.electron_pos) == 0:
            sim.electron_pos = np.array([[0.0, 0.0, 0.0005]])
            sim.electron_vel = np.array([[0.0, 0.0, 1e4]])

    timer = QTimer()
    timer.timeout.connect(timer_event)
    timer.start(16)  # ~60 FPS

    window.show()
    sys.exit(app.exec())