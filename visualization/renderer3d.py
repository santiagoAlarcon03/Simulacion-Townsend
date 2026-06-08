import numpy as np
from PyQt6.QtWidgets import QLabel

# ==========================================
# CLASE: RENDERIZADOR 3D (PyVistaQt + PyQt6)
# ==========================================
class Renderer3D:
    """Gestiona la visualización tridimensional en tiempo real de la simulación.
    
    Esta clase actúa como un envoltorio (wrapper) alrededor de PyVistaQt para incrustar
    una ventana gráfica interactiva de 3D dentro de la interfaz de usuario PyQt6. Se encarga
    de mapear las coordenadas físicas de los electrones y partículas neutras a mallas 
    poligonales (PolyData) representadas como esferas.
    """

    def __init__(self, parent=None) -> None:
        """Inicializa la ventana gráfica 3D y configura los actores iniciales.

        Intenta cargar dinámicamente PyVista. Si no se encuentra instalado, degrada la 
        interfaz de forma segura a un widget `QLabel` con un mensaje de error.

        Args:
            parent (QWidget, optional): Widget padre de PyQt6 para la gestión de memoria de Qt.
        """
        self.available = False
        self.widget = QLabel("3D view requires PyVistaQt.")

        self._plotter = None

        self._electron_mesh = None
        self._electron_actor = None

        self._neutral_mesh = None
        self._neutral_actor = None

        # =====================================================
        # IONES POSITIVOS
        # =====================================================

        self._ion_mesh = None
        self._ion_actor = None

        self._domain_actor = None

        try:
            import pyvista as pv
            from pyvistaqt import QtInteractor
        except Exception as exc:
            self.widget.setText(f"3D view unavailable: {exc}")
            return

        # Configuración del entorno 3D si las librerías están disponibles
        self.available = True
        self._pv = pv

        self._plotter = QtInteractor(parent)
        self.widget = self._plotter

        self._plotter.set_background("black")
        self._plotter.add_axes()

        self._first_render = True

        # =====================================================
        # ELECTRONES
        # =====================================================

        self._electron_mesh = pv.PolyData(
            np.zeros((1, 3))
        )

        self._electron_actor = self._plotter.add_mesh(
            self._electron_mesh,
            render_points_as_spheres=True,
            point_size=12,
            color="yellow",
        )

        # =====================================================
        # GAS NEUTRO
        # =====================================================

        self._neutral_mesh = pv.PolyData(
            np.zeros((1, 3))
        )

        self._neutral_actor = self._plotter.add_mesh(
            self._neutral_mesh,
            render_points_as_spheres=True,
            point_size=7,
            color="dodgerblue",
            opacity=0.35,
        )

        # =====================================================
        # IONES POSITIVOS
        # =====================================================

        self._ion_mesh = pv.PolyData(
            np.zeros((1, 3))
        )

        self._ion_actor = self._plotter.add_mesh(
            self._ion_mesh,
            render_points_as_spheres=True,
            point_size=10,
            color="orangered",
            ambient=0.5,
            diffuse=0.8,
        )

        self._plotter.view_isometric()

        # Establece las dimensiones de la caja por defecto
        self._set_default_domain()
    def update_particles(self,electron_positions,neutral_positions=None,ion_positions=None,) -> None:
    
            if not self.available:

                return

            # =====================================================
            # ELECTRONES
            # =====================================================

            if electron_positions is None or len(electron_positions) == 0:
                electron_points = np.empty((0, 3))
            else:
                electron_points = np.asarray(
                    electron_positions,
                    dtype=float,
                )

            # =====================================================
            # GAS NEUTRO
            # =====================================================

            if neutral_positions is None or len(neutral_positions) == 0:
                neutral_points = np.empty((0, 3))
            else:
                neutral_points = np.asarray(
                    neutral_positions,
                    dtype=float,
                )

            # =====================================================
            # IONES POSITIVOS
            # =====================================================

            if ion_positions is None or len(ion_positions) == 0:
                ion_points = np.empty((0, 3))
            else:
                ion_points = np.asarray(
                    ion_positions,
                    dtype=float,
                )

            # =====================================================
            # ACTUALIZACIÓN VISUAL
            # =====================================================

            self._replace_actor(
                "electron",
                electron_points,
                "yellow",
            )

            self._replace_actor(
                "neutral",
                neutral_points,
                "dodgerblue",
            )

            self._replace_actor(
                "ion",
                ion_points,
                "orangered",
            )

            # =====================================================
            # CÁMARA
            # =====================================================

            if self._first_render:
                self._plotter.reset_camera()
                self._first_render = False

            self._plotter.render()

    def _replace_actor(self, kind: str, points: np.ndarray, color: str) -> None:
        """Remueve de forma eficiente el actor gráfico anterior y dibuja uno nuevo.

        Este método evita la acumulación de memoria limpiando dinámicamente las referencias 
        previas de PyVista antes de inyectar las nuevas nubes de puntos.

        Args:
            kind (str): Identificador del tipo de partícula ("electron" o "neutral").
            points (np.ndarray): Matriz limpia de NumPy con las posiciones (N, 3).
            color (str): Color en formato de texto estándar para las esferas (ej. "yellow").
        """
        actor_attr = f"_{kind}_actor"
        mesh_attr = f"_{kind}_mesh"
        actor = getattr(self, actor_attr)
        
        # Elimina el dibujo previo si existe
        if actor is not None:
            self._plotter.remove_actor(actor)
            
        # Si no hay partículas de esta especie, limpia los atributos de rastreo y sale
        if points.size == 0:
            setattr(self, mesh_attr, None)
            setattr(self, actor_attr, None)
            return
            
        # Crea la nueva estructura geométrica de PyVista
        mesh = self._pv.PolyData(points)
        actor = self._plotter.add_mesh(
            mesh,
            render_points_as_spheres=True,
            point_size=12,
            color=color,
        )
        
        # Almacena las referencias actuales para el siguiente ciclo
        setattr(self, mesh_attr, mesh)
        setattr(self, actor_attr, actor)

    def set_domain(self, xy_extent: float, gap_distance: float) -> None:
        """Establece las fronteras límites del contenedor en la interfaz visual.

        Args:
            xy_extent (float): Extensión máxima desde el origen hacia los ejes X e Y.
            gap_distance (float): Distancia de separación (Eje Z) entre el cátodo y el ánodo.
        """
        if not self.available:
            return
        self._add_domain_box(xy_extent, gap_distance)
        self._set_camera_for_domain(xy_extent, gap_distance)

    def _set_default_domain(self) -> None:
        """Configura un contenedor tridimensional por defecto (0.5cm x 0.5cm x 1.0cm)."""
        self._add_domain_box(0.005, 0.01)
        self._set_camera_for_domain(0.005, 0.01)

    def _add_domain_box(self, xy_extent: float, gap_distance: float) -> None:
        """Dibuja una caja gris semitransparente tipo wireframe que denota los límites físicos.

        Args:
            xy_extent (float): Límites espaciales en los ejes laterales.
            gap_distance (float): Longitud máxima en el eje vertical Z.
        """
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
        """Calcula una distancia y ángulo óptimos para enfocar el contenedor de la simulación.

        Args:
            xy_extent (float): Magnitud lateral del sistema.
            gap_distance (float): Altura del sistema.
        """
        center = (0.0, 0.0, gap_distance * 0.5)
        distance = max(xy_extent, gap_distance) * 6.0
        self._plotter.camera_position = [
            (distance, distance, distance),  # Posición de la cámara en el espacio
            center,                          # Punto de enfoque (Target)
            (0.0, 0.0, 1.0),                 # Vector "Arriba" de la cámara (Eje Z hacia arriba)
        ]
        self._plotter.reset_camera()

    def clear(self) -> None:
        """Limpia por completo las partículas del escenario restableciendo el estado visual."""
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
# CLASE: MOTOR DE SIMULACIÓN DE CHOQUES
# ==========================================
class TownsendSimulation:
    """Motor de cálculo físico microscópico para descargas de tipo Townsend.
    
    Aplica cinemática básica y procesos estocásticos iterativos mediante el método de 
    Monte Carlo simplificado. Simula la aceleración de electrones libres en un campo
    eléctrico uniforme y gestiona colisiones elásticas e ionizaciones multiplicativas 
    (avalancha) con átomos de gas neutros distribuidos aleatoriamente.
    """

    def __init__(self, xy_extent=0.005, gap_distance=0.01, num_neutrals=150):
        """Inicializa las constantes físicas, dimensiones de cámara y condiciones iniciales de partículas.

        Args:
            xy_extent (float, optional): Extensión del plano X/Y (m). Por defecto 0.005 (5mm).
            gap_distance (float, optional): Espacio libre entre placas (m). Por defecto 0.01 (10mm).
            num_neutrals (int, optional): Número estático de partículas de gas neutro en el medio. Por defecto 150.
        """
        self.xy_extent = xy_extent
        self.gap_distance = gap_distance
        
        # Campo eléctrico uniforme orientado en el eje vertical Z (V/m)
        self.E_field = np.array([0.0, 0.0, 5e5]) 
        self.dt = 1e-11  # Intervalo diferencial de tiempo por paso de cálculo (segundos)
        
        # Constantes del Sistema Internacional (SI)
        self.e_charge = 1.602e-19                    # Carga elemental del electrón (C)
        self.e_mass = 9.109e-31                      # Masa en reposo del electrón (kg)
        self.collision_radius = 0.0003                # Radio de sección eficaz para impactos (m)
        self.ionization_energy = 15.6 * self.e_charge # Energía mínima para arrancar un electrón (Argón ~15.6 eV)

        # Distribución espacial homogénea del gas neutro inicial
        self.neutral_pos = (np.random.rand(num_neutrals, 3) - 0.5) * 2 * xy_extent
        self.neutral_pos[:, 2] = np.random.rand(num_neutrals) * gap_distance 

        # Configuración del electrón semilla primario (Inyectado cerca del cátodo en Z=0)
        self.electron_pos = np.array([[0.0, 0.0, 0.0005]])
        self.electron_vel = np.array([[0.0, 0.0, 1e4]])

    def step(self):
        """Calcula el siguiente paso temporal integrando fuerzas físicas y procesando colisiones.

        Avanza la posición de los electrones usando la integración cinemática lineal de Euler.
        Luego, recorre cada electrón para detectar si colisiona con el gas. Dependiendo de la 
        energía disponible, procesa un rebote elástico o una ionización por impacto (Townsend).
        Finalmente elimina los electrones absorbidos por los electrodos metálicos.
        """
        if len(self.electron_pos) == 0:
            return

        # 1. INTEGRACIÓN CINEMÁTICA: Aceleración por campo de Coulomb (a = q*E / m)
        acceleration = (self.e_charge * self.E_field) / self.e_mass
        self.electron_vel += acceleration * self.dt
        self.electron_pos += self.electron_vel * self.dt

        nuevos_electrones = []
        electrones_a_eliminar = []

        # 2. DETECCIÓN Y PROCESAMIENTO ESTOCÁSTICO DE IMPACTOS
        for i, e_pos in enumerate(self.electron_pos):
            # Calcula las distancias euclidianas del electrón 'i' contra todos los átomos neutros
            distancias = np.linalg.norm(self.neutral_pos - e_pos, axis=1)
            choques = np.where(distancias < self.collision_radius)[0]

            if len(choques) > 0:
                # Se detectó colisión: calculamos la magnitud de velocidad y su Energía Cinética ($E_k = 0.5 \cdot m \cdot v^2$)
                v_mag = np.linalg.norm(self.electron_vel[i])
                kinetic_energy = 0.5 * self.e_mass * (v_mag**2)

                if kinetic_energy > self.ionization_energy:
                    # --- CASO A: DESCARGA MULTIPLICATIVA DE TOWNSEND ---
                    # El electrón impacta con suficiente energía para ionizar el átomo.
                    # Nacen 2 electrones libres que comparten la energía excedente en direcciones aleatorias.
                    for _ in range(2):
                        v_random = np.random.normal(0, 1, 3)
                        v_random /= np.linalg.norm(v_random) # Normaliza a vector unitario
                        
                        # Energía remanente dividida de forma equitativa ($v = \sqrt{2 \cdot E_{rem} / 2m}$)
                        v_val = np.sqrt(2 * (kinetic_energy - self.ionization_energy) / (2 * self.e_mass))
                        nuevos_electrones.append(v_random * v_val)
                    
                    electrones_a_eliminar.append(i)
                else:
                    # --- CASO B: COLISIÓN PURAMENTE ELÁSTICA ---
                    # No hay pérdida neta de energía cinética hacia el medio. 
                    # El electrón simplemente es dispersado (scattered) rebotando hacia una dirección al azar.
                    v_random = np.random.normal(0, 1, 3)
                    self.electron_vel[i] = (v_random / np.linalg.norm(v_random)) * v_mag

        # Actualización de estructuras: Remoción de electrones que sufrieron ionización (fueron reemplazados por el par secundario)
        if len(electrones_a_eliminar) > 0:
            self.electron_pos = np.delete(self.electron_pos, electrones_a_eliminar, axis=0)
            self.electron_vel = np.delete(self.electron_vel, electrones_a_eliminar, axis=0)

        # Inserción en bloque de los nuevos electrones multiplicados en el punto exacto de la colisión
        if len(nuevos_electrones) > 0:
            puntos_impacto = np.repeat([e_pos], len(nuevos_electrones), axis=0)
            self.electron_pos = np.vstack([self.electron_pos, puntos_impacto])
            self.electron_vel = np.vstack([self.electron_vel, nuevos_electrones])

        # 3. CONDICIÓN DE FRONTERA DE ABSORCIÓN (Pérdidas en el Ánodo/Cátodo)
        # Filtra y elimina los electrones que impactan contra las placas físicas en Z=0 y Z=gap_distance
        fuera_rango = (self.electron_pos[:, 2] > self.gap_distance) | (self.electron_pos[:, 2] < 0)
        self.electron_pos = self.electron_pos[~fuera_rango]
        self.electron_vel = self.electron_vel[~fuera_rango]


# ==========================================
# CÓDIGO DE EJECUCIÓN (Integración con PyQt6)
# ==========================================
if __name__ == "__main__":
    """Punto de entrada de la aplicación.
    
    Instancia el bucle de eventos de la interfaz gráfica, acopla el motor físico
    `TownsendSimulation` con el motor visual `Renderer3D` y corre un temporizador (QTimer)
    fijo a una tasa de refresco aproximada de 60 FPS (16ms) para procesar los frames.
    """
    import sys
    from PyQt6.QtWidgets import QApplication, QMainWindow
    from PyQt6.QtCore import QTimer

    app = QApplication(sys.argv)
    window = QMainWindow()
    window.setWindowTitle("Simulación de Avalancha Townsend 3D")
    window.resize(800, 600)

    # 1. Instanciación e inicialización del entorno físico y visual
    renderer = Renderer3D(window)
    sim = TownsendSimulation()
    
    window.setCentralWidget(renderer.widget)
    renderer.set_domain(sim.xy_extent, sim.gap_distance)

    # 2. Definición de la subrutina recurrente del reloj (Game Loop de la Simulación)
    def timer_event():
        sim.step()
        # Sincroniza los hilos de datos físicos con el buffer de representación de PyVista
        renderer.update_particles(sim.electron_pos, sim.neutral_pos)
        
        # Inyección cíclica (Mantiene la simulación corriendo indefinidamente si los electrones se agotan)
        if len(sim.electron_pos) == 0:
            sim.electron_pos = np.array([[0.0, 0.0, 0.0005]])
            sim.electron_vel = np.array([[0.0, 0.0, 1e4]])

    # Configuración del reloj interno de PyQt para disparar la física periódicamente
    timer = QTimer()
    timer.timeout.connect(timer_event)
    timer.start(16)  # Ejecución en milisegundos (~60 ciclos por segundo)

    window.show()
    sys.exit(app.exec())