from typing import Optional
import numpy as np

from config import SimulationConfig
from physics import avalanche, breakdown, collisions, constants, field as field_module, ionization, particles
from simulation.state import SimulationState, Stage, STAGE_ORDER


class SimulationEngine:
    """Motor principal de simulación física para avalanchas electrónicas y rupturas dieléctricas.
    
    Esta clase se encarga de gestionar el ciclo de vida, la cinemática y las interacciones 
    de electrones y partículas neutras dentro de un dominio tridimensional bajo la 
    influencia de un campo eléctrico constante.
    """

    def __init__(self, config: SimulationConfig) -> None:
        """Inicializa el motor de simulación con una configuración específica.

        Args:
            config (SimulationConfig): Objeto que contiene los parámetros físicos 
                y de control de la simulación (límites, campo eléctrico, temperaturas, etc.).
        """
        self.config = config
        # Generador de números aleatorios para procesos estocásticos (colisiones, térmicos)
        self.rng = np.random.default_rng(config.seed)
        # Inicializa el campo eléctrico vectorial apuntando en dirección -Z (desde el ánodo al cátodo)
        # de modo que la fuerza F = q*E acelere los electrones (q < 0) hacia el ánodo (+Z)
        # y los iones positivos (q > 0) hacia el cátodo (Z = 0).
        self.field = field_module.ElectricField([0.0, 0.0, -config.electric_field])
        # Estado actual de las partículas en la simulación (inicialmente vacío)
        self.state: Optional[SimulationState] = None
        self.initial_count = config.initial_particles


    def reset(self, stage: Stage, n_particles: int) -> None:
        """Reinicia el estado de la simulación configurando una nueva etapa y partículas libres.

        Args:
            stage (Stage): La etapa de la simulación con la que se desea comenzar.
            n_particles (int): Cantidad base de electrones iniciales.
        """
        # Determina la cantidad real de electrones según la etapa seleccionada
        count = self._initial_count_for_stage(stage, n_particles)

        # Inicializa las posiciones y velocidades vectoriales de los electrones libres
        positions, velocities = particles.initialize_electrons(
            count,
            self.config,
            self.rng,
        )

        # Inicializa el fondo de gas neutro (partículas estables contra las que chocan los electrones)
        neutral_positions, neutral_velocities = (
            particles.initialize_neutral_particles(
                500,  # Cantidad fija de partículas de gas neutro para la simulación
                self.config,
                self.rng,
            )
        )

        # Instancia el contenedor del estado físico actual de la simulación
        self.state = SimulationState(
             positions=positions,
             velocities=velocities,

             neutral_positions=neutral_positions,
              neutral_velocities=neutral_velocities,

             ion_positions=np.zeros((0, 3), dtype=float),
             ion_velocities=np.zeros((0, 3), dtype=float),

             time=0.0,
             stage=stage,
)


    def step(self, dt: float):
        """Avanza la simulación un paso de tiempo diferencial (dt).
        
        Calcula el movimiento de las partículas debido al campo eléctrico, gestiona
        las colisiones elásticas, la dispersión de velocidades, los procesos de
        ionización (efecto Townsend) y calcula la corriente eléctrica resultante.

        Args:
            dt (float): Intervalo de tiempo (delta time) a simular.

        Returns:
            tuple: Un par (state, metrics) donde 'state' es el SimulationState actualizado 
                y 'metrics' es un diccionario con el conteo de electrones y la corriente actual.
                Retorna (None, None) si la simulación no ha sido inicializada.
        """
        if self.state is None:
            return None, None

        state = self.state
        # Si no quedan electrones libres, solo avanza el tiempo y retorna métricas vacías
        if state.positions.size == 0:
            state.time += dt
            return state, {"count": 0, "current": 0.0}

        # --- 1. CINEMÁTICA Y MOVIMIENTO DE PARTÍCULAS ---
        # Calcula la aceleración de los electrones: F = q * E  =>  a = (q * E) / m
        acceleration = particles.acceleration_from_field(
            self.field.vector, constants.ELECTRON_CHARGE, constants.ELECTRON_MASS
        )
        # Integración de Euler simple para actualizar velocidad y posición de los electrones
        state.velocities = state.velocities + acceleration * dt
        state.positions = state.positions + state.velocities * dt

        
        # Movimiento de los iones positivos.
# Por ahora se mueven únicamente con su velocidad térmica inicial.
        # --------------------------------------------------
# MOVIMIENTO DE IONES POSITIVOS
# --------------------------------------------------

        if state.ion_positions.size > 0:

            ion_acceleration = particles.acceleration_from_field(
                self.field.vector,
                constants.ELEMENTARY_CHARGE,
                4.65e-26,  # masa aproximada de N2+
            )

            state.ion_velocities = (
                state.ion_velocities
                + ion_acceleration * dt
            )

            state.ion_positions = (
                state.ion_positions
                + state.ion_velocities * dt
            )
        
        
        # Actualiza la posición del gas neutro (movimiento puramente térmico/difusivo)
        if state.neutral_positions.size > 0:
            state.neutral_positions = state.neutral_positions + state.neutral_velocities * dt

        # --- CONDICIONES DE FRONTERA ---
        # Paredes laterales (X, Y): reflexión elástica para electrones y neutros
        self._reflect_lateral(state.positions, state.velocities)
        self._reflect_lateral(state.neutral_positions, state.neutral_velocities)
        self._reflect_lateral(state.ion_positions, state.ion_velocities)
        # Neutros también rebotan en Z (no son absorbidos por los electrodos)
        self._reflect_z(state.neutral_positions, state.neutral_velocities)

        # Ánodo (Z = gap_distance): absorbe electrones que llegan
        if state.positions.shape[0] > 0:
            absorbed_e = state.positions[:, 2] >= self.config.gap_distance
            if absorbed_e.any():
                keep = ~absorbed_e
                state.positions = state.positions[keep]
                state.velocities = state.velocities[keep]

        # Cátodo (Z = 0): absorbe iones positivos que llegan
        if state.ion_positions.shape[0] > 0:
            absorbed_i = state.ion_positions[:, 2] <= 0.0
            if absorbed_i.any():
                keep = ~absorbed_i
                state.ion_positions = state.ion_positions[keep]
                state.ion_velocities = state.ion_velocities[keep]

        # Re-verifica si tras la absorción quedan electrones en el dominio
        if state.positions.size == 0:
            state.time += dt
            return state, {"count": 0, "ion_count": int(state.ion_positions.shape[0]), "current": 0.0,}

        # --- 2. COLISIONES ELECTRONES-NEUTROS ---
        # Detecta colisiones geométricas basadas en un radio de colisión efectivo
        electron_neutral_mask, neutral_collision_mask, collision_points = collisions.detect_electron_neutral_collisions(
            state.positions,
            state.neutral_positions,
            self.config.neutral_collision_radius,
        )
        state.collision_events = int(electron_neutral_mask.sum())

        # Si hubo choques electron-neutro, dispersa la velocidad del electrón y genera subproductos
        if electron_neutral_mask.any():
            collisions.scatter_velocities(state.velocities, electron_neutral_mask, self.rng)
            new_positions = self._spawn_electrons_from_collisions(collision_points, electron_neutral_mask.sum())
            new_velocities = particles.random_directional_velocities(
                new_positions.shape[0], self.rng, self.config.gas_temperature
            )
            state.positions = np.vstack([state.positions, new_positions])
            state.velocities = np.vstack([state.velocities, new_velocities])

            # Creación de iones positivos resultantes de la colisión (la partícula neutra pierde un electrón)
            new_ion_positions = new_positions.copy()
            # Calcular velocidad térmica realista para los iones usando su masa real (N2+)
            ion_mass = 4.65e-26
            thermal_speed_ion = np.sqrt(constants.BOLTZMANN * (self.config.gas_temperature * 0.05) / ion_mass)
            new_ion_velocities = self.rng.normal(0.0, thermal_speed_ion, size=(new_positions.shape[0], 3))
            state.ion_positions = np.vstack([state.ion_positions, new_ion_positions])
            state.ion_velocities = np.vstack([state.ion_velocities, new_ion_velocities])

        # Respawn de partículas neutras que colisionaron para mantener la densidad del gas constante
        if neutral_collision_mask.any():
            count = int(neutral_collision_mask.sum())
            respawn_positions, respawn_velocities = particles.initialize_neutral_particles(
                count, self.config, self.rng
            )
            state.neutral_positions[neutral_collision_mask] = respawn_positions
            state.neutral_velocities[neutral_collision_mask] = respawn_velocities

        # --- 3. COLISIONES DE FONDO (ESTOCÁSTICAS) ---
        # Muestrea colisiones adicionales basadas puramente en la frecuencia de colisión del medio
        collision_mask = collisions.sample_collisions(
            state.positions.shape[0], self.config.collision_frequency, dt, self.rng
        )
        collisions.scatter_velocities(state.velocities, collision_mask, self.rng)
        state.collision_events += int(collision_mask.sum())

        # --- 4. PROCESO DE IONIZACIÓN (AVALANCHA DE TOWNSEND) ---
        # Calcula la energía cinética de cada electrón en electronvoltios (eV)
        energies_ev = particles.kinetic_energy_ev(state.velocities)
        # Calcula el coeficiente Alfa de Townsend según la presión del gas y el campo eléctrico
        alpha = avalanche.townsend_alpha(
            self.config.townsend_A,
            self.config.townsend_B,
            self.config.electric_field,
            self.config.gas_pressure,
        )
        # Distancia recorrida en el eje Z (dirección del campo principal)
        dz = np.abs(state.velocities[:, 2]) * dt
        # Probabilidad acumulada de impactar e ionizar un átomo del gas neutro
        base_prob = 1.0 - np.exp(-alpha * dz)
        probability = np.clip(base_prob * self.config.ionization_probability, 0.0, 1.0)
        
        # Determina cuáles electrones lograron ionizar un átomo
        ionize_mask = ionization.sample_ionization(
            energies_ev,
            self.config.ionization_energy_ev,
            probability,
            self.rng,
        )
        state.ionization_events = int(ionize_mask.sum())

        # Si hubo ionizaciones y no se ha superado el límite de memoria del sistema, añade nuevos electrones
        # Si hubo ionizaciones y no se ha superado el límite de memoria
        if ionize_mask.any() and state.positions.shape[0] < self.config.max_particles:

            max_new = self.config.max_particles - state.positions.shape[0]
            new_count = min(int(ionize_mask.sum()), max_new)

            # =====================================================
            # NUEVOS ELECTRONES
            # =====================================================

            new_positions = state.positions[ionize_mask][:new_count].copy()

            new_velocities = particles.random_thermal_velocities(
                new_count,
                self.rng,
                self.config.gas_temperature,
            )

            state.positions = np.vstack([
                state.positions,
                new_positions,
            ])

            state.velocities = np.vstack([
                state.velocities,
                new_velocities,
            ])

            # =====================================================
            # NUEVOS IONES POSITIVOS
            # =====================================================

            ion_positions = new_positions.copy()

            # Calcular velocidad térmica realista para los iones usando su masa real (N2+)
            ion_mass = 4.65e-26
            thermal_speed_ion = np.sqrt(constants.BOLTZMANN * (self.config.gas_temperature * 0.05) / ion_mass)
            ion_velocities = self.rng.normal(0.0, thermal_speed_ion, size=(new_count, 3))

            state.ion_positions = np.vstack([
                state.ion_positions,
                ion_positions,
            ])

            state.ion_velocities = np.vstack([
                state.ion_velocities,
                ion_velocities,
            ])
        # --- 5. ACTUALIZACIÓN DEL ESTADO FINAL Y MÉTRICAS ---
        state.time += dt
        state.stage = self._infer_stage(state)  # Reevalúa la fase física actual de la descarga

        # Calcula el área transversal del dominio geométrico
        area = (2.0 * self.config.xy_extent) ** 2
        # Estima la corriente inducida neta (Corriente de Shockley-Ramo simplificada)
        current = particles.estimate_current(
            state.velocities, constants.ELECTRON_CHARGE, area
        )

        return state, {"count": int(state.positions.shape[0]),           # electrones
        "ion_count": int(state.ion_positions.shape[0]),  # iones positivos
        "current": float(current),}


    def _spawn_electrons_from_collisions(self, collision_points: np.ndarray, count: int) -> np.ndarray:
        """Genera y posiciona nuevos electrones secundarios derivados de eventos de colisión.

        Aplica un pequeño desfase geométrico para evitar superposiciones exactas y
        mantiene las coordenadas estrictamente dentro de las fronteras físicas de la cámara.

        Args:
            collision_points (np.ndarray): Matriz de coordenadas de los puntos de choque.
            count (int): Cantidad de electrones a generar.

        Returns:
            np.ndarray: Matriz de posiciones (N, 3) listas para los nuevos electrones.
        """
        if count <= 0 or collision_points.size == 0:
            return np.zeros((0, 3), dtype=float)
        collision_points = np.asarray(collision_points, dtype=float)
        if collision_points.shape[0] != count:
            count = collision_points.shape[0]
            
        # Genera un vector de desplazamiento micrométrico aleatorio
        directions = particles.random_directional_velocities(count, self.rng, self.config.gas_temperature)
        offsets = directions * 1.0e-6
        positions = collision_points[:count] + offsets
        
        # Recorta las posiciones para asegurar que no se salgan del volumen de simulación
        positions[:, 0] = np.clip(positions[:, 0], -self.config.xy_extent, self.config.xy_extent)
        positions[:, 1] = np.clip(positions[:, 1], -self.config.xy_extent, self.config.xy_extent)
        positions[:, 2] = np.clip(positions[:, 2], 0.0, self.config.gap_distance)
        return positions


    def _reflect_into_domain(self, positions: np.ndarray, velocities: np.ndarray) -> None:
        """Mantiene a las partículas dentro del dominio aplicando rebotes elásticos en las paredes.
        
        Si una partícula excede un límite, su posición se refleja de vuelta y el componente 
        de su velocidad en dicho eje se invierte (-1.0).

        Args:
            positions (np.ndarray): Matriz de posiciones a evaluar y modificar in-place.
            velocities (np.ndarray): Matriz de velocidades a evaluar y modificar in-place.
        """
        extent = self.config.xy_extent
        gap = self.config.gap_distance

        # Reflexiones en los ejes X (0) e Y (1)
        for axis in (0, 1):
            upper = positions[:, axis] > extent
            lower = positions[:, axis] < -extent
            if np.any(upper):
                positions[upper, axis] = 2.0 * extent - positions[upper, axis]
                velocities[upper, axis] *= -1.0
            if np.any(lower):
                positions[lower, axis] = -2.0 * extent - positions[lower, axis]
                velocities[lower, axis] *= -1.0

        # Reflexión en el eje Z (2) (Ánodo y Cátodo)
        upper_z = positions[:, 2] > gap
        lower_z = positions[:, 2] < 0.0
        if np.any(upper_z):
            positions[upper_z, 2] = 2.0 * gap - positions[upper_z, 2]
            velocities[upper_z, 2] *= -1.0
        if np.any(lower_z):
            positions[lower_z, 2] = -positions[lower_z, 2]
            velocities[lower_z, 2] *= -1.0

    def _reflect_lateral(self, positions: np.ndarray, velocities: np.ndarray) -> None:
        """Aplica rebote elástico sólo en las paredes laterales X e Y.

        Los electrodos (Z=0 y Z=gap) se tratan por separado con absorción.
        """
        if positions.size == 0:
            return
        extent = self.config.xy_extent
        for axis in (0, 1):
            upper = positions[:, axis] > extent
            lower = positions[:, axis] < -extent
            if np.any(upper):
                positions[upper, axis] = 2.0 * extent - positions[upper, axis]
                velocities[upper, axis] *= -1.0
            if np.any(lower):
                positions[lower, axis] = -2.0 * extent - positions[lower, axis]
                velocities[lower, axis] *= -1.0

    def _reflect_z(self, positions: np.ndarray, velocities: np.ndarray) -> None:
        """Aplica rebote elástico en los límites del eje Z.

        Usado únicamente para partículas neutras que no interactúan con los electrodos.
        """
        if positions.size == 0:
            return
        gap = self.config.gap_distance
        upper_z = positions[:, 2] > gap
        lower_z = positions[:, 2] < 0.0
        if np.any(upper_z):
            positions[upper_z, 2] = 2.0 * gap - positions[upper_z, 2]
            velocities[upper_z, 2] *= -1.0
        if np.any(lower_z):
            positions[lower_z, 2] = -positions[lower_z, 2]
            velocities[lower_z, 2] *= -1.0


    def _initial_count_for_stage(self, stage: Stage, count: int) -> int:
        """Determina un multiplicador de partículas adecuado según la complejidad de la etapa.

        Esto previene que simulaciones en fases avanzadas (como crecimiento exponencial)
        tarden demasiado en arrancar si se inician con muy pocos electrones libres.

        Args:
            stage (Stage): Etapa objetivo.
            count (int): Cantidad de partículas propuestas.

        Returns:
            int: Cantidad de partículas escalada y recomendada.
        """
        if stage in (Stage.AVALANCHE, Stage.EXPONENTIAL_GROWTH):
            return max(count, count * 5)
        if stage in (Stage.SELF_SUSTAINED, Stage.CHARGE_EVOLUTION):
            return max(count, count * 10)
        return count


    def _infer_stage(self, state: SimulationState) -> Stage:
        """Infiere de manera heurística cuál es la fase de la descarga eléctrica actual.

        Analiza hitos como el primer choque, la presencia de ionización, la duplicación
        o multiplicación por 8 de la población inicial de electrones, o si se alcanza
        la condición de ruptura autosostenida.

        Args:
            state (SimulationState): Estado actual con las métricas acumuladas.

        Returns:
            Stage: La etapa más avanzada que cumple las condiciones físicas actuales.
        """
        stage = state.stage
        stage = self._max_stage(stage, Stage.FIELD_ACCELERATION)
        if state.collision_events > 0:
            stage = self._max_stage(stage, Stage.COLLISIONS)
        if state.ionization_events > 0:
            stage = self._max_stage(stage, Stage.IONIZATION)
        if state.positions.shape[0] >= self.initial_count * 2:
            stage = self._max_stage(stage, Stage.AVALANCHE)
        if state.positions.shape[0] >= self.initial_count * 8:
            stage = self._max_stage(stage, Stage.EXPONENTIAL_GROWTH)
        if breakdown.is_self_sustained(self.config):
            stage = self._max_stage(stage, Stage.SELF_SUSTAINED)
        if state.time > 0.0:
            stage = self._max_stage(stage, Stage.CHARGE_EVOLUTION)
        return stage


    def _max_stage(self, current: Stage, candidate: Stage) -> Stage:
        """Compara dos etapas basándose en su orden jerárquico y devuelve la más avanzada.

        Args:
            current (Stage): Etapa actual de la simulación.
            candidate (Stage): Nueva etapa candidata a evaluar.

        Returns:
            Stage: La etapa que se encuentre más adelante en la lista estricta `STAGE_ORDER`.
        """
        if STAGE_ORDER.index(candidate) > STAGE_ORDER.index(current):
            return candidate
        return current


    def add_electron(self, position: list | None = None, velocity: list | None = None) -> None:
        """Inyecta de forma manual o automática un electrón individual en el sistema.

        Si no se especifican coordenadas o velocidades, el sistema calcula una posición
        visible despejada y una velocidad correspondiente a la temperatura del gas.

        Args:
            position (list, optional): Coordenadas tridimensionales [x, y, z].
            velocity (list, optional): Vector de velocidad tridimensional [vx, vy, vz].
        """
        if self.state is None:
            self.reset(Stage.INITIAL_ELECTRONS, 1)
            return

        if self.state.positions.shape[0] >= self.config.max_particles:
            return

        import numpy as _np

        # Configuración de posición
        if position is None:
            pos = self._sample_visible_position()
        else:
            pos = _np.asarray(position, dtype=float).reshape(1, 3)

        # Configuración de velocidad
        if velocity is None:
            vel = particles.random_directional_velocities(
                1, self.rng, self.config.gas_temperature
            )
        else:
            vel = _np.asarray(velocity, dtype=float).reshape(1, 3)

        # Inserción en las matrices globales de estado
        if self.state.positions.size == 0:
            self.state.positions = pos
            self.state.velocities = vel
        else:
            self.state.positions = _np.vstack([self.state.positions, pos])
            self.state.velocities = _np.vstack([self.state.velocities, vel])


    def add_neutral(self, position: list | None = None, velocity: list | None = None) -> None:
        """Añade una partícula neutra de gas para interactuar en el espacio tridimensional.

        Args:
            position (list, optional): Coordenadas tridimensionales de colocación [x, y, z].
            velocity (list, optional): Vector de velocidad inicial [vx, vy, vz].
        """
        if self.state is None:
            self.reset(Stage.INITIAL_ELECTRONS, 1)

        if self.state is None:
            return

        if self.state.neutral_positions.shape[0] >= self.config.max_particles:
            return

        import numpy as _np

        if position is None:
            pos = self._sample_visible_position()
        else:
            pos = _np.asarray(position, dtype=float).reshape(1, 3)

        if velocity is None:
            # Los neutros se mueven un poco más lento (un 50% de la energía térmica base)
            vel = particles.random_thermal_velocities(1, self.rng, self.config.gas_temperature * 0.5)
        else:
            vel = _np.asarray(velocity, dtype=float).reshape(1, 3)

        if self.state.neutral_positions.size == 0:
            self.state.neutral_positions = pos
            self.state.neutral_velocities = vel
        else:
            self.state.neutral_positions = _np.vstack([self.state.neutral_positions, pos])
            self.state.neutral_velocities = _np.vstack([self.state.neutral_velocities, vel])


    def _sample_visible_position(self) -> np.ndarray:
        """Busca estocásticamente una coordenada espacial despejada en la parte superior del dominio.
        
        Intenta hasta 64 veces encontrar un punto que esté lo suficientemente alejado de los
        electrones existentes (para evitar aglomeraciones visuales o físicas inmediatas).

        Returns:
            np.ndarray: Vector de posición (1, 3) en una zona segura y despejada del gap.
        """
        min_distance = max(self.config.xy_extent, self.config.gap_distance) * 0.30
        for _ in range(64):
            candidate = np.array([
                self.rng.uniform(-self.config.xy_extent, self.config.xy_extent),
                self.rng.uniform(-self.config.xy_extent, self.config.xy_extent),
                self.rng.uniform(self.config.gap_distance * 0.70, self.config.gap_distance * 0.95),
            ], dtype=float).reshape(1, 3)
            
            if self.state is None or self.state.positions.size == 0:
                return candidate
                
            distances = np.linalg.norm(self.state.positions - candidate, axis=1)
            if np.all(distances >= min_distance):
                return candidate
                
        # Caso de respaldo (Fallback): Colocación por defecto en el centro-superior si falla la búsqueda.
        return np.array([[0.0, 0.0, self.config.gap_distance * 0.9]], dtype=float)
