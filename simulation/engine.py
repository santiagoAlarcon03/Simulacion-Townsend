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
        """Inicializa el motor de simulación con una configuración específica."""
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.field = field_module.ElectricField([0.0, 0.0, -config.electric_field])
        self.state: Optional[SimulationState] = None
        self.initial_count = config.initial_particles

    def reset(self, stage: Stage, n_particles: int, neutral_count: int = None) -> None:
        """Reinicia el estado de la simulación configurando una nueva etapa y partículas libres."""
        count = self._initial_count_for_stage(stage, n_particles)

        positions, velocities = particles.initialize_electrons(
            count,
            self.config,
            self.rng,
        )

        n_neutrals = neutral_count if neutral_count is not None else 500

        neutral_positions, neutral_velocities = (
            particles.initialize_neutral_particles(
                n_neutrals,
                self.config,
                self.rng,
            )
        )

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
        """Avanza la simulación un paso de tiempo diferencial (dt)."""
        if self.state is None:
            return None, None

        state = self.state
        
        # Mantenemos viva la simulación mientras queden electrones O iones en movimiento
        if state.positions.size == 0 and state.ion_positions.size == 0:
            state.time += dt
            return state, {"count": 0, "ion_count": 0, "current": 0.0}      

        # --- 1. CINEMÁTICA Y MOVIMIENTO DE PARTÍCULAS ---
        if state.positions.size > 0:
            acceleration = particles.acceleration_from_field(
                self.field.vector, constants.ELECTRON_CHARGE, constants.ELECTRON_MASS
            )
            state.velocities = state.velocities + acceleration * dt
            state.positions = state.positions + state.velocities * dt

        # MOVIMIENTO DE IONES POSITIVOS
        if state.ion_positions.size > 0:
            ion_acceleration = particles.acceleration_from_field(
                self.field.vector,
                constants.ELEMENTARY_CHARGE,
                4.65e-26,  # masa aproximada de N2+
            )
            ion_acceleration[2] = -np.abs(ion_acceleration[2]) * 30.0

            state.ion_velocities = state.ion_velocities + ion_acceleration * dt
            state.ion_positions = state.ion_positions + state.ion_velocities * dt
        
        # Movimiento del gas neutro
        if state.neutral_positions.size > 0:
            state.neutral_positions = state.neutral_positions + state.neutral_velocities * dt

        # --- CONDICIONES DE FRONTERA ---
        self._reflect_lateral(state.positions, state.velocities)
        self._reflect_lateral(state.neutral_positions, state.neutral_velocities)
        self._reflect_lateral(state.ion_positions, state.ion_velocities)
        self._reflect_z(state.neutral_positions, state.neutral_velocities)

        # Ánodo (Z = gap_distance): absorbe electrones
        if state.positions.shape[0] > 0:
            absorbed_e = state.positions[:, 2] >= self.config.gap_distance
            if absorbed_e.any():
                keep = ~absorbed_e
                state.positions = state.positions[keep]
                state.velocities = state.velocities[keep]

        # Cátodo (Z = 0): absorbe iones positivos
        if state.ion_positions.shape[0] > 0:
            absorbed_i = state.ion_positions[:, 2] <= 0.0
            if absorbed_i.any():
                keep = ~absorbed_i
                state.ion_positions = state.ion_positions[keep]
                state.ion_velocities = state.ion_velocities[keep]

        # 🎯 CORRECCIÓN INTEGRAL: Se eliminó el return abrupto que cortaba las colisiones y desionizaciones.

        # --- 2. COLISIONES ELECTRONES-NEUTROS ---
        state.collision_events = 0
        if state.positions.size > 0 and state.neutral_positions.size > 0:
            electron_neutral_mask, neutral_collision_mask, collision_points = collisions.detect_electron_neutral_collisions(
                state.positions,
                state.neutral_positions,
                self.config.neutral_collision_radius,
            )
            state.collision_events = int(electron_neutral_mask.sum())

            if electron_neutral_mask.any():
                collisions.scatter_velocities(state.velocities, electron_neutral_mask, self.rng)
                new_positions = self._spawn_electrons_from_collisions(collision_points, electron_neutral_mask.sum())
                new_velocities = particles.random_directional_velocities(
                    new_positions.shape[0], self.rng, self.config.gas_temperature
                )
                state.positions = np.vstack([state.positions, new_positions])
                state.velocities = np.vstack([state.velocities, new_velocities])

                # =====================================================
                # 🛡️ SHIELD 1: DESFASE ANTI-RECOMBINACIÓN EN COLISIONES
                # =====================================================
                current_dtype = state.positions.dtype
                r_recomb = getattr(self.config, 'recombination_radius', 0.001)
                
                # Desfase disperso aleatorio (entre 1.5 y 2.5 veces el radio de recombinación)
                ion_offsets = self.rng.uniform(r_recomb * 1.5, r_recomb * 2.5, size=(new_positions.shape[0], 3))
                ion_offsets *= self.rng.choice([-1, 1], size=(new_positions.shape[0], 3))
                
                # Sumamos el desfase a las posiciones originales
                new_ion_positions = (new_positions.copy() + ion_offsets).astype(current_dtype)
                
                # 🛡️ RECORTE ANTI-ESCAPE: Forzamos a que el eje Z (índice 2) no supere el Ánodo
                new_ion_positions[:, 2] = np.clip(new_ion_positions[:, 2], 0.0, self.config.gap_distance - 1.0e-6)
                
                ion_mass = 4.65e-26
                thermal_speed_ion = np.sqrt(constants.BOLTZMANN * (self.config.gas_temperature * 0.05) / ion_mass)
                new_ion_velocities = self.rng.normal(0.0, thermal_speed_ion, size=(new_positions.shape[0], 3)).reshape(-1, 3).astype(current_dtype)
                
                # Inserción segura en el estado global
                if state.ion_positions is None or state.ion_positions.size == 0:
                    state.ion_positions = new_ion_positions
                    state.ion_velocities = new_ion_velocities
                else:
                    state.ion_positions = np.vstack([state.ion_positions, new_ion_positions])
                    state.ion_velocities = np.vstack([state.ion_velocities, new_ion_velocities])

            if neutral_collision_mask.any():
                count = int(neutral_collision_mask.sum())
                respawn_positions, respawn_velocities = particles.initialize_neutral_particles(
                    count, self.config, self.rng
                )
                state.neutral_positions[neutral_collision_mask] = respawn_positions
                state.neutral_velocities[neutral_collision_mask] = respawn_velocities

        # --- 3. COLISIONES DE FONDO (ESTOCÁSTICAS) ---
        state.collision_events = 0  # Inicialización limpia de eventos de colisión del frame
        if state.positions.size > 0:
            collision_mask = collisions.sample_collisions(
                state.positions.shape[0], self.config.collision_frequency, dt, self.rng
            )
            collisions.scatter_velocities(state.velocities, collision_mask, self.rng)
            state.collision_events += int(collision_mask.sum())

        # --- 4. PROCESO DE IONIZACIÓN (AVALANCHA DE TOWNSEND) ---
        state.ionization_events = 0
        if state.positions.size > 0:
            energies_ev = particles.kinetic_energy_ev(state.velocities)
            alpha = avalanche.townsend_alpha(
                self.config.townsend_A,
                self.config.townsend_B,
                self.config.electric_field,
                self.config.gas_pressure,
            )
            dz = np.abs(state.velocities[:, 2]) * dt
            base_prob = 1.0 - np.exp(-alpha * dz)
            probability = np.clip(base_prob * self.config.ionization_probability, 0.0, 1.0)
            
            ionize_mask = ionization.sample_ionization(
                energies_ev,
                self.config.ionization_energy_ev,
                probability,
                self.rng,
            )
            state.ionization_events = int(ionize_mask.sum())

            if ionize_mask.any() and state.positions.shape[0] < self.config.max_particles:
                max_new = self.config.max_particles - state.positions.shape[0]
                new_count = min(int(ionize_mask.sum()), max_new)

                new_positions = state.positions[ionize_mask][:new_count].copy()
                new_velocities = particles.random_thermal_velocities(
                    new_count, self.rng, self.config.gas_temperature,
                )
            
                state.positions = np.vstack([state.positions, new_positions])
                state.velocities = np.vstack([state.velocities, new_velocities])

                # =====================================================
                # 🛡️ SHIELD 2: DESFASE ANTI-RECOMBINACIÓN EN TOWNSEND
                # =====================================================
                current_dtype = state.positions.dtype
                r_recomb = getattr(self.config, 'recombination_radius', 0.001)
                
                ion_offsets = self.rng.uniform(r_recomb * 1.5, r_recomb * 2.5, size=(new_count, 3))
                ion_offsets *= self.rng.choice([-1, 1], size=(new_count, 3))
                
                ion_positions = (new_positions + ion_offsets).reshape(-1, 3).astype(current_dtype)

                ion_mass = 4.65e-26
                thermal_speed_ion = np.sqrt(constants.BOLTZMANN * self.config.gas_temperature / ion_mass)
                ion_velocities = self.rng.normal(0.0, thermal_speed_ion, size=(new_count, 3)).reshape(-1, 3).astype(current_dtype)
            
                if state.ion_positions is None or state.ion_positions.size == 0:
                    state.ion_positions = ion_positions
                    state.ion_velocities = ion_velocities
                else:
                    state.ion_positions = np.vstack([state.ion_positions, ion_positions])
                    state.ion_velocities = np.vstack([state.ion_velocities, ion_velocities])

        # =====================================================
        # 📡 TELEMETRÍA DE SEGUIMIENTO (PRE-POST RECOMBINACIÓN)
        # =====================================================
        iones_antes = state.ion_positions.shape[0] if state.ion_positions.size > 0 else 0
        
        # Evaluamos la recombinación una ÚNICA vez por frame
        self.check_recombination()
        
        iones_despues = state.ion_positions.shape[0] if state.ion_positions.size > 0 else 0
        
        if iones_antes > 0 or state.ionization_events > 0:
            print(f"📡 [TRACKER] Creados Townsend: {state.ionization_events} | En memoria ANTES de recombinar: {iones_antes} | DESPUÉS: {iones_despues}")
        # =====================================================

        # --- 5. ACTUALIZACIÓN DEL ESTADO FINAL Y MÉTRICAS ---
        state.time += dt
        state.stage = self._infer_stage(state)

        area = (2.0 * self.config.xy_extent) ** 2
        current = 0.0
        if state.positions.size > 0:
            current = particles.estimate_current(state.velocities, constants.ELECTRON_CHARGE, area)

        e_len = int(state.positions.shape[0]) if state.positions.size > 0 else 0
        i_len = int(state.ion_positions.shape[0]) if state.ion_positions.size > 0 else 0

        return state, {"count": e_len, "ion_count": i_len, "current": float(current)}

    def _spawn_electrons_from_collisions(self, collision_points: np.ndarray, count: int) -> np.ndarray:
        if count <= 0 or collision_points.size == 0:
            return np.zeros((0, 3), dtype=float)
        collision_points = np.asarray(collision_points, dtype=float)
        if collision_points.shape[0] != count:
            count = collision_points.shape[0]
            
        directions = particles.random_directional_velocities(count, self.rng, self.config.gas_temperature)
        offsets = directions * 1.0e-6
        positions = collision_points[:count] + offsets
        
        positions[:, 0] = np.clip(positions[:, 0], -self.config.xy_extent, self.config.xy_extent)
        positions[:, 1] = np.clip(positions[:, 1], -self.config.xy_extent, self.config.xy_extent)
        positions[:, 2] = np.clip(positions[:, 2], 0.0, self.config.gap_distance)
        return positions

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
                velocities[upper, axis] *= -0.2
            if np.any(lower):
                positions[lower, axis] = -2.0 * extent - positions[lower, axis]
                velocities[lower, axis] *= -0.2

    def _reflect_z(self, positions: np.ndarray, velocities: np.ndarray) -> None:
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
        if stage in (Stage.AVALANCHE, Stage.EXPONENTIAL_GROWTH):
            return max(count, count * 5)
        if stage in (Stage.SELF_SUSTAINED, Stage.CHARGE_EVOLUTION):
            return max(count, count * 10)
        return count

    def _infer_stage(self, state: SimulationState) -> Stage:
        stage = state.stage
        stage = self._max_stage(stage, Stage.FIELD_ACCELERATION)
        if state.collision_events > 0:
            stage = self._max_stage(stage, Stage.COLLISIONS)
        if getattr(state, 'ionization_events', 0) > 0:
            stage = self._max_stage(stage, Stage.IONIZATION)
        if state.positions.size > 0 and state.positions.shape[0] >= self.initial_count * 2:
            stage = self._max_stage(stage, Stage.AVALANCHE)
        if state.positions.size > 0 and state.positions.shape[0] >= self.initial_count * 8:
            stage = self._max_stage(stage, Stage.EXPONENTIAL_GROWTH)
        if breakdown.is_self_sustained(self.config):
            stage = self._max_stage(stage, Stage.SELF_SUSTAINED)
        if state.time > 0.0:
            stage = self._max_stage(stage, Stage.CHARGE_EVOLUTION)
        return stage

    def _max_stage(self, current: Stage, candidate: Stage) -> Stage:
        if STAGE_ORDER.index(candidate) > STAGE_ORDER.index(current):
            return candidate
        return current

    def add_electron(self, position: list | None = None, velocity: list | None = None) -> None:
        """Inyecta un electrón garantizando una posición inicial baja (Cátodo Z=0) para dar tiempo a colisionar."""
        if self.state is None:
            self.reset(Stage.INITIAL_ELECTRONS, 1)
            return

        if self.state.positions.size > 0 and self.state.positions.shape[0] >= self.config.max_particles:
            return

        # Forzamos la inyección en la base (Z próximo a 0) para que recorra todo el GAP acelerando
        if position is None:
            pos = np.array([[
                self.rng.uniform(-self.config.xy_extent * 0.5, self.config.xy_extent * 0.5),
                self.rng.uniform(-self.config.xy_extent * 0.5, self.config.xy_extent * 0.5),
                self.config.gap_distance * 0.05  # ⚡ Próximo al cátodo
            ]], dtype=float)
        else:
            pos = np.asarray(position, dtype=float).reshape(1, 3)

        if velocity is None:
            vel = particles.random_directional_velocities(1, self.rng, self.config.gas_temperature)
        else:
            vel = np.asarray(velocity, dtype=float).reshape(1, 3)

        if self.state.positions.size == 0:
            self.state.positions = pos
            self.state.velocities = vel
        else:
            self.state.positions = np.vstack([self.state.positions, pos])
            self.state.velocities = np.vstack([self.state.velocities, vel])

    def add_neutral(self, position: list | None = None, velocity: list | None = None) -> None:
        if self.state is None:
            self.reset(Stage.INITIAL_ELECTRONS, 1)
            return

        if self.state.neutral_positions.shape[0] >= self.config.max_particles:
            return

        if position is None:
            pos = np.array([[
                self.rng.uniform(-self.config.xy_extent, self.config.xy_extent),
                self.rng.uniform(-self.config.xy_extent, self.config.xy_extent),
                self.rng.uniform(0.0, self.config.gap_distance)
            ]], dtype=float)
        else:
            pos = np.asarray(position, dtype=float).reshape(1, 3)

        if velocity is None:
            vel = particles.random_thermal_velocities(1, self.rng, self.config.gas_temperature * 0.5)
        else:
            vel = np.asarray(velocity, dtype=float).reshape(1, 3)

        if self.state.neutral_positions.size == 0:
            self.state.neutral_positions = pos
            self.state.neutral_velocities = vel
        else:
            self.state.neutral_positions = np.vstack([self.state.neutral_positions, pos])
            self.state.neutral_velocities = np.vstack([self.state.neutral_velocities, vel])

    def check_recombination(self):
        """Detecta la desionización comparando las matrices de posiciones de NumPy de manera dinámica."""
        state = self.state
        safe_dtype = state.positions.dtype if state.positions.size > 0 else np.float32
        
        if state.positions.size == 0 or state.ion_positions.size == 0:
            state.recombined_pos = np.empty((0, 3), dtype=safe_dtype)
            return

        recombined_this_frame = 0
        e_indices_a_borrar = []
        ion_indices_a_borrar = []
        puntos_de_impacto = []

        radius = getattr(self.config, 'recombination_radius', 0.001)
        threshold_sq = radius ** 2
        electrones_disponibles = np.ones(state.positions.shape[0], dtype=bool)

        for ion_idx, ion_pos in enumerate(state.ion_positions):
            if not np.any(electrones_disponibles):
                break
            
            deltas = state.positions[electrones_disponibles] - ion_pos
            distances_sq = np.sum(deltas * deltas, axis=1)
            
            if distances_sq.size == 0:
                break

            closest_rel_idx = int(np.argmin(distances_sq))
            
            if distances_sq[closest_rel_idx] <= threshold_sq:
                indices_reales = np.where(electrones_disponibles)[0]
                closest_real_idx = indices_reales[closest_rel_idx]
                
                e_indices_a_borrar.append(closest_real_idx)
                ion_indices_a_borrar.append(ion_idx)
                puntos_de_impacto.append(ion_pos)
                recombined_this_frame += 1
                electrones_disponibles[closest_real_idx] = False

        if recombined_this_frame > 0:
            state.recombined_pos = np.array(puntos_de_impacto, dtype=state.positions.dtype)

            mask_e = np.ones(state.positions.shape[0], dtype=bool)
            mask_e[e_indices_a_borrar] = False
            
            mask_ion = np.ones(state.ion_positions.shape[0], dtype=bool)
            mask_ion[ion_indices_a_borrar] = False

            state.positions = state.positions[mask_e]
            state.velocities = state.velocities[mask_e]
            
            state.ion_positions = state.ion_positions[mask_ion]
            state.ion_velocities = state.ion_velocities[mask_ion]
            
            if hasattr(state, 'total_recombinations'):
                state.total_recombinations += recombined_this_frame
            else:
                state.total_recombinations = recombined_this_frame
                
            print(f"--> ¡ÉXITO! Se recombinaron {recombined_this_frame} partículas reales en este paso.")
        else:
            state.recombined_pos = np.empty((0, 3), dtype=safe_dtype)