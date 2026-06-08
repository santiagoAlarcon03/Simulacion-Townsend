from typing import Optional

import numpy as np

from config import SimulationConfig
from physics import avalanche, breakdown, collisions, constants, field as field_module, ionization, particles
from simulation.state import SimulationState, Stage, STAGE_ORDER


class SimulationEngine:
    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.field = field_module.ElectricField([0.0, 0.0, config.electric_field])
        self.state: Optional[SimulationState] = None
        self.initial_count = config.initial_particles


        
    def reset(self, stage: Stage, n_particles: int) -> None:
     count = self._initial_count_for_stage(stage, n_particles)

    # Electrones iniciales
     positions, velocities = particles.initialize_electrons(
        count,
        self.config,
        self.rng,
    )

    # Gas neutro inicial
     neutral_positions, neutral_velocities = (
        particles.initialize_neutral_particles(
            500,  # puedes probar 100, 500, 1000, etc.
            self.config,
            self.rng,
        )
    )

     self.state = SimulationState(
        positions=positions,
        velocities=velocities,
        neutral_positions=neutral_positions,
        neutral_velocities=neutral_velocities,
        time=0.0,
        stage=stage,
    )

     self.initial_count = count



    def step(self, dt: float):
        if self.state is None:
            return None, None

        state = self.state
        if state.positions.size == 0:
            state.time += dt
            return state, {"count": 0, "current": 0.0}

        acceleration = particles.acceleration_from_field(
            self.field.vector, constants.ELECTRON_CHARGE, constants.ELECTRON_MASS
        )
        state.velocities = state.velocities + acceleration * dt
        state.positions = state.positions + state.velocities * dt

        if state.neutral_positions.size > 0:
            state.neutral_positions = state.neutral_positions + state.neutral_velocities * dt

        self._reflect_into_domain(state.positions, state.velocities)
        self._reflect_into_domain(state.neutral_positions, state.neutral_velocities)

        if state.positions.size == 0:
            state.time += dt
            return state, {"count": 0, "current": 0.0}

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

        if neutral_collision_mask.any():
            count = int(neutral_collision_mask.sum())
            respawn_positions, respawn_velocities = particles.initialize_neutral_particles(
                count, self.config, self.rng
            )
            state.neutral_positions[neutral_collision_mask] = respawn_positions
            state.neutral_velocities[neutral_collision_mask] = respawn_velocities

        collision_mask = collisions.sample_collisions(
            state.positions.shape[0], self.config.collision_frequency, dt, self.rng
        )
        collisions.scatter_velocities(state.velocities, collision_mask, self.rng)
        state.collision_events += int(collision_mask.sum())

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
                new_count, self.rng, self.config.gas_temperature
            )
            state.positions = np.vstack([state.positions, new_positions])
            state.velocities = np.vstack([state.velocities, new_velocities])

        state.time += dt
        state.stage = self._infer_stage(state)

        area = (2.0 * self.config.xy_extent) ** 2
        current = particles.estimate_current(
            state.velocities, constants.ELECTRON_CHARGE, area
        )

        return state, {"count": int(state.positions.shape[0]), "current": float(current)}

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

    def _reflect_into_domain(self, positions: np.ndarray, velocities: np.ndarray) -> None:
        """Keep particles inside the visual domain by reflecting them on the walls."""
        extent = self.config.xy_extent
        gap = self.config.gap_distance

        for axis in (0, 1):
            upper = positions[:, axis] > extent
            lower = positions[:, axis] < -extent
            if np.any(upper):
                positions[upper, axis] = 2.0 * extent - positions[upper, axis]
                velocities[upper, axis] *= -1.0
            if np.any(lower):
                positions[lower, axis] = -2.0 * extent - positions[lower, axis]
                velocities[lower, axis] *= -1.0

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
        if STAGE_ORDER.index(candidate) > STAGE_ORDER.index(current):
            return candidate
        return current

    def add_electron(self, position: list | None = None, velocity: list | None = None) -> None:
        """Add a single electron to the current simulation state.

        If `position` or `velocity` are not provided, generate them using
        the same initialization strategy as `initialize_electrons`.
        """
        if self.state is None:
            # If there's no state yet, initialize a single electron at default stage
            self.reset(Stage.INITIAL_ELECTRONS, 1)
            return

        if self.state.positions.shape[0] >= self.config.max_particles:
            return

        import numpy as _np

        # create position
        if position is None:
            pos = self._sample_visible_position()
        else:
            pos = _np.asarray(position, dtype=float).reshape(1, 3)

        # create velocity
        if velocity is None:
            # assign a random direction with thermal magnitude
            vel = particles.random_directional_velocities(
                1, self.rng, self.config.gas_temperature
            )
        else:
            vel = _np.asarray(velocity, dtype=float).reshape(1, 3)

        if self.state.positions.size == 0:
            self.state.positions = pos
            self.state.velocities = vel
        else:
            self.state.positions = _np.vstack([self.state.positions, pos])
            self.state.velocities = _np.vstack([self.state.velocities, vel])

    def add_neutral(self, position: list | None = None, velocity: list | None = None) -> None:
        """Add a neutral particle that floats randomly inside the 3D cube."""
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
        """Sample a position that is not too close to existing particles."""
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
        # fallback: place it at the center-top if we fail to find a clear spot
        return np.array([[0.0, 0.0, self.config.gap_distance * 0.9]], dtype=float)
