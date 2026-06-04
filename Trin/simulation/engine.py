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
        positions, velocities = particles.initialize_electrons(count, self.config, self.rng)
        self.state = SimulationState(
            positions=positions,
            velocities=velocities,
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

        in_bounds = (
            (np.abs(state.positions[:, 0]) <= self.config.xy_extent)
            & (np.abs(state.positions[:, 1]) <= self.config.xy_extent)
            & (state.positions[:, 2] >= 0.0)
            & (state.positions[:, 2] <= self.config.gap_distance)
        )
        state.positions = state.positions[in_bounds]
        state.velocities = state.velocities[in_bounds]

        if state.positions.size == 0:
            state.time += dt
            return state, {"count": 0, "current": 0.0}

        collision_mask = collisions.sample_collisions(
            state.positions.shape[0], self.config.collision_frequency, dt, self.rng
        )
        collisions.scatter_velocities(state.velocities, collision_mask, self.rng)
        state.collision_events = int(collision_mask.sum())

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
