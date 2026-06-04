from dataclasses import dataclass
from typing import Optional

from simulation.state import Stage


@dataclass
class SimulationConfig:
    dt: float = 1e-10
    frame_dt_ms: int = 33
    gap_distance: float = 0.01
    xy_extent: float = 0.005
    electric_field: float = 1.0e5
    initial_particles: int = 200
    max_particles: int = 20000
    gas_pressure: float = 100.0
    gas_temperature: float = 300.0
    townsend_A: float = 15.0
    townsend_B: float = 365.0
    secondary_emission_gamma: float = 0.02
    ionization_energy_ev: float = 15.6
    collision_frequency: float = 5.0e10
    ionization_probability: float = 0.02
    seed: Optional[int] = 1234
    start_stage: Stage = Stage.INITIAL_ELECTRONS
