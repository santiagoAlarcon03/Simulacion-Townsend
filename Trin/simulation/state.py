from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np


class Stage(Enum):
    INITIAL_ELECTRONS = auto()
    FIELD_ACCELERATION = auto()
    COLLISIONS = auto()
    IONIZATION = auto()
    AVALANCHE = auto()
    EXPONENTIAL_GROWTH = auto()
    SELF_SUSTAINED = auto()
    CHARGE_EVOLUTION = auto()
    VISUALIZATION = auto()


STAGE_ORDER = [
    Stage.INITIAL_ELECTRONS,
    Stage.FIELD_ACCELERATION,
    Stage.COLLISIONS,
    Stage.IONIZATION,
    Stage.AVALANCHE,
    Stage.EXPONENTIAL_GROWTH,
    Stage.SELF_SUSTAINED,
    Stage.CHARGE_EVOLUTION,
    Stage.VISUALIZATION,
]


STAGE_LABELS = {
    Stage.INITIAL_ELECTRONS: "Initial electrons",
    Stage.FIELD_ACCELERATION: "Field acceleration",
    Stage.COLLISIONS: "Collisions with gas",
    Stage.IONIZATION: "Impact ionization",
    Stage.AVALANCHE: "Electron avalanche",
    Stage.EXPONENTIAL_GROWTH: "Exponential growth",
    Stage.SELF_SUSTAINED: "Self sustained discharge",
    Stage.CHARGE_EVOLUTION: "Charge and current evolution",
    Stage.VISUALIZATION: "Visualization",
}


@dataclass
class SimulationState:
    positions: np.ndarray
    velocities: np.ndarray
    time: float
    stage: Stage
    collision_events: int = 0
    ionization_events: int = 0
    neutral_positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=float))
    neutral_velocities: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=float))
